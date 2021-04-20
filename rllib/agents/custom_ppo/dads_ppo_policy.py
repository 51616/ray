'''Based on https://github.com/google-research/dads'''

from gym.spaces import Tuple, Discrete, Dict
import logging
import numpy as np
import tree
from collections import defaultdict

import ray
from ray.rllib.agents.qmix.mixers import VDNMixer, QMixer
from ray.rllib.agents.qmix.model import RNNModel, _get_size
from ray.rllib.agents.ppo.ppo_torch_policy import ppo_surrogate_loss, vf_preds_fetches, ValueNetworkMixin, kl_and_loss_stats
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.env.wrappers.group_agents_wrapper import GROUP_REWARDS
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import _unpack_obs, restore_original_dimensions
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    convert_to_torch_tensor, explained_variance, sequence_mask, \
    convert_to_non_torch_type
from ray.rllib.agents.custom_ppo.skill_dynamics_model import SkillDynamics
from ray.rllib.agents.custom_ppo.model import PPODADSAgent
from ray.rllib.execution.rollout_ops import StandardizeFields
from ray.rllib.utils.sgd import minibatches, standardize


# Torch must be installed.
torch, nn = try_import_torch(error=True)

logger = logging.getLogger(__name__)

class DADSPPOPolicy(TorchPolicy):
    def __init__(self, obs_space, action_space, config):

        # update policy attr for loss calculation
        self.framework = config['framework'] = 'torch'
        self.kl_coeff = config['kl_coeff']
        self.kl_target = config['kl_target']
        self.entropy_coeff = config['entropy_coeff']
        self.cur_lr = config['lr']
        # setup ._value() for gae computation
        self.setup_value(config)
        self.dist_class, logit_dim = ModelCatalog.get_action_dist(
                    action_space, config["model"], framework='torch')

        self.model = PPODADSAgent(obs_space, action_space, logit_dim,
                                  config['model'], name='ppo_dads_agent')

        super().__init__(obs_space, action_space, config,
                         model=self.model,
                         loss=None, # we calculate the loss inside learn_on_batch
                         action_distribution_class=self.dist_class
                         )
        # Merge Model's view requirements into Policy's.
        self.view_requirements.update(self.model.view_requirements)
        # Perform test runs through postprocessing- and loss functions.
        self._initialize_loss_from_dummy_batch(
            auto_remove_unneeded_view_reqs=True,
            stats_fn=None,
        )
        self.ppo_opt = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.minibatch_size = config['sgd_minibatch_size']
        self.ppo_epochs = config['ppo_epochs']
        self.dynamics_epochs = config['dynamics_epochs']
        self.use_dynamics = config['use_dynamics']
        # create dynamics model
        if self.use_dynamics:
            self.dynamics = SkillDynamics(self.config)
            self.dynamics_opt = torch.optim.Adam(self.dynamics.parameters(), lr=self.config["dynamics_lr"])

    @override(TorchPolicy)
    def postprocess_trajectory(self,
                                   sample_batch,
                                   other_agent_batches=None,
                                   episode=None):
        return sample_batch

    @override(TorchPolicy)
    def compute_actions_from_input_dict(self,
                                        input_dict,
                                        explore=None,
                                        timestep=None,
                                        episodes=None,
                                        **kwargs):

        explore = explore if explore is not None else self.config["explore"]
        with torch.no_grad():
            input_dict = self._lazy_tensor_dict(input_dict)
            # state_batches for RNN
            state_batches = [
                input_dict[k] for k in input_dict.keys() if "state_in" in k[:8]
            ]
            seq_lens = np.array([1] * len(input_dict["obs"])) if state_batches else None
            dist_inputs, state_out = self.model(input_dict, state_batches, seq_lens)
            action_dist = self.dist_class(dist_inputs, self.model)
            # Get the exploration action from the forward results.
            actions, logp = \
                self.exploration.get_exploration_action(
                    action_distribution=action_dist,
                    timestep=timestep,
                    explore=explore)
            # add extra info to the trajectory
            extra_info = {}
            # get values from the critic after doing inference for the actions
            extra_info[SampleBatch.VF_PREDS] = self.model.value_function()
            # Action-dist inputs.
            if dist_inputs is not None:
                extra_info[SampleBatch.ACTION_DIST_INPUTS] = dist_inputs

            # Action-logp and action-prob.
            if logp is not None:
                extra_info[SampleBatch.ACTION_PROB] = \
                    torch.exp(logp.float())
                extra_info[SampleBatch.ACTION_LOGP] = logp

            # Update our global timestep by the batch size.
            self.global_timestep += len(input_dict[SampleBatch.CUR_OBS])

            return convert_to_non_torch_type((actions, state_out, extra_info))


    @override(TorchPolicy)
    def learn_on_batch(self, train_batch):
        # print(type(train_batch))
        # Turn the values into tensors
        # train_batch_tensor = self._lazy_tensor_dict(train_batch)
        # train_batch_tensor = train_batch_tensor
        # restore_original_dimensions()
        # print(train_batch_tensor.keys())
        # update the skill dynamics

        # Set Model to train mode.
        if self.model:
            self.model.train()
        if self.dynamics:
            self.dynamics.train()
        

        stats = defaultdict(int)
        if self.use_dynamics:
            c = 0
            for ep in range(self.dynamics_epochs):
                for mb in minibatches(train_batch, self.minibatch_size): # minibatches(train_batch.copy(), self.minibatch_size)
                    c += 1
                    mb["is_training"] = True
                    minibatch = self._lazy_tensor_dict(mb)

                    obs = _unpack_obs(minibatch['obs'],
                                        self.model.options['orig_obs_space'],torch)
                    next_obs = _unpack_obs(minibatch['new_obs'],
                                        self.model.options['orig_obs_space'],torch)
                    dynamics_obs = obs['dynamics_obs']
                    next_dynamics_obs = next_obs['dynamics_obs'] - obs['dynamics_obs']
                    z = obs['z']

                    log_prob = self.dynamics.get_log_prob(dynamics_obs,z,next_dynamics_obs,training=True)
                    dynamics_loss = -torch.mean(log_prob)
                    orth_loss = self.dynamics.orthogonal_regularization()
                    l2_loss = self.dynamics.l2_regularization()
                    if self.config['dynamics_orth_reg']:
                        dynamics_loss += orth_loss
                    if self.config['dynamics_l2_reg'] and not self.config['dynamics_spectral_norm']:
                        dynamics_loss += l2_loss
                    self.dynamics_opt.zero_grad()
                    dynamics_loss.backward()
                    if self.config['grad_clip']:
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.dynamics.parameters(), self.config['grad_clip'])
                    self.dynamics_opt.step()
                    stats['dynamics_loss'] += dynamics_loss.item()
                    stats['orth_loss'] += orth_loss.item()
                    stats['l2_loss'] += l2_loss.item()
            stats['dynamics_loss'] /= c
            stats['orth_loss'] /= c
            stats['l2_loss'] /= c

            self.dynamics.eval()
            # compute intrinsic reward
            with torch.no_grad():
                batch = self._lazy_tensor_dict(train_batch)
                obs = _unpack_obs(batch['obs'],
                                        self.model.options['orig_obs_space'],torch)
                next_obs = _unpack_obs(batch['new_obs'],
                                    self.model.options['orig_obs_space'],torch)
                z = obs['z']
                dynamics_obs = obs['dynamics_obs']
                next_dynamics_obs = next_obs['dynamics_obs'] - obs['dynamics_obs']
                
                dads_reward, info = self.dynamics.compute_reward(dynamics_obs,z,next_dynamics_obs)
                dads_reward = self.config['dads_reward_scale'] * dads_reward.numpy()
                # # replace the reward column in train_batch
                # print(train_batch['rewards'].shape)
                train_batch['rewards'] = dads_reward
                stats['avg_dads_reward'] = dads_reward.mean()
                stats['num_skills_higher_prob'] = info['num_higher_prob']

        # calculate GAE for dads reward here?
        trajs = train_batch.split_by_episode()
        processed_trajs = []
        for traj in trajs:
            processed_trajs.append(compute_gae_for_sample_batch(self,traj))
        batch = SampleBatch.concat_samples(processed_trajs)
        
        # train_batch = compute_gae_for_sample_batch(self, self._lazy_numpy_dict(train_batch))
        # train_batch = self._lazy_tensor_dict(train_batch)
        # update agent using RL algo
        # split to minibatches
        c = 0
        for ep in range(self.ppo_epochs):
            # batch.shuffle()
            for mb in minibatches(batch, self.minibatch_size):
                c += 1
                mb["is_training"] = True
                # minibatch = mb.copy()
                mb['advantages'] = standardize(mb['advantages'])
                minibatch = self._lazy_tensor_dict(mb)
                # compute the loss
                loss_out = ppo_surrogate_loss(self,self.model,self.dist_class,minibatch)
                # compute gradient
                self.ppo_opt.zero_grad()
                # the learning_rate is already used in ppo_surrogate_loss
                loss_out.backward()
                # grad norm
                if self.config['grad_clip']:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['grad_clip'])
                self.ppo_opt.step()
                # log stats
                stats['ppo_loss'] += loss_out.item()
        stats['ppo_loss'] /= c
        # add more info about the loss
        stats.update(kl_and_loss_stats(self, train_batch)) 

                #  {
                #     "loss": loss_out.item(),
                #     'test': 1
                #     # "grad_norm": grad_norm
                #     # if isinstance(grad_norm, float) else grad_norm.item(),
                # }
        return {LEARNER_STATS_KEY: stats}

    def setup_value(self, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.
            if config["_use_trajectory_view_api"]:
                def value(**input_dict):
                    model_out, _ = self.model.from_batch(
                        convert_to_torch_tensor(input_dict, self.device),
                        is_training=False)
                    # [0] = remove the batch dim.
                    return self.model.value_function()[0]
        else:
            def value(*args, **kwargs):
                return 0.0

        self._value = value