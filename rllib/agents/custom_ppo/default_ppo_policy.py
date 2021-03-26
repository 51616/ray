from gym.spaces import Tuple, Discrete, Dict
import logging
import numpy as np
import tree

import ray
from ray.rllib.agents.ppo.ppo_torch_policy import ppo_surrogate_loss, vf_preds_fetches,\
ValueNetworkMixin, kl_and_loss_stats, setup_mixins, LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin
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
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    convert_to_torch_tensor, explained_variance, sequence_mask, \
    convert_to_non_torch_type
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size


# Torch must be installed.
torch, nn = try_import_torch(error=True)

logger = logging.getLogger(__name__)

class DefaultPPOPolicy(TorchPolicy, LearningRateSchedule, EntropyCoeffSchedule,
                   KLCoeffMixin, ValueNetworkMixin):
    def __init__(self, obs_space, action_space, config,
                 model, loss, action_distribution_class):
        # update policy attr for loss calculation
        self.framework = config['framework'] = 'torch'
        self.kl_coeff = config['kl_coeff']
        self.kl_target = config['kl_target']
        self.entropy_coeff = config['entropy_coeff']
        self.cur_lr = config['lr']
        # setup ._value() for gae computation
        # self.setup_value(config)
        # self.dist_class, logit_dim = ModelCatalog.get_action_dist(
        #             action_space, config["model"], framework='torch')
        assert getattr(self, 'model',None), f'The agent\' model has to be initialized before this.'
        # self.model = ModelCatalog.get_model_v2(
        #             obs_space=obs_space,
        #             action_space=action_space,
        #             num_outputs=logit_dim,
        #             model_config=config["model"],
        #             framework='torch')

        super().__init__(obs_space, action_space, config,
                         model=self.model,
                         loss=ppo_surrogate_loss,
                         action_distribution_class=self.dist_class,
                         max_seq_len=config['model']['max_seq_len']
                         )
        
        # Merge Model's view requirements into Policy's.
        self.view_requirements.update(self.model.view_requirements)
        # init mixins
        setup_mixins(self, obs_space, action_space, config)
        # Perform test runs through postprocessing- and loss functions.
        self._initialize_loss_from_dummy_batch(
            auto_remove_unneeded_view_reqs=True,
            stats_fn=kl_and_loss_stats,
        )
        self.global_timestep = 0
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    @override(TorchPolicy)
    def postprocess_trajectory(self,
                                   sample_batch,
                                   other_agent_batches=None,
                                   episode=None):
        # '''
        # Calculate GAE in postprocess
        # '''
        # with torch.no_grad():
        #     # Call super's postprocess_trajectory first.
        #     # sample_batch = super().postprocess_trajectory(
        #     #     sample_batch, other_agent_batches, episode)
        #     return compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)
        raise NotImplementedError()

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
            self._is_recurrent = state_batches is not None and state_batches != []
            self.model.eval()
            # print(len(input_dict['obs']))

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


    # @override(TorchPolicy)
    # def learn_on_batch(self, postprocessed_batch):
    #     raise NotImplementedError()
        
        # pad_batch_to_sequences_of_same_size(
        #     postprocessed_batch,
        #     max_seq_len=self.max_seq_len,
        #     shuffle=False,
        #     batch_divisibility_req=self.batch_divisibility_req,
        #     view_requirements=self.view_requirements,
        # )

        # # Set Model to train mode.
        # if self.model:
        #     self.model.train()
        # # Turn the values into tensors
        # postprocessed_batch["is_training"] = True
        # train_batch = self._lazy_tensor_dict(postprocessed_batch)
        # print(postprocessed_batch.keys())
        # print(postprocessed_batch['agent_index'].shape)
        # print(train_batch['obs'].shape)
        # # print(train_batch[''])
        # # compute the loss
        # loss_out = ppo_surrogate_loss(self,self.model,self.dist_class,train_batch)
        # # compute gradient
        # self.optimizer.zero_grad()
        # loss_out.backward()
        # # grad norm
        # if self.config['grad_clip']:
        #     grad_norm = nn.utils.clip_grad_norm_(
        #         self.model.parameters(), self.config['grad_clip'])
        # self.optimizer.step()
        # # log stats
        # # stats = {
        # #     "loss": loss_out.item(),
        # #     'test': 1
        # #     # "grad_norm": grad_norm
        # #     # if isinstance(grad_norm, float) else grad_norm.item(),
        # # }
        # stats = kl_and_loss_stats(self, train_batch)
        # return {LEARNER_STATS_KEY: stats}

    # def setup_value(self, config):
    #     # When doing GAE, we need the value function estimate on the
    #     # observation.
    #     if config["use_gae"]:
    #         # Input dict is provided to us automatically via the Model's
    #         # requirements. It's a single-timestep (last one in trajectory)
    #         # input_dict.
    #         if config["_use_trajectory_view_api"]:
    #             def value(**input_dict):
    #                 model_out, _ = self.model.from_batch(
    #                     convert_to_torch_tensor(input_dict, self.device),
    #                     is_training=False)
    #                 # [0] = remove the batch dim.
    #                 return self.model.value_function()[0]
    #     else:
    #         def value(*args, **kwargs):
    #             return 0.0

    #     self._value = value
