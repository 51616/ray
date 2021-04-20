from gym.spaces import Tuple, Discrete, Dict
import logging
import numpy as np
import tree

import ray
from ray.rllib.agents.qmix.mixers import VDNMixer, QMixer
from ray.rllib.agents.qmix.model import RNNModel, _get_size
from ray.rllib.agents.ppo.ppo_torch_policy import ppo_surrogate_loss, vf_preds_fetches,\
ValueNetworkMixin, kl_and_loss_stats, setup_mixins, LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.agents.custom_ppo.default_ppo_policy import DefaultPPOPolicy
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.env.wrappers.group_agents_wrapper import GROUP_REWARDS
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    convert_to_torch_tensor, explained_variance, sequence_mask, \
    convert_to_non_torch_type
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.utils.sgd import minibatches, standardize

# Torch must be installed.
torch, nn = try_import_torch(error=True)

logger = logging.getLogger(__name__)

class VanillaPPOPolicy(DefaultPPOPolicy):
    def __init__(self, obs_space, action_space, config):
        # update policy attr for loss calculation
        self.minibatch_size = config['sgd_minibatch_size']
        self.ppo_epochs = config['ppo_epochs']
        self.dist_class, logit_dim = ModelCatalog.get_action_dist(
                    action_space, config["model"], framework='torch')
        self.model = ModelCatalog.get_model_v2(
                    obs_space=obs_space,
                    action_space=action_space,
                    num_outputs=logit_dim,
                    model_config=config["model"],
                    framework='torch')

        super().__init__(obs_space, action_space, config,
                         model=self.model,
                         loss=ppo_surrogate_loss,
                         action_distribution_class=self.dist_class
                         )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    @override(TorchPolicy)
    def postprocess_trajectory(self,
                                   sample_batch,
                                   other_agent_batches=None,
                                   episode=None):
        '''
        Calculate GAE in postprocess
        '''
        with torch.no_grad():
            # Call super's postprocess_trajectory first.
            # sample_batch = super().postprocess_trajectory(
            #     sample_batch, other_agent_batches, episode)
            return compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)


    @override(TorchPolicy)
    def learn_on_batch(self, postprocessed_batch):
        # Set Model to train mode.
        if self.model:
            self.model.train()

        for k, v in postprocessed_batch.items():
            if 'state_in' in k[:8]:
                # assume all traj has the same length
                postprocessed_batch[k] = np.tile(v, (postprocessed_batch.count//v.shape[0],1))
            # print(k, len(postprocessed_batch[k]))
        postprocessed_batch.seq_lens = None # remove to use .copy()

        c = 0
        for ep in range(self.ppo_epochs):
            for mb in minibatches(postprocessed_batch, self.minibatch_size):
                c += 1
                # pad batch for rnn
                pad_batch_to_sequences_of_same_size(
                    mb,
                    max_seq_len=self.max_seq_len,
                    shuffle=False,
                    batch_divisibility_req=self.batch_divisibility_req,
                    view_requirements=self.view_requirements,
                )
                mb["is_training"] = True
                # minibatch = mb.copy()
                mb['advantages'] = standardize(mb['advantages'])
                minibatch = self._lazy_tensor_dict(mb)
                # compute the loss
                loss = ppo_surrogate_loss(self,self.model,self.dist_class,minibatch)
                # compute gradient
                self.optimizer.zero_grad()
                loss.backward()
                # grad norm
                # apply_grad_clipping(self, self.optimizer, loss)
                if self.config['grad_clip']:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['grad_clip'])
                self.optimizer.step()
                # log stats
                # stats['ppo_loss'] += loss_out.item()
        # stats['ppo_loss'] /= c
        # add more info about the loss
        # TODO: move this to inner loop and use average instead (0)
        # stats.update(kl_and_loss_stats(self, train_batch)) 
        # compute the loss
        
        # log stats
        # stats = {
        #     "loss": loss_out.item(),
        #     'test': 1
        #     # "grad_norm": grad_norm
        #     # if isinstance(grad_norm, float) else grad_norm.item(),
        # }
        # TODO: move this to inner loop and use average instead (0)
        
        stats = kl_and_loss_stats(self, postprocessed_batch)
        return {LEARNER_STATS_KEY: stats}
