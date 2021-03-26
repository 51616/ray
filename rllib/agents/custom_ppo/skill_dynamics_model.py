'''Based on https://github.com/google-research/dads'''

from gym import spaces
import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
D = torch.distributions

class SkillDynamics(nn.Module):
    """The default skill dynamics model for DADS"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        options = self.options = config['model']['custom_model_config']
        obs_space = options['dynamics_obs']
        hidden_dim = options['hidden_dim']
        self.num_experts  = options['num_experts']
        num_hiddens = options['num_hiddens']
        self.z_dim = options['z_dim']
        self.num_reps = options['num_reps']
        self.z_range = options['z_range']
        self.z_type = options['z_type']

        obs_shape = obs_space.shape
        if isinstance(options['orig_obs_space'], spaces.Dict):
            obs_shape = options['orig_obs_space']['dynamics_obs'].shape
        self.obs_dim = int(np.prod(obs_shape))
        input_dim = self.obs_dim + self.z_dim
        print(f'dynamics input dim: {input_dim}')

        self.bn_in = nn.BatchNorm1d(self.obs_dim)
        self.bn_target = nn.BatchNorm1d(self.obs_dim, affine=False)

        hiddens = [SlimFC(input_dim,hidden_dim,activation_fn=nn.ReLU,
                          initializer=lambda w: nn.init.orthogonal_(w,1.0))]

        for _ in range(num_hiddens-1):
            hiddens.append(SlimFC(hidden_dim,hidden_dim,activation_fn=nn.ReLU,
                                  initializer=lambda w: nn.init.orthogonal_(w,1.0)))
        self.hiddens = nn.Sequential(*hiddens)

        self.logits = SlimFC(hidden_dim + self.z_dim, self.num_experts,
                            initializer=lambda w: nn.init.orthogonal_(w,0.01)) # nn.Linear(hidden_dim, self.num_experts)
        
        self.means = SlimFC(hidden_dim + self.z_dim, self.num_experts * self.obs_dim,
                            initializer=lambda w: nn.init.orthogonal_(w,0.01)) # nn.Linear(hidden_dim, self.num_experts * self.obs_dim)
        if config.get('dynamics_spectral_norm',None):
            # print(self.hiddens._modules)
            # print(self.hiddens._modules['0']._model._modules['0'])
            self.hiddens._modules['0']._model._modules['0'] = nn.utils.spectral_norm(self.hiddens._modules['0']._model._modules['0'])
            self.hiddens._modules['1']._model._modules['0'] = nn.utils.spectral_norm(self.hiddens._modules['1']._model._modules['0'])
            self.logits._model._modules['0'] = nn.utils.spectral_norm(self.logits._model._modules['0'])
            self.means._model._modules['0'] = nn.utils.spectral_norm(self.means._model._modules['0'])

    def orthogonal_regularization(self):
        reg = 1e-4
        orth_loss = torch.zeros(1)
        layers = [self.logits,self.means]
        if self.config['dynamics_reg_hiddens']:
            layers.append(self.hiddens)
        for layer in layers:
            for name, param in layer.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0])
                    orth_loss = orth_loss + (reg * sym.abs().sum())
        return torch.sum(orth_loss)

    def l2_regularization(self):
        reg = 1e-4
        l2_loss = torch.zeros(1)
        for name, param in self.hiddens.named_parameters():
            if 'bias' not in name:
                l2_loss = l2_loss + (0.5 * reg * torch.sum(torch.pow(param, 2)))
        return torch.sum(l2_loss)


    def forward(self, obs, z, training=False):
        # obs = batch_norm(obs)
        self.bn_in.train(mode=training)
        norm_obs = self.bn_in(obs)
        inp = torch.cat([norm_obs,z],axis=-1)
        x = self.hiddens(inp)
        logits = self.logits(torch.cat([x,z],axis=-1)) # [batch,num_experts]
        means = self.means(torch.cat([x,z],axis=-1))
        means = means.reshape(obs.shape[0], self.num_experts, self.obs_dim)
        return logits, means

    def get_distribution(self, obs, z, training=False):
        logits, means = self.forward(obs,z,training)
        mix = D.Categorical(logits) 
        comp = D.Independent(D.Normal(means, 1.0),1)
        return D.MixtureSameFamily(mix, comp)

    def get_log_prob(self, obs, z, next_obs, training=False):
        gmm = self.get_distribution(obs,z,training)
        self.bn_target.train(mode=training)
        norm_next_obs = self.bn_target(next_obs)
        return gmm.log_prob(norm_next_obs)

    def compute_reward(self, obs, z, next_obs):
        num_reps = self.num_reps if self.z_type=='cont' else self.z_dim
        if self.z_type=='cont':
            alt_obs = obs.repeat(num_reps,1) # [Batch_size*num_reps, obs_dim]
            alt_next_obs = next_obs.repeat(num_reps,1)
            # continuous uniform
            alt_skill = np.random.uniform(self.z_range[0],self.z_range[1],size=[alt_obs.shape[0],self.z_dim]).astype(np.float32)
        elif self.z_type=='discrete':
            alt_obs = obs.repeat(self.z_dim,1)
            alt_next_obs = next_obs.repeat(self.z_dim,1)
            alt_skill = np.tile(np.eye(self.z_dim),[obs.shape[0],1]).astype(np.float32)
        alt_skill = torch.from_numpy(alt_skill)

        log_prob = self.get_log_prob(obs,z,next_obs,training=False) # [Batch_size]
        log_prob = log_prob.reshape(obs.shape[0],1)
        alt_log_prob = self.get_log_prob(alt_obs,alt_skill,alt_next_obs,training=False) # [Batch_size*num_reps]
        # alt_log_prob = torch.cat(torch.split(alt_log_prob, num_reps,dim=0),dim=0) # [Batch_size, num_reps]
        alt_log_prob = alt_log_prob.reshape(obs.shape[0],num_reps) # [Batch_size, num_reps]
        # print(log_prob.shape)
        # print(alt_log_prob.shape)
        diff = alt_log_prob - log_prob

        reward = np.log(num_reps+1) - np.log(1 + np.exp(np.clip(
                diff, -50, 50)).sum(axis=-1))
        # print(reward.shape)
        return reward, {'log_prob':log_prob,'alt_log_prob':alt_log_prob,'num_higher_prob':((-diff)>=0).sum().item()}
