from gym import spaces
import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
D = torch.distributions
class PPODADSAgent(TorchModelV2, nn.Module):
    """Torch version of FastModel (tf)."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.options = options = model_config['custom_model_config']
        self.z_dist = D.Uniform(options['z_range'][0],options['z_range'][1])
        self.z_dim = options['z_dim']
        hidden_dim = options['hidden_dim']
        num_hiddens = options['num_hiddens']

        obs_shape = obs_space.shape
        if isinstance(options.get('orig_obs_space',None), spaces.Dict):
            obs_shape = options['orig_obs_space']['obs'].shape

        input_dim = int(np.prod(obs_shape)) + self.z_dim
        print(f'input dim: {input_dim}')

        hiddens = [SlimFC(input_dim,hidden_dim,activation_fn=nn.ReLU,
                          initializer=normc_initializer(1.0))]
        for _ in range(num_hiddens-1):
            hiddens.append(SlimFC(hidden_dim, hidden_dim, activation_fn=nn.ReLU,
                                  initializer=normc_initializer(1.0)))
        self.hiddens = nn.Sequential(*hiddens)
        self.logits = SlimFC(hidden_dim + self.z_dim, num_outputs,
                             initializer=lambda w: nn.init.orthogonal_(w,0.01))
        self.values = SlimFC(hidden_dim + self.z_dim, 1,
                             initializer=lambda w: nn.init.orthogonal_(w,1.0))

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        assert isinstance(input_dict['obs'], dict)
        obs_flat = input_dict['obs']['obs']
        # else:
        #     obs_flat = input_dict['obs_flat']
        # in_state = torch.stack(state,dim=0)
        # print(state[0])
        # print(obs_flat)
        assert 'z' in input_dict['obs'], 'z must be in input_dict'
        z = input_dict['obs']['z']
        inp = torch.cat([obs_flat,z],axis=-1)
        # inp = obs_flat
        
        # print(inp)
        self._hidden_out = torch.cat([self.hiddens(inp),z],axis=-1)
        logits = self.logits(self._hidden_out)
        return logits, state.copy()

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self.values(self._hidden_out), [-1])