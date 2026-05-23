import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class MAPPOModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 1. 解析我们在 marl_env 中定义的 Dict 空间维度
        # local_obs 维度是 19， global_state 维度是 26
        local_obs_dim = obs_space.original_space["obs"].shape[0]
        global_state_dim = obs_space.original_space["global_state"].shape[0]

        # 2. 构建 Actor (策略网络) - 也就是无人机的大脑
        # 它只允许看到局部雷达信息 (local_obs_dim)
        self.actor = nn.Sequential(
            nn.Linear(local_obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, num_outputs) # num_outputs 是连续动作空间决定的高斯分布参数
        )

        # 3. 构建 Critic (价值网络) - 也就是全知的上帝
        # 它直接读取毫无遮盖的全局物理态势 (global_state_dim)
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1) # 最终输出一个标量：当前局势的胜率打分 (Value)
        )

        # 内部缓存，用于给 RLlib 传递 Value
        self._last_value = None

    def forward(self, input_dict, state, seq_lens):
        """前向传播：每次环境 step 后，RLlib 会把观测字典送到这里"""
        
        # 将输入字典拆包
        local_obs = input_dict["obs"]["obs"]
        global_state = input_dict["obs"]["global_state"]

        # 让 Critic 计算当前局势得分并缓存起来
        self._last_value = self.critic(global_state).squeeze(1)

        # 让 Actor 根据局部观测输出飞行推杆动作
        action_logits = self.actor(local_obs)

        return action_logits, state

    def value_function(self):
        """RLlib 的底层 PPO 算法会调用这个函数来获取局势打分"""
        return self._last_value