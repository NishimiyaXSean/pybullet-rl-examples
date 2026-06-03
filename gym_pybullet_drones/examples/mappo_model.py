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
    
    # ==================== 新增：模型外科手术 (权重动态补齐) ====================
    def load_state_dict(self, state_dict, strict=True):
        """
        拦截 PyTorch 的权重加载过程。
        当检测到动作空间从 11 扩展到 13 时，保留前 11 个动作的旧权重，并为新动作赋予随机初始权重。
        """
        # 你的 Actor 网络最后一层叫 actor.4 (Linear 层)
        target_weight_key = "actor.4.weight"
        target_bias_key = "actor.4.bias"
        
        if target_weight_key in state_dict:
            old_w = state_dict[target_weight_key]
            old_b = state_dict[target_bias_key]
            
            # 获取当前新模型 (已构建为 13 维) 的权重矩阵形状
            current_w = self.actor[4].weight
            current_b = self.actor[4].bias
            
            # 如果发现旧权重维度 (11) 小于当前维度 (13)
            if old_w.shape[0] < current_w.shape[0]:
                print(f"\n[模型外科手术] 检测到动作空间扩展: {old_w.shape[0]} -> {current_w.shape[0]}")
                print("正在自动将旧技能权重迁移至新网络，并为新动作初始化权重...")
                
                # 1. 复制当前新网络（已经随机初始化好）的权重作为底板
                new_w = current_w.clone().detach()
                new_b = current_b.clone().detach()
                
                # 2. 将旧模型的经验（前 11 个动作的权重）精准覆盖到前 11 个位置
                new_w[:old_w.shape[0], :] = old_w
                new_b[:old_b.shape[0]] = old_b
                
                # 3. 把修补好的权重写回状态字典，骗过 PyTorch 的严格检查
                state_dict[target_weight_key] = new_w
                state_dict[target_bias_key] = new_b
                
        # 放行，交给 PyTorch 底层继续正常加载
        return super().load_state_dict(state_dict, strict=strict)
