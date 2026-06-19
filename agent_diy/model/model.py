import torch
import torch.nn as nn
from agent_diy.conf.conf import Config

def make_fc_layer(in_features, out_features):
    """创建正交初始化的线性层。"""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc

class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "independent_ppo"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION # 15维特征 [cite: 1, 2]
        hidden_dim = 64
        mid_dim = 32
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # Actor骨干网络：决策动作概率
        self.actor_backbone = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )
        self.actor_head = make_fc_layer(mid_dim, action_num)

        # Critic骨干网络：评估状态价值
        self.critic_backbone = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, mid_dim),
            nn.ReLU(),
        )
        self.critic_head = make_fc_layer(mid_dim, value_num)

    def forward(self, obs, inference=False):
        # 两个分支独立计算
        actor_hidden = self.actor_backbone(obs)
        logits = self.actor_head(actor_hidden)

        critic_hidden = self.critic_backbone(obs)
        value = self.critic_head(critic_hidden)
        
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()