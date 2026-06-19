import numpy as np
import math
import collections
from agent_diy.conf.conf import Config

class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.last_dist_to_target = 999
        self.history_action = -1    # 历史动作，抑制抖动
        self.last_pos = None    # 上一步坐标
        self.trajectory_queue = collections.deque(maxlen=5)     # 短时轨迹记忆

        # 记录上一步的电量，初始值设为最大电量 250
        self.last_battery = 250
        self.min_battery_since_last_charge = 250

    def feature_process(self, env_obs, last_action, astar_action, astar_dist, has_target):
        self.step_no += 1
        frame_state = env_obs['observation']['frame_state']
        hero = frame_state['heroes']
        
        # 1. 基础状态归一化
        battery_ratio = hero['battery'] / max(1, hero['battery_max'])
        package_ratio = len(hero['packages']) / 3.0
        
        # 2. A* 导航特征 (Feature Injection核心)
        astar_guidance = np.zeros(8, dtype=np.float32)
        if astar_action != -1:
            astar_guidance[astar_action] = 1.0
            
        dist_norm = min(astar_dist / 128.0, 1.0) 
        
        # 3. 官方无人机(NPC)斥力场特征
        npcs = frame_state.get('npcs', [])
        min_npc_dist = 999
        npc_dx, npc_dz = 0, 0
        
        for npc in npcs:
            # 使用切比雪夫距离评估威胁程度
            dist = max(abs(npc['pos']['x'] - hero['pos']['x']), abs(npc['pos']['z'] - hero['pos']['z']))
            if dist < min_npc_dist:
                min_npc_dist = dist
                # 计算相对方向向量，并用 sign 函数提取纯方向 (-1, 0, 1)
                npc_dx = np.sign(npc['pos']['x'] - hero['pos']['x'])
                npc_dz = np.sign(npc['pos']['z'] - hero['pos']['z'])
                
        # 斥力场强度：距离越近，斥力越大。超过 4 格则视为安全
        repulsion_magnitude = 0.0
        if min_npc_dist <= 4:
            # 距离为 1 时(碰撞边缘)，斥力为 1.0；距离 4 时，斥力为 0.2
            repulsion_magnitude = 1.0 / max(min_npc_dist, 1.0)
            
        npc_features = [repulsion_magnitude, npc_dx, npc_dz]
        
        # 4. 拼接特征 (维度变更为：2 + 8 + 1 + 1 + 3 = 15维)
        feature = np.concatenate([
            [battery_ratio, package_ratio], 
            astar_guidance, 
            [dist_norm],
            [1.0 if has_target else 0.0],
            npc_features  # 注入 NPC 斥力特征
        ]).astype(np.float32)

        # 5. 合法动作掩码 (防止撞墙 + 防止主动撞击动态 NPC)
        local_map = env_obs['observation']['map_info']
        legal_action = np.ones(Config.ACTION_NUM, dtype=np.float32)
        directions = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]
        
        npc_positions = [(npc['pos']['x'], npc['pos']['z']) for npc in npcs]

        for act, (dx, dz) in enumerate(directions):
            # 屏蔽撞墙
            if local_map[10+dz][10+dx] == 0:
                legal_action[act] = 0.0 
                continue 
            # 防止斜穿
            if dx != 0 and dz != 0:
                # 检查相邻的两个正交方向的格子
                val_x = local_map[10][10+dx]
                val_z = local_map[10+dz][10]
                # 如果相邻的两堵墙都是障碍物，说明是死角，禁止斜穿
                if val_x == 0 and val_z == 0:
                    legal_action[act] = 0.0
                    continue
                
            # 允许极限操作
            next_x = hero['pos']['x'] + dx
            next_z = hero['pos']['z'] + dz
            for nx, nz in npc_positions:
                if max(abs(next_x - nx), abs(next_z - nz)) <= 1:
                    legal_action[act] = 0.0
                    break

        self.min_battery_since_last_charge = min(self.min_battery_since_last_charge, hero['battery'])

        # 6. 奖励塑造 (Dense Reward)
        reward = 0.0
        
        # 投递成功奖励
        if hasattr(self, 'last_delivered') and hero['delivered'] > self.last_delivered:
            reward += 10.0
        self.last_delivered = hero['delivered']

        # 充电成功
        if hero['battery'] > self.last_battery + 50:
            # 消耗了电量才给奖励
            if self.min_battery_since_last_charge <= 100:
                reward += 5.0  
            # 充满电后，重置最低水位线
            self.min_battery_since_last_charge = hero['battery']
            
        self.last_battery = hero['battery']
        
        # 靠近目标的小奖励 (势能奖励)
        if has_target and astar_dist < self.last_dist_to_target:
            reward += 0.05
        elif has_target and astar_dist > self.last_dist_to_target:
            reward -= 0.05
        self.last_dist_to_target = astar_dist
        
        # 转向惩罚逻辑
        if self.history_action != -1 and last_action != -1:
            if self.history_action != last_action:
                reward -= 0.02 
        self.history_action = last_action
        
        # 动态障碍物规避
        if min_npc_dist <= 1:
            reward -= 20.0  # 距离 <= 1 死亡惩罚
        else:   # 指数衰减
            alpha = 3.0 
            beta = 2.0
            reward -= alpha * math.exp(-beta * (min_npc_dist - 1))

        # 智能防停滞、防震荡与转向逻辑
        current_pos = (hero['pos']['x'], hero['pos']['z'])
        
        # 防停滞惩罚(撞墙检测)
        if self.last_pos == current_pos:
            reward -= 0.1  
            
        # 防震荡惩罚(历史轨迹重合检测)
        elif current_pos in self.trajectory_queue:
            # 给予中等惩罚，逼迫探索
            reward -= 0.05 
            # 此时免除转向惩罚，鼓励它打大方向盘脱困
        
        else:
            reward+=0
            
                    
        # 更新状态与记忆
        self.last_pos = current_pos
        self.history_action = last_action
        # 压入记忆队列
        self.trajectory_queue.append(current_pos)

        # 生存与耗时惩罚
        reward -= 0.01

        return feature, legal_action, [reward]