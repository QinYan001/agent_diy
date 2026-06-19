#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import numpy as np
import itertools
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model
from agent_diy.feature.navigation import GlobalMapTracker, AStarPlanner

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
         # 分离 Actor 和 Critic 的学习率
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.model.actor_backbone.parameters(), 'lr': 0.0003},
                {'params': self.model.actor_head.parameters(), 'lr': 0.0003},
                {'params': self.model.critic_backbone.parameters(), 'lr': 0.001},
                {'params': self.model.critic_head.parameters(), 'lr': 0.001}
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()

        # 基础记忆与物理规划组件
        self.last_action = -1
        self.map_tracker = GlobalMapTracker()
        self.planner = AStarPlanner()
        
        # 高层事件驱动与决策惯性组件
        self.target_pos = None
        self.target_type = None
        self.commitment_steps = 0  # 决策坚持步数，防止目标震荡
        self.last_pkg_count = 0    # 包裹变动探测

        #  A* 距离缓存池
        self.a_star_cache = {}
        
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1
        self.map_tracker = GlobalMapTracker() 

        # 缓存高层目标和触发重规划的记忆状态
        self.target_pos = None
        self.target_type = None       # 记录当前目标的类型: 'charger', 'warehouse', 'station', 'frontier'
        self.commitment_steps = 0
        self.last_pkg_count = 0
        # 每局重置缓存
        self.a_star_cache.clear()

    def _manhattan_dist(self, pos1, pos2):
        return abs(pos1['x'] - pos2['x']) + abs(pos1['z'] - pos2['z'])

    def _is_unsafe(self, pos, npcs, threshold=4):
        # 评估目标点周围的环境风险
        for npc in npcs:
            dist = max(abs(pos[0] - npc['pos']['x']), abs(pos[1] - npc['pos']['z']))
            if dist <= threshold:
                return True
        return False

    def _get_a_star_dist(self, start_pos, end_pos, is_static=False):
        """带缓存的 A* 寻路。is_static 为 True 时表示终点是不动的（驿站/充电桩）"""
        if start_pos == end_pos:
            return 0
            
        cache_key = (start_pos, end_pos)
        if is_static and cache_key in self.a_star_cache:
            return self.a_star_cache[cache_key]
            
        _, dist = self.planner.plan(self.map_tracker.grid, start_pos, end_pos)
        
        if is_static and dist != -1:
            self.a_star_cache[cache_key] = dist
            
        return dist

    def observation_process(self, env_obs):
        obs = env_obs['observation']
        frame_state = obs['frame_state']
        hero = frame_state['heroes']
        hero_pos = hero['pos']
        organs = frame_state['organs']
        local_map = obs['map_info']
        battery = hero['battery']

        battery_max = hero.get('battery_max', 250)
        
        # 1. 更新全局已知地图
        self.map_tracker.update_map(hero_pos, local_map)

        # 空间膨胀：NPC 注入全局地图
        npcs = frame_state.get('npcs', [])
        temp_obstacles = []

        for npc in npcs:
            nx, nz = npc['pos']['x'], npc['pos']['z']
            # 扩大影响范围至 7x7，形成多级势场
            for dx in range(-3, 4):
                for dz in range(-3, 4):
                    tx, tz = nx + dx, nz + dz
                    if 0 <= tx < self.map_tracker.map_size and 0 <= tz < self.map_tracker.map_size:
                        # 记录原本地图状态以便恢复
                        temp_obstacles.append(((tx, tz), self.map_tracker.grid[tz][tx]))
                        
                        # 计算切比雪夫距离（即方块距离）
                        dist = max(abs(dx), abs(dz))
                        
                        # 按照距离赋予不同的 A* 寻路代价 [cite: 1]
                        if dist <= 1:
                            cost = 0  # 绝对死区（碰撞判定范围）
                        elif dist == 2:
                            cost = 20 # 极高风险区（NPC 下一步可能到达的边缘）
                        elif dist == 3:
                            cost = 10 # 中高风险区
                        else:
                            cost = 5  # 警戒区
                        
                        # 采用最大代价值覆盖（防止多个 NPC 区域重叠时代价被小的冲掉）
                        current_val = self.map_tracker.grid[tz][tx]
                        if current_val != 0: # 不要覆盖真实的建筑障碍物
                            self.map_tracker.grid[tz][tx] = max(current_val, cost)
        
        # 2. 高层决策
        current_pkgs = hero['packages']
        if len(current_pkgs) != self.last_pkg_count:
            self.commitment_steps = 0 # 触发器：装货/卸货完毕，立刻强制重规划
            self.last_pkg_count = len(current_pkgs)

        # 提取所有可以充电的器官 (1: 仓库, 2: 充电桩)
        energy_providers = [o for o in organs if o['sub_type'] in [1, 2]]
        
        # 计算动态电量警戒线 (曼哈顿距离预估 + 安全冗余)
        if energy_providers:
            min_provider_manhattan = min([self._manhattan_dist(hero_pos, p['pos']) for p in energy_providers])
            dynamic_battery_threshold = min_provider_manhattan * 1.5 + 20
        else:
            dynamic_battery_threshold = 0

        need_replan = False
        
        # 消耗决策惯性承诺
        if self.commitment_steps > 0:
            self.commitment_steps -= 1
        else:
            need_replan = True # 承诺时间到期，放开重新思考
            
        # 触发器1：动态电量报警兜底
        if battery <= dynamic_battery_threshold and self.target_type != 'supply':
            if self.target_type == 'station' and battery > self._manhattan_dist(hero_pos, {'x': self.target_pos[0], 'z': self.target_pos[1]}) + 5:
                pass 
            else:
                need_replan = True
                self.commitment_steps = 0
        
        # 触发器2：极低绝对电量救命线
        if battery < 40:
            need_replan = True
            self.commitment_steps = 0

        # 触发器3：当前目标达成或意外变得极度危险
        if self.target_pos:
            dist_to_target = max(abs(hero_pos['x'] - self.target_pos[0]), abs(hero_pos['z'] - self.target_pos[1]))
            if dist_to_target <= 1 and self.target_type == 'supply':
                need_replan = True
                self.commitment_steps = 0
            elif dist_to_target == 0 and self.target_type != 'supply':
                need_replan = True
                self.commitment_steps = 0
            elif self.target_type == 'station' and self._is_unsafe(self.target_pos, npcs):
                need_replan = True
                self.commitment_steps = 0
        else:
            need_replan = True

        has_target = True if self.target_pos is not None else False

        # 仅当触发重规划时，才执行耗时的搜索逻辑
        if need_replan:
            self.target_pos = None
            self.target_type = None
            has_target = False

            start_pos = (hero_pos['x'], hero_pos['z'])

            # --- 优先级 1：紧急补给充电 ---
            if battery <= dynamic_battery_threshold or battery < 40:
                best_dist = float('inf')
                best_provider_pos = None
                
                for p in energy_providers:
                    p_pos = (p['pos']['x'], p['pos']['z'])
                    _, true_dist = self.planner.plan(self.map_tracker.grid, start_pos, p_pos)
                    # 从仓库和充电桩里选一条真实路径最短的
                    if true_dist != -1 and true_dist < best_dist:
                        best_dist = true_dist
                        best_provider_pos = p_pos
                        
                if best_provider_pos:
                    self.target_pos = best_provider_pos
                    self.target_type = 'supply'
                    has_target = True

            # --- 优先级 2：带货送单 (带全链路能耗审查) ---
            if not has_target and len(current_pkgs) > 0:
                # 获取所有需要送达的独立驿站坐标
                stations = []
                for pkg_id in set(current_pkgs): 
                    for organ in organs:
                        if organ['sub_type'] == 3 and organ['config_id'] == pkg_id:
                            st_pos = (organ['pos']['x'], organ['pos']['z'])
                            if not self._is_unsafe(st_pos, npcs):
                                stations.append(st_pos)
                            break 
                
                providers = [(p['pos']['x'], p['pos']['z']) for p in energy_providers]
                
                best_route = None
                max_delivered = -1
                min_cost = float('inf')

                # 穷举所有送货顺序 (最多 3! = 6 种)
                for perm in itertools.permutations(stations):
                    sim_pos = start_pos
                    sim_bat = battery
                    delivered = 0
                    cost = 0
                    valid = True
                    route_nodes = []

                    for st in perm:
                        dist_to_st = self._get_a_star_dist(sim_pos, st, is_static=False)
                        if dist_to_st == -1:
                            valid = False; break

                        # 评估送达后去最近充电桩的兜底代价
                        dist_to_provider = min([self._get_a_star_dist(st, p, is_static=True) for p in providers]) if providers else 0
                        
                        # 安全判断：直接飞过去电量够不够回程？(预留 20 格安全机动冗余)
                        if sim_bat - dist_to_st < dist_to_provider + 10:
                            # 尝试顺路插入充电点
                            best_mid_provider = None
                            min_mid_cost = float('inf')
                            
                            for p in providers:
                                d1 = self._get_a_star_dist(sim_pos, p, is_static=False)
                                if d1 != -1 and sim_bat > d1 + 10: # 当前电量够去这个充电桩
                                    d2 = self._get_a_star_dist(p, st, is_static=True)
                                    if d2 != -1 and 250 - d2 >= dist_to_provider + 20:
                                        if d1 + d2 < min_mid_cost:
                                            min_mid_cost = d1 + d2
                                            best_mid_provider = p
                                            
                            if best_mid_provider is None:
                                valid = False; break # 怎么飞都会坠机
                                
                            # 采纳顺路充电方案
                            d1 = self._get_a_star_dist(sim_pos, best_mid_provider, is_static=False)
                            cost += d1
                            sim_pos = best_mid_provider
                            sim_bat = 250 
                            route_nodes.append(('supply', best_mid_provider))
                            
                            d2 = self._get_a_star_dist(sim_pos, st, is_static=True)
                            cost += d2
                            sim_pos = st
                            sim_bat -= d2
                            delivered += 1
                            route_nodes.append(('station', st))
                        else:
                            # 直飞方案
                            cost += dist_to_st
                            sim_pos = st
                            sim_bat -= dist_to_st
                            delivered += 1
                            route_nodes.append(('station', st))

                    if valid and delivered > 0:
                        if delivered > max_delivered or (delivered == max_delivered and cost < min_cost):
                            max_delivered = delivered
                            min_cost = cost
                            best_route = route_nodes
                                
                    if best_route:
                        self.target_type, self.target_pos = best_route[0]
                        has_target = True
                        self.commitment_steps = self._get_a_star_dist(start_pos, self.target_pos, is_static=False) + 5
                    else:
                        # 发现无论送哪单都会中途坠毁 -> 被迫去寻找补给
                        best_dist = float('inf')
                        best_provider_pos = None
                        for p in energy_providers:
                            p_pos = (p['pos']['x'], p['pos']['z'])
                            _, true_dist = self.planner.plan(self.map_tracker.grid, start_pos, p_pos)
                            if true_dist != -1 and true_dist < best_dist:
                                best_dist = true_dist
                                best_provider_pos = p_pos
                        if best_provider_pos:
                            self.target_pos = best_provider_pos
                            self.target_type = 'supply'
                            has_target = True
            

            # --- 优先级 3：空仓回补货 ---
            if not has_target and len(current_pkgs) == 0:
                best_dist = float('inf')
                best_warehouse_pos = None
                for organ in organs:
                    if organ['sub_type'] == 1:
                        w_pos = (organ['pos']['x'], organ['pos']['z'])
                        _, true_dist = self.planner.plan(self.map_tracker.grid, start_pos, w_pos)
                        if true_dist != -1 and true_dist < best_dist:
                            best_dist = true_dist
                            best_warehouse_pos = w_pos
                if best_warehouse_pos:
                    self.target_pos = best_warehouse_pos
                    self.target_type = 'supply'
                    has_target = True

            # --- 优先级 4：区域探索 ---
            if not has_target:
                frontier_pos = self.map_tracker.find_nearest_frontier((hero_pos['x'], hero_pos['z']))
                if frontier_pos:
                    self.target_pos = frontier_pos
                    self.target_type = 'frontier'
                    has_target = True

        # 3. A* 唯一目标的局部微调避障
        astar_act, astar_dist = -1, 999
        if self.target_pos:
            start_pos = (hero_pos['x'], hero_pos['z'])
            astar_act, astar_dist = self.planner.plan(self.map_tracker.grid, start_pos, self.target_pos)

        # 寻路完成后清除临时障碍物和代价值
        for (tx, tz), old_val in temp_obstacles:
            self.map_tracker.grid[tz][tx] = old_val
            
        # 4. 传入预处理器
        feature, legal_action, reward = self.preprocessor.feature_process(
            env_obs, 
            self.last_action, 
            astar_act, 
            astar_dist, 
            has_target
        )
        
        obs_data = ObsData(feature=list(feature), legal_action=list(legal_action))
        remain_info = {"reward": reward}
        
        return obs_data, remain_info

    def _forward(self, feature, legal_action):
        self.model.set_eval_mode()
        obs_t = (
            torch.tensor(np.array([feature]), dtype=torch.float32)
            .view(1, -1)
            .to(self.device)
        )
        with torch.no_grad():
            logits, value = self.model(obs_t, inference=True)
        return logits.cpu().numpy()[0], value.cpu().numpy()[0]

    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action
        logits, value = self._forward(feature, legal_action)
        legal_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_np)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)
        return [ActData(action=[action], d_action=[d_action], prob=list(prob), value=value)]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        if obs_data is None: return 0
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def _legal_soft_max(self, logits, legal_action):
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        if use_max: return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))