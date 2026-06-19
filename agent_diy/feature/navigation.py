import numpy as np
import heapq
from collections import deque

class GlobalMapTracker:
    """动态维护 128x128 全局地图"""
    def __init__(self, map_size=128):
        self.map_size = map_size
        # -1: 未知, 0: 障碍物, 1: 可通行, >1: 高风险高代价区域
        self.grid = np.full((map_size, map_size), -1, dtype=np.int8)
        
    def update_map(self, center_pos, local_map_21x21):
        """利用 21x21 的局部视野更新全局地图"""
        cx, cz = center_pos['x'], center_pos['z']
        for i in range(21):
            for j in range(21):
                gx = cx - 10 + j
                gz = cz - 10 + i
                if 0 <= gx < self.map_size and 0 <= gz < self.map_size:
                    # 仅在未知或原先是普通地形时更新，避免覆盖临时高代价标记
                    if self.grid[gz][gx] <= 1: 
                        self.grid[gz][gx] = local_map_21x21[i][j]

    def find_nearest_frontier(self, start_pos):
        """BFS 寻找最近的边界点"""
        queue = deque([start_pos])
        visited = set([start_pos])
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        while queue:
            cx, cz = queue.popleft()
            if self.grid[cz][cx] == 1:
                for dx, dz in directions:
                    nx, nz = cx + dx, cz + dz
                    if 0 <= nx < self.map_size and 0 <= nz < self.map_size:
                        if self.grid[nz][nx] == -1:
                            return (cx, cz)
                            
                for dx, dz in directions:
                    nx, nz = cx + dx, cz + dz
                    if 0 <= nx < self.map_size and 0 <= nz < self.map_size:
                        if self.grid[nz][nx] >= 1 and (nx, nz) not in visited:
                            visited.add((nx, nz))
                            queue.append((nx, nz))
        return None

class AStarPlanner:
    """A* 寻路算法 (支持 8 方向，带地形代价优化)"""
    def __init__(self):
        self.directions = [
            (1, 0), (1, -1), (0, -1), (-1, -1), 
            (-1, 0), (-1, 1), (0, 1), (1, 1)
        ]
        
    def heuristic(self, current, goal, start=None):
        h = max(abs(current[0] - goal[0]), abs(current[1] - goal[1]))
        if start is not None:
            dx1 = current[0] - goal[0]
            dz1 = current[1] - goal[1]
            dx2 = start[0] - goal[0]
            dz2 = start[1] - goal[1]
            cross = abs(dx1 * dz2 - dx2 * dz1)
            h += cross * 0.001
        return h

    def plan(self, grid, start, goal):
        if start == goal:
            return -1, 0
            
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal, start)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break
                
            for act_idx, (dx, dz) in enumerate(self.directions):
                neighbor = (current[0] + dx, current[1] + dz)
                
                if not (0 <= neighbor[0] < 128 and 0 <= neighbor[1] < 128):
                    continue
                
                cell_value = grid[neighbor[1]][neighbor[0]]
                if cell_value == 0: # 明确的物理障碍或致命区
                    continue
                # 防止斜穿
                if dx != 0 and dz != 0:
                    val_x = grid[current[1]][current[0] + dx]
                    val_z = grid[current[1] + dz][current[0]]
                    if val_x == 0 and val_z == 0:
                        continue # 视为障碍，不加入路径节点
                    
                # 获取地形代价。-1(未知)视为正常代价 1，>1 的数值就是避险代价
                step_cost = cell_value if cell_value > 1 else 1
                tentative_g_score = g_score[current] + step_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = (current, act_idx)
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal, start)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        if goal not in came_from:
            return -1, 999
            
        curr = goal
        path_length = 0
        first_action = -1
        while curr != start:
            prev, act = came_from[curr]
            first_action = act
            curr = prev
            path_length += 1
            
        return first_action, path_length