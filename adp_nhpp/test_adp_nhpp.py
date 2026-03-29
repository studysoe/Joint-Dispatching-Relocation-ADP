# ========== 第一部分：核心数据结构 ==========

import heapq
import random
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Deque, Any
from enum import Enum
import numpy as np
from collections import deque
import os
import pandas as pd
from datetime import datetime, date
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# ========== 1. 枚举类型定义 ==========

class PassengerType(Enum):
    """乘客类型枚举"""
    EXPRESS = "E"     # 快车专用乘客
    DUAL = "D"        # 双类型兼容乘客
    PREMIUM = "P"     # 专车专用乘客

class CarState(Enum):
    """车辆状态枚举"""
    IDLE = "idle"              # 空闲
    OCCUPIED = "occupied"      # 占用中（载客）
    RELOCATING = "relocating"  # 重新调度中（空驶）

class DecisionType(Enum):
    """决策类型枚举"""
    DISPATCHING = "dispatching"    # 派单决策
    REBALANCING = "rebalancing"    # 重新平衡决策

# ========== 2. 数据类定义 ==========

@dataclass
class Passenger:
    """乘客数据类"""
    passenger_id: int           # 乘客ID
    arrival_logic_time: float   # 到达逻辑时间（秒）
    origin: int                # 起点区域
    destination: int           # 终点区域
    passenger_type: PassengerType  # 乘客类型
    
    def __str__(self) -> str:
        return (f"Passenger {self.passenger_id}: Type={self.passenger_type.value}, "
                f"Origin={self.origin}, Destination={self.destination}, "
                f"ArrivalTime={self.arrival_logic_time:.2f}s")

@dataclass
class CarOnTheWay:
    """在途车辆数据类（用于优先队列）"""
    arrival_logic_time: float   # 到达逻辑时间
    origin: int                # 起点
    destination: int           # 终点
    vehicle_type: str          # 车辆类型："E"=快车，"P"=专车
    car_state: CarState        # 车辆状态
    
    def __lt__(self, other):
        # 用于优先队列排序：到达时间越早，优先级越高
        return self.arrival_logic_time < other.arrival_logic_time
    
    def __str__(self) -> str:
        return (f"CarOnTheWay: Type={self.vehicle_type}, State={self.car_state.value}, "
                f"Origin={self.origin}, Destination={self.destination}, "
                f"ArrivalTime={self.arrival_logic_time:.2f}s")

@dataclass
class NetworkStats:
    """网络统计信息数据类"""
    total_express_idle: int      # 空闲快车总数
    total_premium_idle: int      # 空闲专车总数
    total_express_occupied: int  # 占用快车总数
    total_premium_occupied: int  # 占用专车总数
    total_express_relocating: int  # 重新调度快车总数
    total_premium_relocating: int  # 重新调度专车总数
    arrival_queue_size: int      # 在途车辆队列大小

# ========== 3. 状态类定义 ==========

@dataclass
class PreDecisionState:
    """
    获取完整的决策前状态向量 S_t
    
    Args:
        network: 网络对象，用于获取车辆分布
        
    Returns:
        决策前状态向量 S_t
    """
    state_type: DecisionType     # 决策类型
    passenger_info: Optional[Tuple[int, int]] = None  # 乘客信息：(起点, 终点)，仅用于派单决策
    vehicle_completion_info: Optional[Tuple[int, str]] = None  # 车辆完成信息：(区域, 车辆类型)，仅用于重新平衡决策
    
    def get_state_vector(self, network) -> np.ndarray:
        """获取决策前状态向量 S_t，完全按照论文4.1节编码"""
        n_nodes = network.n_nodes
        vector = []
        
        # 1. 乘客信息 S_t^1（论文4.1节(i)）
        if self.passenger_info:
            origin, destination = self.passenger_info
            od_vector = np.zeros(n_nodes * n_nodes)
            index = origin * n_nodes + destination
            od_vector[index] = 1
            vector.extend(od_vector)
        else:
            vector.extend([0] * (n_nodes * n_nodes))
        
        # 2. 车辆完成信息 S_t^3（论文4.1节(iii)）
        if self.vehicle_completion_info:
            zone, vehicle_type = self.vehicle_completion_info
            # 按照论文：长度为2*|I|的向量
            completion_vector = np.zeros(2 * n_nodes)
            if vehicle_type == "E":
                completion_vector[zone] = 1.0  # Express部分
            else:  # "P"
                completion_vector[n_nodes + zone] = 1.0  # Premium部分
            vector.extend(completion_vector)
        else:
            vector.extend([0] * (2 * n_nodes))
        
        # 3. 车辆分布向量 S_t^2（论文4.1节(ii)）
        vehicle_dist_vector = network.get_vehicle_distribution_vector()
        vector.extend(vehicle_dist_vector)
        
        return np.array(vector, dtype=np.float32)
    
    def __str__(self) -> str:
        info = f"PreDecisionState: Type={self.state_type.value}"
        if self.passenger_info:
            origin, destination = self.passenger_info
            info += f", Passenger: O={origin}, D={destination}"
        if self.vehicle_completion_info:
            zone, v_type = self.vehicle_completion_info
            info += f", VehicleCompletion: Zone={zone}, Type={v_type}"
        return info

@dataclass  
class PostDecisionState:
    """决策后状态 S_t^a - 只包含车辆分布"""
    vehicle_distribution_vector: np.ndarray  # 车辆分布向量
    
    def get_state_vector(self) -> np.ndarray:
        """获取决策后状态向量（用于神经网络V_θ输入）"""
        # S_t^a 就是车辆分布向量本身
        return self.vehicle_distribution_vector
    
    def __str__(self) -> str:
        return f"PostDecisionState: Vector shape={self.vehicle_distribution_vector.shape}"

# ========== 4. 收益计算器类 ==========

class RevenueCalculator:
    """收益计算器类"""
    
    def __init__(self):
        # 定价参数（基于滴滴2023年定价）
        self.express_base_fare = 10.0  # 快车基础费（元）
        self.premium_base_fare = 15.0  # 专车基础费（元）
        self.express_rate_per_km = 2.40  # 快车每公里费率（元/公里）
        self.premium_rate_per_km = 3.60  # 专车每公里费率（元/公里）
        self.average_speed = 25.0  # 平均速度（公里/小时）
        self.relocation_cost_factor = 0.2  # 重新平衡成本系数
        
    def calculate_trip_revenue(self, 
                            origin: int, 
                            destination: int, 
                            vehicle_type: str,
                            travel_time_seconds: float) -> float:
        """
        计算行程收益 - 使用实际行程时间
        
        Args:
            origin: 起点区域
            destination: 终点区域
            vehicle_type: 车辆类型 ("E"或"P")
            travel_time_seconds: 实际行程时间（秒），应该>0
            
        Returns:
            行程收益（元）
        """
        # 只检查负值，不检查0值（因为即使是区域内出行，时间也应该>0）
        if travel_time_seconds < 0:
            return 0.0
            
        # 将行程时间转换为小时
        travel_time_hours = travel_time_seconds / 3600.0
        
        # 计算行程距离（公里）: 距离 = 速度 × 时间
        distance_km = self.average_speed * travel_time_hours
        
        # 计算收益
        if vehicle_type == "E":  # 快车
            revenue = self.express_base_fare + self.express_rate_per_km * distance_km
        else:  # 专车
            revenue = self.premium_base_fare + self.premium_rate_per_km * distance_km
        
        # 确保收益非负
        revenue = max(0.0, revenue)
        
        return revenue 
    
    def calculate_relocation_cost(self,
                                origin: int,
                                destination: int,
                                vehicle_type: str,
                                travel_time_seconds: float) -> float:
        """
        计算重新平衡成本 - 使用实际行程时间
        
        注意：重新平衡时，如果起终点相同，成本应该为0（留在原地没有成本）
        
        Args:
            origin: 起点区域
            destination: 目标区域
            vehicle_type: 车辆类型 ("E"或"P")
            travel_time_seconds: 实际行程时间（秒）
                
        Returns:
            重新平衡成本（元），负值表示成本
        """
        if origin == destination:
            # 留在原地：没有行程时间，也没有成本
            return 0.0
            
        # 只检查负值
        if travel_time_seconds < 0:
            return 0.0
                
        # 重新平衡成本 = 重新平衡系数 × 对应行程收益
        trip_revenue = self.calculate_trip_revenue(
            origin, destination, vehicle_type, travel_time_seconds
        )
        cost = -self.relocation_cost_factor * trip_revenue
        
        return cost

# ========== 5. 动作评估结果类 ==========

@dataclass
class ActionEvaluation:
    """动作评估结果，包含Q值计算所需的所有信息"""
    action: Any  # 具体动作
    travel_time_seconds: float  # 生成的行程时间
    estimated_immediate_reward: float  # 估计的即时收益/成本
    post_decision_state: PostDecisionState  # 决策后状态 S_t^a
    
    def __str__(self) -> str:
        return (f"ActionEvaluation: TravelTime={self.travel_time_seconds:.1f}s, "
                f"ImmediateReward={self.estimated_immediate_reward:.2f}")

# ========== 6. 神经网络定义 ==========

class PostDecisionValueNetwork(nn.Module):
    """决策后状态价值网络 V_θ(S^a) — 与训练代码架构一致"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super(PostDecisionValueNetwork, self).__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.0)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x

# ========== 7. Network类 ==========

class Network:
    """网络类，管理节点间的行程时间和车辆状态"""
    
    def __init__(self, n_nodes: int, express_num: int, premium_num: int, travel_matrix: np.ndarray):
        """
        初始化网络模型
        
        Args:
            n_nodes: 区域数量
            express_num: 快车数量
            premium_num: 专车数量
            travel_matrix: 行程时间矩阵（分钟），包括对角线上区域内行程时间
        """
        self.n_nodes = n_nodes
        self.express_num = express_num
        self.premium_num = premium_num
        self.total_cars = express_num + premium_num
        
        # 行程时间矩阵（分钟）- 包括对角线上区域内行程时间
        self.travel_matrix = travel_matrix
        
        # 行程率参数存储（用于时间生成）
        self.travel_rates = {}
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                # 存储平均行程时间（分钟），包括i=j的情况
                self.travel_rates[(i, j)] = self.travel_matrix[i][j]
        
        # 收益计算器
        self.revenue_calculator = RevenueCalculator()
        
        # 初始化车辆状态
        self._initialize_vehicle_distribution()
        
        # 使用优先队列维护在途车辆，按到达逻辑时间排序
        self.arrival_queue = []
        
        # 统计信息
        self.total_revenue = 0.0
        self.total_relocation_cost = 0.0
        self.served_passengers = 0
        self.rejected_passengers = 0

    def _initialize_vehicle_distribution(self):
        """初始化车辆分布"""
        # 初始化快车分布 - 均匀分配到各个区域
        express_cars = [self.express_num // self.n_nodes] * self.n_nodes
        remaining_express = self.express_num % self.n_nodes
        for i in range(remaining_express):
            express_cars[random.randint(0, self.n_nodes - 1)] += 1
        
        # 初始化专车分布 - 均匀分配到各个区域
        premium_cars = [self.premium_num // self.n_nodes] * self.n_nodes
        remaining_premium = self.premium_num % self.n_nodes
        for i in range(remaining_premium):
            premium_cars[random.randint(0, self.n_nodes - 1)] += 1
        
        # 初始化车辆状态字典
        self.express_idle_num = {i: express_cars[i] for i in range(self.n_nodes)}
        self.premium_idle_num = {i: premium_cars[i] for i in range(self.n_nodes)}
        
        # 初始化占用和重新调度车辆数量
        self.express_occupy_num = {(i, j): 0 for i in range(self.n_nodes) for j in range(self.n_nodes)}
        self.premium_occupy_num = {(i, j): 0 for i in range(self.n_nodes) for j in range(self.n_nodes)}
        self.express_relocation_num = {(i, j): 0 for i in range(self.n_nodes) for j in range(self.n_nodes)}
        self.premium_relocation_num = {(i, j): 0 for i in range(self.n_nodes) for j in range(self.n_nodes)}
        
        # 初始化收益统计
        self.revenue_by_type = {"express": 0.0, "premium": 0.0}
        self.cost_by_type = {"express": 0.0, "premium": 0.0}

    def update_express_from_relocation_to_idle(self, origin: int, destination: int) -> bool:
        """更新快车从重新调度到空闲状态"""
        key = (origin, destination)
        if key in self.express_relocation_num and self.express_relocation_num[key] > 0:
            self.express_relocation_num[key] -= 1
            self.express_idle_num[destination] += 1
            return True
        return False

    def update_premium_from_relocation_to_idle(self, origin: int, destination: int) -> bool:
        """更新专车从重新调度到空闲状态"""
        key = (origin, destination)
        if key in self.premium_relocation_num and self.premium_relocation_num[key] > 0:
            self.premium_relocation_num[key] -= 1
            self.premium_idle_num[destination] += 1
            return True
        return False

    def update_express_from_occupied_to_idle(self, origin: int, destination: int) -> bool:
        """更新快车从占用到空闲状态"""
        key = (origin, destination)
        if key in self.express_occupy_num and self.express_occupy_num[key] > 0:
            self.express_occupy_num[key] -= 1
            self.express_idle_num[destination] += 1
            return True
        return False

    def update_premium_from_occupied_to_idle(self, origin: int, destination: int) -> bool:
        """更新专车从占用到空闲状态"""
        key = (origin, destination)
        if key in self.premium_occupy_num and self.premium_occupy_num[key] > 0:
            self.premium_occupy_num[key] -= 1
            self.premium_idle_num[destination] += 1
            return True
        return False

    # ========== 队列操作方法 ==========
    
    def insert_arrival_queue(self, origin: int, destination: int, vehicle_type: str, car_state: CarState, arrival_logic_time: float):
        """
        按照 arrival_logic_time 升序插入到优先队列中
        
        Args:
            origin: 起点区域
            destination: 终点区域
            vehicle_type: 车辆类型 ("E"或"P")
            car_state: 车辆状态
            arrival_logic_time: 到达逻辑时间
        """
        car_on_the_way = CarOnTheWay(
            arrival_logic_time=arrival_logic_time,
            origin=origin,
            destination=destination,
            vehicle_type=vehicle_type,
            car_state=car_state
        )
        heapq.heappush(self.arrival_queue, car_on_the_way)
    
    def pop_arrival_queue(self) -> Optional[CarOnTheWay]:
        """删除并返回到达逻辑时间最早的车辆记录"""
        if self.arrival_queue:
            return heapq.heappop(self.arrival_queue)
        return None

    def get_earliest_arrival_time(self) -> float:
        """获取最早的车辆到达逻辑时间，如果没有则返回无穷大"""
        if self.arrival_queue:
            return self.arrival_queue[0].arrival_logic_time
        return float('inf')

    # ========== 核心方法：动作评估 ==========
    
    def evaluate_dispatching_action(self, origin: int, destination: int, vehicle_type: str) -> ActionEvaluation:
        """
        评估派单动作 - 生成行程时间、计算即时收益、并生成决策后状态
        
        Args:
            origin: 起点区域
            destination: 终点区域
            vehicle_type: 车辆类型 ("E"或"P")
            
        Returns:
            ActionEvaluation 包含行程时间、估计即时收益和决策后状态
        """
        # 1. 生成行程时间（包括origin==destination的情况，使用随机生成）
        travel_time_seconds = self.get_travel_time(origin, destination)
        
        # 2. 计算估计即时收益
        estimated_revenue = self.revenue_calculator.calculate_trip_revenue(
            origin, destination, vehicle_type, travel_time_seconds
        )
        
        # 3. 构建动作表示
        action = 0 if vehicle_type == "E" else 1  # 0=快车, 1=专车
        
        # 4. 模拟执行动作，生成决策后状态 S_t^a
        post_state_vector = self._simulate_dispatching_action(origin, destination, vehicle_type)
        post_decision_state = PostDecisionState(post_state_vector)
        
        return ActionEvaluation(
            action=action,
            travel_time_seconds=travel_time_seconds,
            estimated_immediate_reward=estimated_revenue,
            post_decision_state=post_decision_state
        )
    
    def evaluate_rebalancing_action(self, origin: int, destination: int, vehicle_type: str) -> ActionEvaluation:
        """
        评估重新平衡动作 - 生成行程时间、计算即时成本、并生成决策后状态
        
        Args:
            origin: 起点区域（车辆当前位置）
            destination: 目标区域
            vehicle_type: 车辆类型 ("E"或"P")
            
        Returns:
            ActionEvaluation 包含行程时间、估计即时成本和决策后状态
        """
        if origin == destination:
            # 留在原地：没有行程时间，没有成本
            travel_time_seconds = 0
            estimated_cost = 0.0
        else:
            # 正常重新平衡调度：计算行程时间和成本
            travel_time_seconds = self.get_travel_time(origin, destination)
            estimated_cost = self.revenue_calculator.calculate_relocation_cost(
                origin, destination, vehicle_type, travel_time_seconds
            )
        
        # 3. 构建动作表示
        action = destination  # 目标区域编号
        
        # 4. 模拟执行动作，生成决策后状态 S_t^a
        post_state_vector = self._simulate_rebalancing_action(origin, destination, vehicle_type)
        post_decision_state = PostDecisionState(post_state_vector)
        
        return ActionEvaluation(
            action=action,
            travel_time_seconds=travel_time_seconds,
            estimated_immediate_reward=estimated_cost,  # 注意：重新平衡时是成本，可能是负值
            post_decision_state=post_decision_state
        )
    
    def _simulate_dispatching_action(self, origin: int, destination: int, vehicle_type: str) -> np.ndarray:
        """
        模拟派单动作，返回决策后状态向量（不实际修改网络状态）
        
        Args:
            origin: 起点区域
            destination: 终点区域
            vehicle_type: 车辆类型 ("E"或"P")
            
        Returns:
            模拟执行动作后的车辆分布向量
        """
        # 创建当前车辆分布的深拷贝
        express_idle_copy = self.express_idle_num.copy()
        premium_idle_copy = self.premium_idle_num.copy()
        express_occupy_copy = self.express_occupy_num.copy()
        premium_occupy_copy = self.premium_occupy_num.copy()
        
        # 模拟状态更新（不区分origin==destination的情况）
        if vehicle_type == "E":
            if express_idle_copy[origin] > 0:
                express_idle_copy[origin] -= 1
                express_occupy_copy[(origin, destination)] += 1
        else:  # Premium
            if premium_idle_copy[origin] > 0:
                premium_idle_copy[origin] -= 1
                premium_occupy_copy[(origin, destination)] += 1
        
        # 从模拟状态构建向量
        return self._build_vehicle_distribution_vector(
            express_idle_copy, premium_idle_copy,
            express_occupy_copy, premium_occupy_copy,
            self.express_relocation_num, self.premium_relocation_num
        )
    
    def _simulate_rebalancing_action(self, origin: int, destination: int, vehicle_type: str) -> np.ndarray:
        """
        模拟重新平衡动作，返回决策后状态向量（不实际修改网络状态）
        
        Args:
            origin: 起点区域
            destination: 目标区域
            vehicle_type: 车辆类型 ("E"或"P")
            
        Returns:
            模拟执行动作后的车辆分布向量
        """
        # 创建当前车辆分布的深拷贝
        express_idle_copy = self.express_idle_num.copy()
        premium_idle_copy = self.premium_idle_num.copy()
        express_relocation_copy = self.express_relocation_num.copy()
        premium_relocation_copy = self.premium_relocation_num.copy()
        
        # 模拟状态更新（不区分origin==destination的情况）
        if vehicle_type == "E":
            if express_idle_copy[origin] > 0:
                express_idle_copy[origin] -= 1
                express_relocation_copy[(origin, destination)] += 1
        else:  # Premium
            if premium_idle_copy[origin] > 0:
                premium_idle_copy[origin] -= 1
                premium_relocation_copy[(origin, destination)] += 1
        
        # 从模拟状态构建向量
        return self._build_vehicle_distribution_vector(
            express_idle_copy, premium_idle_copy,
            self.express_occupy_num, self.premium_occupy_num,
            express_relocation_copy, premium_relocation_copy
        )
    
    def _build_vehicle_distribution_vector(self, 
                                          express_idle: Dict,
                                          premium_idle: Dict,
                                          express_occupy: Dict,
                                          premium_occupy: Dict,
                                          express_relocation: Dict,
                                          premium_relocation: Dict) -> np.ndarray:
        """从给定的状态字典构建车辆分布向量"""
        vector = []
        
        # 1. 空闲车辆数量
        for i in range(self.n_nodes):
            vector.append(express_idle[i])
        for i in range(self.n_nodes):
            vector.append(premium_idle[i])
        
        # 2. 占用车辆数量（按OD对，包括origin==destination的情况）
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                vector.append(express_occupy[(i, j)])
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                vector.append(premium_occupy[(i, j)])
        
        # 3. 重新调度车辆数量（按OD对，包括origin==destination的情况）
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                vector.append(express_relocation[(i, j)])
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                vector.append(premium_relocation[(i, j)])
        
        return np.array(vector, dtype=np.float32)

    # ========== 核心方法：动作执行 ==========
    
    def execute_dispatching_action(self, origin: int, destination: int, vehicle_type: str, 
                                travel_time_seconds: float, current_time: float) -> Tuple[bool, float]:
        """
        执行派单动作 - 使用之前评估时生成的行程时间
        
        Args:
            origin: 起点区域
            destination: 终点区域
            vehicle_type: 车辆类型 ("E"或"P")
            travel_time_seconds: 之前评估时生成的行程时间（对于起终点相同，也是随机生成的正值）
            current_time: 当前逻辑时间
            
        Returns:
            (success, actual_revenue)
        """
        success = False
        actual_revenue = 0.0
        
        if vehicle_type == "E":
            if self.express_idle_num[origin] > 0:
                # 更新车辆状态（包括origin==destination的情况）
                self.express_idle_num[origin] -= 1
                self.express_occupy_num[(origin, destination)] += 1
                
                # 计算实际收益（基于实际行程时间）
                actual_revenue = self.revenue_calculator.calculate_trip_revenue(
                    origin, destination, "E", travel_time_seconds
                )
                
                # 添加到在途队列（包括origin==destination的情况）
                arrival_time = current_time + travel_time_seconds
                self.insert_arrival_queue(origin, destination, "E", CarState.OCCUPIED, arrival_time)
                
                # 更新统计
                self.total_revenue += actual_revenue
                self.revenue_by_type["express"] += actual_revenue
                self.served_passengers += 1
                
                success = True
        else:  # Premium
            if self.premium_idle_num[origin] > 0:
                # 更新车辆状态（包括origin==destination的情况）
                self.premium_idle_num[origin] -= 1
                self.premium_occupy_num[(origin, destination)] += 1
                
                # 计算实际收益
                actual_revenue = self.revenue_calculator.calculate_trip_revenue(
                    origin, destination, "P", travel_time_seconds
                )
                
                # 添加到在途队列（包括origin==destination的情况）
                arrival_time = current_time + travel_time_seconds
                self.insert_arrival_queue(origin, destination, "P", CarState.OCCUPIED, arrival_time)
                
                # 更新统计
                self.total_revenue += actual_revenue
                self.revenue_by_type["premium"] += actual_revenue
                self.served_passengers += 1
                
                success = True
        
        return success, actual_revenue
    
    def execute_rebalancing_action(self, origin: int, destination: int, vehicle_type: str,
                                travel_time_seconds: float, current_time: float) -> Tuple[bool, float]:
        """
        执行重新平衡动作 - 使用之前评估时生成的行程时间
        
        Args:
            origin: 起点区域
            destination: 目标区域
            vehicle_type: 车辆类型 ("E"或"P")
            travel_time_seconds: 之前评估时生成的行程时间
            current_time: 当前逻辑时间
            
        Returns:
            (success, actual_cost)
        """
        success = False
        actual_cost = 0.0
        
        if origin == destination:
            # 留在原地：不更新车辆状态，不添加到队列，立即返回
            return True, 0.0
            
        # 正常重新平衡调度
        if vehicle_type == "E":
            if self.express_idle_num[origin] > 0:
                # 更新车辆状态
                self.express_idle_num[origin] -= 1
                self.express_relocation_num[(origin, destination)] += 1
                
                # 计算实际成本（负值）
                actual_cost = self.revenue_calculator.calculate_relocation_cost(
                    origin, destination, "E", travel_time_seconds
                )
                
                # 添加到在途队列
                arrival_time = current_time + travel_time_seconds
                self.insert_arrival_queue(origin, destination, "E", CarState.RELOCATING, arrival_time)
                
                # 更新统计
                self.total_relocation_cost += actual_cost
                self.cost_by_type["express"] += actual_cost
                
                success = True
        else:  # Premium
            if self.premium_idle_num[origin] > 0:
                # 更新车辆状态
                self.premium_idle_num[origin] -= 1
                self.premium_relocation_num[(origin, destination)] += 1
                
                # 计算实际成本
                actual_cost = self.revenue_calculator.calculate_relocation_cost(
                    origin, destination, "P", travel_time_seconds
                )
                
                # 添加到在途队列
                arrival_time = current_time + travel_time_seconds
                self.insert_arrival_queue(origin, destination, "P", CarState.RELOCATING, arrival_time)
                
                # 更新统计
                self.total_relocation_cost += actual_cost
                self.cost_by_type["premium"] += actual_cost
                
                success = True
        
        return success, actual_cost

    # ========== 行程时间生成 ==========
    
    def get_travel_time(self, origin: int, destination: int) -> float:
        """
        根据指数分布生成行程时间（逻辑时间，单位：秒）
        
        Args:
            origin: 起点区域
            destination: 终点区域
            
        Returns:
            行程时间（秒）
        """
        # 获取平均行程时间（分钟），包括origin==destination的情况
        mean_travel_time_minutes = self.travel_rates[(origin, destination)]
        
        # 确保平均行程时间至少为0.1分钟（6秒），避免数值问题
        if mean_travel_time_minutes <= 0:
            mean_travel_time_minutes = 0.1
        
        # 对于指数分布，λ = 1/平均时间
        travel_rate = 1.0 / mean_travel_time_minutes
        
        # 生成指数分布的行程时间（分钟）
        u = random.random()
        travel_time_minutes = -math.log(1 - u) / travel_rate
        
        # 确保生成的行程时间至少为0.01分钟（0.6秒）
        travel_time_minutes = max(travel_time_minutes, 0.01)
        
        # 转换为秒（因为环境中使用秒作为逻辑时间单位）
        travel_time_seconds = travel_time_minutes * 60
        
        return travel_time_seconds
    
    # ========== 状态向量获取 ==========
    
    def get_vehicle_distribution_vector(self) -> np.ndarray:
        """获取当前实际的车辆分布向量"""
        return self._build_vehicle_distribution_vector(
            self.express_idle_num, self.premium_idle_num,
            self.express_occupy_num, self.premium_occupy_num,
            self.express_relocation_num, self.premium_relocation_num
        )
    
    # ========== 其他方法 ==========
    
    def get_idle_vehicles_at_node(self, node: int, vehicle_type: str = None) -> int:
        """获取指定节点的空闲车辆数量"""
        if vehicle_type == "E":
            return self.express_idle_num.get(node, 0)
        elif vehicle_type == "P":
            return self.premium_idle_num.get(node, 0)
        else:
            return self.express_idle_num.get(node, 0) + self.premium_idle_num.get(node, 0)
    
    def has_idle_vehicles_at_node(self, node: int, vehicle_type: str = None) -> bool:
        """检查指定节点是否有空闲车辆"""
        return self.get_idle_vehicles_at_node(node, vehicle_type) > 0
    
    def reset(self):
        """重置网络状态（用于新的episode）"""
        # 重新初始化车辆分布
        self._initialize_vehicle_distribution()
        
        # 清空在途队列
        self.arrival_queue = []
        
        # 重置统计信息
        self.total_revenue = 0.0
        self.total_relocation_cost = 0.0
        self.revenue_by_type = {"express": 0.0, "premium": 0.0}
        self.cost_by_type = {"express": 0.0, "premium": 0.0}
        self.served_passengers = 0
        self.rejected_passengers = 0

# ========== 8. 基于真实数据的乘客生成器 ==========

class RealDataPassengerGenerator:
    """基于真实数据的乘客生成器"""
    
    def __init__(self, data_folder: str, n_nodes: int, zone_mapping: Dict[int, int]):
        """
        初始化基于真实数据的乘客生成器
        """
        self.data_folder = data_folder
        self.n_nodes = n_nodes
        self.zone_mapping = zone_mapping
        
        # 获取所有可用的日期文件
        self.available_dates = self._get_available_dates()
        
        # 乘客队列
        self.passenger_queue = deque()
        self.next_id = 0
        
        # 乘客类型生成概率 [快车, 双类型, 专车]
        self.passenger_type_probs = [0.5, 0.3, 0.2]
        
        # 当前处理的天数据
        self.current_date = None
        self.current_date_data = None
        
        # 统计信息
        self.generation_stats = {
            'total_generated': 0,
            'total_trips_in_date': 0
        }
    
    def _get_available_dates(self) -> List[date]:
        """获取文件夹中所有可用的日期"""
        dates = []
        if not os.path.exists(self.data_folder):
            print(f"数据文件夹不存在: {self.data_folder}")
            return dates
        
        for filename in os.listdir(self.data_folder):
            if filename.startswith("trips_") and filename.endswith("_17_19_core.parquet"):
                # 从文件名提取日期: trips_2025-11-03_17_19_core.parquet
                try:
                    date_str = filename.split('_')[1]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    dates.append(file_date)
                except (IndexError, ValueError):
                    continue
        
        dates.sort()
        return dates
    
    def _generate_passenger_type(self) -> PassengerType:
        """生成乘客类型"""
        passenger_type_idx = np.random.choice(
            [0, 1, 2],  # 0=EXPRESS, 1=DUAL, 2=PREMIUM
            p=self.passenger_type_probs
        )
        
        if passenger_type_idx == 0:
            return PassengerType.EXPRESS
        elif passenger_type_idx == 1:
            return PassengerType.DUAL
        else:
            return PassengerType.PREMIUM
    
    def load_date_data(self, target_date: date) -> bool:
        """
        加载指定日期的数据
        """
        # 查找对应日期的文件
        filename = f"trips_{target_date}_17_19_core.parquet"
        filepath = os.path.join(self.data_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"找不到日期文件: {filepath}")
            return False
        
        try:
            # 读取数据
            df = pd.read_parquet(filepath)
            
            # 确保时间戳是datetime类型
            # 如果时间戳已经是数值类型，转换为datetime
            if not pd.api.types.is_datetime64_any_dtype(df['tpep_pickup_datetime']):
                # 尝试转换为datetime，假设是毫秒时间戳
                try:
                    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], unit='ms')
                except:
                    # 如果失败，尝试其他转换方式
                    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
            
            # 存储原始数据
            self.current_date = target_date
            self.current_date_data = df
            
            # 重置统计
            self.generation_stats['total_trips_in_date'] = len(df)
            
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def generate_passengers_from_current_date(self):
        """
        从当前加载的日期数据生成乘客队列
        """
        if self.current_date_data is None:
            print("没有加载日期数据")
            return
        
        # 重置队列
        self.passenger_queue = deque()
        self.next_id = 0
        self.generation_stats['total_generated'] = 0
        
        df = self.current_date_data
        
        # 1. 找出最早的上车时间作为开始时间
        start_time = df['tpep_pickup_datetime'].min()
        
        # 转换开始时间为毫秒时间戳
        if isinstance(start_time, pd.Timestamp):
            start_time_ms = start_time.value // 10**6  # 纳秒转毫秒
        elif hasattr(start_time, 'timestamp'):
            start_time_ms = int(start_time.timestamp() * 1000)
        else:
            start_time_ms = int(start_time)
        
        # 2. 处理每条行程记录
        passengers = []
        
        for idx, row in df.iterrows():
            # 提取起终点区域
            pu_location = row['PULocationID']
            do_location = row['DOLocationID']
            
            # 映射到仿真节点
            origin = self.zone_mapping.get(pu_location, -1)
            destination = self.zone_mapping.get(do_location, -1)
            
            # 跳过无效区域
            if origin == -1 or destination == -1:
                continue
            
            # 获取上车时间
            pickup_time = row['tpep_pickup_datetime']
            
            # 转换上车时间为毫秒时间戳
            if isinstance(pickup_time, pd.Timestamp):
                pickup_time_ms = pickup_time.value // 10**6  # 纳秒转毫秒
            elif hasattr(pickup_time, 'timestamp'):
                pickup_time_ms = int(pickup_time.timestamp() * 1000)
            else:
                pickup_time_ms = int(pickup_time)
            
            # 计算到达逻辑时间（相对于开始时间的秒数）
            arrival_time = (pickup_time_ms - start_time_ms) / 1000.0
            
            # 确保 arrival_time 是浮点数
            arrival_time = float(arrival_time)
            
            # 生成乘客类型（基于固定概率）
            passenger_type = self._generate_passenger_type()
            
            # 创建乘客对象
            passenger = Passenger(
                passenger_id=self.next_id,
                arrival_logic_time=arrival_time,
                origin=origin,
                destination=destination,
                passenger_type=passenger_type
            )
            
            passengers.append(passenger)
            self.next_id += 1
        
        # 3. 按到达时间排序
        passengers.sort(key=lambda x: x.arrival_logic_time)
        
        # 4. 重新分配ID（按时间顺序）
        for i, passenger in enumerate(passengers):
            passenger.passenger_id = i
            self.passenger_queue.append(passenger)
        
        self.generation_stats['total_generated'] = len(self.passenger_queue)
        print(f"  生成 {len(self.passenger_queue)} 个乘客")
    
    def get_next_passenger(self) -> Optional[Passenger]:
        """获取下一个乘客"""
        if self.passenger_queue:
            return self.passenger_queue.popleft()
        return None
    
    def get_next_passenger_time(self) -> float:
        """获取下一个乘客的逻辑到达时间"""
        if self.passenger_queue:
            return self.passenger_queue[0].arrival_logic_time
        else:
            return float('inf')
    
    def get_queue_length(self) -> int:
        """获取队列长度"""
        return len(self.passenger_queue)
    
    def get_available_dates(self) -> List[date]:
        """获取所有可用日期"""
        return self.available_dates.copy()
    
    def reset(self):
        """重置生成器（清空队列）"""
        self.passenger_queue = deque()
        self.next_id = 0
        self.generation_stats = {
            'total_generated': 0,
            'total_trips_in_date': self.generation_stats.get('total_trips_in_date', 0)
        }

# ========== 9. 训练好的神经网络智能体（测试模式） ==========

class TrainedNeuralADPAgent:
    """训练好的神经网络决策器（仅推理模式）"""
    
    def __init__(self, model_path: str, post_state_dim: int, env: Any,
                 epsilon_end: float = 0.01, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化训练好的智能体
        
        Args:
            model_path: 训练好的模型路径
            post_state_dim: 决策后状态维度
            env: 环境实例
            epsilon_end: 测试时的探索率
            device: 计算设备
        """
        self.post_state_dim = post_state_dim
        self.env = env
        self.epsilon = epsilon_end
        self.device = device
        
        # 初始化神经网络
        self.value_network = PostDecisionValueNetwork(post_state_dim).to(device)
        
        # 加载训练好的模型参数
        self.load_trained_model(model_path)
        
        # 设置为评估模式
        self.value_network.eval()
        
        print(f"训练好的智能体已加载，设备: {device}")
        print(f"探索率固定为: {epsilon_end}")
    
    def load_trained_model(self, model_path: str):
        """加载训练好的模型参数"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
            print(f"模型已从 {model_path} 加载")
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    def _compute_q_value_from_eval(self, action_eval: ActionEvaluation) -> float:
        """
        从已计算的ActionEvaluation计算Q值：Q(S, a) = R_inst(S, a) + V_θ(S^a)
        """
        post_state_tensor = torch.FloatTensor(
            action_eval.post_decision_state.get_state_vector()
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.value_network(post_state_tensor).item()

        return action_eval.estimated_immediate_reward + value

    def select_action(self, current_pre_state: PreDecisionState) -> Tuple[Any, ActionEvaluation, Dict[Any, float]]:
        """
        选择动作（纯贪心策略：始终选择Q值最大的动作）
        """
        action_evals = self.env.evaluate_actions()

        if not action_evals:
            return None, None, {}

        all_q_values = {}
        for action_eval in action_evals:
            q_value = self._compute_q_value_from_eval(action_eval)
            all_q_values[action_eval.action] = q_value

        # 纯贪心：始终选择Q值最大的动作（如有并列，按评估顺序取第一个）
        selected_action = max(all_q_values, key=all_q_values.get)

        selected_action_eval = None
        for action_eval in action_evals:
            if action_eval.action == selected_action:
                selected_action_eval = action_eval
                break

        selected_action = selected_action_eval.action if selected_action_eval else None
        return selected_action, selected_action_eval, all_q_values
# ========== 10. 基于真实数据的测试环境 ==========

class TestRideHailingEnvRealData:
    """基于真实数据和训练好的神经网络的测试环境"""
    
    def __init__(self, n_nodes: int, express_num: int, premium_num: int, 
                 travel_matrix: np.ndarray,
                 data_folder: str,
                 zone_mapping: Dict[int, int],
                 trained_agent: TrainedNeuralADPAgent):
        """
        初始化测试环境（使用真实数据）
        """
        self.n_nodes = n_nodes
        self.express_num = express_num
        self.premium_num = premium_num
        self.travel_matrix = travel_matrix
        self.data_folder = data_folder
        self.zone_mapping = zone_mapping
        self.trained_agent = trained_agent

        # 初始化网络
        self.network = Network(n_nodes, express_num, premium_num, travel_matrix)
        
        # 初始化真实数据乘客生成器
        self.passenger_generator = RealDataPassengerGenerator(
            data_folder=data_folder,
            n_nodes=n_nodes,
            zone_mapping=zone_mapping
        )
        
        # 当前状态和统计
        self.current_time = 0.0
        self.current_pre_state = None
        self.current_passenger = None
        self.current_vehicle_completion = None
        self.episode_revenue = 0.0
        self.episode_cost = 0.0
        self.episode_served = 0
        self.episode_rejected = 0

    def _get_next_event(self):
        # Episode结束条件：乘客队列为空且没有在途车辆
        
        # 乘客队列为空
        if self.passenger_generator.get_queue_length() == 0:
            # 检查是否还有在途车辆
            next_vehicle_time = self.network.get_earliest_arrival_time()
            if next_vehicle_time < float('inf'):
                return next_vehicle_time, 'vehicle_arrival'
            else:
                # 既没有乘客，也没有在途车辆，episode结束
                return float('inf'), 'episode_end'
        else:
            # 正常处理事件
            next_passenger_time = self.passenger_generator.get_next_passenger_time()
            next_vehicle_time = self.network.get_earliest_arrival_time()
            
            next_event_time = min(next_passenger_time, next_vehicle_time)
            if next_event_time == next_passenger_time:
                return next_passenger_time, 'passenger_arrival'
            else:
                return next_vehicle_time, 'vehicle_arrival'
    
    def _handle_passenger_arrival(self, passenger: Passenger) -> Tuple[Optional[PreDecisionState], bool]:
        """
        处理乘客到达事件
        
        Returns:
            (decision_state, need_decision)
        """
        origin = passenger.origin
        destination = passenger.destination
        passenger_type = passenger.passenger_type
        
        # 根据乘客类型处理
        if passenger_type == PassengerType.EXPRESS:
            # 快车专用乘客
            if self.network.has_idle_vehicles_at_node(origin, "E"):
                # 有快车可用，自动分配
                travel_time = self.network.get_travel_time(origin, destination)
                success, revenue = self.network.execute_dispatching_action(
                    origin, destination, "E", travel_time, self.current_time
                )
                if success:
                    self.episode_revenue += revenue
                    self.episode_served += 1
                else:
                    self.episode_rejected += 1
                return None, False
            else:
                # 无快车可用，乘客流失
                self.episode_rejected += 1
                return None, False
                
        elif passenger_type == PassengerType.PREMIUM:
            # 专车专用乘客
            if self.network.has_idle_vehicles_at_node(origin, "P"):
                # 有专车可用，自动分配
                travel_time = self.network.get_travel_time(origin, destination)
                success, revenue = self.network.execute_dispatching_action(
                    origin, destination, "P", travel_time, self.current_time
                )
                if success:
                    self.episode_revenue += revenue
                    self.episode_served += 1
                else:
                    self.episode_rejected += 1
                return None, False
            else:
                # 无专车可用，乘客流失
                self.episode_rejected += 1
                return None, False
                
        else:  # PassengerType.DUAL
            # 双类型乘客
            has_express = self.network.has_idle_vehicles_at_node(origin, "E")
            has_premium = self.network.has_idle_vehicles_at_node(origin, "P")
            
            if has_express and has_premium:
                # 两种车都有，需要决策
                self.current_passenger = passenger
                decision_state = PreDecisionState(
                    state_type=DecisionType.DISPATCHING,
                    passenger_info=(origin, destination)
                )
                return decision_state, True
            elif has_express:
                # 只有快车可用，自动分配快车
                travel_time = self.network.get_travel_time(origin, destination)
                success, revenue = self.network.execute_dispatching_action(
                    origin, destination, "E", travel_time, self.current_time
                )
                if success:
                    self.episode_revenue += revenue
                    self.episode_served += 1
                else:
                    self.episode_rejected += 1
                return None, False
            elif has_premium:
                # 只有专车可用，自动分配专车
                travel_time = self.network.get_travel_time(origin, destination)
                success, revenue = self.network.execute_dispatching_action(
                    origin, destination, "P", travel_time, self.current_time
                )
                if success:
                    self.episode_revenue += revenue
                    self.episode_served += 1
                else:
                    self.episode_rejected += 1
                return None, False
            else:
                # 两种车都无，乘客流失
                self.episode_rejected += 1
                return None, False
    
    def _handle_vehicle_arrival(self, car: CarOnTheWay) -> Tuple[Optional[PreDecisionState], bool]:
        """
        处理车辆到达事件
        """
        origin = car.origin
        destination = car.destination
        vehicle_type = car.vehicle_type
        car_state = car.car_state
        
        # 根据车辆状态处理
        if car_state == CarState.RELOCATING:
            # 重新调度车辆到达，变为空闲
            if vehicle_type == "E":
                self.network.update_express_from_relocation_to_idle(origin, destination)
            else:
                self.network.update_premium_from_relocation_to_idle(origin, destination)
            return None, False
                
        elif car_state == CarState.OCCUPIED:
            # 载客车辆到达，完成行程 - TRIGGERS RELOCATION DECISION
            if vehicle_type == "E":
                self.network.update_express_from_occupied_to_idle(origin, destination)
            else:
                self.network.update_premium_from_occupied_to_idle(origin, destination)
        
            
            # 只有载客车辆完成行程时才触发重新平衡决策
            self.current_vehicle_completion = (destination, vehicle_type)
            decision_state = PreDecisionState(
                state_type=DecisionType.REBALANCING,
                vehicle_completion_info=(destination, vehicle_type)
            )
            return decision_state, True
        
        return None, False
    
    def evaluate_actions(self) -> List[ActionEvaluation]:
        """
        评估所有可能的动作（供智能体调用）
        
        Returns:
            所有可能动作的评估结果列表
        """
        if self.current_pre_state is None:
            return []
        
        action_evals = []
        
        if self.current_pre_state.state_type == DecisionType.DISPATCHING:
            # 派单决策：评估派快车和派专车
            origin, destination = self.current_pre_state.passenger_info
            
            # 评估派快车
            if self.network.has_idle_vehicles_at_node(origin, "E"):
                action_eval = self.network.evaluate_dispatching_action(origin, destination, "E")
                action_evals.append(action_eval)
            
            # 评估派专车
            if self.network.has_idle_vehicles_at_node(origin, "P"):
                action_eval = self.network.evaluate_dispatching_action(origin, destination, "P")
                action_evals.append(action_eval)
        
        else:  # REBALANCING
            # 重新平衡决策：评估所有目标区域（包括当前区域）
            zone, vehicle_type = self.current_pre_state.vehicle_completion_info
            
            if self.network.has_idle_vehicles_at_node(zone, vehicle_type):
                # 评估所有可能的区域（包括zone自身）
                for destination in range(self.n_nodes):
                    action_eval = self.network.evaluate_rebalancing_action(zone, destination, vehicle_type)
                    action_evals.append(action_eval)
        
        return action_evals
    
    def get_current_pre_state(self) -> Optional[PreDecisionState]:
        """获取当前的决策前状态"""
        return self.current_pre_state
    
    def run_episode_for_date(self, target_date: date) -> Dict[str, Any]:
        """
        为指定日期运行一个episode
        """
        # 重置环境
        self.current_time = 0.0
        self.current_pre_state = None
        self.current_passenger = None
        self.current_vehicle_completion = None
        self.episode_revenue = 0.0
        self.episode_cost = 0.0
        self.episode_served = 0
        self.episode_rejected = 0
        
        # 加载指定日期的数据
        if not self.passenger_generator.load_date_data(target_date):
            print(f"无法加载日期 {target_date} 的数据")
            return None
        
        # 从数据生成乘客队列
        self.passenger_generator.generate_passengers_from_current_date()
        
        # 重置网络状态
        self.network.reset()
        
        while True:
            # 获取下一个事件
            next_event_time, event_type = self._get_next_event()
            
            # 检查是否结束
            if event_type == 'episode_end':
                break
            
            # 更新时间
            self.current_time = next_event_time
            
            # 处理事件
            if event_type == 'passenger_arrival':
                passenger = self.passenger_generator.get_next_passenger()
                decision_state, need_decision = self._handle_passenger_arrival(passenger)
                
                if need_decision:
                    self.current_pre_state = decision_state
                    # 使用训练好的智能体进行决策
                    action, action_eval, _ = self.trained_agent.select_action(decision_state)
                    
                    if action is not None and action_eval is not None:
                        # 执行动作
                        if decision_state.state_type == DecisionType.DISPATCHING:
                            origin, destination = decision_state.passenger_info
                            vehicle_type = "E" if action == 0 else "P"
                            
                            success, revenue = self.network.execute_dispatching_action(
                                origin, destination, vehicle_type,
                                action_eval.travel_time_seconds, self.current_time
                            )
                            
                            if success:
                                self.episode_revenue += revenue
                                self.episode_served += 1
                            else:
                                self.episode_rejected += 1
                    
                    # 重置当前状态
                    self.current_pre_state = None
                    self.current_passenger = None
                
            elif event_type == 'vehicle_arrival':
                car = self.network.pop_arrival_queue()
                decision_state, need_decision = self._handle_vehicle_arrival(car)
                
                if need_decision:
                    self.current_pre_state = decision_state
                    # 使用训练好的智能体进行决策
                    action, action_eval, _ = self.trained_agent.select_action(decision_state)
                    
                    if action is not None and action_eval is not None:
                        # 执行动作
                        zone, vehicle_type = decision_state.vehicle_completion_info
                        destination = action  # 动作就是目标区域
                        
                        if destination != zone:
                            success, cost = self.network.execute_rebalancing_action(
                                zone, destination, vehicle_type,
                                action_eval.travel_time_seconds, self.current_time
                            )
                            
                            if success:
                                self.episode_cost += cost
                    
                    # 重置当前状态
                    self.current_pre_state = None
                    self.current_vehicle_completion = None
        
        # 收集统计信息
        total_profit = self.episode_revenue + self.episode_cost
        
        episode_info = {
            'date': target_date,
            'profit': total_profit,
            'revenue': self.episode_revenue,
            'cost': self.episode_cost,
            'served': self.episode_served,
            'rejected': self.episode_rejected,
            'total_in_data': self.passenger_generator.generation_stats['total_trips_in_date']
        }
        
        # 输出当天结果
        print(f"{target_date}: 收益={total_profit:.2f}元, 数据总量={self.passenger_generator.generation_stats['total_trips_in_date']}")
        
        return episode_info
    
    def reset(self):
        """重置环境"""
        self.current_time = 0.0
        self.current_pre_state = None
        self.current_passenger = None
        self.current_vehicle_completion = None
        self.episode_revenue = 0.0
        self.episode_cost = 0.0
        self.episode_served = 0
        self.episode_rejected = 0
        
        # 重置网络和乘客生成器
        self.network.reset()
        self.passenger_generator.reset()

# ========== 11. 测试实验运行器 ==========

class TestExperimentRunner:
    """测试实验运行器"""
    
    def __init__(self, test_env: TestRideHailingEnvRealData):
        """
        初始化实验运行器
        """
        self.test_env = test_env
        
        # 实验统计
        self.all_results = []
        self.summary_stats = {
            'total_profit': 0.0,
            'total_revenue': 0.0,
            'total_cost': 0.0,
            'total_served': 0,
            'total_rejected': 0,
            'total_in_data': 0
        }
    
    def run_test_suite(self, dates_to_test: List[date] = None):
        """
        运行测试套件
        
        Args:
            dates_to_test: 要测试的日期列表，如果为None则测试所有可用日期
        """
        # 获取所有可用日期
        available_dates = self.test_env.passenger_generator.get_available_dates()
        
        if not available_dates:
            return
        
        if dates_to_test is None:
            dates_to_test = available_dates
        
        print(f"开始运行 {len(dates_to_test)} 天的数据...")
        print("="*80)
        
        for target_date in dates_to_test:
            # 运行一个episode
            episode_result = self.test_env.run_episode_for_date(target_date)
            if episode_result:
                self.all_results.append(episode_result)
                
                # 更新汇总统计
                self.summary_stats['total_profit'] += episode_result['profit']
                self.summary_stats['total_revenue'] += episode_result['revenue']
                self.summary_stats['total_cost'] += episode_result['cost']
                self.summary_stats['total_served'] += episode_result['served']
                self.summary_stats['total_rejected'] += episode_result['rejected']
                self.summary_stats['total_in_data'] += episode_result['total_in_data']
        
        print("="*80)
        print("实验完成")
    
    def _print_summary(self):
        """打印实验总结"""
        if not self.all_results:
            return
        
        print("\n" + "="*80)
        avg_profit = self.summary_stats['total_profit'] / len(self.all_results)
        print(f"平均收益: ¥{avg_profit:.2f}")
        print("="*80)
    
    def get_daily_statistics(self):
        """获取每日详细统计"""
        return self.all_results
    
    def get_summary_statistics(self):
        """获取汇总统计"""
        return self.summary_stats


# ========== 12. 主程序（测试模式） ==========

def main_test():
    """主程序 - 测试模式"""
    print("="*80)
    print("神经网络双服务网约车系统 - 测试模式")
    print("使用训练好的神经网络和真实数据")
    print("="*80)
    
    # 加载训练好的模型
    model_path = "adp_17_19_model_ratio_0_9_20260328_004957.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 配置参数（与训练时相同）
    n_nodes = 9
    express_num = 270 
    premium_num = 30  
    
    print(f"\n车队规模配置:")
    print(f"  Express快车: {express_num}辆")
    print(f"  Premium专车: {premium_num}辆")
    print(f"  总车队规模: {express_num + premium_num}辆")
    
    # 行程时间矩阵（分钟）
    travel_matrix = np.array([
        [ 6.46,  14.83,   7.824, 29.463, 12.757, 20.332, 13.342, 14.41,  12.668],
        [12.743,  9.379, 17.15,  17.791, 10.018, 13.632, 16.319, 20.545, 12.437],
        [ 9.09,  21.585,  4.976, 36.239, 18.118, 25.383, 15.384, 10.56,  18.118],
        [26.704, 17.536, 30.826,  6.864, 19.824, 16.,    20.172, 23.118, 19.47 ],
        [10.335, 11.357, 14.843, 19.069,  8.952, 17.463, 19.998, 22.937, 15.533],
        [16.623, 11.494, 20.532, 13.672, 13.707,  9.512, 10.622, 15.011,  9.354],
        [10.43,  15.497, 13.229, 20.801, 17.366, 12.439,  5.625,  7.075,  8.228],
        [12.883, 21.483, 10.22,  28.771, 22.309, 18.252,  7.147,  4.954, 13.189],
        [10.133, 11.745, 13.84,  19.511, 12.766, 12.391,  8.802, 13.124,  7.842]
    ])
    # 计算决策后状态维度
    post_state_dim = 2 * n_nodes + 4 * n_nodes * n_nodes
    
    # 真实数据配置 — 基于脚本文件位置定位，兼容任意 cwd
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    data_folder = os.path.join(_project_root, "data", "filtered_daily_trips_17_19_core_zones")
    
    # 区域映射：真实Taxi Zone ID -> 仿真节点ID (0-8)
    zone_mapping = {
        237: 0,  # 区域237映射到节点0
        161: 1,  # 区域161映射到节点1
        236: 2,  # 区域236映射到节点2
        186: 3,  # 区域186映射到节点3
        162: 4,  # 区域162映射到节点4
        230: 5,  # 区域230映射到节点5
        142: 6,  # 区域142映射到节点6
        239: 7,  # 区域239映射到节点7
        163: 8   # 区域163映射到节点8
    }
    
    print(f"\n真实数据配置:")
    print(f"  数据文件夹: {data_folder}")
    print(f"  区域映射: {zone_mapping}")
    
    # 检查数据文件夹是否存在
    if not os.path.exists(data_folder):
        print(f"数据文件夹不存在: {data_folder}")
        return None
    
    # 创建测试智能体（加载训练好的模型）
    trained_agent = TrainedNeuralADPAgent(
        model_path=model_path,
        post_state_dim=post_state_dim,
        env=None,
        epsilon_end=0.01,
        device=device
    )
    
    # 创建测试环境
    test_env = TestRideHailingEnvRealData(
        n_nodes=n_nodes,
        express_num=express_num,
        premium_num=premium_num,
        travel_matrix=travel_matrix,
        data_folder=data_folder,
        zone_mapping=zone_mapping,
        trained_agent=trained_agent
    )
    
    # 设置智能体中的环境引用
    trained_agent.env = test_env
    
    # 检查可用天数
    available_dates = test_env.passenger_generator.get_available_dates()
    if not available_dates:
        print("没有找到可用的日期数据文件")
        return None
    
    print(f"\n找到 {len(available_dates)} 天的数据")
    print(f"日期范围: {min(available_dates)} 到 {max(available_dates)}")
    print("="*80)
    
    # 创建测试运行器
    runner = TestExperimentRunner(test_env)
    
    # 运行测试（使用所有可用日期）
    results = runner.run_test_suite(dates_to_test=available_dates)
    
    return results


if __name__ == "__main__":
    # 设置基础日志（仅控制台输出）
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 运行测试
    results = main_test()
    
    if results:
        print(f"\n总共测试了 {len(results)} 天的数据")
        print("="*80)
        print("神经网络测试完成")
        print("="*80)
