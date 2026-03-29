# ========== 第一部分：核心数据结构 ==========

import heapq
import random
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Deque, Any
from enum import Enum
import numpy as np
from collections import deque, defaultdict  # 添加defaultdict
import logging  # 添加日志模块

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
    
    注意：这个向量是给智能体作为观察输入的，包含[决策类型, 乘客信息S_t^1, 
        车辆完成信息S_t^3, 车辆分布S_t^2]。
    神经网络 V_θ 不应该直接使用这个向量，V_θ只接收决策后状态 S_t^a。
    
    智能体使用 S_t 来计算 Q(S_t, a) = R_inst(S_t, a) + V_θ(S_t^a)
    
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
        
        # 添加日志记录器
        self.logger = logging.getLogger('environment')
        
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
            self.logger.warning(f"警告：行程时间为负值 {travel_time_seconds:.1f}s，收益设为0")
            return 0.0
            
        # 将行程时间转换为小时
        travel_time_hours = travel_time_seconds / 3600.0
        
        # 计算行程距离（公里）: 距离 = 速度 × 时间
        distance_km = self.average_speed * travel_time_hours
        
        # 计算收益
        if vehicle_type == "E":  # 快车
            revenue = self.express_base_fare + self.express_rate_per_km * distance_km
            self.logger.debug(f"快车收益计算: 时间={travel_time_hours:.3f}h, 距离={distance_km:.2f}km, "
                            f"收益=¥{self.express_base_fare:.1f} + ¥{self.express_rate_per_km:.2f}*{distance_km:.2f} = ¥{revenue:.2f}")
        else:  # 专车
            revenue = self.premium_base_fare + self.premium_rate_per_km * distance_km
            self.logger.debug(f"专车收益计算: 时间={travel_time_hours:.3f}h, 距离={distance_km:.2f}km, "
                            f"收益=¥{self.premium_base_fare:.1f} + ¥{self.premium_rate_per_km:.2f}*{distance_km:.2f} = ¥{revenue:.2f}")
        
        # 确保收益非负
        revenue = max(0.0, revenue)
        
        # 记录区域内出行的情况
        if origin == destination:
            self.logger.debug(f"区域内出行收益计算: 区域{origin}, 行程时间={travel_time_seconds:.1f}s, 收益=¥{revenue:.2f}")
        
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
            self.logger.warning(f"警告：行程时间为负值 {travel_time_seconds:.1f}s，成本设为0")
            return 0.0
                
        # 重新平衡成本 = 重新平衡系数 × 对应行程收益
        trip_revenue = self.calculate_trip_revenue(
            origin, destination, vehicle_type, travel_time_seconds
        )
        cost = -self.relocation_cost_factor * trip_revenue
        
        # 记录区域内出行的情况
        if origin == destination:
            self.logger.debug(f"区域内出行重新平衡成本: 区域{origin}, 行程时间={travel_time_seconds:.1f}s, "
                            f"收益=¥{trip_revenue:.2f}, 成本=¥{cost:.2f}")
        
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

# ========== 6. 经验回放数据类 ==========

@dataclass
class Experience:
    """经验回放数据 - 按照算法存储 (S_t^a, S_{t+1}, r_t^trans)
    
    关键改动：新增 next_action_evals 字段，存储 S_{t+1} 时所有可行动作的
    (R_inst, post_state_vector) 对，使 learn() 不再依赖当前环境状态。
    """
    post_state: PostDecisionState  # 决策后状态 S_t^a
    transition_reward: float = 0.0  # 转移奖励 R_trans(S_t^a, W_{t+1})
    next_action_evals: Optional[List[Tuple[float, np.ndarray]]] = None  # S_{t+1}所有可行动作的 (R_inst, post_state_vector)
    done: bool = False  # 是否结束
    
    def __str__(self) -> str:
        n_next = len(self.next_action_evals) if self.next_action_evals else 0
        return (f"Experience: TransReward={self.transition_reward:.2f}, "
                f"NextActions={n_next}, Done={self.done}")

# ========== 以下为第二个代码中的EfficientNHPPPassengerGenerator类 ==========
# 注意：这里先声明类，但实际的完整类定义将在第三部分完全替换

class EfficientNHPPPassengerGenerator:
    """高效NHPP乘客生成器 - 使用nhpp包生成非齐次泊松过程"""
    pass  # 具体实现在第三部分完全替换

# ========== 第二部分：Network类 ==========

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
        
        # 添加日志记录器
        self.logger = logging.getLogger('environment')
        
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

    def print_vehicle_statistics(self):
        """记录车辆统计信息到日志"""
        stats = self.get_network_statistics()
        self.logger.info(f"车辆统计:")
        self.logger.info(f"  空闲快车: {stats.total_express_idle}")
        self.logger.info(f"  空闲专车: {stats.total_premium_idle}")
        self.logger.info(f"  占用快车: {stats.total_express_occupied}")
        self.logger.info(f"  占用专车: {stats.total_premium_occupied}")
        self.logger.info(f"  调度快车: {stats.total_express_relocating}")
        self.logger.info(f"  调度专车: {stats.total_premium_relocating}")
        self.logger.info(f"  在途队列: {stats.arrival_queue_size}")

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
            
    def peek_arrival_queue(self) -> Optional[CarOnTheWay]:
        """查看但不删除到达逻辑时间最早的车辆记录"""
        if self.arrival_queue:
            return self.arrival_queue[0]
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
                                travel_time_seconds: float, current_time: float) -> Tuple[bool, float, PostDecisionState]:
        """
        执行派单动作 - 使用之前评估时生成的行程时间
        
        Args:
            origin: 起点区域
            destination: 终点区域
            vehicle_type: 车辆类型 ("E"或"P")
            travel_time_seconds: 之前评估时生成的行程时间（对于起终点相同，也是随机生成的正值）
            current_time: 当前逻辑时间
            
        Returns:
            (success, actual_revenue, post_decision_state)
        """
        success = False
        actual_revenue = 0.0
        post_decision_state = None
        
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
                
                # 获取真实的决策后状态 S_t^a
                post_state_vector = self.get_vehicle_distribution_vector()
                post_decision_state = PostDecisionState(post_state_vector)
                
                # 更新统计
                self.total_revenue += actual_revenue
                self.revenue_by_type["express"] += actual_revenue
                self.served_passengers += 1
                
                # 记录区域内出行
                if origin == destination:
                    self.logger.info(f"区域内出行派单: 快车服务区域内出行{origin}，行程时间{travel_time_seconds:.1f}s，收益¥{actual_revenue:.2f}")
                
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
                
                # 获取真实的决策后状态 S_t^a
                post_state_vector = self.get_vehicle_distribution_vector()
                post_decision_state = PostDecisionState(post_state_vector)
                
                # 更新统计
                self.total_revenue += actual_revenue
                self.revenue_by_type["premium"] += actual_revenue
                self.served_passengers += 1
                
                # 记录区域内出行
                if origin == destination:
                    self.logger.info(f"区域内出行派单: 专车服务区域内出行{origin}，行程时间{travel_time_seconds:.1f}s，收益¥{actual_revenue:.2f}")
                
                success = True
        
        return success, actual_revenue, post_decision_state
    
    def execute_rebalancing_action(self, origin: int, destination: int, vehicle_type: str,
                                travel_time_seconds: float, current_time: float) -> Tuple[bool, float, PostDecisionState]:
        """
        执行重新平衡动作 - 使用之前评估时生成的行程时间
        
        Args:
            origin: 起点区域
            destination: 目标区域
            vehicle_type: 车辆类型 ("E"或"P")
            travel_time_seconds: 之前评估时生成的行程时间
            current_time: 当前逻辑时间
            
        Returns:
            (success, actual_cost, post_decision_state)
        """
        success = False
        actual_cost = 0.0
        post_decision_state = None
        
        if origin == destination:
            # 留在原地：不更新车辆状态，不添加到队列，立即返回
            post_state_vector = self.get_vehicle_distribution_vector()
            post_decision_state = PostDecisionState(post_state_vector)
            return True, 0.0, post_decision_state
            
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
                
                # 获取真实的决策后状态 S_t^a
                post_state_vector = self.get_vehicle_distribution_vector()
                post_decision_state = PostDecisionState(post_state_vector)
                
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
                
                # 获取真实的决策后状态 S_t^a
                post_state_vector = self.get_vehicle_distribution_vector()
                post_decision_state = PostDecisionState(post_state_vector)
                
                # 更新统计
                self.total_relocation_cost += actual_cost
                self.cost_by_type["premium"] += actual_cost
                
                success = True
        
        return success, actual_cost, post_decision_state

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
            self.logger.warning(f"行程时间矩阵中({origin}, {destination})的值为{mean_travel_time_minutes}，设为0.1分钟")
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
        
        # 记录起终点相同的情况
        if origin == destination:
            self.logger.debug(f"区域内出行时间生成: 区域{origin}, "
                            f"平均时间={mean_travel_time_minutes:.2f}min, "
                            f"生成时间={travel_time_minutes:.2f}min = {travel_time_seconds:.1f}s")
        
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
    
    def get_total_idle_vehicles(self, vehicle_type: str = None) -> int:
        """获取总的空闲车辆数量"""
        if vehicle_type == "E":
            return sum(self.express_idle_num.values())
        elif vehicle_type == "P":
            return sum(self.premium_idle_num.values())
        else:
            return sum(self.express_idle_num.values()) + sum(self.premium_idle_num.values())
    
    def get_total_occupied_vehicles(self, vehicle_type: str = None) -> int:
        """获取总的占用车辆数量"""
        if vehicle_type == "E":
            return sum(self.express_occupy_num.values())
        elif vehicle_type == "P":
            return sum(self.premium_occupy_num.values())
        else:
            return sum(self.express_occupy_num.values()) + sum(self.premium_occupy_num.values())
    
    def get_total_relocation_vehicles(self, vehicle_type: str = None) -> int:
        """获取总的调度车辆数量"""
        if vehicle_type == "E":
            return sum(self.express_relocation_num.values())
        elif vehicle_type == "P":
            return sum(self.premium_relocation_num.values())
        else:
            return sum(self.express_relocation_num.values()) + sum(self.premium_relocation_num.values())
    
    def get_network_statistics(self) -> NetworkStats:
        """获取网络统计信息"""
        return NetworkStats(
            total_express_idle=self.get_total_idle_vehicles("E"),
            total_premium_idle=self.get_total_idle_vehicles("P"),
            total_express_occupied=self.get_total_occupied_vehicles("E"),
            total_premium_occupied=self.get_total_occupied_vehicles("P"),
            total_express_relocating=self.get_total_relocation_vehicles("E"),
            total_premium_relocating=self.get_total_relocation_vehicles("P"),
            arrival_queue_size=len(self.arrival_queue)
        )
    
    def get_financial_statistics(self) -> Dict[str, float]:
        """获取财务统计信息"""
        total_profit = self.total_revenue + self.total_relocation_cost
        
        return {
            "total_revenue": self.total_revenue,
            "total_relocation_cost": self.total_relocation_cost,
            "total_profit": total_profit,
            "revenue_express": self.revenue_by_type["express"],
            "revenue_premium": self.revenue_by_type["premium"],
            "cost_express": self.cost_by_type["express"],
            "cost_premium": self.cost_by_type["premium"],
            "avg_revenue_per_trip": self.total_revenue / max(1, self.served_passengers)
        }
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        total_passengers = self.served_passengers + self.rejected_passengers
        return {
            "served_passengers": self.served_passengers,
            "rejected_passengers": self.rejected_passengers,
            "total_passengers": total_passengers,
            "service_rate": self.served_passengers / max(1, total_passengers),
            "express_utilization": self.get_total_occupied_vehicles("E") / max(1, self.express_num),
            "premium_utilization": self.get_total_occupied_vehicles("P") / max(1, self.premium_num)
        }
    
    def validate_vehicle_count(self) -> bool:
        """验证车辆总数是否保持不变"""
        total_express = (self.get_total_idle_vehicles("E") + 
                        self.get_total_occupied_vehicles("E") + 
                        self.get_total_relocation_vehicles("E"))
        total_premium = (self.get_total_idle_vehicles("P") + 
                        self.get_total_occupied_vehicles("P") + 
                        self.get_total_relocation_vehicles("P"))
        total = total_express + total_premium
        
        is_valid = total == self.total_cars
        return is_valid
    
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
# ========== 第三部分：使用nhpp包的高效NHPP乘客生成器 ==========

import numpy as np
import random
import time
from collections import deque, defaultdict
from typing import Optional, Dict, List, Tuple

try:
    from nhpp import NonHomogeneousPoissonProcess
    NHPP_AVAILABLE = True
except ImportError:
    NHPP_AVAILABLE = False
    print("警告: 未安装nhpp包，将使用备用thinning算法")
    print("请安装: pip install nhpp")

class EfficientNHPPPassengerGenerator:
    """
    高效NHPP乘客生成器 - 使用nhpp包生成非齐次泊松过程
    """
    
    def __init__(self, n_nodes: int, passenger_generation_logic_time: float, 
                 nhpp_parameters: dict, use_nhpp_package: bool = True):
        """
        初始化高效NHPP乘客生成器
        
        Args:
            n_nodes: 节点数量（应为9，对应9个区域）
            passenger_generation_logic_time: 乘客生成的逻辑时间限制（秒）
            nhpp_parameters: NHPP参数字典，格式为{(i,j): (coeff, lambda_max)}
                            coeff: [a0, a1, a2, a3] 对应 λ(t) = a0 + a1·t + a2·t² + a3·t³
                            lambda_max: 最大到达率（乘客/秒）
            use_nhpp_package: 是否使用nhpp包（如果可用）
        """
        self.n_nodes = n_nodes
        self.passenger_generation_logic_time = passenger_generation_logic_time
        self.nhpp_parameters = nhpp_parameters
        
        # 分离系数和λ_max
        self.coefficients_dict = {}
        self.lambda_max_dict = {}
        
        for (i, j), (coeff, lambda_max) in nhpp_parameters.items():
            self.coefficients_dict[(i, j)] = coeff
            self.lambda_max_dict[(i, j)] = lambda_max
        
        # 检查nhpp包是否可用
        self.use_nhpp = use_nhpp_package and NHPP_AVAILABLE
        
        # 乘客类型生成概率 [快车, 双类型, 专车]
        self.passenger_type_probs = [0.5, 0.3, 0.2]
        
        # 乘客队列
        self.passenger_queue = deque()
        self.next_id = 0
        
        # 生成统计
        self.generation_stats = {
            'total_generated': 0,
            'express_count': 0,
            'dual_count': 0, 
            'premium_count': 0,
            'same_origin_destination_count': 0,
            'od_counts': defaultdict(int),
            'expected_total_passengers': 0.0,
            'actual_total_passengers': 0,
            'generation_time': 0.0,
            'od_generation_counts': defaultdict(int),
            'total_lambda_max': sum(lambda_max for lambda_max in self.lambda_max_dict.values()),
            'same_od_details': {},
            'using_nhpp_package': self.use_nhpp
        }
        
        # 添加日志记录器
        import logging
        self.logger = logging.getLogger('environment')
        
        self.logger.info(f"高效NHPP乘客生成器初始化完成")
        self.logger.info(f"使用{'nhpp包' if self.use_nhpp else '备用thinning算法'}")
        self.logger.info(f"OD对数量: {len(self.coefficients_dict)}")
        self.logger.info(f"总λ_max: {self.generation_stats['total_lambda_max']:.6f} 乘客/秒")
        self.logger.info(f"理论最大乘客数/2小时: {self.generation_stats['total_lambda_max'] * 7200:.0f}")
        
        if not self.use_nhpp and use_nhpp_package:
            self.logger.warning("nhpp包不可用，将使用备用thinning算法")
            self.logger.info("建议安装: pip install nhpp")
    
    def _create_lambda_function(self, coeff):
        """创建λ(t)函数"""
        a0, a1, a2, a3 = coeff
        
        def lambda_func(t):
            # λ(t) = a0 + a1·t + a2·t² + a3·t³
            return max(0.0, a0 + a1*t + a2*t*t + a3*t*t*t)
        
        return lambda_func
    
    def _generate_with_nhpp_package(self, i: int, j: int) -> np.ndarray:
        """
        使用nhpp包生成NHPP事件
        
        Args:
            i: 起点区域索引
            j: 终点区域索引
            
        Returns:
            事件时间数组（秒）
        """
        coeff = self.coefficients_dict.get((i, j))
        if coeff is None:
            return np.array([])
        
        # 创建λ(t)函数
        lambda_func = self._create_lambda_function(coeff)
        
        try:
            # 使用nhpp包生成事件
            nhpp = NonHomogeneousPoissonProcess(
                lambda_func, 
                t_min=0, 
                t_max=self.passenger_generation_logic_time
            )
            
            events = nhpp.generate()
            
            # 确保返回的是numpy数组
            if isinstance(events, list):
                events = np.array(events)
            elif events is None:
                events = np.array([])
            
            return events
            
        except Exception as e:
            self.logger.error(f"使用nhpp包生成OD({i}→{j})时出错: {e}")
            return np.array([])
    
    def _generate_with_backup_thinning(self, i: int, j: int) -> np.ndarray:
        """
        备用thinning算法生成NHPP事件
        
        Args:
            i: 起点区域索引
            j: 终点区域索引
            
        Returns:
            事件时间数组（秒）
        """
        coeff = self.coefficients_dict.get((i, j))
        lambda_max = self.lambda_max_dict.get((i, j), 0.0)
        
        if coeff is None or lambda_max <= 0:
            return np.array([])
        
        # 创建λ(t)函数
        lambda_func = self._create_lambda_function(coeff)
        
        # 简单thinning算法
        events = []
        t = 0.0
        
        # 设置最大迭代次数
        max_iterations = int(lambda_max * self.passenger_generation_logic_time * 3)
        
        for _ in range(max_iterations):
            # 生成指数分布的时间间隔
            u = np.random.random()
            interval = -np.log(1 - u) / lambda_max
            t += interval
            
            if t >= self.passenger_generation_logic_time:
                break
            
            # 计算接受概率
            lambda_current = lambda_func(t)
            accept_prob = lambda_current / lambda_max if lambda_max > 0 else 0
            accept_prob = min(1.0, max(0.0, accept_prob))
            
            if np.random.random() < accept_prob:
                events.append(t)
        
        return np.array(events)
    
    def _compute_expected_passengers(self):
        """计算期望乘客数 - 使用数值积分"""
        total_expected = 0.0
        
        T = self.passenger_generation_logic_time
        
        for (i, j), coeff in self.coefficients_dict.items():
            a0, a1, a2, a3 = coeff
            
            # 对λ(t)在[0,T]上数值积分
            # ∫[0,T] (a0 + a1·t + a2·t² + a3·t³) dt
            # = a0·T + a1·T²/2 + a2·T³/3 + a3·T⁴/4
            
            if T > 0:
                integral = a0*T + a1*(T**2)/2 + a2*(T**3)/3 + a3*(T**4)/4
                total_expected += max(0, integral)
        
        self.generation_stats['expected_total_passengers'] = total_expected
        
        # 计算区域内出行的期望值
        same_od_expected = 0.0
        for i in range(self.n_nodes):
            if (i, i) in self.coefficients_dict:
                a0, a1, a2, a3 = self.coefficients_dict[(i, i)]
                integral = a0*T + a1*(T**2)/2 + a2*(T**3)/3 + a3*(T**4)/4
                same_od_expected += max(0, integral)
        
        self.generation_stats['expected_same_od_passengers'] = same_od_expected
        self.generation_stats['same_od_proportion'] = same_od_expected / total_expected if total_expected > 0 else 0
        
        self.logger.info(f"计算期望乘客数: {total_expected:.1f}")
        self.logger.info(f"其中区域内出行期望: {same_od_expected:.1f} ({self.generation_stats['same_od_proportion']*100:.1f}%)")
    
    def generate_nhpp_passengers_fast(self) -> int:
        """
        快速生成NHPP乘客
        
        Returns:
            生成的乘客总数
        """
        start_time = time.time()
        
        self.logger.info(f"开始生成NHPP乘客 (时间范围: {self.passenger_generation_logic_time:.1f}s)")
        self.logger.info(f"使用{'nhpp包' if self.use_nhpp else '备用thinning算法'}")
        self.logger.info(f"总λ_max: {self.generation_stats['total_lambda_max']:.6f} 乘客/秒")
        
        # 清空队列和统计
        self.passenger_queue = deque()
        self.next_id = 0
        self._reset_statistics()
        
        # 计算期望值
        self._compute_expected_passengers()
        
        # 并行处理OD对（使用线程池）
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            all_arrivals = []
            od_keys = list(self.coefficients_dict.keys())
            
            # 使用线程池并行生成
            with ThreadPoolExecutor(max_workers=min(8, len(od_keys))) as executor:
                # 提交所有任务
                future_to_od = {}
                for od_key in od_keys:
                    i, j = od_key
                    if self.use_nhpp:
                        future = executor.submit(self._generate_with_nhpp_package, i, j)
                    else:
                        future = executor.submit(self._generate_with_backup_thinning, i, j)
                    future_to_od[future] = od_key
                
                # 收集结果
                for future in as_completed(future_to_od):
                    od_key = future_to_od[future]
                    i, j = od_key
                    try:
                        arrivals = future.result()
                        if arrivals is not None and len(arrivals) > 0:
                            # 转换为(时间, i, j)格式
                            for t in arrivals:
                                all_arrivals.append((t, i, j))
                            
                            self.generation_stats['od_generation_counts'][od_key] = len(arrivals)
                            
                            # 记录生成较多的OD对
                            if len(arrivals) > 5:
                                lambda_max = self.lambda_max_dict.get((i, j), 0)
                                self.logger.debug(f"OD({i}→{j}): λ_max={lambda_max:.6f}, 生成{len(arrivals)}名乘客")
                    except Exception as e:
                        self.logger.error(f"生成OD({i}→{j})时出错: {e}")
        except ImportError:
            # 如果没有concurrent.futures，使用串行生成
            self.logger.warning("未找到concurrent.futures，使用串行生成")
            all_arrivals = []
            
            for (i, j) in self.coefficients_dict.keys():
                if self.use_nhpp:
                    arrivals = self._generate_with_nhpp_package(i, j)
                else:
                    arrivals = self._generate_with_backup_thinning(i, j)
                
                if arrivals is not None and len(arrivals) > 0:
                    for t in arrivals:
                        all_arrivals.append((t, i, j))
                    
                    self.generation_stats['od_generation_counts'][(i, j)] = len(arrivals)
                    
                    if len(arrivals) > 5:
                        lambda_max = self.lambda_max_dict.get((i, j), 0)
                        self.logger.debug(f"OD({i}→{j}): λ_max={lambda_max:.6f}, 生成{len(arrivals)}名乘客")
        
        # 按到达时间排序
        all_arrivals.sort(key=lambda x: x[0])
        
        self.logger.info(f"共生成{len(all_arrivals)}个候选到达时间，正在分配乘客类型...")
        
        # 批量创建乘客
        for arrival_time, origin, destination in all_arrivals:
            # 生成乘客类型
            passenger_type_idx = np.random.choice([0, 1, 2], p=self.passenger_type_probs)
            
            if passenger_type_idx == 0:
                passenger_type = PassengerType.EXPRESS
            elif passenger_type_idx == 1:
                passenger_type = PassengerType.DUAL
            else:
                passenger_type = PassengerType.PREMIUM
            
            # 创建乘客
            passenger = Passenger(
                passenger_id=self.next_id,
                arrival_logic_time=arrival_time,
                origin=origin,
                destination=destination,
                passenger_type=passenger_type
            )
            
            # 添加到队列
            self.passenger_queue.append(passenger)
            self.next_id += 1
            
            # 更新统计
            self._update_statistics(passenger)
        
        # 更新统计
        self.generation_stats['actual_total_passengers'] = len(self.passenger_queue)
        self.generation_stats['generation_time'] = time.time() - start_time
        
        # 记录结果
        self._log_generation_summary()
        
        return len(self.passenger_queue)
    
    def generate_nhpp_passengers(self) -> int:
        """主生成函数（向后兼容）"""
        return self.generate_nhpp_passengers_fast()
    
    def _update_statistics(self, passenger):
        """更新统计信息"""
        self.generation_stats['total_generated'] += 1
        
        # 乘客类型统计
        if passenger.passenger_type == PassengerType.EXPRESS:
            self.generation_stats['express_count'] += 1
        elif passenger.passenger_type == PassengerType.DUAL:
            self.generation_stats['dual_count'] += 1
        elif passenger.passenger_type == PassengerType.PREMIUM:
            self.generation_stats['premium_count'] += 1
        
        # OD对统计
        od_key = (passenger.origin, passenger.destination)
        self.generation_stats['od_counts'][od_key] += 1
        
        # 起终点相同的乘客统计
        if passenger.origin == passenger.destination:
            self.generation_stats['same_origin_destination_count'] += 1
            # 记录具体的区域内出行
            self.generation_stats['same_od_details'][passenger.origin] = \
                self.generation_stats['same_od_details'].get(passenger.origin, 0) + 1
    
    def _reset_statistics(self):
        """重置统计"""
        self.generation_stats = {
            'total_generated': 0,
            'express_count': 0,
            'dual_count': 0,
            'premium_count': 0,
            'same_origin_destination_count': 0,
            'od_counts': defaultdict(int),
            'expected_total_passengers': 0.0,
            'actual_total_passengers': 0,
            'generation_time': 0.0,
            'od_generation_counts': defaultdict(int),
            'total_lambda_max': self.generation_stats.get('total_lambda_max', 0),
            'same_od_details': {},
            'expected_same_od_passengers': 0.0,
            'same_od_proportion': 0.0,
            'using_nhpp_package': self.use_nhpp
        }
    
    def _log_generation_summary(self):
        """记录生成摘要"""
        total_gen = self.generation_stats['total_generated']
        expected = self.generation_stats['expected_total_passengers']
        actual = self.generation_stats['actual_total_passengers']
        gen_time = self.generation_stats['generation_time']
        total_lambda_max = self.generation_stats['total_lambda_max']
        same_od_count = self.generation_stats['same_origin_destination_count']
        
        self.logger.info("\n" + "="*60)
        self.logger.info(f"NHPP乘客生成完成 ({'使用nhpp包' if self.use_nhpp else '使用备用算法'})")
        self.logger.info("-"*60)
        self.logger.info(f"生成耗时: {gen_time:.3f}秒")
        self.logger.info(f"系统总λ_max: {total_lambda_max:.6f} 乘客/秒")
        self.logger.info(f"理论最大乘客数: {total_lambda_max * self.passenger_generation_logic_time:.0f}")
        self.logger.info(f"理论期望乘客数: {expected:.1f}")
        self.logger.info(f"实际生成乘客数: {actual}")
        
        if expected > 0:
            diff = actual - expected
            diff_percent = diff / expected * 100
            self.logger.info(f"差异: {diff:+.1f} ({diff_percent:+.2f}%)")
        
        if total_gen > 0:
            # 乘客类型分布
            express_count = self.generation_stats['express_count']
            dual_count = self.generation_stats['dual_count']
            premium_count = self.generation_stats['premium_count']
            
            self.logger.info(f"乘客类型分布:")
            self.logger.info(f"  快车专用: {express_count} ({express_count/total_gen*100:.1f}%)")
            self.logger.info(f"  双类型: {dual_count} ({dual_count/total_gen*100:.1f}%)")
            self.logger.info(f"  专车专用: {premium_count} ({premium_count/total_gen*100:.1f}%)")
            
            self.logger.info(f"起终点相同的乘客: {same_od_count} ({same_od_count/total_gen*100:.1f}%)")
            
            # 验证OD 0->0（区域237->237）
            if (0, 0) in self.generation_stats['od_generation_counts']:
                od_00_count = self.generation_stats['od_generation_counts'][(0, 0)]
                lambda_max_00 = self.lambda_max_dict.get((0, 0), 0)
                self.logger.info(f"验证OD 237->237 (索引0->0):")
                self.logger.info(f"  λ_max: {lambda_max_00:.6f} 乘客/秒")
                self.logger.info(f"  生成乘客数: {od_00_count}")
        
        self.logger.info(f"当前队列长度: {len(self.passenger_queue)}")
        self.logger.info("="*60)
    
    # 乘客队列操作方法
    def get_next_passenger(self) -> Optional[Passenger]:
        """获取下一个乘客"""
        if self.passenger_queue:
            return self.passenger_queue.popleft()
        return None
    
    def peek_next_passenger(self) -> Optional[Passenger]:
        """查看下一个乘客但不移除"""
        if self.passenger_queue:
            return self.passenger_queue[0]
        return None
    
    def get_queue_length(self) -> int:
        """获取队列长度"""
        return len(self.passenger_queue)
    
    def get_next_passenger_time(self) -> float:
        """获取下一个乘客的到达时间"""
        if self.passenger_queue:
            return self.passenger_queue[0].arrival_logic_time
        else:
            return float('inf')
    
    def print_statistics(self):
        """打印统计信息"""
        self._log_generation_summary()
    
    def reset(self):
        """重置生成器"""
        self.passenger_queue = deque()
        self.next_id = 0
        self._reset_statistics()
        self.logger.info("NHPP乘客生成器已重置")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return self.generation_stats.copy()
    
    def get_lambda_max_for_od(self, i: int, j: int) -> float:
        """获取指定OD对的λ_max"""
        return self.lambda_max_dict.get((i, j), 0.0)
    
    def get_coefficients_for_od(self, i: int, j: int) -> Optional[np.ndarray]:
        """获取指定OD对的多项式系数"""
        return self.coefficients_dict.get((i, j)) 
# ========== 第四部分：强化学习环境 ==========

import gym
from gym import spaces

class EfficientNHPPRideHailingEnv(gym.Env):
    """双服务网约车环境 - 使用高效NHPP乘客生成器"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, n_nodes: int, express_num: int, premium_num: int, 
                 travel_matrix: np.ndarray, nhpp_parameters: dict,
                 passenger_generation_time: float, use_nhpp_package: bool = True):
        """
        初始化环境（使用NHPP乘客生成器）
        
        Args:
            n_nodes: 区域数量
            express_num: 快车数量
            premium_num: 专车数量
            travel_matrix: 行程时间矩阵（包括对角线上的区域内行程时间）
            nhpp_parameters: NHPP参数字典，格式为{(i,j): (coeff, lambda_max)}
                            coeff: [a0, a1, a2, a3] 对应 λ(t) = a0 + a1·t + a2·t² + a3·t³
                            lambda_max: 最大到达率（乘客/秒）
            passenger_generation_time: 乘客生成时间（秒）
            use_nhpp_package: 是否使用nhpp包（如果可用）
        """
        super(EfficientNHPPRideHailingEnv, self).__init__()

        self.logger = logging.getLogger('environment')
        self.summary_logger = logging.getLogger('summary')

        self.n_nodes = n_nodes
        self.express_num = express_num
        self.premium_num = premium_num
        self.travel_matrix = travel_matrix
        self.nhpp_parameters = nhpp_parameters
        self.passenger_generation_time = passenger_generation_time

        
        # 初始化网络
        self.network = Network(n_nodes, express_num, premium_num, travel_matrix)
        
        # 初始化NHPP乘客生成器（替换原来的SimplePassengerGenerator）
        self.passenger_generator = EfficientNHPPPassengerGenerator(
            n_nodes=n_nodes,
            passenger_generation_logic_time=passenger_generation_time,
            nhpp_parameters=nhpp_parameters,
            use_nhpp_package=use_nhpp_package
        )
        
        # 状态空间定义
        # 计算状态向量维度
        vehicle_dist_dim = self._get_vehicle_distribution_dim()
        # 注意：乘客信息S_t^1是n_nodes * n_nodes维，包括origin==destination的情况
        pre_state_dim = (n_nodes * n_nodes) + (2 * n_nodes) + vehicle_dist_dim
        
        # 状态空间（连续的数值向量）
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(pre_state_dim,),  # 使用决策前状态作为观察
            dtype=np.float32
        )
        
        # 动作空间
        # 注意：动作空间是动态的，取决于当前状态类型
        # 这里先定义最大动作空间
        self.max_action_space = spaces.Discrete(max(n_nodes, 2))
        
        # 当前状态和统计
        self.current_time = 0.0
        self.current_pre_state = None
        self.current_passenger = None
        self.current_vehicle_completion = None
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_revenue = 0.0
        self.episode_cost = 0.0
        self.episode_served = 0
        self.episode_rejected = 0
        
        # 经验缓存（用于训练）
        self.current_experience = None
        
        self.logger.info(f"高效NHPP网约车环境初始化完成")
        self.logger.info(f"使用{'nhpp包' if use_nhpp_package else '备用thinning算法'}")
        self.logger.info(f"NHPP参数数量: {len(nhpp_parameters)}")
        
    def _get_vehicle_distribution_dim(self) -> int:
        """计算车辆分布向量的维度"""
        n_nodes = self.n_nodes
        # 空闲车辆: n_nodes * 2 (快车+专车)
        # 占用车辆: n_nodes * n_nodes * 2 (所有OD对 × 2种车辆，包括origin==destination)
        # 调度车辆: n_nodes * n_nodes * 2 (所有OD对 × 2种车辆，包括origin==destination)
        return (n_nodes * 2) + (n_nodes * n_nodes * 2) + (n_nodes * n_nodes * 2)
    
    def _get_next_event(self):
        # 乘客生成器使用独立的时间生成了乘客
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
    
    def _handle_passenger_arrival(self, passenger: Passenger) -> Tuple[PreDecisionState, bool]:
        """
        处理乘客到达事件
        
        Returns:
            (decision_state, need_decision)
        """
        origin = passenger.origin
        destination = passenger.destination
        passenger_type = passenger.passenger_type
        
        # 记录乘客信息（包括是否起终点相同）
        if origin == destination:
            self.logger.info(f"乘客{passenger.passenger_id}: 区域内出行（起终点相同）区域{origin}，类型={passenger_type.value}")
        
        # 根据乘客类型处理
        if passenger_type == PassengerType.EXPRESS:
            # 快车专用乘客
            if self.network.has_idle_vehicles_at_node(origin, "E"):
                # 有快车可用，自动分配
                action_eval = self.network.evaluate_dispatching_action(origin, destination, "E")
                success, revenue, post_state = self.network.execute_dispatching_action(
                    origin, destination, "E", 
                    action_eval.travel_time_seconds, 
                    self.current_time
                )
                if success:
                    is_same = "（起终点相同）" if origin == destination else ""
                    self.logger.info(f"乘客{passenger.passenger_id}: 快车专用乘客{is_same}，自动分配快车，行程时间{action_eval.travel_time_seconds:.1f}s，收益¥{revenue:.2f}")
                    self.episode_revenue += revenue
                    self.episode_reward += revenue
                    self.episode_served += 1
                return None, False
            else:
                # 无快车可用，乘客流失
                is_same = "（起终点相同）" if origin == destination else ""
                self.logger.info(f"乘客{passenger.passenger_id}: 快车专用乘客{is_same}，无快车可用，乘客流失")
                self.episode_rejected += 1
                return None, False
                
        elif passenger_type == PassengerType.PREMIUM:
            # 专车专用乘客
            if self.network.has_idle_vehicles_at_node(origin, "P"):
                # 有专车可用，自动分配
                action_eval = self.network.evaluate_dispatching_action(origin, destination, "P")
                success, revenue, post_state = self.network.execute_dispatching_action(
                    origin, destination, "P", 
                    action_eval.travel_time_seconds, 
                    self.current_time
                )
                if success:
                    is_same = "（起终点相同）" if origin == destination else ""
                    self.logger.info(f"乘客{passenger.passenger_id}: 专车专用乘客{is_same}，自动分配专车，行程时间{action_eval.travel_time_seconds:.1f}s，收益¥{revenue:.2f}")
                    self.episode_revenue += revenue
                    self.episode_reward += revenue
                    self.episode_served += 1
                return None, False
            else:
                # 无专车可用，乘客流失
                is_same = "（起终点相同）" if origin == destination else ""
                self.logger.info(f"乘客{passenger.passenger_id}: 专车专用乘客{is_same}，无专车可用，乘客流失")
                self.episode_rejected += 1
                return None, False
                
        else:  # PassengerType.DUAL
            # 双类型乘客
            has_express = self.network.has_idle_vehicles_at_node(origin, "E")
            has_premium = self.network.has_idle_vehicles_at_node(origin, "P")
            
            if has_express and has_premium:
                # 两种车都有，需要决策
                is_same = "（起终点相同）" if origin == destination else ""
                self.logger.info(f"乘客{passenger.passenger_id}: 双类型乘客{is_same}，两种车都可用，需要决策")
                self.current_passenger = passenger
                decision_state = PreDecisionState(
                    state_type=DecisionType.DISPATCHING,
                    passenger_info=(origin, destination)
                )
                return decision_state, True
            elif has_express:
                # 只有快车可用，自动分配快车
                action_eval = self.network.evaluate_dispatching_action(origin, destination, "E")
                success, revenue, post_state = self.network.execute_dispatching_action(
                    origin, destination, "E", 
                    action_eval.travel_time_seconds, 
                    self.current_time
                )
                if success:
                    is_same = "（起终点相同）" if origin == destination else ""
                    self.logger.info(f"乘客{passenger.passenger_id}: 双类型乘客{is_same}，只有快车可用，自动分配快车，行程时间{action_eval.travel_time_seconds:.1f}s，收益¥{revenue:.2f}")
                    self.episode_revenue += revenue
                    self.episode_reward += revenue
                    self.episode_served += 1
                return None, False
            elif has_premium:
                # 只有专车可用，自动分配专车
                action_eval = self.network.evaluate_dispatching_action(origin, destination, "P")
                success, revenue, post_state = self.network.execute_dispatching_action(
                    origin, destination, "P", 
                    action_eval.travel_time_seconds, 
                    self.current_time
                )
                if success:
                    is_same = "（起终点相同）" if origin == destination else ""
                    self.logger.info(f"乘客{passenger.passenger_id}: 双类型乘客{is_same}，只有专车可用，自动分配专车，行程时间{action_eval.travel_time_seconds:.1f}s，收益¥{revenue:.2f}")
                    self.episode_revenue += revenue
                    self.episode_reward += revenue
                    self.episode_served += 1
                return None, False
            else:
                # 两种车都无，乘客流失
                is_same = "（起终点相同）" if origin == destination else ""
                self.logger.info(f"乘客{passenger.passenger_id}: 双类型乘客{is_same}，两种车都无，乘客流失")
                self.episode_rejected += 1
                return None, False
    
    def _handle_vehicle_arrival(self, car: CarOnTheWay) -> Tuple[PreDecisionState, bool]:
        """
        处理车辆到达事件
        
        注意：不再对起终点相同的行程做特殊处理
        """
        origin = car.origin
        destination = car.destination
        vehicle_type = car.vehicle_type
        car_state = car.car_state
        
        # 记录是否起终点相同
        is_same = "（起终点相同）" if origin == destination else ""
        self.logger.info(f"车辆到达{is_same}: {vehicle_type}车从{origin}到{destination}, 状态={car_state.value}")
        
        # 根据车辆状态处理
        if car_state == CarState.RELOCATING:
            # 重新调度车辆到达，变为空闲
            if vehicle_type == "E":
                success = self.network.update_express_from_relocation_to_idle(origin, destination)
            else:
                success = self.network.update_premium_from_relocation_to_idle(origin, destination)
            
            if success:
                self.logger.info(f"  → {vehicle_type}车完成重新调度，变为空闲在区域{destination}")
                # 重新调度车辆到达后不触发重新平衡决策（no back-to-back relocation规则）
                return None, False
                
        elif car_state == CarState.OCCUPIED:
            # 载客车辆到达，完成行程 - TRIGGERS RELOCATION DECISION
            if vehicle_type == "E":
                success = self.network.update_express_from_occupied_to_idle(origin, destination)
            else:
                success = self.network.update_premium_from_occupied_to_idle(origin, destination)
            
            if success:
                is_same = "（起终点相同）" if origin == destination else ""
                self.logger.info(f"  → {vehicle_type}车完成乘客行程{is_same}，变为空闲在区域{destination}（触发重新平衡决策）")
                
                # 只有载客车辆完成行程时才触发重新平衡决策
                self.current_vehicle_completion = (destination, vehicle_type)
                decision_state = PreDecisionState(
                    state_type=DecisionType.REBALANCING,
                    vehicle_completion_info=(destination, vehicle_type)
                )
                return decision_state, True
        
        return None, False
    
    def step(self, action: Any, action_info: Optional[Dict] = None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作
        """
        if self.current_pre_state is None:
            raise ValueError("没有待处理的决策状态")
        
        # 执行动作
        reward = 0.0
        post_state = None
        
        if self.current_pre_state.state_type == DecisionType.DISPATCHING:
            # 派单决策
            origin, destination = self.current_pre_state.passenger_info
            
            # 记录是否区域内出行
            is_same = origin == destination
            
            # 从action解析车辆类型
            if action == 0:  # 派快车
                vehicle_type = "E"
            else:  # 派专车
                vehicle_type = "P"
            
            # 检查动作有效性
            if not self.network.has_idle_vehicles_at_node(origin, vehicle_type):
                # 动作无效：没有对应类型的车辆
                same_str = "（区域内出行）" if is_same else ""
                self.logger.info(f"警告：动作无效！区域{origin}没有{vehicle_type}车可用{same_str}")
                reward = 0.0
                self.episode_rejected += 1
            else:
                # 执行动作
                if action_info and 'travel_time_seconds' in action_info:
                    # 使用智能体提供的行程时间
                    travel_time_seconds = action_info['travel_time_seconds']
                else:
                    # 如果没有提供，则重新生成（不推荐）
                    self.logger.info(f"警告：智能体未提供行程时间，重新生成")
                    travel_time_seconds = self.network.get_travel_time(origin, destination)
                
                # 执行动作
                success, revenue, post_state = self.network.execute_dispatching_action(
                    origin, destination, vehicle_type,
                    travel_time_seconds,
                    self.current_time
                )
                
                if success:
                    reward = revenue
                    self.episode_revenue += revenue
                    self.episode_served += 1
                    same_str = "（区域内出行）" if is_same else ""
                    self.logger.info(f"执行派单决策{same_str}: 派{vehicle_type}车服务乘客，行程时间{travel_time_seconds:.1f}s，收益¥{revenue:.2f}")
                else:
                    # 如果执行失败，乘客流失
                    reward = 0.0
                    self.episode_rejected += 1
                    same_str = "（区域内出行）" if is_same else ""
                    self.logger.info(f"执行派单决策失败{same_str}: 无{vehicle_type}车可用，乘客流失")
        
        else:  # REBALANCING
            # 重新平衡决策
            zone, vehicle_type = self.current_pre_state.vehicle_completion_info
            destination = action  # 动作就是目标区域
            
            # 记录是否留在原地
            is_same = zone == destination
            
            # 检查动作有效性
            if destination < 0 or destination >= self.n_nodes:
                # 动作无效：目标区域不存在
                self.logger.info(f"警告：动作无效！目标区域{destination}不存在")
                reward = 0.0
            elif not self.network.has_idle_vehicles_at_node(zone, vehicle_type):
                # 动作无效：没有对应类型的车辆
                same_str = "（留在原地）" if is_same else ""
                self.logger.info(f"警告：动作无效！区域{zone}没有{vehicle_type}车可用{same_str}")
                reward = 0.0
            else:
                # 执行动作
                if action_info and 'travel_time_seconds' in action_info:
                    # 使用智能体提供的行程时间
                    travel_time_seconds = action_info['travel_time_seconds']
                else:
                    # 如果没有提供，则重新生成（不推荐）
                    travel_time_seconds = 0 if is_same else self.network.get_travel_time(zone, destination)
                
                # 执行动作
                success, cost, post_state = self.network.execute_rebalancing_action(
                    zone, destination, vehicle_type,
                    travel_time_seconds,
                    self.current_time
                )
                
                if success:
                    reward = cost  # 注意：成本是负值
                    self.episode_cost += cost
                    if is_same:
                        self.logger.info(f"执行重新平衡决策: {vehicle_type}车保持在区域{zone}，无成本")
                    else:
                        self.logger.info(f"执行重新平衡决策: {vehicle_type}车从区域{zone}重新调度到区域{destination}，行程时间{travel_time_seconds:.1f}s，成本¥{cost:.2f}")
                else:
                    reward = 0.0
                    self.logger.info(f"执行重新平衡决策失败")
        
        # 更新总奖励
        self.episode_reward += reward
        
        # 存储经验（仅保留 post_state，其余字段由 Trainer 回填）
        if post_state is not None:
            self.current_experience = Experience(
                post_state=post_state,
            )
        
        # 重置当前状态
        self.current_pre_state = None
        self.current_passenger = None
        self.current_vehicle_completion = None
        
        # 推进到下一个事件
        return self._advance_to_next_event()
    
    def _advance_to_next_event(self) -> Tuple[np.ndarray, float, bool, dict]:
        """
        推进到下一个事件
        
        Returns:
            (observation, reward, done, info)
        """
        while True:
            # 获取下一个事件
            next_event_time, event_type = self._get_next_event()
            
            # 更新时间
            time_elapsed = next_event_time - self.current_time
            self.current_time = next_event_time
            
            # 处理事件
            if event_type == 'episode_end':
                # Episode结束条件：乘客队列为空且没有在途车辆
                done = True
                observation = self._get_observation(None)
                
                # 如果有未完成的经验，标记为结束
                if self.current_experience is not None:
                    self.current_experience.done = True
                
                info = self._get_info()
                
                # 统计起终点相同的乘客服务情况
                total_same_od = self.passenger_generator.generation_stats.get('same_origin_destination_count', 0)
                self.logger.info(f"Episode结束: 当前时间={self.current_time:.2f}s, 已服务乘客={self.episode_served}")
                if total_same_od > 0:
                    self.logger.info(f"  其中起终点相同的乘客: 生成{total_same_od}个，服务{self.episode_served}个，流失{self.episode_rejected}个")
                
                # 将摘要信息输出到控制台
                self.summary_logger.info(f"Episode结束: 服务={self.episode_served}, 流失={self.episode_rejected}, "
                                        f"收入={self.episode_revenue:.1f}, 成本={self.episode_cost:.1f}, "
                                        f"利润={self.episode_revenue + self.episode_cost:.1f}")
                return observation, 0.0, done, info
                    
            elif event_type == 'passenger_arrival':
                # 乘客到达
                passenger = self.passenger_generator.get_next_passenger()
                decision_state, need_decision = self._handle_passenger_arrival(passenger)
                
                if need_decision:
                    # 需要决策
                    done = False
                    self.current_pre_state = decision_state
                    observation = self._get_observation(decision_state)
                    info = self._get_info()
                    return observation, 0.0, done, info
                # 否则继续处理下一个事件
                
            elif event_type == 'vehicle_arrival':
                # 车辆到达
                car = self.network.pop_arrival_queue()
                decision_state, need_decision = self._handle_vehicle_arrival(car)
                
                if need_decision:
                    # 需要决策
                    done = False
                    self.current_pre_state = decision_state
                    observation = self._get_observation(decision_state)
                    info = self._get_info()
                    return observation, 0.0, done, info
                # 否则继续处理下一个事件
            else:
                # 不应该到达这里，但为了安全起见
                self.logger.warning(f"未知的事件类型 '{event_type}'，结束episode")
                done = True
                observation = self._get_observation(None)
                info = self._get_info()
                return observation, 0.0, done, info
        
    def _get_observation(self, decision_state: Optional[PreDecisionState]) -> np.ndarray:
        """获取观察（决策前状态向量）"""
        if decision_state is None:
            # 如果没有决策状态，返回零向量
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            return decision_state.get_state_vector(self.network)
    
    def _get_info(self) -> dict:
        """获取环境信息"""
        # 获取起终点相同乘客的统计
        total_same_od = self.passenger_generator.generation_stats.get('same_origin_destination_count', 0)
        same_od_stats = {
            'same_od_generated': total_same_od,
            'same_od_served': 0,  # 这个需要额外跟踪，这里暂时为0
            'same_od_rejected': 0  # 这个需要额外跟踪，这里暂时为0
        }
        
        # 获取NHPP生成器的统计
        nhpp_stats = self.passenger_generator.generation_stats
        
        return {
            'current_time': self.current_time,
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'episode_revenue': self.episode_revenue,
            'episode_cost': self.episode_cost,
            'episode_served': self.episode_served,
            'episode_rejected': self.episode_rejected,
            'vehicle_stats': self.network.get_network_statistics(),
            'current_experience': self.current_experience,
            'same_od_stats': same_od_stats,
            'nhpp_stats': nhpp_stats,
            'total_lambda_max': nhpp_stats.get('total_lambda_max', 0),
            'expected_passengers': nhpp_stats.get('expected_total_passengers', 0)
        }
    
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            observation: 初始观察
        """
        # 重置网络
        self.network.reset()
        
        # 重置NHPP乘客生成器（生成新的NHPP乘客）
        self.passenger_generator.reset()
        self.passenger_generator.generate_nhpp_passengers_fast()
        
        # 重置状态变量
        self.current_time = 0.0
        self.current_pre_state = None
        self.current_passenger = None
        self.current_vehicle_completion = None
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_revenue = 0.0
        self.episode_cost = 0.0
        self.episode_served = 0
        self.episode_rejected = 0
        self.current_experience = None
        
        # 将重置信息记录到日志
        self.logger.info("\n" + "="*60)
        self.logger.info("NHPP网约车环境已重置")
        self.logger.info(f"使用{'nhpp包' if self.passenger_generator.use_nhpp else '备用thinning算法'}")
        
        # 记录NHPP统计
        stats = self.passenger_generator.generation_stats
        self.logger.info(f"NHPP参数: {len(self.nhpp_parameters)}个OD对")
        self.logger.info(f"总λ_max: {stats.get('total_lambda_max', 0):.6f} 乘客/秒")
        self.logger.info(f"期望乘客数: {stats.get('expected_total_passengers', 0):.1f}")
        self.logger.info(f"实际生成乘客数: {stats.get('actual_total_passengers', 0)}")
        self.logger.info("="*60)
        
        # 推进到第一个事件
        observation, _, _, _ = self._advance_to_next_event()
        return observation
    
    def render(self, mode='human'):
        """渲染环境状态 - 记录到日志而不输出到控制台"""
        if mode == 'human':
            self.logger.info(f"\n当前时间: {self.current_time:.2f}s")
            self.logger.info(f"仿真进度: {self.current_time/self.passenger_generation_time*100:.1f}%")
            self.logger.info(f"步数: {self.episode_step}")
            self.logger.info(f"累计奖励: {self.episode_reward:.2f}")
            self.logger.info(f"累计收入: {self.episode_revenue:.2f}")
            self.logger.info(f"累计成本: {self.episode_cost:.2f}")
            self.logger.info(f"服务乘客: {self.episode_served}, 流失乘客: {self.episode_rejected}")
            
            # 显示NHPP乘客生成统计
            stats = self.passenger_generator.generation_stats
            remaining_passengers = self.passenger_generator.get_queue_length()
            total_generated = stats.get('actual_total_passengers', 0)
            if total_generated > 0:
                self.logger.info(f"NHPP生成: {total_generated}乘客, 剩余: {remaining_passengers}")
                self.logger.info(f"系统λ_max: {stats.get('total_lambda_max', 0):.6f}乘客/秒")
            
            if self.current_pre_state is not None:
                self.logger.info(f"当前决策状态: {self.current_pre_state}")
            
            # 调用network的print_vehicle_statistics方法，它现在会记录到日志
            self.network.print_vehicle_statistics()
            
            if self.current_experience is not None:
                self.logger.info(f"上一个经验: {self.current_experience}")
    
    def close(self):
        """关闭环境"""
        pass
    
    def get_valid_actions(self) -> List:
        """获取当前有效的动作列表"""
        if self.current_pre_state is None:
            return []
        
        if self.current_pre_state.state_type == DecisionType.DISPATCHING:
            # 派单决策：0=快车，1=专车
            return [0, 1]
        else:  # REBALANCING
            # 重新平衡决策：可以选择任何区域（包括当前区域）
            return list(range(self.n_nodes))
    
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
    
    def get_experience(self) -> Optional[Experience]:
        """获取当前的经验（如果有的话）"""
        return self.current_experience

# 为了向后兼容，保留原类名（可选）
DualServiceRideHailingEnv = EfficientNHPPRideHailingEnv
# ========== 第五部分：智能体与神经网络 ==========

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import numpy as np
import wandb

# ========== 1. 神经网络定义 ==========

class PostDecisionValueNetwork(nn.Module):
    """决策后状态价值网络 V_θ(S^a)
    
    改进：
    - 输入端添加 LayerNorm 做状态归一化
    - 输出层使用 xavier_uniform_ 初始化（无激活函数）
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super(PostDecisionValueNetwork, self).__init__()
        
        # 输入归一化层
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # 激活函数
        self.activation = nn.ReLU()
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.0)
        
        # 输出层使用 xavier（无激活函数）
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入归一化
        x = self.input_norm(x)
        
        # 隐藏层
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # 输出层（无激活函数）
        x = self.output_layer(x)
        return x

# ========== 2. 经验回放缓冲区 ==========

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        """添加经验到缓冲区"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

# ========== 3. 神经网络ADP智能体 ==========

class NeuralADPAgent:
    """基于神经网络的近似动态规划智能体 (Double DQN)

    关键特性：
    1. Double DQN: 用当前网络选择动作，用目标网络评估价值（减少过估计）
    2. learn() 完全基于存储数据计算 TD 目标（不再依赖当前环境状态）
    3. ε 衰减移到 decay_epsilon() 由 Trainer 在每个决策步调用
    4. 使用 MSE Loss
    """

    def __init__(self, 
                 post_state_dim: int,
                 pre_state_dim: int,
                 env: DualServiceRideHailingEnv,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 50000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 tau: float = 0.005,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 wandb_config: dict = None):
        self.post_state_dim = post_state_dim
        self.pre_state_dim = pre_state_dim
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.device = device
        
        self.logger = logging.getLogger('agent')
        self.summary_logger = logging.getLogger('summary')
        
        # 初始化网络
        self.value_network = PostDecisionValueNetwork(post_state_dim).to(device)
        self.target_network = PostDecisionValueNetwork(post_state_dim).to(device)
        self.target_network.load_state_dict(self.value_network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练计数器 — 对应算法中的 C_update
        self.update_counter = 0
        self.train_step = 0
        self.total_reward = 0.0
        
        self.logger.info(f"智能体初始化完成，设备: {device}")
        self.summary_logger.info(f"智能体初始化: 状态维度(前{pre_state_dim}/后{post_state_dim}), 设备:{device}")

    def select_action(self, current_pre_state: PreDecisionState) -> Tuple[Any, ActionEvaluation, Dict[Any, float]]:
        """
        选择动作（ε-greedy策略） - Algorithm 1 第5-11行
        
        对每个可行动作 a:
            Q̂(S_t, a; θ) = R_inst(S_t, a) + V_θ(S_t^a)
        然后按 ε-greedy 选择。
        
        注意：这里的 Q̂ 里 V_θ 不需要乘 γ，γ 只在 TD 目标中使用一次。
        """
        action_evals = self.env.evaluate_actions()
        if not action_evals:
            return None, None, {}
        
        # 批量计算所有动作的 Q̂ 值
        post_state_vectors = []
        for ae in action_evals:
            post_state_vectors.append(ae.post_decision_state.get_state_vector())
        
        batch_tensor = torch.FloatTensor(np.array(post_state_vectors)).to(self.device)
        
        with torch.no_grad():
            values = self.value_network(batch_tensor).squeeze(-1).cpu().numpy()
        
        all_q_values = {}
        for i, ae in enumerate(action_evals):
            # Q̂(S_t, a) = R_inst(S_t, a) + V_θ(S_t^a)  — 无 γ
            q = ae.estimated_immediate_reward + values[i]
            all_q_values[ae.action] = q
        
        # ε-greedy
        if random.random() < self.epsilon:
            selected_action_eval = random.choice(action_evals)
            self.logger.debug(f"探索: 随机选择动作 {selected_action_eval.action}")
        else:
            max_q = max(all_q_values.values())
            max_actions = [a for a, q in all_q_values.items() if q == max_q]
            selected_action_id = random.choice(max_actions)
            selected_action_eval = next(ae for ae in action_evals if ae.action == selected_action_id)
            self.logger.debug(f"利用: 选择动作 {selected_action_id}, Q={max_q:.4f}")
        
        return selected_action_eval.action, selected_action_eval, all_q_values
    
    def store_experience(self, experience: Experience):
        """存储经验到回放缓冲区 — Algorithm 1 第14行"""
        self.replay_buffer.add(experience)
    
    def learn(self) -> Optional[float]:
        """
        从经验回放缓冲区学习 — Algorithm 1 第16-28行 (Double DQN)

        Double DQN 核心思想：
        - 用当前网络（value_network）选择最优动作 a* = argmax_a'[R_inst + V_θ(S'^{a'})]
        - 用目标网络（target_network）评估该动作的价值 V_{θ⁻}(S'^{a*})
        - TD 目标：y_i = r_trans + γ * [R_inst(a*) + V_{θ⁻}(S'^{a*})]
        - 使用 MSE Loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        # 准备训练数据
        post_state_vectors = []
        target_values = []

        for exp in batch:
            post_state_vectors.append(exp.post_state.get_state_vector())

            if exp.done or exp.next_action_evals is None or len(exp.next_action_evals) == 0:
                # 终止状态：y_i = r_i^trans
                y_i = exp.transition_reward
            else:
                # 准备下一状态所有可行动作的数据
                next_post_vecs = []
                next_inst_rewards = []
                for (r_inst, post_vec) in exp.next_action_evals:
                    next_inst_rewards.append(r_inst)
                    next_post_vecs.append(post_vec)

                next_batch = torch.FloatTensor(np.array(next_post_vecs)).to(self.device)

                with torch.no_grad():
                    # Double DQN: 用当前网络选择动作
                    current_values = self.value_network(next_batch).squeeze(-1).cpu().numpy()
                    q_values_current = [r_s + v for r_s, v in zip(next_inst_rewards, current_values)]
                    best_action_idx = int(np.argmax(q_values_current))

                    # Double DQN: 用目标网络评估该动作的价值
                    target_values_net = self.target_network(next_batch).squeeze(-1).cpu().numpy()
                    max_next_q = next_inst_rewards[best_action_idx] + target_values_net[best_action_idx]

                # y_i = r_trans + γ * max_next_q
                y_i = exp.transition_reward + self.gamma * max_next_q

            target_values.append(y_i)
        
        # 转换为张量
        post_state_tensor = torch.FloatTensor(np.array(post_state_vectors)).to(self.device)
        target_tensor = torch.FloatTensor(target_values).unsqueeze(1).to(self.device)

        # 前向传播
        predicted_values = self.value_network(post_state_tensor)

        # Huber Loss（对异常值更鲁棒）
        # delta设置：当|pred-target| > delta时使用线性损失，否则使用平方损失
        # 根据loss范围600-1200，设置delta=8.0（正常误差范围内用平方损失，异常值用线性损失）
        loss = F.huber_loss(predicted_values, target_tensor, delta=8.0)

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新计数器
        self.train_step += 1
        self.update_counter += 1

        # 记录loss到wandb
        loss_value = loss.item()

        # 额外监控：计算如果用MSE会是多少（用于对比）
        mse_loss_value = F.mse_loss(predicted_values, target_tensor).item()
        diff_abs_max = torch.abs(predicted_values - target_tensor).max().item()

        wandb.log({
            'train/loss': loss_value,                    # Huber Loss
            'train/mse_loss_reference': mse_loss_value,  # MSE Loss作为参考
            'train/max_abs_diff': diff_abs_max,          # 最大绝对差值
            'train/epsilon': self.epsilon,
            'train/train_step': self.train_step
        }, step=self.train_step)
        
        # 目标网络软更新
        if self.update_counter >= self.target_update_freq:
            self._update_target_network()
            self.update_counter = 0
        
        self.logger.debug(f"训练步数 {self.train_step}: 损失 = {loss_value:.6f}")
        
        return loss_value
    
    def _update_target_network(self):
        """软更新目标网络：θ⁻ ← τθ + (1-τ)θ⁻"""
        for target_param, param in zip(self.target_network.parameters(), 
                                      self.value_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        if self.train_step % 1000 == 0:
            self.logger.info(f"目标网络已更新（训练步数: {self.train_step}，tau={self.tau}）")
            # 记录目标网络更新到wandb
            wandb.log({'train/target_update': self.train_step}, step=self.train_step)
    
    def decay_epsilon(self):
        """衰减探索率 — 由 Trainer 在每个决策步结束时调用"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    
    def get_training_info(self) -> Dict[str, Any]:
        return {
            'train_step': self.train_step,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer)
        }
    
    def print_training_info(self):
        info = self.get_training_info()
        self.logger.info(f"训练信息: 步数={info['train_step']}, ε={info['epsilon']:.4f}, 缓冲区={info['buffer_size']}")
    
    def reset_episode_stats(self):
        self.total_reward = 0.0

# ========== 4. 训练器类 ==========

class Trainer:
    """训练器类 — 管理训练流程
    
    关键修复：
    - 正确回填经验的 transition_reward 和 next_action_evals
    - ε 衰减在每个决策步结束后执行
    """
    
    def __init__(self, 
                 agent: NeuralADPAgent,
                 env: DualServiceRideHailingEnv,
                 num_episodes: int = 1000,
                 save_freq: int = 100,
                 log_freq: int = 10,
                 wandb_project: str = "ride-hailing-adp",
                 wandb_run_name: str = None):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.save_freq = save_freq
        self.log_freq = log_freq
        
        self.logger = logging.getLogger('trainer')
        self.summary_logger = logging.getLogger('summary')
        
        # 初始化wandb run
        if wandb_run_name is None:
            wandb_run_name = f"run_{np.random.randint(1000)}"
        
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "num_episodes": num_episodes,
                "learning_rate": agent.optimizer.param_groups[0]['lr'],
                "gamma": agent.gamma,
                "epsilon_start": agent.epsilon,
                "epsilon_end": agent.epsilon_end,
                "epsilon_decay": agent.epsilon_decay,
                "buffer_capacity": agent.replay_buffer.buffer.maxlen,
                "batch_size": agent.batch_size,
                "target_update_freq": agent.target_update_freq,
                "tau": agent.tau,
                "post_state_dim": agent.post_state_dim,
                "pre_state_dim": agent.pre_state_dim,
                "env_nodes": env.n_nodes,
                "env_express_num": env.express_num,
                "env_premium_num": env.premium_num,
                "passenger_generation_time": env.passenger_generation_time
            }
        )
        
        self.logger.info(f"训练器初始化完成，计划训练 {num_episodes} 回合")
        self.summary_logger.info(f"开始训练，共 {num_episodes} 回合，wandb项目: {wandb_project}")
    
    def _collect_action_evals(self) -> List[Tuple[float, np.ndarray]]:
        """收集当前状态所有可行动作的 (R_inst, post_state_vector) 对"""
        action_evals = self.env.evaluate_actions()
        if not action_evals:
            return []
        result = []
        for ae in action_evals:
            result.append((ae.estimated_immediate_reward, 
                          ae.post_decision_state.get_state_vector().copy()))
        return result
    
    def train_episode(self) -> Dict[str, float]:
        """
        训练一个回合 — 修复经验回填逻辑
        
        按照算法：
        1. 选择动作 → 执行动作 → 得到 S_t^a（post_state）
        2. 环境推进 → 得到 S_{t+1} 和 transition_reward
        3. 存储经验 (S_t^a, S_{t+1}, r_trans) 到 replay buffer
        
        实现方式：
        - 维护 prev_experience 追踪上一个未完成的经验
        - 在下一个决策点到来时，回填 transition_reward 和 next_action_evals
        """
        observation = self.env.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        self.agent.reset_episode_stats()
        
        # 追踪上一次决策的经验，等待回填
        prev_post_state: Optional[PostDecisionState] = None
        accumulated_transition_reward: float = 0.0  # 从上一个决策后状态到当前的累积转移奖励
        
        while not done:
            current_pre_state = self.env.get_current_pre_state()
            
            if current_pre_state is None:
                observation, reward, done, info = self.env._advance_to_next_event()
                episode_reward += reward
                continue
            
            # ========== 回填上一个经验并存储 ==========
            if prev_post_state is not None:
                # 收集当前状态 S_{t+1} 的所有可行动作评估
                next_action_evals = self._collect_action_evals()
                
                # 创建完整经验并存储
                experience = Experience(
                    post_state=prev_post_state,
                    transition_reward=accumulated_transition_reward,
                    next_action_evals=next_action_evals if next_action_evals else None,
                    done=False  
                )
                self.agent.store_experience(experience)
                
                # 学习
                if len(self.agent.replay_buffer) >= self.agent.batch_size:
                    self.agent.learn()
            
            # ========== 选择并执行动作 ==========
            action, action_eval, q_values = self.agent.select_action(current_pre_state)
            
            if action is None:
                observation, reward, done, info = self.env._advance_to_next_event()
                episode_reward += reward
                continue
            
            action_info = {
                'travel_time_seconds': action_eval.travel_time_seconds,
                'estimated_immediate_reward': action_eval.estimated_immediate_reward
            }
            
            # 执行动作 → 得到即时奖励 + 环境推进到下一个决策点
            observation, immediate_reward, done, info = self.env.step(action, action_info)
            episode_reward += immediate_reward
            
            # 从环境获取本次决策的 post_state
            env_exp = self.env.get_experience()
            if env_exp is not None:
                # 保存 post_state，等下一个决策点到来时回填
                prev_post_state = env_exp.post_state
                accumulated_transition_reward = 0.0
            else:
                prev_post_state = None
                accumulated_transition_reward = 0.0
            
            # ε 衰减 — 每个决策步结束后
            self.agent.decay_epsilon()
            
            episode_steps += 1
        
        # ========== Episode 结束：存储最后一个经验（done=True）==========
        if prev_post_state is not None:
            experience = Experience(
                post_state=prev_post_state,
                transition_reward=accumulated_transition_reward,
                next_action_evals=None,
                done=True
            )
            self.agent.store_experience(experience)
            
            if len(self.agent.replay_buffer) >= self.agent.batch_size:
                self.agent.learn()
        
        # 记录episode指标到wandb
        wandb.log({
            'episode/reward': episode_reward,
            'episode/steps': episode_steps,
            'episode/revenue': self.env.episode_revenue,
            'episode/cost': self.env.episode_cost,
            'episode/profit': self.env.episode_revenue + self.env.episode_cost,
            'episode/served': self.env.episode_served,
            'episode/rejected': self.env.episode_rejected,
            'episode/service_rate': self.env.episode_served / max(1, self.env.episode_served + self.env.episode_rejected),
            'episode/epsilon': self.agent.epsilon,
            'episode/buffer_size': len(self.agent.replay_buffer)
        }, step=self.agent.train_step)
        
        episode_info = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'episode_revenue': self.env.episode_revenue,
            'episode_cost': self.env.episode_cost,
            'episode_served': self.env.episode_served,
            'episode_rejected': self.env.episode_rejected,
            'total_profit': self.env.episode_revenue + self.env.episode_cost
        }
        
        return episode_info
    
    def train(self):
        """执行训练"""
        self.logger.info("="*60)
        self.logger.info("开始训练")
        self.logger.info("="*60)
        self.summary_logger.info("训练开始")
        
        for episode in range(1, self.num_episodes + 1):
            episode_info = self.train_episode()
            
            self.logger.info(f"回合 {episode}/{self.num_episodes}: "
                           f"奖励={episode_info['episode_reward']:.2f}, "
                           f"利润={episode_info['total_profit']:.2f}, "
                           f"服务={episode_info['episode_served']}, "
                           f"流失={episode_info['episode_rejected']}")
            
            if episode % self.log_freq == 0:
                agent_info = self.agent.get_training_info()
                self.summary_logger.info(
                    f"回合{episode:3d}: "
                    f"奖励={episode_info['episode_reward']:7.1f} "
                    f"利润={episode_info['total_profit']:7.1f} "
                    f"服务={episode_info['episode_served']:4d} "
                    f"流失={episode_info['episode_rejected']:4d} "
                    f"ε={self.agent.epsilon:.3f}"
                )
        
        self.logger.info("="*60)
        self.logger.info("训练完成")
        self.logger.info("="*60)
        self.summary_logger.info("训练完成")
        
        
        # 关闭wandb run
        wandb.finish()
        
        self.print_training_summary()
    
    def print_training_summary(self):
        """打印训练总结（仅控制台输出，数据从wandb查看）"""
        self.logger.info("="*60)
        self.logger.info("训练总结")
        self.logger.info("="*60)
        self.logger.info(f"总回合数: {self.num_episodes}")
        self.logger.info(f"最终探索率: {self.agent.epsilon:.3f}")
        self.logger.info(f"最终缓冲区大小: {len(self.agent.replay_buffer)}")
        self.logger.info(f"训练步数: {self.agent.train_step}")
        self.logger.info("\n详细训练曲线请查看wandb项目")
        
        self.summary_logger.info("="*40)
        self.summary_logger.info("训练总结")
        self.summary_logger.info("="*40)
        self.summary_logger.info(f"总回合数: {self.num_episodes}")
        self.summary_logger.info(f"最终探索率: {self.agent.epsilon:.3f}")
        self.summary_logger.info(f"详细训练曲线请查看wandb项目")
# ========== 第六部分：日志配置 ==========

import os
import logging
from datetime import datetime
from typing import Dict
import wandb

class LoggerConfig:
    """日志配置类 - 统一的日志系统设置（仅保留控制台输出）"""
    
    # 类变量，存储已创建的logger引用
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def setup_loggers(cls, log_dir: str = "logs", level: int = logging.INFO) -> Dict[str, logging.Logger]:
        """
        设置统一的日志系统（仅保留控制台输出，所有详细日志通过wandb记录）
        
        Args:
            log_dir: 日志目录（保留参数但不再使用，为兼容性保留）
            level: 日志级别
            
        Returns:
            Dict[str, logging.Logger]: 配置好的logger字典
        """
        # 定义要创建的logger名称
        logger_names = ['trainer', 'environment', 'agent', 'summary']
        
        # 清除所有现有logger的handler（防止重复）
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = False  # 禁止传播到父logger
        
        # 日志格式定义
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        summary_formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 配置各个logger
        cls._loggers = {}
        
        # 1. 训练器logger（控制台输出）
        trainer_logger = logging.getLogger('trainer')
        trainer_console = logging.StreamHandler()
        trainer_console.setLevel(level)
        trainer_console.setFormatter(console_formatter)
        trainer_logger.addHandler(trainer_console)
        cls._loggers['trainer'] = trainer_logger
        
        # 2. 环境logger（控制台输出）
        env_logger = logging.getLogger('environment')
        env_console = logging.StreamHandler()
        env_console.setLevel(level)
        env_console.setFormatter(console_formatter)
        env_logger.addHandler(env_console)
        cls._loggers['environment'] = env_logger
        
        # 3. 智能体logger（控制台输出）
        agent_logger = logging.getLogger('agent')
        agent_console = logging.StreamHandler()
        agent_console.setLevel(level)
        agent_console.setFormatter(console_formatter)
        agent_logger.addHandler(agent_console)
        cls._loggers['agent'] = agent_logger
        
        # 4. 摘要logger（简洁的控制台输出）
        summary_logger = logging.getLogger('summary')
        summary_console = logging.StreamHandler()
        summary_console.setLevel(level)
        summary_console.setFormatter(summary_formatter)
        summary_logger.addHandler(summary_console)
        cls._loggers['summary'] = summary_logger
        
        # 记录初始化信息
        summary_logger.info(f"日志系统已初始化，仅保留控制台输出")
        summary_logger.info(f"日志级别: {logging.getLevelName(level)}")
        summary_logger.info(f"详细训练曲线和指标请查看wandb项目")
        
        return cls._loggers
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取已配置的logger
        
        Args:
            name: logger名称 ('trainer', 'environment', 'agent', 'summary')
            
        Returns:
            logging.Logger: 对应的logger实例
            
        Raises:
            ValueError: 如果请求的logger不存在
        """
        if name not in cls._loggers:
            raise ValueError(f"Logger '{name}' 未配置。请先调用 setup_loggers()")
        return cls._loggers[name]
    
    @classmethod
    def add_console_handler(cls, logger_name: str, level: int = logging.INFO):
        """
        为指定logger添加控制台输出（如果尚未添加）
        
        Args:
            logger_name: logger名称
            level: 日志级别
        """
        if logger_name in cls._loggers:
            logger = cls._loggers[logger_name]
            # 检查是否已有StreamHandler
            has_stream = any(
                isinstance(handler, logging.StreamHandler) 
                for handler in logger.handlers
            )
            if not has_stream:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(level)
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                ))
                logger.addHandler(console_handler)
    
    @classmethod
    def cleanup(cls):
        """清理所有logger的handler并关闭wandb"""
        # 关闭wandb（如果处于活动状态）
        try:
            if wandb.run is not None:
                wandb.finish()
        except:
            pass
        
        # 清理logger handlers
        for logger_name in ['trainer', 'environment', 'agent', 'summary']:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()
# ========== 第七部分：主程序和测试 ==========

import logging
import time
from typing import Dict, Any
import wandb

# ========== 基于真实出租车数据的NHPP配置类（按秒计算的λ(t)函数） ==========

class RealNHPPConfig:
    """真实NHPP参数配置类 - 基于按秒计算的λ(t)函数"""
    
    def __init__(self, quick_test: bool = True):
        """
        初始化配置
        
        Args:
            quick_test: 是否快速测试模式
        """
        # 原始区域ID和映射
        self.original_zone_ids = [237, 161, 236, 186, 162, 230, 142, 239, 163]
        self.n_nodes = 9
        
        # 创建映射
        self.zone_to_idx = {zone: idx for idx, zone in enumerate(self.original_zone_ids)}
        self.idx_to_zone = {idx: zone for idx, zone in enumerate(self.original_zone_ids)}
        
        # 行程时间矩阵
        self.travel_matrix =  np.array([
    [6.095065, 12.676830, 7.456200, 27.558737, 10.594408, 18.981833, 13.565680, 14.377516, 11.535740],
    [11.123316, 8.745464, 15.501824, 16.953431, 9.221930, 12.725887, 15.589845, 19.934882, 11.458760],
    [8.708179, 18.470793, 4.822331, 32.131676, 15.598182, 24.639176, 15.033392, 10.513293, 17.185670],
    [24.104568, 15.698473, 28.693086, 7.422782, 17.223955, 14.270898, 18.512715, 21.920932, 17.156832],
    [9.197727, 10.546554, 13.072132, 17.778294, 8.060552, 15.498720, 19.464867, 21.402553, 13.607127],
    [14.984137, 10.033276, 19.728165, 12.377795, 12.237704, 9.081808, 10.825063, 15.109510, 9.267567],
    [10.434873, 14.605556, 13.125660, 19.957931, 15.867941, 11.334748, 5.550570, 6.831122, 7.922796],
    [12.661859, 19.881672, 9.886090, 25.166458, 20.431189, 16.644135, 6.938669, 4.771336, 12.904756],
    [9.457781, 10.383094, 12.952297, 18.012084, 11.087675, 11.520476, 9.454008, 13.270505, 6.973626]
])
        
        # 车辆数量配置
        if quick_test:
            self.express_num = 20  # 快速测试
            self.premium_num = 20  # 快速测试
            self.num_episodes = 10  # 快速测试回合数
            self.passenger_generation_time = 600.0  # 10分钟快速测试
        else:
            self.express_num = 240  # 完整训练
            self.premium_num = 60 # 完整训练
            self.num_episodes = 100  # 训练回合数
            self.passenger_generation_time = 7200.0  # 2小时完整测试
        
        # NHPP参数存储 - 使用按秒计算的λ(t)函数
        # 格式: {(i,j): (coeff, lambda_max)}，其中coeff是[a0, a1, a2, a3]对应λ(t)=a0+a1*t+a2*t²+a3*t³
        self.nhpp_parameters = {}
        
        # 初始化NHPP参数（81个OD对）
        self._initialize_nhpp_parameters()
        
        # 计算总统计
        self._calculate_statistics()
        
        # 训练参数
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.999970127261722
        self.buffer_capacity = 50000
        self.batch_size = 64
        self.target_update_freq = 4000
        self.tau = 0.005
        
        # 训练器参数
        self.save_freq = 10
        self.log_freq = 50
        
        # wandb配置
        self.wandb_project = "ride-hailing-adp"
        self.wandb_run_name = None  # 将在运行时生成
        
        self.quick_test = quick_test
        
    def _initialize_nhpp_parameters(self):
        """初始化NHPP参数 - 基于按秒计算的λ(t)函数"""
        # 格式: λ(t) = a0 + a1·t + a2·t² + a3·t³，其中t∈[0,7200]秒
        
        od_params_data = [
            # OD 237 -> 各个区域
            (237, 237, [4.288519e-06, -2.689440e-07, 4.249501e-09, 6.457106e-11], 1.825308e-02),
            (237, 161, [-4.626345e-06, 6.697208e-08, -1.070724e-10, 4.115956e-11], 9.126209e-03),
            (237, 236, [-1.182392e-06, -2.562051e-07, 5.005277e-09, 9.565997e-11], 2.572618e-02),
            (237, 186, [4.682650e-06, -1.109498e-07, 5.610211e-10, 6.897683e-12], 2.419063e-03),
            (237, 162, [-6.024571e-06, 6.068810e-08, 8.704002e-10, 2.569841e-11], 7.473166e-03),
            (237, 230, [-1.873998e-05, 4.862790e-07, -2.684213e-09, 1.749939e-11], 5.144042e-03),
            (237, 142, [8.689646e-06, -2.252779e-07, 1.816463e-09, 3.252245e-11], 1.042366e-02),
            (237, 239, [4.429280e-06, -1.496491e-07, 1.519578e-09, 2.077645e-11], 5.776408e-03),
            (237, 163, [-1.144856e-06, -3.780439e-09, 4.021873e-10, 1.961877e-11], 4.777839e-03),
            
            # OD 161 -> 各个区域
            (161, 237, [3.472105e-07, -4.941717e-08, 5.406386e-10, 6.598167e-11], 1.458492e-02),
            (161, 161, [-1.018171e-05, 3.583876e-07, -4.070351e-09, 4.457773e-11], 9.628790e-03),
            (161, 236, [-4.964172e-06, 2.323562e-08, 1.840490e-09, 4.232474e-11], 1.280434e-02),
            (161, 186, [5.255078e-07, 4.462124e-08, -1.504787e-09, 3.165679e-11], 6.837867e-03),
            (161, 162, [3.134278e-06, -1.147202e-07, 9.523395e-10, 1.470773e-11], 3.671841e-03),
            (161, 230, [-2.878093e-05, 8.468082e-07, -5.826712e-09, 2.726767e-11], 7.187542e-03),
            (161, 142, [9.402835e-06, -2.976492e-07, 2.999388e-09, 2.325727e-11], 8.512134e-03),
            (161, 239, [-8.379972e-06, 1.765043e-07, -3.476024e-10, 1.885440e-11], 5.562879e-03),
            (161, 163, [-2.599387e-06, 7.131556e-08, -3.431487e-10, 2.229608e-11], 5.277144e-03),
            
            # OD 236 -> 各个区域
            (236, 237, [-3.545301e-05, 8.040671e-07, -2.056319e-09, 8.819911e-11], 2.587264e-02),
            (236, 161, [-7.432181e-06, 1.601056e-07, -5.098230e-10, 2.236382e-11], 5.722213e-03),
            (236, 236, [-1.710715e-05, 4.385007e-07, -2.589361e-09, 7.243819e-11], 1.626894e-02),
            (236, 186, [-2.655691e-06, 8.416503e-08, -7.122386e-10, 6.049314e-12], 1.306652e-03),
            (236, 162, [6.193484e-07, -1.121463e-07, 1.951590e-09, 1.279575e-11], 4.697139e-03),
            (236, 230, [-1.338168e-05, 3.337041e-07, -1.785231e-09, 1.061783e-11], 3.114949e-03),
            (236, 142, [1.970654e-05, -4.994402e-07, 3.244396e-09, 3.501387e-11], 1.248006e-02),
            (236, 239, [4.128990e-07, -8.260660e-08, 1.809341e-09, 2.512357e-11], 7.701243e-03),
            (236, 163, [-5.808738e-06, 1.746543e-07, -1.567351e-09, 2.114130e-11], 4.566520e-03),
            
            # OD 186 -> 各个区域
            (186, 237, [1.576523e-06, -4.988972e-08, 3.726592e-10, 7.431860e-12], 1.781046e-03),
            (186, 161, [-3.368880e-06, 1.054401e-07, -1.232179e-09, 1.895229e-11], 4.093694e-03),
            (186, 236, [4.157142e-06, -1.427216e-07, 1.333047e-09, 3.838626e-12], 1.644879e-03),
            (186, 186, [-2.918445e-06, 7.099382e-08, -2.390545e-10, 4.544796e-12], 1.535697e-03),
            (186, 162, [5.208157e-06, -1.661684e-07, 1.541470e-09, 6.414841e-12], 2.687504e-03),
            (186, 230, [-1.239635e-05, 3.668534e-07, -2.621098e-09, 2.576806e-11], 5.865758e-03),
            (186, 142, [4.354207e-06, -1.354523e-07, 1.241482e-09, 5.211136e-12], 2.309800e-03),
            (186, 239, [-2.261560e-06, 4.694380e-08, -7.835045e-11, 4.997391e-12], 1.499862e-03),
            (186, 163, [-2.486778e-06, 3.916172e-08, -1.920848e-11, 1.254435e-11], 2.977322e-03),
            
            # OD 162 -> 各个区域
            (162, 237, [-4.147758e-06, -3.772395e-08, 2.588504e-09, 2.900237e-11], 1.025584e-02),
            (162, 161, [-7.788776e-06, 1.831703e-07, -8.585733e-10, 1.750935e-11], 4.362336e-03),
            (162, 236, [2.627539e-07, -2.411035e-08, 1.135312e-09, 2.812335e-11], 9.350095e-03),
            (162, 186, [-1.876813e-06, 1.630817e-08, 3.744094e-10, 1.459987e-11], 3.932941e-03),
            (162, 162, [2.392937e-06, -1.595325e-07, 2.547587e-09, 7.287699e-12], 4.151390e-03),
            (162, 230, [-2.192118e-05, 5.798147e-07, -2.920828e-09, 1.518187e-11], 6.028436e-03),
            (162, 142, [4.363591e-06, -1.545254e-07, 1.763856e-09, 1.259974e-11], 4.530688e-03),
            (162, 239, [-4.166526e-06, 1.254784e-07, -6.617667e-10, 1.141594e-11], 3.447187e-03),
            (162, 163, [-3.199967e-06, 4.481227e-08, 3.828355e-10, 9.360973e-12], 3.218061e-03),
            
            # OD 230 -> 各个区域
            (230, 237, [5.499063e-06, -2.586584e-07, 3.177738e-09, 8.185053e-12], 4.295474e-03),
            (230, 161, [-2.796452e-06, 6.537343e-08, -3.991822e-10, 1.741435e-11], 3.761499e-03),
            (230, 236, [5.377071e-06, -1.695466e-07, 1.596249e-09, 1.140365e-11], 4.001733e-03),
            (230, 186, [-3.950692e-06, 1.455167e-07, -1.470108e-09, 2.118264e-11], 4.575451e-03),
            (230, 162, [-5.874426e-06, 1.546964e-07, -9.074263e-10, 1.008862e-11], 2.503051e-03),
            (230, 230, [-1.740744e-05, 4.807491e-07, -2.587522e-09, 1.752019e-11], 6.076448e-03),
            (230, 142, [1.107320e-06, -7.050116e-08, 1.236275e-09, 2.131709e-11], 6.030266e-03),
            (230, 239, [4.344823e-06, -1.781532e-07, 2.212181e-09, 7.496671e-12], 3.490403e-03),
            (230, 163, [4.025765e-06, -1.642983e-07, 1.821601e-09, 1.097705e-11], 3.693648e-03),
            
            # OD 142 -> 各个区域
            (142, 237, [7.150659e-06, -2.490867e-07, 3.161889e-09, 1.496048e-11], 7.726075e-03),
            (142, 161, [-3.894388e-06, 1.122167e-07, -9.890127e-10, 1.756062e-11], 3.793093e-03),
            (142, 236, [-5.133085e-06, 1.411129e-07, -9.406289e-10, 2.464557e-11], 5.410782e-03),
            (142, 186, [-1.848661e-06, 8.040470e-08, -1.011235e-09, 7.260366e-12], 1.568239e-03),
            (142, 162, [-3.828699e-06, 9.905553e-08, -6.529523e-10, 1.098436e-11], 2.372621e-03),
            (142, 230, [-2.098277e-05, 6.235109e-07, -4.580695e-09, 2.174582e-11], 4.862090e-03),
            (142, 142, [-4.645113e-06, 1.314205e-07, -6.362314e-10, 2.197503e-11], 5.705123e-03),
            (142, 239, [7.770008e-06, -2.456481e-07, 2.486476e-09, 2.835362e-11], 9.068540e-03),
            (142, 163, [7.131891e-07, -2.772992e-08, 7.729083e-10, 1.332730e-11], 5.054186e-03),
            
            # OD 239 -> 各个区域
            (239, 237, [-9.327763e-06, 2.426117e-07, -1.456856e-09, 2.038103e-11], 4.747620e-03),
            (239, 161, [2.984133e-06, -1.064756e-07, 8.315010e-10, 9.358917e-12], 2.425228e-03),
            (239, 236, [-1.773589e-06, 4.795929e-09, 7.062432e-10, 2.174994e-11], 6.018498e-03),
            (239, 186, [1.313769e-06, -3.687938e-08, 2.827993e-10, 1.810669e-12], 6.966122e-04),
            (239, 162, [1.595291e-06, -5.582179e-08, 4.226275e-10, 4.888671e-12], 1.254451e-03),
            (239, 230, [-1.768897e-05, 4.472078e-07, -2.572915e-09, 1.418036e-11], 3.770382e-03),
            (239, 142, [1.551186e-05, -4.423415e-07, 3.652768e-09, 3.188613e-11], 1.125356e-02),
            (239, 239, [1.038816e-05, -2.790386e-07, 1.939109e-09, 2.106009e-11], 6.767739e-03),
            (239, 163, [3.049822e-06, -1.025645e-07, 9.879905e-10, 9.574753e-12], 2.744785e-03),
            
            # OD 163 -> 各个区域
            (163, 237, [2.533698e-07, -6.089925e-08, 1.718918e-09, 2.437903e-11], 8.064508e-03),
            (163, 161, [5.630440e-08, 3.619569e-10, -1.569541e-10, 2.528417e-11], 5.461380e-03),
            (163, 236, [4.579425e-06, -2.013888e-07, 3.071956e-09, 1.871742e-11], 7.827069e-03),
            (163, 186, [6.268557e-06, -1.648915e-07, 9.444878e-10, 1.057653e-11], 2.950162e-03),
            (163, 162, [4.645113e-06, -1.580243e-07, 1.725722e-09, 8.748108e-12], 3.718161e-03),
            (163, 230, [-3.278793e-05, 9.212004e-07, -5.592299e-09, 2.365097e-11], 7.858258e-03),
            (163, 142, [3.763011e-06, -1.039319e-07, 1.226610e-09, 1.880922e-11], 6.884514e-03),
            (163, 239, [-7.976457e-07, 1.316115e-08, 9.423819e-11, 1.826959e-11], 4.315735e-03),
            (163, 163, [3.115510e-06, -8.767400e-08, 6.307024e-10, 9.049834e-12], 2.487967e-03),
        ]
        
        # 转换为索引格式并存储
        for orig_zone, dest_zone, coeff, lambda_max in od_params_data:
            orig_idx = self.zone_to_idx[orig_zone]
            dest_idx = self.zone_to_idx[dest_zone]
            self.nhpp_parameters[(orig_idx, dest_idx)] = (np.array(coeff), lambda_max)
        
        # 验证数据完整性
        if len(self.nhpp_parameters) != 81:
            logging.warning(f"NHPP参数数量不完整: {len(self.nhpp_parameters)}/81")
        else:
            logging.info(f"成功加载81个OD对的NHPP参数")
    
    def _calculate_statistics(self):
        """计算NHPP统计信息"""
        # 计算总λ_max
        lambda_values = [lambda_max for _, lambda_max in self.nhpp_parameters.values()]
        self.total_lambda_max = sum(lambda_values)
        
        # 计算区域内出行统计
        self.same_zone_od_count = len([(i,j) for i,j in self.nhpp_parameters.keys() if i == j])
        
        # 区域内出行的λ_max总和
        self.same_zone_lambda_max = sum(
            lambda_max for (i,j), (_, lambda_max) in self.nhpp_parameters.items() if i == j
        )
        
        # 计算2小时期望乘客数（近似）
        self.expected_passengers_2h = self.total_lambda_max * 7200  # λ_max * 7200秒
        
        # 区域内出行期望乘客数
        self.expected_same_od_2h = self.same_zone_lambda_max * 7200
        
        # 区域内出行比例
        self.same_od_proportion = self.same_zone_lambda_max / self.total_lambda_max if self.total_lambda_max > 0 else 0
        
        # 验证区域内出行OD 0->0
        if (0, 0) in self.nhpp_parameters:
            coeff_00, lambda_max_00 = self.nhpp_parameters[(0, 0)]
            a0, a1, a2, a3 = coeff_00
            logging.debug(f"OD 0->0参数: λ(t) = {a0:.2e} + {a1:.2e}·t + {a2:.2e}·t² + {a3:.2e}·t³")
            logging.debug(f"OD 0->0 λ_max: {lambda_max_00:.6f}")
        
    def print_summary(self):
        """打印配置摘要"""
        logger = logging.getLogger('summary')
        
        logger.info("\n" + "="*60)
        logger.info("真实出租车数据NHPP参数配置摘要")
        logger.info("λ(t) = a0 + a1·t + a2·t² + a3·t³，t∈[0,7200]秒")
        logger.info("="*60)
        
        logger.info(f"区域配置:")
        logger.info(f"  原始区域ID: {self.original_zone_ids}")
        logger.info(f"  区域数量: {self.n_nodes}")
        
        logger.info(f"NHPP参数:")
        logger.info(f"  OD对总数: {len(self.nhpp_parameters)}")
        logger.info(f"  区域内出行OD对: {self.same_zone_od_count}")
        logger.info(f"  系统最大总到达率: {self.total_lambda_max:.6f} 乘客/秒")
        logger.info(f"  系统最大总到达率: {self.total_lambda_max*3600:.2f} 乘客/小时")
        logger.info(f"  2小时期望乘客数: {self.expected_passengers_2h:.0f} 乘客")
        logger.info(f"  区域内出行λ_max: {self.same_zone_lambda_max:.6f} 乘客/秒")
        logger.info(f"  区域内出行比例: {self.same_od_proportion*100:.1f}%")
        logger.info(f"  区域内出行期望: {self.expected_same_od_2h:.0f} 乘客/2小时")
        
        # 示例OD对
        logger.info(f"\n关键OD对验证:")
        # OD 237->237（区域内出行）
        test_od = (0, 0)  # 237->237
        if test_od in self.nhpp_parameters:
            coeff, lambda_max = self.nhpp_parameters[test_od]
            a0, a1, a2, a3 = coeff
            logger.info(f"  OD 237->237 (索引0->0):")
            logger.info(f"    λ(t) = {a0:.2e} + {a1:.2e}·t + {a2:.2e}·t² + {a3:.2e}·t³")
            logger.info(f"    λ_max: {lambda_max:.6f} 乘客/秒")
            logger.info(f"    期望乘客数/2小时: {lambda_max * 7200:.0f}")
            
            # 计算在t=0和t=7200时的λ值
            λ0 = a0
            λ7200 = a0 + a1*7200 + a2*(7200**2) + a3*(7200**3)
            logger.info(f"    λ(0) = {λ0:.6f}, λ(7200) = {λ7200:.6f}")
        
        # OD 237->161（跨区域出行）
        test_od2 = (0, 1)  # 237->161
        if test_od2 in self.nhpp_parameters:
            coeff, lambda_max = self.nhpp_parameters[test_od2]
            a0, a1, a2, a3 = coeff
            logger.info(f"  OD 237->161 (索引0->1):")
            logger.info(f"    λ(t) = {a0:.2e} + {a1:.2e}·t + {a2:.2e}·t² + {a3:.2e}·t³")
            logger.info(f"    λ_max: {lambda_max:.6f} 乘客/秒")
        
        logger.info(f"仿真配置:")
        logger.info(f"  快车数量: {self.express_num}")
        logger.info(f"  专车数量: {self.premium_num}")
        logger.info(f"  总车辆数: {self.express_num + self.premium_num}")
        logger.info(f"  乘客生成时间: {self.passenger_generation_time}秒 ({self.passenger_generation_time/60:.1f}分钟)")
        logger.info(f"  模式: {'快速测试' if self.quick_test else '完整训练'}")
        
        logger.info(f"训练参数:")
        logger.info(f"  学习率: {self.learning_rate}")
        logger.info(f"  折扣因子: {self.gamma}")
        logger.info(f"  初始探索率: {self.epsilon_start}")
        logger.info(f"  目标更新频率: {self.target_update_freq}")
        logger.info(f"  批次大小: {self.batch_size}")
        logger.info(f"  缓冲区容量: {self.buffer_capacity}")
        
        logger.info(f"wandb配置:")
        logger.info(f"  项目名称: {self.wandb_project}")
        
        if self.quick_test:
            logger.info(f"  训练回合数: {self.num_episodes}")
            logger.info(f"  保存频率: {self.save_freq}")
            logger.info(f"  日志频率: {self.log_freq}")
        
        logger.info("="*60)
    
    def get_nhpp_parameters(self):
        """获取NHPP参数"""
        return self.nhpp_parameters
    
    def get_coefficients_dict(self):
        """获取多项式系数字典"""
        coeff_dict = {}
        for (i, j), (coeff, _) in self.nhpp_parameters.items():
            coeff_dict[(i, j)] = coeff
        return coeff_dict
    
    def get_lambda_max_dict(self):
        """获取λ_max字典"""
        lambda_dict = {}
        for (i, j), (_, lambda_max) in self.nhpp_parameters.items():
            lambda_dict[(i, j)] = lambda_max
        return lambda_dict

# ========== 使用真实出租车数据的主程序 ==========

def run_real_nhpp_test(quick_test: bool = True):
    """
    运行真实出租车数据测试
    
    Args:
        quick_test: 是否快速测试模式
    """
    summary_logger = logging.getLogger('summary')
    
    summary_logger.info("\n" + "="*60)
    summary_logger.info(f"真实出租车数据{'快速' if quick_test else '完整'}测试")
    summary_logger.info("="*60)
    
    # 创建真实出租车数据配置
    config = RealNHPPConfig(quick_test=quick_test)
    config.print_summary()
    
    # 创建环境 - 使用新的EfficientNHPPRideHailingEnv类
    env = EfficientNHPPRideHailingEnv(
        n_nodes=config.n_nodes,
        express_num=config.express_num,
        premium_num=config.premium_num,
        travel_matrix=config.travel_matrix,
        nhpp_parameters=config.get_nhpp_parameters(),
        passenger_generation_time=config.passenger_generation_time,
        use_nhpp_package=True
    )
    
    # 计算状态维度
    vehicle_dist_dim = env._get_vehicle_distribution_dim()
    pre_state_dim = (config.n_nodes * config.n_nodes) + (2 * config.n_nodes) + vehicle_dist_dim
    post_state_dim = vehicle_dist_dim
    
    summary_logger.info(f"状态维度: 决策前={pre_state_dim}, 决策后={post_state_dim}")
    
    # 创建智能体
    agent = NeuralADPAgent(
        post_state_dim=post_state_dim,
        pre_state_dim=pre_state_dim,
        env=env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        buffer_capacity=config.buffer_capacity,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_freq,
        tau=config.tau
    )
    
    # 创建训练器
    trainer = Trainer(
        agent=agent,
        env=env,
        num_episodes=config.num_episodes,
        save_freq=config.save_freq,
        log_freq=config.log_freq,
        wandb_project=config.wandb_project,
        wandb_run_name=f"{'quick' if quick_test else 'full'}_{config.num_episodes}ep_{config.passenger_generation_time/60:.0f}min"
    )
    
    # 开始训练
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    summary_logger.info(f"\n真实出租车数据测试完成，总耗时: {training_time:.2f}秒")
    summary_logger.info(f"平均每回合耗时: {training_time/config.num_episodes:.2f}秒")
    summary_logger.info(f"训练曲线请查看wandb项目: {config.wandb_project}")

def validate_real_nhpp_generation():
    """验证真实出租车数据NHPP乘客生成"""
    summary_logger = logging.getLogger('summary')
    
    summary_logger.info("\n" + "="*60)
    summary_logger.info("验证真实出租车数据NHPP乘客生成")
    summary_logger.info("λ(t) = a0 + a1·t + a2·t² + a3·t³，t∈[0,7200]秒")
    summary_logger.info("="*60)
    
    # 创建配置
    config = RealNHPPConfig(quick_test=False)  # 使用完整2小时数据
    
    # 创建生成器
    generator = EfficientNHPPPassengerGenerator(
        n_nodes=config.n_nodes,
        passenger_generation_logic_time=7200.0,  # 2小时完整测试
        nhpp_parameters=config.get_nhpp_parameters(),
        use_nhpp_package=True
    )
    
    # 生成乘客
    summary_logger.info("开始生成乘客...")
    num_passengers = generator.generate_nhpp_passengers_fast()
    
    # 分析结果
    stats = generator.generation_stats
    expected = stats['expected_total_passengers']
    actual = stats['actual_total_passengers']
    gen_time = stats['generation_time']
    total_lambda_max = stats['total_lambda_max']
    
    summary_logger.info(f"\n验证结果:")
    summary_logger.info(f"  系统总λ_max: {total_lambda_max:.6f} 乘客/秒")
    summary_logger.info(f"  理论最大乘客数/2小时: {total_lambda_max * 7200:.0f}")
    summary_logger.info(f"  理论期望乘客数 (2小时): {expected:.1f}")
    summary_logger.info(f"  实际生成乘客数 (2小时): {actual}")
    
    if expected > 0:
        diff = actual - expected
        diff_percent = diff / expected * 100
        summary_logger.info(f"  差异: {diff:+.1f} ({diff_percent:+.2f}%)")
    
    summary_logger.info(f"  生成耗时: {gen_time:.2f}秒")
    summary_logger.info(f"  生成速率: {actual/gen_time:.1f} 乘客/秒")
    
    # 验证区域内出行
    same_od_count = stats['same_origin_destination_count']
    if same_od_count > 0:
        summary_logger.info(f"\n区域内出行验证:")
        summary_logger.info(f"  生成区域内出行乘客: {same_od_count} ({same_od_count/actual*100:.1f}%)")
        
        # 验证OD 237->237
        od_00_count = stats['od_generation_counts'].get((0, 0), 0)
        lambda_max_00 = config.get_lambda_max_dict().get((0, 0), 0)
        coeff_00 = config.get_coefficients_dict().get((0, 0))
        
        if coeff_00 is not None:
            a0, a1, a2, a3 = coeff_00
            summary_logger.info(f"  OD 237->237 (区域0->0):")
            summary_logger.info(f"    λ(t) = {a0:.2e} + {a1:.2e}·t + {a2:.2e}·t² + {a3:.2e}·t³")
            summary_logger.info(f"    λ_max: {lambda_max_00:.6f} 乘客/秒")
            
            # 计算在t=0和t=7200时的λ值
            λ0 = a0
            λ7200 = a0 + a1*7200 + a2*(7200**2) + a3*(7200**3)
            summary_logger.info(f"    λ(0) = {λ0:.6f}, λ(7200) = {λ7200:.6f}")
            
            expected_00 = lambda_max_00 * 7200
            summary_logger.info(f"    期望乘客数: {expected_00:.0f}")
            summary_logger.info(f"    实际生成: {od_00_count}名乘客")
            
            if expected_00 > 0:
                diff_00 = od_00_count - expected_00
                summary_logger.info(f"    差异: {diff_00:+.0f} ({diff_00/expected_00*100:+.1f}%)")
    
    # 统计前5个生成最多的OD对
    od_counts = sorted(stats['od_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
    summary_logger.info(f"\n生成最多的前5个OD对:")
    for (i, j), count in od_counts:
        orig_zone = config.idx_to_zone[i]
        dest_zone = config.idx_to_zone[j]
        lambda_max = config.get_lambda_max_dict().get((i, j), 0)
        coeff = config.get_coefficients_dict().get((i, j))
        if coeff is not None:
            a0, a1, a2, a3 = coeff
            summary_logger.info(f"  OD {orig_zone}->{dest_zone} (索引{i}->{j}):")
            summary_logger.info(f"    生成{count}名乘客, λ_max={lambda_max:.6f}")
            summary_logger.info(f"    λ(t) = {a0:.2e} + {a1:.2e}·t + {a2:.2e}·t² + {a3:.2e}·t³")
    
    summary_logger.info("="*60)

# ========== 主程序入口 ==========

def main():
    """主程序 - 完整训练"""
    # 获取摘要日志器用于控制台输出
    summary_logger = logging.getLogger('summary')
    
    summary_logger.info("\n" + "="*60)
    summary_logger.info("完整训练模式")
    summary_logger.info("="*60)
    
    # 运行完整训练（2小时，100回合）
    run_real_nhpp_test(quick_test=False)

def quick_test():
    """快速测试 - 小规模运行"""
    # 获取摘要日志器用于控制台输出
    summary_logger = logging.getLogger('summary')
    
    summary_logger.info("\n" + "="*60)
    summary_logger.info("快速测试模式")
    summary_logger.info("="*60)
    
    # 运行快速测试（10分钟，10回合）
    run_real_nhpp_test(quick_test=True)

if __name__ == "__main__":
    # 设置基本日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置自定义日志系统（仅控制台输出）
    loggers = LoggerConfig.setup_loggers(level=logging.INFO)
    
    # 获取摘要日志器
    summary_logger = logging.getLogger('summary')
    
    summary_logger.info("="*60)
    summary_logger.info("真实出租车数据网约车调度系统")
    summary_logger.info("基于按秒计算的λ(t)函数，81个OD对完整参数")
    summary_logger.info("λ(t) = a0 + a1·t + a2·t² + a3·t³，t∈[0,7200]秒")
    summary_logger.info("="*60)
    
    # 显示模式选择
    try:
        print("\n选择操作模式:")
        print("1 - 验证NHPP乘客生成 (2小时完整验证)")
        print("2 - 快速测试 (10分钟，10回合)")
        print("3 - 完整训练 (2小时，100回合)")
        print("4 - 自定义模式")
        
        mode = input("\n请选择模式 (1/2/3/4): ").strip()
        
        if mode == "4":
            print("\n自定义模式配置:")
            quick_input = input("快速测试模式? (y/n): ").strip().lower()
            quick_test_mode = quick_input == 'y'
            
            if not quick_test_mode:
                episodes_input = input("训练回合数 (默认100): ").strip()
                episodes = int(episodes_input) if episodes_input else 100
                
                time_input = input("乘客生成时间(秒，默认7200): ").strip()
                gen_time = float(time_input) if time_input else 7200.0
                
                summary_logger.info(f"自定义模式: 快速测试={quick_test_mode}, 回合数={episodes}, 生成时间={gen_time}秒")
            
    except EOFError:
        mode = "1"
        summary_logger.warning("使用默认模式 1 (验证NHPP生成)")
    
    try:
        if mode == "1":
            validate_real_nhpp_generation()
        elif mode == "2":
            quick_test()
        elif mode == "3":
            main()
        elif mode == "4":
            quick_test()  # 自定义模式简化处理
        else:
            summary_logger.warning(f"未知模式 '{mode}'，使用默认值")
            validate_real_nhpp_generation()
    except KeyboardInterrupt:
        summary_logger.info("\n\n程序被用户中断")
    except Exception as e:
        summary_logger.error(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        summary_logger.info("\n" + "="*60)
        summary_logger.info("程序执行完成")
        summary_logger.info("="*60)
        
        # 清理日志系统
        LoggerConfig.cleanup()       