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

# ========== 2. 数据类定义 ==========

@dataclass
class Passenger:
    """乘客数据类"""
    passenger_id: int           # 乘客ID
    arrival_logic_time: float   # 到达逻辑时间（秒）
    origin: int                # 起点区域
    destination: int           # 终点区域
    passenger_type: PassengerType  # 乘客类型

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

# ========== 3. 固定策略决策器 ==========

class FixedPolicyController:
    """基于流体近似求解的固定策略决策器"""
    
    def __init__(self, n_nodes: int, 
                 delta_ij: np.ndarray,  # 调度矩阵（双兼容乘客分配给Express的概率）
                 phi_E: np.ndarray,      # Express车队重定位策略
                 phi_P: np.ndarray):     # Premium车队重定位策略
        """
        初始化固定策略控制器
        """
        self.n_nodes = n_nodes
        self.delta_ij = delta_ij
        self.phi_E = phi_E
        self.phi_P = phi_P
        
        # 严格使用策略矩阵，不做裁剪或归一化修正
        
        # 验证矩阵维度
        assert delta_ij.shape == (n_nodes, n_nodes), f"delta_ij shape mismatch: {delta_ij.shape}"
        assert phi_E.shape == (n_nodes, n_nodes), f"phi_E shape mismatch: {phi_E.shape}"
        assert phi_P.shape == (n_nodes, n_nodes), f"phi_P shape mismatch: {phi_P.shape}"

        # 严格校验策略矩阵，不做任何数值修正
        if np.any((self.delta_ij < 0) | (self.delta_ij > 1)):
            raise ValueError("delta_ij must be within [0, 1] for direct policy sampling")

        if np.any(self.phi_E < 0) or np.any(self.phi_P < 0):
            raise ValueError("phi_E and phi_P must be non-negative probability matrices")

        if not np.allclose(np.sum(self.phi_E, axis=1), 1.0, atol=1e-8):
            raise ValueError("Each row of phi_E must sum to 1.0 for direct policy sampling")

        if not np.allclose(np.sum(self.phi_P, axis=1), 1.0, atol=1e-8):
            raise ValueError("Each row of phi_P must sum to 1.0 for direct policy sampling")
    
    def get_dispatching_decision(self, origin: int, destination: int) -> str:
        """
        获取派单决策（针对双类型乘客）
        """
        # 严格使用策略矩阵中的原始概率
        prob_express = self.delta_ij[origin, destination]
            
        # 根据概率做出决策
        if random.random() < prob_express:
            decision = "E"
        else:
            decision = "P"
            
        return decision
    
    def get_rebalancing_decision(self, zone: int, vehicle_type: str) -> int:
        """
        获取重平衡决策
        """
        if vehicle_type == "E":
            # 获取Express重定位概率分布
            probs = self.phi_E[zone, :].copy()
        else:  # "P"
            # 获取Premium重定位概率分布
            probs = self.phi_P[zone, :].copy()
        
        # 严格使用策略矩阵中的原始概率分布
        destination = np.random.choice(self.n_nodes, p=probs)
        
        return destination

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

# ========== 5. Network类 ==========

class Network:
    """网络类，管理节点间的行程时间和车辆状态"""
    
    def __init__(self, n_nodes: int, express_num: int, premium_num: int, travel_matrix: np.ndarray):
        """
        初始化网络模型
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
        
        # 固定策略控制器（稍后设置）
        self.controller = None
        
        # 初始化车辆状态
        self._initialize_vehicle_distribution()
        
        # 使用优先队列维护在途车辆，按到达逻辑时间排序
        self.arrival_queue = []
        
        # 统计信息
        self.total_revenue = 0.0
        self.total_relocation_cost = 0.0
        self.served_passengers = 0
        self.rejected_passengers = 0

    def set_controller(self, controller):
        """设置固定策略控制器"""
        self.controller = controller

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

    # ========== 核心方法：动作执行 ==========
    
    def execute_dispatching_action(self, origin: int, destination: int, vehicle_type: str, 
                                travel_time_seconds: float, current_time: float) -> Tuple[bool, float]:
        """
        执行派单动作 - 使用之前评估时生成的行程时间
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

# ========== 6. 基于真实数据的乘客生成器 ==========

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

# ========== 7. 基于固定策略的环境（使用真实数据） ==========

class FixedPolicyRideHailingEnvRealData:
    """基于真实数据和固定策略的双服务网约车环境"""
    
    def __init__(self, n_nodes: int, express_num: int, premium_num: int, 
                 travel_matrix: np.ndarray,
                 data_folder: str,
                 zone_mapping: Dict[int, int],
                 delta_ij: np.ndarray, phi_E: np.ndarray, phi_P: np.ndarray):
        """
        初始化环境（使用真实数据）
        """
        self.n_nodes = n_nodes
        self.express_num = express_num
        self.premium_num = premium_num
        self.travel_matrix = travel_matrix
        self.data_folder = data_folder
        self.zone_mapping = zone_mapping

        # 初始化网络
        self.network = Network(n_nodes, express_num, premium_num, travel_matrix)
        
        # 初始化固定策略控制器
        self.controller = FixedPolicyController(n_nodes, delta_ij, phi_E, phi_P)
        self.network.set_controller(self.controller)
        
        # 初始化真实数据乘客生成器
        self.passenger_generator = RealDataPassengerGenerator(
            data_folder=data_folder,
            n_nodes=n_nodes,
            zone_mapping=zone_mapping
        )
        
        # 当前状态和统计
        self.current_time = 0.0
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
    
    def _handle_passenger_arrival(self, passenger: Passenger) -> bool:
        """
        处理乘客到达事件
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
                return True
            else:
                # 无快车可用，乘客流失
                self.episode_rejected += 1
                return True
                
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
                return True
            else:
                # 无专车可用，乘客流失
                self.episode_rejected += 1
                return True
                
        else:  # PassengerType.DUAL
            # 双类型乘客
            has_express = self.network.has_idle_vehicles_at_node(origin, "E")
            has_premium = self.network.has_idle_vehicles_at_node(origin, "P")
            
            if has_express and has_premium:
                # 两种车都有，使用固定策略决策
                vehicle_type = self.controller.get_dispatching_decision(origin, destination)
                travel_time = self.network.get_travel_time(origin, destination)
                success, revenue = self.network.execute_dispatching_action(
                    origin, destination, vehicle_type, travel_time, self.current_time
                )
                if success:
                    self.episode_revenue += revenue
                    self.episode_served += 1
                else:
                    self.episode_rejected += 1
                return True
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
                return True
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
                return True
            else:
                # 两种车都无，乘客流失
                self.episode_rejected += 1
                return True
    
    def _handle_vehicle_arrival(self, car: CarOnTheWay) -> bool:
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
            return True
                
        elif car_state == CarState.OCCUPIED:
            # 载客车辆到达，完成行程 - TRIGGERS RELOCATION DECISION
            if vehicle_type == "E":
                self.network.update_express_from_occupied_to_idle(origin, destination)
            else:
                self.network.update_premium_from_occupied_to_idle(origin, destination)
            
            # 使用固定策略进行重平衡决策
            target_zone = self.controller.get_rebalancing_decision(destination, vehicle_type)
            
            if target_zone == destination:
                # 留在原地，无成本
                return True
            else:
                # 重新平衡调度
                travel_time = self.network.get_travel_time(destination, target_zone)
                success, cost = self.network.execute_rebalancing_action(
                    destination, target_zone, vehicle_type, travel_time, self.current_time
                )
                if success:
                    self.episode_cost += cost
                return True
        
        return True
    
    def run_episode_for_date(self, target_date: date) -> Dict[str, Any]:
        """
        为指定日期运行一个episode
        """
        # 重置环境
        self.current_time = 0.0
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
                self._handle_passenger_arrival(passenger)
                
            elif event_type == 'vehicle_arrival':
                car = self.network.pop_arrival_queue()
                self._handle_vehicle_arrival(car)
        
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
        print(f"{target_date}: 收益={total_profit:.2f}, 收入={self.episode_revenue:.2f}, "
              f"成本={self.episode_cost:.2f}, 服务={self.episode_served}, "
              f"流失={self.episode_rejected}, 数据总数={self.passenger_generator.generation_stats['total_trips_in_date']}")
        
        return episode_info
    
    def reset(self):
        """重置环境"""
        self.current_time = 0.0
        self.episode_revenue = 0.0
        self.episode_cost = 0.0
        self.episode_served = 0
        self.episode_rejected = 0
        
        # 重置网络和乘客生成器
        self.network.reset()
        self.passenger_generator.reset()

# ========== 8. 实验运行器（使用真实数据） ==========

class FixedPolicyExperimentRunnerRealData:
    """使用真实数据的固定策略实验运行器"""
    
    def __init__(self, env: FixedPolicyRideHailingEnvRealData):
        """
        初始化实验运行器
        """
        self.env = env
        
        # 实验统计
        self.all_results = []
    
    def run_experiment(self, use_all_dates: bool = True):
        """
        运行实验
        """
        # 获取所有可用日期
        available_dates = self.env.passenger_generator.get_available_dates()
        
        if not available_dates:
            print("没有可用的日期数据")
            return
        
        print(f"开始运行 {len(available_dates)} 天的数据...")
        print("="*80)
        
        # 使用所有可用日期
        dates_to_run = available_dates
        
        for target_date in dates_to_run:
            # 运行一个episode
            episode_result = self.env.run_episode_for_date(target_date)
            if episode_result:
                self.all_results.append(episode_result)
        
        print("="*80)
        print("实验完成")
        
        # 输出汇总结果
        self._print_summary()
        
        return self.all_results
    
    def _print_summary(self):
        """打印实验总结"""
        if not self.all_results:
            return
        
        # 提取关键指标
        profits = [r['profit'] for r in self.all_results]
        revenues = [r['revenue'] for r in self.all_results]
        costs = [r['cost'] for r in self.all_results]
        served = [r['served'] for r in self.all_results]
        rejected = [r['rejected'] for r in self.all_results]
        total_in_data = [r['total_in_data'] for r in self.all_results]
        
        print("\n汇总统计:")
        print(f"总天数: {len(self.all_results)}")
        print(f"总收益: {sum(profits):.2f}")
        print(f"平均每日收益: {np.mean(profits):.2f} ± {np.std(profits):.2f}")
        print(f"平均每日收入: {np.mean(revenues):.2f} ± {np.std(revenues):.2f}")
        print(f"平均每日成本: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
        print(f"总服务乘客: {sum(served)}")
        print(f"总流失乘客: {sum(rejected)}")
        print(f"总数据乘客数: {sum(total_in_data)}")
        print(f"总体服务率: {sum(served)/max(1, sum(served)+sum(rejected)):.2%}")

# ========== 9. 主程序（使用真实数据） ==========

def load_fixed_policy_matrices(filepath: str = None):
    """
    从训练文件加载固定策略矩阵

    参数:
    filepath: 策略矩阵文件路径，默认为 fluid_approximation_hpp/fluid_policy_hpp.npz
    """
    import os

    # 默认路径
    if filepath is None:
        # 尝试多个可能的路径
        possible_paths = [
            "fluid_approximation_hpp/fluid_policy_hpp_17_19.npz",
            "fluid_policy_hpp_17_19.npz",
            os.path.join(os.path.dirname(__file__), "fluid_policy_hpp_17_19.npz"),
            "fluid_approximation_hpp/fluid_policy_hpp.npz",
            "fluid_policy_hpp.npz",
            os.path.join(os.path.dirname(__file__), "fluid_policy_hpp.npz")
        ]
        for p in possible_paths:
            if os.path.exists(p):
                filepath = p
                break
        else:
            raise FileNotFoundError(
                f"找不到策略矩阵文件，请先运行 train_fluid_approximation_hpp_17_19.py 生成策略矩阵"
            )

    # 从文件加载策略矩阵
    data = np.load(filepath)
    delta_ij = data['delta_ij']
    phi_E = data['phi_E']
    phi_P = data['phi_P']
    n_nodes = delta_ij.shape[0]

    print(f"策略矩阵已从 {filepath} 加载")

    # 验证策略矩阵
    print("验证策略矩阵...")

    # 仅验证重定位概率和是否为1
    print("\nExpress车队重定位策略每行概率和:")
    for i in range(n_nodes):
        row_sum = np.sum(phi_E[i, :])
        if abs(row_sum - 1.0) > 0.001:
            print(f"警告: phi_E 第 {i} 行概率和不为1 (和为 {row_sum:.4f})")
        print(f"phi_E[{i}, :] 概率和 = {row_sum:.6f}")

    # 输出调度矩阵摘要
    print("\n调度矩阵 δ_ij 摘要:")
    print("(只显示小于1的值，大于0.95的显示为1.0000)")
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and delta_ij[i, j] < 0.95:
                print(f"  δ[{i},{j}] = {delta_ij[i,j]:.4f}")

    # 输出重定位策略摘要
    print("\nExpress车队重定位策略摘要:")
    print("(只显示有显著重定位的区域)")
    for i in range(n_nodes):
        has_relocation = False
        for j in range(n_nodes):
            if i != j and phi_E[i, j] > 0.01:
                has_relocation = True

        if has_relocation:
            print(f"\n区域 {i} → 其他区域:")
            for j in range(n_nodes):
                if i != j and phi_E[i, j] > 0.0001:
                    print(f"  到区域 {j}: {phi_E[i,j]:.4f}")
            print(f"  留在区域 {i}: {phi_E[i,i]:.4f}")

    print("\nPremium车队重定位策略摘要:")
    print("所有Premium车辆都留在原地")

    return delta_ij, phi_E, phi_P
def main_real_data():
    """主程序（使用真实数据）"""
    print("="*80)
    print("真实数据固定策略双服务网约车系统")
    print("使用纽约TLC真实出租车数据和流体近似求解策略")
    print("="*80)
    
    # 加载固定策略矩阵（使用实际求解结果）
    delta_ij, phi_E, phi_P = load_fixed_policy_matrices()
    
    # 配置参数
    n_nodes = 9
    express_num = 240  # α=0.8, N=300 → 0.8*300=240
    premium_num = 60   # (1-α)=0.2, N=300 → 0.2*300=60
    
    print(f"车队规模配置:")
    print(f"  Express快车: {express_num}辆")
    print(f"  Premium专车: {premium_num}辆")
    print(f"  总车队规模: {express_num + premium_num}辆")
    
    # 行程时间矩阵（分钟）- 使用提供的固定数据
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
    
    print(f"真实数据配置:")
    print(f"  数据文件夹: {data_folder}")
    print(f"  区域映射: {zone_mapping}")
    
    # 检查数据文件夹是否存在
    if not os.path.exists(data_folder):
        print(f"数据文件夹不存在: {data_folder}")
        print(f"请确保已运行数据预处理脚本并生成数据文件夹")
        return None
    
    # 创建环境
    env = FixedPolicyRideHailingEnvRealData(
        n_nodes=n_nodes,
        express_num=express_num,
        premium_num=premium_num,
        travel_matrix=travel_matrix,
        data_folder=data_folder,
        zone_mapping=zone_mapping,
        delta_ij=delta_ij,
        phi_E=phi_E,
        phi_P=phi_P
    )
    
    # 检查可用天数
    available_dates = env.passenger_generator.get_available_dates()
    if not available_dates:
        print("没有找到可用的日期数据文件")
        return None
    
    print(f"找到 {len(available_dates)} 天的数据")
    print(f"日期范围: {min(available_dates)} 到 {max(available_dates)}")
    print("="*80)
    
    # 创建实验运行器
    runner = FixedPolicyExperimentRunnerRealData(env)
    
    # 运行实验（使用所有可用日期）
    results = runner.run_experiment(use_all_dates=True)
    
    return results

if __name__ == "__main__":
    results = main_real_data()
    
    if results:
        print(f"\n总共运行了 {len(results)} 天的数据")