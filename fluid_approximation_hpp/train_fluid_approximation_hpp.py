import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.stats import dirichlet

def generate_random_q_matrices(num_zones=9, concentration_params=None):
    """
    使用狄克雷分布随机生成三个q矩阵，满足 qE_ij + qP_ij + qD_ij = 1
    
    参数:
    num_zones: 区域数量
    concentration_params: 狄克雷分布的浓度参数 [α_E, α_P, α_D]
                         默认对应概率 [0.5, 0.3, 0.2]
    """
    np.random.seed(40)  # 为了可重复性
    
    # 设置浓度参数（控制概率分布的形状）
    # 较大的值使分布更集中在均值附近，较小的值使分布更分散
    if concentration_params is None:
        # 对应期望概率 [0.5, 0.3, 0.2]
        concentration_params = [5.0, 3.0, 2.0]  # 总和为10，控制方差
    
    q_E = np.zeros((num_zones, num_zones))
    q_P = np.zeros((num_zones, num_zones))
    q_D = np.zeros((num_zones, num_zones))
    
    for i in range(num_zones):
        for j in range(num_zones):
            # 使用狄克雷分布生成和为1的三个随机数
            random_vector = dirichlet.rvs(concentration_params, size=1)[0]
            
            # 分配概率（可以按任意顺序，这里按E, P, D顺序）
            q_E[i,j] = random_vector[0]  # Express乘客概率
            q_P[i,j] = random_vector[1]  # Premium乘客概率
            q_D[i,j] = random_vector[2]  # Dual-compatible乘客概率
            
            # 添加区域特定的模式（可选）
            # 例如，对角线上的区域可能有不同的分布
            if i == j:
                # 相同区域：增加双兼容乘客的概率
                adjusted_vector = dirichlet.rvs([3.0, 2.0, 5.0], size=1)[0]
                q_E[i,j] = adjusted_vector[0]
                q_P[i,j] = adjusted_vector[1]
                q_D[i,j] = adjusted_vector[2]
    
    # 验证约束
    print("随机生成的q矩阵验证:")
    print("-" * 40)
    for i in range(3):  # 只检查前3个区域作为示例
        for j in range(3):
            sum_q = q_E[i,j] + q_P[i,j] + q_D[i,j]
            print(f"q_E[{i},{j}]={q_E[i,j]:.3f} + q_P[{i},{j}]={q_P[i,j]:.3f} + "
                  f"q_D[{i},{j}]={q_D[i,j]:.3f} = {sum_q:.6f}")
    print("-" * 40)
    
    # 计算统计信息
    mean_E = np.mean(q_E)
    mean_P = np.mean(q_P)
    mean_D = np.mean(q_D)
    print(f"平均值: E={mean_E:.3f}, P={mean_P:.3f}, D={mean_D:.3f}")
    print(f"平均值总和: {mean_E+mean_P+mean_D:.6f}")
    
    return q_E, q_P, q_D

def solve_two_fleet_optimization(num_zones=9, alpha=0.8, N=500):
    """
    使用狄克雷分布生成的随机q矩阵求解双车队优化问题
    """
    # 创建模型
    model = gp.Model("TwoFleetOptimization")
    
    # ========== 固定参数 ==========
    # 1/μ_ij（单位是十分钟）
    frction_mu_ij = np.array([
        [6.095065, 12.676830, 7.456200, 27.558737, 10.594408, 18.981833, 13.565680, 14.377516, 11.535740],
        [11.123316, 8.745464, 15.501824, 16.953431, 9.221930, 12.725887, 15.589845, 19.934882, 11.458760],
        [8.708179, 18.470793, 4.822331, 32.131676, 15.598182, 24.639176, 15.033392, 10.513293, 17.185670],
        [24.104568, 15.698473, 28.693086, 7.422782, 17.223955, 14.270898, 18.512715, 21.920932, 17.156832],
        [9.197727, 10.546554, 13.072132, 17.778294, 8.060552, 15.498720, 19.464867, 21.402553, 13.607127],
        [14.984137, 10.033276, 19.728165, 12.377795, 12.237704, 9.081808, 10.825063, 15.109510, 9.267567],
        [10.434873, 14.605556, 13.125660, 19.957931, 15.867941, 11.334748, 5.550570, 6.831122, 7.922796],
        [12.661859, 19.881672, 9.886090, 25.166458, 20.431189, 16.644135, 6.938669, 4.771336, 12.904756],
        [9.457781, 10.383094, 12.952297, 18.012084, 11.087675, 11.520476, 9.454008, 13.270505, 6.973626]
    ]) / 10
    
    mu_ij = 1.0 / frction_mu_ij
    
    # p_ij 和 λ_i
    p_ij = np.array([
        [0.208592, 0.108899, 0.298686, 0.019514, 0.083720, 0.049807, 0.105044, 0.068534, 0.057204],
        [0.227981, 0.096316, 0.187767, 0.081521, 0.052825, 0.084004, 0.113097, 0.076954, 0.079535],
        [0.325079, 0.074293, 0.210677, 0.015343, 0.058412, 0.033019, 0.126783, 0.103275, 0.053118],
        [0.082594, 0.150199, 0.070052, 0.042521, 0.109820, 0.255430, 0.089018, 0.064546, 0.135821],
        [0.218182, 0.093013, 0.194290, 0.088956, 0.071675, 0.101728, 0.096018, 0.068670, 0.067468],
        [0.110667, 0.109905, 0.103810, 0.119810, 0.066095, 0.112381, 0.176571, 0.098476, 0.102286],
        [0.160912, 0.085743, 0.135635, 0.023955, 0.058318, 0.099290, 0.120767, 0.206674, 0.108706],
        [0.130968, 0.063471, 0.172771, 0.015340, 0.032790, 0.086098, 0.270757, 0.147843, 0.079962],
        [0.177841, 0.124832, 0.165746, 0.055398, 0.074511, 0.130656, 0.129760, 0.100493, 0.040765]
    ])
    
    lambda_i = np.array([
        0.024857,  # 237
        0.011875,  # 161
        0.023431,  # 236
        0.005790,  # 186
        0.008067,  # 162
        0.010641,  # 230
        0.015862,  # 142
        0.012096,  # 239
        0.008873   # 163
    ])
    
    lambda_ij = lambda_i[:, None] * p_ij * 2000 / N
    
    # ========== 收益参数 ==========
    p0, rho, v = 10, 2.4, 25
    gamma_E = p0 + rho * frction_mu_ij * v / 60 * 10
    gamma_P = p0 * 1.5 + 1.5 * rho * frction_mu_ij * v / 60 * 10
    
    # ========== 重定位成本 ==========
    beta_E = gamma_E * 0.2
    beta_P = gamma_P * 0.2
    np.fill_diagonal(beta_E, 0)  # 对角线设置为0，因为同一区域内重定位没有成本
    np.fill_diagonal(beta_P, 0)
    
    # ========== 使用狄克雷分布生成随机q矩阵 ==========
    print("=" * 80)
    print("使用狄克雷分布生成随机q矩阵")
    print("=" * 80)
    
    # 生成随机q矩阵
    q_E, q_P, q_D = generate_random_q_matrices(num_zones=num_zones)
    
    # ========== 定义变量 ==========
    a_E = [model.addVar(lb=0, ub=1, name=f"a_E_{i}") for i in range(num_zones)]
    a_P = [model.addVar(lb=0, ub=1, name=f"a_P_{i}") for i in range(num_zones)]
    a_EP = [model.addVar(lb=0, ub=1, name=f"a_EP_{i}") for i in range(num_zones)]
    
    f_E = [[model.addVar(lb=0, name=f"f_E_{i}_{j}") for j in range(num_zones)] for i in range(num_zones)]
    f_P = [[model.addVar(lb=0, name=f"f_P_{i}_{j}") for j in range(num_zones)] for i in range(num_zones)]
    r_E = [[model.addVar(lb=0, name=f"r_E_{i}_{j}") for j in range(num_zones)] for i in range(num_zones)]
    r_P = [[model.addVar(lb=0, name=f"r_P_{i}_{j}") for j in range(num_zones)] for i in range(num_zones)]
    
    model.update()
    
    # ========== 目标函数 ==========
    revenue = sum(gamma_E[i, j] * mu_ij[i, j] * f_E[i][j] + gamma_P[i, j] * mu_ij[i, j] * f_P[i][j]
                  for i in range(num_zones) for j in range(num_zones))
    
    cost = sum(beta_E[i, j] * mu_ij[i, j] * r_E[i][j] + beta_P[i, j] * mu_ij[i, j] * r_P[i][j]
               for i in range(num_zones) for j in range(num_zones) if i != j)
    
    model.setObjective(revenue - cost, GRB.MAXIMIZE)
    
    # ========== 约束条件 ==========
    # 可用性约束
    for i in range(num_zones):
        model.addConstr(a_E[i] <= a_EP[i])
        model.addConstr(a_P[i] <= a_EP[i])
        model.addConstr(a_E[i] + a_P[i] >= a_EP[i])
    
    # 车队规模约束
    model.addConstr(sum(f_E[i][j] for i in range(num_zones) for j in range(num_zones)) +
                    sum(r_E[i][j] for i in range(num_zones) for j in range(num_zones)) == alpha)
    
    model.addConstr(sum(f_P[i][j] for i in range(num_zones) for j in range(num_zones)) +
                    sum(r_P[i][j] for i in range(num_zones) for j in range(num_zones)) == 1 - alpha)
    
    # 流量平衡约束
    for i in range(num_zones):
        # Express
        out_E = sum(mu_ij[i, j] * f_E[i][j] for j in range(num_zones)) + \
                sum(mu_ij[i, j] * r_E[i][j] for j in range(num_zones) if j != i)
        in_E = sum(mu_ij[l, i] * r_E[l][i] for l in range(num_zones) if l != i) + \
               sum(mu_ij[l, i] * f_E[l][i] for l in range(num_zones))
        model.addConstr(out_E == in_E)
        
        # Premium
        out_P = sum(mu_ij[i, j] * f_P[i][j] for j in range(num_zones)) + \
                sum(mu_ij[i, j] * r_P[i][j] for j in range(num_zones) if j != i)
        in_P = sum(mu_ij[l, i] * r_P[l][i] for l in range(num_zones) if l != i) + \
               sum(mu_ij[l, i] * f_P[l][i] for l in range(num_zones))
        model.addConstr(out_P == in_P)
    
    # 需求分配约束
    for i in range(num_zones):
        for j in range(num_zones):
            # 总需求约束（论文中的式33c）
            model.addConstr(mu_ij[i, j] * f_E[i][j] + mu_ij[i, j] * f_P[i][j] ==
                            lambda_ij[i, j] * q_D[i, j] * a_EP[i] +
                            lambda_ij[i, j] * q_E[i, j] * a_E[i] +
                            lambda_ij[i, j] * q_P[i, j] * a_P[i])
            
            # Express下限（论文中的式33d）
            model.addConstr(lambda_ij[i, j] * q_D[i, j] * (a_EP[i] - a_P[i]) +
                            lambda_ij[i, j] * q_E[i, j] * a_E[i] <= mu_ij[i, j] * f_E[i][j])
            
            # Premium下限（论文中的式33e）
            model.addConstr(lambda_ij[i, j] * q_D[i, j] * (a_EP[i] - a_E[i]) +
                            lambda_ij[i, j] * q_P[i, j] * a_P[i] <= mu_ij[i, j] * f_P[i][j])
            
            # Express上限（论文中的式33f）
            model.addConstr(mu_ij[i, j] * f_E[i][j] <=
                            lambda_ij[i, j] * q_D[i, j] * (a_E[i] + a_P[i] - a_EP[i]) +
                            lambda_ij[i, j] * q_D[i, j] * (a_EP[i] - a_P[i]) +
                            lambda_ij[i, j] * q_E[i, j] * a_E[i])
            
            # Premium上限（论文中的式33g）
            model.addConstr(mu_ij[i, j] * f_P[i][j] <=
                            lambda_ij[i, j] * q_D[i, j] * (a_E[i] + a_P[i] - a_EP[i]) +
                            lambda_ij[i, j] * q_D[i, j] * (a_EP[i] - a_E[i]) +
                            lambda_ij[i, j] * q_P[i, j] * a_P[i])
    
    # ========== 求解 ==========
    model.setParam('OutputFlag', 0)  # 不输出求解过程
    model.setParam('TimeLimit', 300)
    model.optimize()
    
    # ========== 结果输出 ==========
    if model.status == GRB.OPTIMAL:
        # 收集结果
        results = {
            'objective': model.objVal,
            'a_E': [a_E[i].x for i in range(num_zones)],
            'a_P': [a_P[i].x for i in range(num_zones)],
            'a_EP': [a_EP[i].x for i in range(num_zones)],
            'f_E': [[f_E[i][j].x for j in range(num_zones)] for i in range(num_zones)],
            'f_P': [[f_P[i][j].x for j in range(num_zones)] for i in range(num_zones)],
            'r_E': [[r_E[i][j].x for j in range(num_zones)] for i in range(num_zones)],
            'r_P': [[r_P[i][j].x for j in range(num_zones)] for i in range(num_zones)],
            'mu_ij': mu_ij,
            'lambda_ij': lambda_ij
        }
        
        # ========== 计算调度矩阵 δ_ij (根据论文中的式34a) ==========
        delta_ij = np.zeros((num_zones, num_zones))

        for i in range(num_zones):
            for j in range(num_zones):
                numerator = mu_ij[i, j] * results['f_E'][i][j] - \
                            lambda_ij[i, j] * q_E[i, j] * results['a_E'][i] - \
                            lambda_ij[i, j] * q_D[i, j] * (results['a_EP'][i] - results['a_P'][i])

                denominator = lambda_ij[i, j] * q_D[i, j] * (results['a_E'][i] + results['a_P'][i] - results['a_EP'][i])

                if denominator > 1e-10:  # 避免除以零
                    delta_ij[i, j] = numerator / denominator
                else:
                    delta_ij[i, j] = 0.0

        # 裁剪delta_ij到[0, 1]范围（概率必须在此范围内）
        delta_ij = np.clip(delta_ij, 0.0, 1.0)
        
        # ========== 计算重定位策略 φ_E 和 φ_P (根据论文中的式32a) ==========
        phi_E = np.zeros((num_zones, num_zones))
        phi_P = np.zeros((num_zones, num_zones))

        # Express车队重定位策略
        for i in range(num_zones):
            # 计算到达区域i的总Express服务流量
            total_inflow_E = sum(mu_ij[l, i] * results['f_E'][l][i] for l in range(num_zones))

            if total_inflow_E > 1e-10:
                for j in range(num_zones):
                    if i != j:
                        phi_E[i, j] = mu_ij[i, j] * results['r_E'][i][j] / total_inflow_E

            # 计算保留在区域i的比例（式32b）
            phi_E[i, i] = 1 - sum(phi_E[i, j] for j in range(num_zones) if j != i)

        # Premium车队重定位策略
        for i in range(num_zones):
            # 计算到达区域i的总Premium服务流量
            total_inflow_P = sum(mu_ij[l, i] * results['f_P'][l][i] for l in range(num_zones))

            if total_inflow_P > 1e-10:
                for j in range(num_zones):
                    if i != j:
                        phi_P[i, j] = mu_ij[i, j] * results['r_P'][i][j] / total_inflow_P

            # 计算保留在区域i的比例（式32b）
            phi_P[i, i] = 1 - sum(phi_P[i, j] for j in range(num_zones) if j != i)

        # 确保phi_E和phi_P非负且每行和为1
        phi_E = np.clip(phi_E, 0.0, 1.0)
        phi_P = np.clip(phi_P, 0.0, 1.0)

        # 归一化每行，确保和为1
        for i in range(num_zones):
            row_sum_E = np.sum(phi_E[i, :])
            if row_sum_E > 0:
                phi_E[i, :] /= row_sum_E
            else:
                phi_E[i, i] = 1.0  # 如果全为0，则留在原地

            row_sum_P = np.sum(phi_P[i, :])
            if row_sum_P > 0:
                phi_P[i, :] /= row_sum_P
            else:
                phi_P[i, i] = 1.0  # 如果全为0，则留在原地
        
        # ========== 输出三个矩阵 ==========
        print("\n" + "=" * 80)
        print("双车队优化结果 - 三个关键矩阵")
        print("=" * 80)

        print("\n1. 调度矩阵 δ_ij (Dispatching Policy):")
        print("   (双兼容乘客分配给Express车队的概率)")
        print("-" * 80)
        for i in range(num_zones):
            for j in range(num_zones):
                if i != j:  # 只显示跨区域的值
                    print(f"  δ[{i},{j}] = {delta_ij[i,j]:.4f}", end="  ")
            if i < num_zones-1:
                print()
        print("\n" + "-" * 80)

        print("\n2. Express车队重定位策略 φ_E:")
        print("   (到达区域i的Express车辆被重定位到区域j的比例)")
        print("-" * 80)
        for i in range(num_zones):
            print(f"  区域 {i} → 其他区域:")
            for j in range(num_zones):
                if i != j and phi_E[i,j] > 1e-4:  # 只显示有显著值的重定位
                    print(f"    到区域 {j}: {phi_E[i,j]:.4f}")
            if phi_E[i,i] > 1e-4:  # 显示保留在原地的比例
                print(f"    留在区域 {i}: {phi_E[i,i]:.4f}")
            if i < num_zones-1:
                print()
        print("-" * 80)

        print("\n3. Premium车队重定位策略 φ_P:")
        print("   (到达区域i的Premium车辆被重定位到区域j的比例)")
        print("-" * 80)
        for i in range(num_zones):
            print(f"  区域 {i} → 其他区域:")
            for j in range(num_zones):
                if i != j and phi_P[i,j] > 1e-4:  # 只显示有显著值的重定位
                    print(f"    到区域 {j}: {phi_P[i,j]:.4f}")
            if phi_P[i,i] > 1e-4:  # 显示保留在原地的比例
                print(f"    留在区域 {i}: {phi_P[i,i]:.4f}")
            if i < num_zones-1:
                print()
        print("-" * 80)
        
        # 返回三个矩阵
        return {
            'delta_ij': delta_ij,  # 调度矩阵
            'phi_E': phi_E,        # Express重定位策略
            'phi_P': phi_P,        # Premium重定位策略
            'q_E': q_E,            # 随机生成的q矩阵
            'q_P': q_P,
            'q_D': q_D
        }
    else:
        print(f"求解失败，状态: {model.status}")
        return None


def save_policy_matrices(results: dict, filepath: str = "fluid_policy_hpp_17_19.npz"):
    """
    保存策略矩阵到文件

    参数:
    results: solve_two_fleet_optimization返回的结果字典
    filepath: 保存文件路径
    """
    if results is None:
        print("无法保存：求解结果为空")
        return False

    np.savez(
        filepath,
        delta_ij=results['delta_ij'],
        phi_E=results['phi_E'],
        phi_P=results['phi_P'],
        q_E=results['q_E'],
        q_P=results['q_P'],
        q_D=results['q_D']
    )
    print(f"策略矩阵已保存到: {filepath}")
    return True


def load_policy_matrices(filepath: str = "fluid_policy_hpp_17_19.npz"):
    """
    从文件加载策略矩阵

    参数:
    filepath: 策略矩阵文件路径

    返回:
    delta_ij, phi_E, phi_P 三个策略矩阵
    """
    import os
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"策略矩阵文件不存在: {filepath}")

    data = np.load(filepath)
    delta_ij = data['delta_ij']
    phi_E = data['phi_E']
    phi_P = data['phi_P']

    print(f"策略矩阵已从 {filepath} 加载")
    return delta_ij, phi_E, phi_P


# 简短的调用代码
if __name__ == "__main__":
    print("双车队优化求解器 - 输出三个关键矩阵")
    print("=" * 80)

    # 设置参数
    alpha = 0.8  # Express车队比例
    N = 300     # 总车队规模

    # 求解
    results = solve_two_fleet_optimization(alpha=alpha, N=N)

    # 保存策略矩阵到文件
    if results is not None:
        save_policy_matrices(results, "fluid_approximation_hpp/fluid_policy_hpp.npz")
    