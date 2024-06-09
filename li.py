import cvxpy as cp
import numpy as np

# 参数设置
np.random.seed(1)
demand_min = 50  # 需求下限https://github.com/nbr123/54/tree/main
demand_max = 100  # 需求上限
initial_stock = 20  # 初始库存

# 第一阶段决策变量
x = cp.Variable()  # 初始订购量

# 第二阶段决策变量
y = cp.Variable()  # 补充订购量

# 不确定性集合（在这里，我们假设需求在[demand_min, demand_max]范围内）
demand = cp.Parameter(nonneg=True)

# 目标函数
# 目标是最小化订购和持有成本，假设持有成本为1，订购成本为2
cost = 2 * x + 2 * y

# 约束条件
constraints = [
    x + y >= demand,  # 满足需求
    x >= 0,
    y >= 0
]

# 定义问题
problem = cp.Problem(cp.Minimize(cost), constraints)

# 求解问题并考虑最坏情况
worst_cost = float('inf')
worst_demand = 0
for d in np.linspace(demand_min, demand_max, 100):
    demand.value = d
    problem.solve()
    if cost.value < worst_cost:
        worst_cost = cost.value
        worst_demand = d

# 输出结果
print(f"最优初始订购量: {x.value}")
print(f"最优补充订购量: {y.value} (针对最坏情况需求 {worst_demand})")
print(f"最坏情况下的总成本: {worst_cost}")
