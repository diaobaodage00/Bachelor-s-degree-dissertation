import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import random
from g2p421_inventory import RefineryInventoryModel
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import pandas as pd
import os
from datetime import datetime
import warnings
import torch

# 添加全局随机种子设置
RANDOM_SEED = 42

# 设置全局随机种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

warnings.filterwarnings("ignore")  # 忽略告警

class RefineryEnv(gym.Env):
    """炼油厂强化学习环境"""
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, planning_horizon=12, render_mode=None, carbon_weight=0.3):
        """
        初始化炼油厂环境
        
        参数:
            planning_horizon: 规划周期数(如12个月)
            render_mode: 渲染模式('human'或'rgb_array')
            carbon_weight: 碳排放权重参数，控制碳排放在奖励函数中的权重
        """
        super(RefineryEnv, self).__init__()
        self.planning_horizon = planning_horizon
        self.current_period = 0
        self.render_mode = render_mode
        self.carbon_weight = carbon_weight
        
        # 初始化单周期模型，传递碳排放权重
        self.model = RefineryInventoryModel(time_periods=1, random_seed=RANDOM_SEED, carbon_weight=self.carbon_weight)
        
        # 定义观测空间 - 产品库存、价格和需求上下限
        num_products = len(self.model.Des)  # 16个产品
        
        # 观测空间包含:
        # 1. 产品库存水平 (16个产品)
        # 2. 产品价格 (16个产品)
        # 3. 产品需求区间 (16个产品 x 2 - 上下限)
        # 4. 当前周期 (1个值)
        obs_dim = num_products * 4 + 1
        
        # 估算观测上下限
        max_inventory = np.array([self.model.max_inventory.get(p, 50000) for p in self.model.Des],dtype=np.float32)
        max_price = np.array([self.model.PS.get(p, 2000) * 1.5 for p in self.model.Des],dtype=np.float32)
        max_demand_low = np.array([self.model.D_min.get(p, 0) * 1.5 for p in self.model.Des],dtype=np.float32)
        max_demand_high = np.array([self.model.D_max.get(p, 5000) * 1.5 for p in self.model.Des],dtype=np.float32)
        
        obs_low = np.concatenate([
            np.zeros(num_products,dtype=np.float32),  # 库存下限
            np.zeros(num_products,dtype=np.float32),  # 价格下限
            np.zeros(num_products,dtype=np.float32),  # 需求下限的下限
            np.zeros(num_products,dtype=np.float32),  # 需求上限的下限
            np.array([0], dtype=np.float32)  # 当前周期
        ])
        
        obs_high = np.concatenate([
            max_inventory,  # 库存上限
            max_price,      # 价格上限
            max_demand_low, # 需求下限的上限
            max_demand_high,# 需求上限的上限
            np.array([planning_horizon], dtype=np.float32)  # 当前周期上限
        ])
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # 定义三层混合动作空间
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]), 
            high=np.array([1, 1, 1, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 历史数据记录，添加碳排放信息
        self.history = {
            'profits': [],
            'inventory': {},
            'sales': {},
            'production': {},
            'unit_modes': {},
            'actions': [],
            'rewards': [],
            'carbon_emissions': [],        # 新增：记录碳排放
            'weighted_profits': []         # 新增：记录考虑碳排放的加权利润
        }
        
        for p in self.model.Des:
            self.history['inventory'][p] = []
            self.history['sales'][p] = []
            self.history['production'][p] = []
        
        for u in self.model.u_process4:
            self.history['unit_modes'][u] = []
    
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        # 如果未提供种子，使用全局随机种子
        if seed is None:
            seed = RANDOM_SEED
            
        super().reset(seed=seed)
        
        # 重置当前周期
        self.current_period = 0
        
        # 重置模型，传递碳排放权重
        self.model = RefineryInventoryModel(time_periods=1, random_seed=seed, carbon_weight=self.carbon_weight)
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 生成新的市场波动
        self.model.generate_market_fluctuations()
        
        # 清空历史记录，包括碳排放记录
        self.history = {
            'profits': [],
            'revenues': [],          
            'material_costs': [],    
            'processing_costs': [],  
            'inventory': {},
            'sales': {},
            'production': {},
            'unit_modes': {},
            'actions': [],
            'rewards': [],
            'carbon_emissions': [],        # 新增：记录碳排放
            'weighted_profits': []         # 新增：记录考虑碳排放的加权利润
        }
        
        for p in self.model.Des:
            self.history['inventory'][p] = []
            self.history['sales'][p] = []
            self.history['production'][p] = []
        
        for u in self.model.u_process4:
            self.history['unit_modes'][u] = []
        
        # 获取初始观测
        observation = self._get_observation()
        
        # 返回初始观测和信息
        info = {'current_period': self.current_period}
        return observation, info
    
    def step(self, action):
        """
        执行一个动作并转移到下一个状态
        
        参数:
            action: 动作字典，包含运行模式、物料流量和分配比例
        
        返回:
            observation: 新的观测
            reward: 奖励(当期利润减去碳排放成本)
            terminated: 是否结束
            truncated: 是否提前结束
            info: 额外信息
        """
        # 记录动作
        self.history['actions'].append(action.copy())
        
        # 应用动作到模型
        self._apply_action(action)
        
        # 求解优化模型
        status = self.model.solve()
        
        # 处理求解结果
        if status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            # 获取结果
            results = self.model.get_results()
            
            # 检查是否包含所有必要的财务键
            has_profit = 'profit' in results['solution']
            has_revenue = 'revenue' in results['solution']
            has_material_cost = 'material_cost' in results['solution']
            has_processing_cost = 'processing_cost' in results['solution']
            has_carbon_emission = 'period_carbon_emission' in results['solution']
            has_weighted_objective = 'weighted_objective' in results['solution']
            
            # 获取原始财务数据
            profit_value = results['solution']['profit'].get(0, 0) if has_profit else 0
            revenue_value = results['solution']['revenue'].get(0, 0) if has_revenue else 0
            material_cost_value = results['solution']['material_cost'].get(0, 0) if has_material_cost else 0
            processing_cost_value = results['solution']['processing_cost'].get(0, 0) if has_processing_cost else 0
            carbon_emission = results['solution']['period_carbon_emission'].get(0, 0) if has_carbon_emission else 0
            weighted_profit = results['solution']['weighted_objective'] if has_weighted_objective else profit_value
            
            # 计算考虑碳排放的奖励
            max_reward = 5e8  # 你可以根据历史数据或业务经验调整
            reward = weighted_profit / max_reward  # 归一化奖励
            
            # 记录最终的财务数据和碳排放数据
            self.history['profits'].append(profit_value)
            self.history['rewards'].append(reward)
            self.history['revenues'].append(revenue_value)
            self.history['material_costs'].append(material_cost_value)
            self.history['processing_costs'].append(processing_cost_value)
            self.history['carbon_emissions'].append(carbon_emission)
            self.history['weighted_profits'].append(weighted_profit)
            
            for p in self.model.Des:
                self.history['inventory'][p].append(results['solution']['inventory'].get((p, 1), 0))
                self.history['sales'][p].append(results['solution']['sales'].get((p, 0), 0))
                self.history['production'][p].append(results['solution']['production'].get((p, 0), 0))
            
            for u in self.model.u_process4:
                mode = 'B'
                for m in self.model.mo:
                    if results['solution']['unit_mode'].get((u, m, 0), 0) >= 0.5:
                        mode = m
                self.history['unit_modes'][u].append(mode)
            
            # 更新模型初始库存
            for product in self.model.Des:
                self.model.initial_inventory[product] = results['solution']['inventory'].get((product, 1), 
                                                    self.model.initial_inventory[product])
        else:
            # 模型不可解，给予负奖励
            reward = -1.0
            print(f"警告: 周期{self.current_period+1}模型求解失败，状态: {status}")
            # 记录负奖励
            self.history['rewards'].append(reward)
            self.history['profits'].append(0.0)
            self.history['revenues'].append(0.0)
            self.history['material_costs'].append(0.0)
            self.history['processing_costs'].append(0.0) 
            self.history['carbon_emissions'].append(0.0)  # 添加碳排放记录
            self.history['weighted_profits'].append(reward)  # 使用相同的负奖励
            
            for p in self.model.Des:
                self.history['inventory'][p].append(self.model.initial_inventory[p])  # 保持库存不变
                self.history['sales'][p].append(0.0)  # 销量为0
                self.history['production'][p].append(0.0)  # 产量为0
        
            for u in self.model.u_process4:
                last_mode = self.history['unit_modes'][u][-1] if self.history['unit_modes'][u] else 'B'
                self.history['unit_modes'][u].append(last_mode)
        
        # 更新当前周期
        self.current_period += 1
        
        # 判断是否结束
        terminated = self.current_period >= self.planning_horizon
        truncated = False
        
        # 如果未结束，生成新的市场环境
        if not terminated:
            self._update_market_conditions()
        
        # 获取新的观测
        observation = self._get_observation()
        
        # 准备额外信息
        info = {
            'model_status': status,
            'current_period': self.current_period,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染环境状态"""
        if self.render_mode is None:
            return
        
        if not self.history['profits']:
            return  # 无数据可渲染
        
        # 创建图表
        fig, axes = plt.subplots(5, 1, figsize=(12, 20))  # 增加一个子图用于碳排放
        
        # 1. 利润曲线
        axes[0].plot(self.history['profits'], marker='o', color='blue', label='Original Profit')
        axes[0].plot(self.history['weighted_profits'], marker='s', color='green', linestyle='--', 
                   label='Consider emission Profit')
        axes[0].set_title('Each Period Profit')
        axes[0].set_xlabel('Period')
        axes[0].set_ylabel('Profit')
        axes[0].legend()
        axes[0].grid(True)
        
        # 2. 主要产品库存
        main_products = ['W92', 'W95', 'JET', 'W00']
        for p in main_products:
            if p in self.model.Des:
                axes[1].plot(self.history['inventory'][p], marker='o', label=f'{p}库存')
        
        axes[1].set_title('Inventory of Main Products')
        axes[1].set_xlabel('Period')
        axes[1].set_ylabel('Inventory')
        axes[1].legend()
        axes[1].grid(True)
        
        # 3. 主要产品销售量
        for p in main_products:
            if p in self.model.Des:
                axes[2].plot(self.history['sales'][p], marker='o', label=f'{p}Sales')
        
        axes[2].set_title('Sales of Main Products')
        axes[2].set_xlabel('Period')
        axes[2].set_ylabel('Sales')
        axes[2].legend()
        axes[2].grid(True)
        
        # 4. 装置运行模式
        for i, u in enumerate(self.model.u_process4):
            y_values = [1 if mode == 'B' else 0 for mode in self.history['unit_modes'][u]]
            axes[3].step(range(len(y_values)), y_values, label=f'{u}Mode', where='post')
        
        axes[3].set_title('Mode (0=A, 1=B)')
        axes[3].set_xlabel('Period')
        axes[3].set_yticks([0, 1])
        axes[3].set_yticklabels(['A', 'B'])
        axes[3].legend()
        axes[3].grid(True)
        
        # 5. 碳排放曲线
        if self.history['carbon_emissions']:
            axes[4].bar(range(len(self.history['carbon_emissions'])), self.history['carbon_emissions'], 
                      color='brown', alpha=0.7)
            axes[4].set_title('Carbon Emission Each Period')
            axes[4].set_xlabel('Period')
            axes[4].set_ylabel('Carbon Emission')
            axes[4].grid(True)
        
        plt.tight_layout()
        
        if self.render_mode == 'human':
            plt.show()
        elif self.render_mode == 'rgb_array':
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close()
            return image
    
    def _get_observation(self):
        """获取当前环境状态的观测"""
        # 获取产品库存
        inventory = np.array([self.model.initial_inventory[p] for p in self.model.Des])
        
        # 获取产品价格
        prices = np.array([self.model.price_fluctuation.get(p, [0])[0] for p in self.model.Des])
        
        # 获取产品需求区间
        demand_min = np.array([self.model.min_demand_fluctuation.get(p, [0])[0] for p in self.model.Des])
        demand_max = np.array([self.model.max_demand_fluctuation.get(p, [0])[0] for p in self.model.Des])
        
        # 当前周期(归一化)
        period = np.array([self.current_period])
        
        # 组合成完整观测
        observation = np.concatenate([inventory, prices, demand_min, demand_max, period])
        # 修复浮点精度问题 - 确保观察值在合法范围内
        observation = np.clip(observation, 0, None)  # 所有负值强制变为0
        return observation.astype(np.float32)
    
    def _apply_action(self, action):
        """应用RL代理的动作到优化模型
            动作含义:
        - action[0:3]: 装置运行模式(FC1, FC2, HCU)，0=模式A，1=模式B
        - action[3]: H6O_FC1_IN/H6O分流比，控制H6O在FC1和FC2之间的分配
        - action[4]: HCJ_JET_IN/HCJ分流比，控制航煤和柴油产品的产量比例
        - action[5]: R3S_PL1_IN/R3S分流比，控制R3S物料的分流
        - action[6]: IV2_FC1_IN/IV2分流比，控制IV2到FC1装置的比例
        - action[7]: C1L_GF1_IN/C1L分流比，控制C1L在GF1和GF2之间的分配
        - action[8]: HCO_FC1_IN/HCO分流比，控制HCO在FC1和FC2之间的分配
        """
        # 1. 设置装置运行模式 (前3个值)
        units = list(self.model.u_process4)  # ['FC1', 'FC2', 'HCU']
        for i, unit in enumerate(units):
            # 二值化模式选择(>=0.5为模式B)
            mode_choice = 'B' if action[i] >= 0.5 else 'A'
            
            # 添加约束强制模式选择
            for mo_val in self.model.mo:
                if mo_val == mode_choice:
                    self.model.model.addConstr(
                        self.model.x[unit, mo_val, 0] >= 0.5,
                        name=f"soft_mode_{unit}_{mo_val}"
                    )
                else:
                    self.model.model.addConstr(
                        self.model.x[unit, mo_val, 0] <= 0.49999999999999999,
                        name=f"soft_mode_{unit}_{mo_val}"
                    )
        
        #H6O_FC1_IN/H6O - 控制H6O在FC1和FC2之间的分配
        h6o_fc1_ratio = action[3]  
        self.model.model.addConstr(
            self.model.MS['H6O_FC1_IN', 0] >= (h6o_fc1_ratio - 1e-6) * self.model.MS['H6O', 0],
            name=f"soft_min_H6O_FC1_split"
        )
        self.model.model.addConstr(
            self.model.MS['H6O_FC1_IN', 0] <= (h6o_fc1_ratio + 1e-6) * self.model.MS['H6O', 0],
            name=f"soft_max_H6O_FC1_split"
        )
        # HCJ_JET_IN/HCJ - 控制航煤和柴油产品的产量比例
        hcj_jet_ratio = action[4]  
        self.model.model.addConstr(
            self.model.MS['HCJ_JET_IN', 0] >= (hcj_jet_ratio - 1e-6) * self.model.MS['HCJ', 0],
            name=f"soft_min_HCJ_JET_split"
        )
        self.model.model.addConstr(
            self.model.MS['HCJ_JET_IN', 0] <= (hcj_jet_ratio + 1e-6) * self.model.MS['HCJ', 0],
            name=f"soft_max_HCJ_JET_split"
        )
        # R3S分配比例(R3S_PL1_IN/R3S, R3S_EBN_IN/R3S)
        r3s_pl1_ratio = action[5]  # R3S到PL1的比例
        
        # 软约束：允许极小的偏差
        self.model.model.addConstr(
            self.model.MS['R3S_PL1_IN', 0] >= (r3s_pl1_ratio - 1e-6) * self.model.MS['R3S', 0],
            name=f"soft_min_R3S_PL1_split"
        )
        self.model.model.addConstr(
            self.model.MS['R3S_PL1_IN', 0] <= (r3s_pl1_ratio + 1e-6) * self.model.MS['R3S', 0],
            name=f"soft_max_R3S_PL1_split"
        )
        # IV2_FC1_IN/IV2 - 控制IV2到FC1装置的比例
        iv2_fc1_ratio = action[6] 
        self.model.model.addConstr(
            self.model.MS['IV2_FC1_IN', 0] >= (iv2_fc1_ratio - 1e-6) * self.model.MS['IV2', 0],
            name=f"soft_min_IV2_FC1_split"
        )
        self.model.model.addConstr(
            self.model.MS['IV2_FC1_IN', 0] <= (iv2_fc1_ratio + 1e-6) * self.model.MS['IV2', 0],
            name=f"soft_max_IV2_FC1_split"
        )
        # C1L_GF1_IN/C1L - 控制C1L在GF1和GF2之间的分配
        c1l_gf1_ratio = action[7]  
        self.model.model.addConstr(
            self.model.MS['C1L_GF1_IN', 0] >= (c1l_gf1_ratio - 1e-6) * self.model.MS['C1L', 0],
            name=f"soft_min_C1L_GF1_split"
        )
        self.model.model.addConstr(
            self.model.MS['C1L_GF1_IN', 0] <= (c1l_gf1_ratio + 1e-6) * self.model.MS['C1L', 0],
            name=f"soft_max_C1L_GF1_split"
        )
        # #HCO_FC1_IN/HCO - 控制HCO在FC1和FC2之间的分配
        hco_fc1_ratio = action[8]  
        self.model.model.addConstr(
            self.model.MS['HCO_FC1_IN', 0] >= (hco_fc1_ratio - 1e-6) * self.model.MS['HCO', 0],
            name=f"soft_min_HCO_FC1_split"
        )
        self.model.model.addConstr(
            self.model.MS['HCO_FC1_IN', 0] <= (hco_fc1_ratio + 1e-6) * self.model.MS['HCO', 0],
            name=f"soft_max_HCO_FC1_split"
        )
    def _update_market_conditions(self):
        """更新下一周期的市场条件"""
        # 保存当前模型的初始库存
        old_inventory = {}
        for product in self.model.Des:
            old_inventory[product] = self.model.initial_inventory[product]
        
        # 创建新的优化模型，传递基础随机种子和碳排放权重
        self.model = RefineryInventoryModel(
            time_periods=max(12, self.planning_horizon), # 确保足够的周期数据
            random_seed=RANDOM_SEED, 
            carbon_weight=self.carbon_weight
        )
        
        # 设置基于当前周期的随机种子，但使用全局种子作为基础
        deterministic_seed = RANDOM_SEED + self.current_period  # 每个周期使用不同但可复现的种子
        random.seed(deterministic_seed)
        np.random.seed(deterministic_seed)

        # 生成新的市场波动(价格和需求)
        self.model.generate_market_fluctuations()
        
        # 恢复库存设置
        for product in self.model.Des:
            self.model.initial_inventory[product] = old_inventory[product]
        
        # 创建单周期模型(保留多周期模型的波动数据)
        multi_period_fluctuations = {
            'min_demand': self.model.min_demand_fluctuation,
            'max_demand': self.model.max_demand_fluctuation,
            'price': self.model.price_fluctuation,
            'crude_cost': self.model.MX1_cost_fluctuation
        }
        
        # 创建单周期模型，传递碳排放权重
        self.model = RefineryInventoryModel(time_periods=1, carbon_weight=self.carbon_weight)
        
        # 恢复库存设置
        for product in self.model.Des:
            self.model.initial_inventory[product] = old_inventory[product]
        
        # 将多周期波动数据映射到单周期模型，使用安全的索引访问方式
        self.model.min_demand_fluctuation = {}
        self.model.max_demand_fluctuation = {}
        self.model.price_fluctuation = {}
        
        # 确保安全访问波动数据
        for product in self.model.Des:
            # 获取多周期数据列表
            min_demand_list = multi_period_fluctuations['min_demand'].get(product, [0])
            max_demand_list = multi_period_fluctuations['max_demand'].get(product, [0])
            price_list = multi_period_fluctuations['price'].get(product, [0])
            
            # 如果列表为空，使用默认值
            if not min_demand_list:
                min_demand_list = [0]
            if not max_demand_list:
                max_demand_list = [0]
            if not price_list:
                price_list = [0]
            
            # 使用模运算确保索引在有效范围内
            period_index = self.current_period % len(min_demand_list)
            
            # 安全地获取波动值
            min_demand_value = min_demand_list[period_index]
            max_demand_value = max_demand_list[period_index]
            price_value = price_list[period_index]
            
            # 设置当前周期的波动值
            self.model.min_demand_fluctuation[product] = [min_demand_value]
            self.model.max_demand_fluctuation[product] = [max_demand_value]
            self.model.price_fluctuation[product] = [price_value]
        
        # 设置原油成本波动
        crude_cost_list = multi_period_fluctuations['crude_cost']
        # 默认原油成本后期可引入Fluctuatation
        self.model.MX1_cost_fluctuation = [self.model.base_MX1_cost]
        
        # 记录更新 - 替换为更详细的输出
        print(f"\n周期 {self.current_period+1} 市场条件更新完成")
        print("-" * 60)
        print(f"{'产品':<6} {'价格波动':<12} {'最小需求波动':<14} {'最大需求波动':<14}")
        print("-" * 60)

        # 按字母顺序排序产品名，便于查看
        sorted_products = sorted(self.model.Des)
        for product in sorted_products:
            # 安全地获取波动值，默认为0
            price_val = self.model.price_fluctuation.get(product, [0])[0]
            min_demand_val = self.model.min_demand_fluctuation.get(product, [0])[0]
            max_demand_val = self.model.max_demand_fluctuation.get(product, [0])[0]
            
            # 打印当前产品的波动数据
            print(f"{product:<6} {price_val:<12.2f} {min_demand_val:<14.2f} {max_demand_val:<14.2f}")
    
    def save_history(self, filename='refinery_rl_history.xlsx'):
        """保存历史数据到Excel文件"""
        # 创建DataFrame
        # 处理时期利润和决策数据
        results_df = pd.DataFrame({
            'Period': range(1, len(self.history['profits'])+1),
            'Profit': self.history['profits'],
            'Reward': self.history['rewards'],
            'Revenue': self.history['revenues'],       
            'Material_Cost': self.history['material_costs'], 
            'Processing_Cost': self.history['processing_costs'],
            'Carbon_Emission': self.history['carbon_emissions'],  # 新增：记录碳排放
            'Weighted_Profit': self.history['weighted_profits']   # 新增：记录考虑碳排放的加权利润
        })
        
        # 添加装置模式
        for u in self.model.u_process4:
            results_df[f'{u}_Mode'] = self.history['unit_modes'][u]
        
        # 添加主要产品数据
        main_products = ['W92', 'W95', 'JET', 'W00']
        for p in main_products:
            if p in self.model.Des:
                results_df[f'{p}_Inventory'] = self.history['inventory'][p]
                results_df[f'{p}_Sales'] = self.history['sales'][p] 
                results_df[f'{p}_Production'] = self.history['production'][p]
        
        # 添加动作数据
        actions_data = {}
        try:
                if self.history['actions']:
                    # 装置模式
                    for i, unit in enumerate(['FC1', 'FC2', 'HCU']):
                        actions_data[f'{unit}_Mode_Action'] = [
                            'B' if a[i] > 0.5 else 'A' for a in self.history['actions']
                        ]

                    # 分配比例
                    split_names = ['H6O_FC1/H6O', 
                                    'HCJ_JET/HCJ', 
                                    'R3S_PL1/R3S',
                                    'IV2_FC1/IV2',
                                    'C1L_GF1/C1L',
                                    'HCO_FC1/HCO'
                    ]
                    for i, name in enumerate(split_names):
                        actions_data[f'Split_{name}'] = [a[i+3] for a in self.history['actions']] 
                # 合并到结果DataFrame
                for k, v in actions_data.items():
                    if len(v) == len(results_df):
                        results_df[k] = v
        except Exception as e:
            print(f"处理动作数据时出错: {e}")
        
        # 保存到Excel
        results_df.to_excel(filename, index=False)
        print(f"历史数据已保存到 {filename}")

def print_financial_summary(env):
    """打印各周期的详细财务明细"""
    
    if not env.history['profits']:
        print("没有可用的财务数据")
        return
    
    print("\n" + "="*80)
    print("各周期财务明细与碳排放")
    print("="*80)
    
    print("\n各周期利润明细:")
    print("-"*80)
    print(f"{'周期':<6} {'收入':<12} {'原材料成本':<12} {'加工成本':<12} {'净利润':<12} {'碳排放':<12} {'加权利润':<12}")
    print("-"*80)
    
    total_revenue = 0
    total_material_cost = 0
    total_processing_cost = 0
    total_profit = 0
    total_carbon_emission = 0
    total_weighted_profit = 0
    
    for t in range(len(env.history['profits'])):
        revenue = env.history['revenues'][t]
        material_cost = env.history['material_costs'][t]
        processing_cost = env.history['processing_costs'][t]
        profit = env.history['profits'][t]
        carbon_emission = env.history['carbon_emissions'][t] if t < len(env.history['carbon_emissions']) else 0
        weighted_profit = env.history['weighted_profits'][t] if t < len(env.history['weighted_profits']) else 0
        
        total_revenue += revenue
        total_material_cost += material_cost
        total_processing_cost += processing_cost
        total_profit += profit
        total_carbon_emission += carbon_emission
        total_weighted_profit += weighted_profit
        
        print(f"{t+1:<6} {revenue:<12.2f} {material_cost:<12.2f} {processing_cost:<12.2f} {profit:<12.2f} {carbon_emission:<12.2f} {weighted_profit:<12.2f}")
    
    print("-"*80)
    print(f"{'总计':<6} {total_revenue:<12.2f} {total_material_cost:<12.2f} {total_processing_cost:<12.2f} {total_profit:<12.2f} {total_carbon_emission:<12.2f} {total_weighted_profit:<12.2f}")
    print("-"*80)
    
    # 添加利润比例分析
    if total_revenue > 0:
        material_cost_ratio = total_material_cost / total_revenue * 100
        processing_cost_ratio = total_processing_cost / total_revenue * 100
        profit_ratio = total_profit / total_revenue * 100
        carbon_impact_ratio = (total_profit - total_weighted_profit) / total_revenue * 100 if total_profit != total_weighted_profit else 0
        
        print(f"\n成本及利润分析 (占收入百分比):")
        print(f"原材料成本: {material_cost_ratio:.2f}%")
        print(f"加工成本: {processing_cost_ratio:.2f}%")
        print(f"碳排放影响: {carbon_impact_ratio:.2f}%")
        print(f"原始净利润: {profit_ratio:.2f}%")
        print(f"考虑碳排放的净利润: {(total_weighted_profit/total_revenue*100):.2f}%")
        
    # 添加碳排放分析
    print(f"\n碳排放分析:")
    print(f"总碳排放量: {total_carbon_emission:.2f}")
    print(f"碳排放权重: {env.carbon_weight}")
    print(f"碳排放成本估算: {(total_profit - total_weighted_profit):.2f}")

# 训练函数
def train_refinery_agent(total_timesteps=100000, planning_horizon=12, log_dir="./logs", 
                        seed=RANDOM_SEED, carbon_weight=0.3):
    """
    训练炼油厂强化学习代理
    
    参数:
        total_timesteps: 总训练步数
        planning_horizon: 规划周期数
        log_dir: 日志目录
        seed: 随机种子，默认使用全局种子
        carbon_weight: 碳排放权重，控制碳排放在奖励函数中的权重
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 创建训练环境，传递碳排放权重
    env = RefineryEnv(planning_horizon=planning_horizon, carbon_weight=carbon_weight)
    
    # 检查环境是否符合gymnasium接口标准
    check_env(env, warn=True)
    
    # 创建时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = f"{log_dir}/refinery_{timestamp}"
    os.makedirs(log_path, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建PPO模型，添加种子设置
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_path,
        learning_rate=0.001,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.2,
        seed=seed
    )
    callback = TensorboardCallback()
    # 训练模型
    model.learn(total_timesteps=total_timesteps, 
                tb_log_name=f"PPO_refinery_{planning_horizon}periods"
                ,callback=callback)
    
    # 保存模型
    model_path = f"{log_path}/ppo_refinery_model.zip"
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    return model, env, model_path

# 评估函数
def evaluate_refinery_agent(model_path, episodes=10, planning_horizon=12, render_mode='human', 
                           seed=RANDOM_SEED, carbon_weight=0.3):
    """
    评估训练好的炼油厂强化学习代理
    
    参数:
        model_path: 模型路径
        episodes: 评估回合数
        planning_horizon: 规划周期数
        render_mode: 渲染模式
        seed: 随机种子，默认使用全局种子
        carbon_weight: 碳排放权重，控制碳排放在奖励函数中的权重
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 创建评估环境，传递碳排放权重
    env = RefineryEnv(
        planning_horizon=planning_horizon, 
        render_mode=render_mode, 
        carbon_weight=carbon_weight
    )
    
    # 加载训练好的模型
    model = PPO.load(model_path)
    
    # 评估结果
    rewards = []
    
    for episode in range(episodes):
        # 重置环境，使用可复现的种子
        episode_seed = seed + episode  # 每个回合使用不同但可控的种子
        obs, _ = env.reset(seed=episode_seed)
        done = False
        total_reward = 0
        
        # 运行一个完整回合
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        # 渲染最终结果
        if render_mode == 'human':
            env.render()
        
        # 保存历史数据
        env.save_history(f"refinery_eval_episode_{episode+1}.xlsx")
        
        # 记录回合奖励
        rewards.append(total_reward)
        print(f"回合 {episode+1} 总奖励: {total_reward:.2f}")
    
    # 打印评估结果
    print(f"\n==== 评估结果 ====")
    print(f"回合数: {episodes}")
    print(f"平均总奖励: {np.mean(rewards):.2f}")
    print(f"最大总奖励: {np.max(rewards):.2f}")
    print(f"最小总奖励: {np.min(rewards):.2f}")
    print(f"标准差: {np.std(rewards):.2f}")
    
    return env
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_counter = 0
        
    def _on_step(self) -> bool:
        # 每100步记录一次数据
        if self.step_counter % 100 == 0:
            self.logger.record('custom/reward', self.training_env.get_attr('history')[0]['rewards'][-1])
        self.step_counter += 1
        return True
# 主函数
if __name__ == "__main__":
    # 可以在这里修改全局随机种子
    custom_seed = 42  # 可以修改为任何整数
    carbon_weight = 0.1  # 设置碳排放权重
    
    # 设置自定义随机种子
    random.seed(custom_seed)
    np.random.seed(custom_seed)
    torch.manual_seed(custom_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(custom_seed)
    
    # 训练模型
    model, train_env, model_path = train_refinery_agent(
        total_timesteps=100000,
        planning_horizon=12,
        log_dir="./refinery_rl_logs",
        seed=custom_seed,
        carbon_weight=carbon_weight
    )
    
    # 评估模型
    eval_env = evaluate_refinery_agent(
        model_path=model_path,
        episodes=1,
        planning_horizon=18,
        render_mode='human',
        seed=custom_seed,
        carbon_weight=carbon_weight
    )
    
    # 打印详细财务数据和碳排放数据
    print("\n===== 详细财务与碳排放报告 =====")
    print_financial_summary(eval_env)