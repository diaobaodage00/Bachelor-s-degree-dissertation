import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
import pandas as pd
import func_load_para
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 添加全局随机种子常量
RANDOM_SEED = 42

# 设置全局随机种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class RefineryInventoryModel:
    def __init__(self, time_periods=1, carbon_weight=0.3, random_seed=RANDOM_SEED):
        """初始化炼油厂库存管理模型"""
        # 设置随机种子
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.model = gp.Model("Jiujiang_Refinery_inventory_model")
        self.time_periods = time_periods
        self.periods = range(time_periods)
        self.periods_extended = range(time_periods+1)
        self.carbon_weight = carbon_weight  # 碳排放权重参数
        
        # 定义集合
        self.define_sets()
        
        # 加载参数
        self.load_parameters()
        
        # 引入波动因素
        self.generate_market_fluctuations()
        
        # 创建模型变量
        self.create_variables()
        
        # 设置目标函数和约束条件
        self.setup_objective_and_constraints()
    
    def define_sets(self):
        """定义模型中的各个集合"""
        # 操作模式
        self.mo = ['A', 'B']
        
        # 流股集合
        self.s = ['MX1', 'BHR', 'BME', 'BH2', 'BRF', 'BMT', 'BEB', 'W92', 'W95', 'JET', 'W00', 'EEN', 'BNZ', 'XYL', 'PLG', 'PPP',
        'CCK', 'SULH', 'NH3', 'STR', 'A60', 'PGS', 'PUL', 'AG1', 'WN1', 'KE1', 'LD1', 'HD1', 'V11', 'V21', 'V31', 'V41',
        'VR1', 'AG2', 'WN2', 'KE3', 'LD2', 'HD2', 'V12', 'V22', 'V32', 'VR2', 'HG9', 'CDG', 'CDT', 'CDL', 'R1G', 'RSL',
        'TOP', 'R1S', 'RFH', 'R1M', 'R1L', 'R1N', 'HR1', 'BNZ_AEX_OUT', 'XYL_AEX_OUT', 'RF1', 'HAR', 'HS3', 'R3S', 'C1L',
        'C1N', 'C1D', 'C1S', 'C1K', 'HS4', 'R4S', 'C2L', 'C2N', 'C2D', 'C2S', 'C2K', 'KLR', 'PLS', 'HCG', 'HLG', 'HCL', 
        'HLN', 'HHN', 'HCJ', 'HCD', 'HCO', 'HS5', 'R5S', 'K1L', 'K1N', 'K1D', 'K1O', 'K1K', 'D1O', 'DAA_SAU_OUT', 'H3S',
        'QT6', 'H6N', 'H6D', 'H6O', 'HS6', 'R6S', 'H1J', 'HS7', 'R7S', 'H2N', 'H2E', 'H2D', 'H2K', 'R8S', 'H1N', 'H3N',
        'HS9', 'R9S', 'H4N', 'H4D', 'R0S', 'H5J', 'GAS_HF7_OUT', 'SH7', 'SZB', 'PH2_HF8_OUT', 'H8N', 'H8O', 'GAS_GF1_OUT',
        'NC3_GF1_OUT', 'RC3_GF1_OUT', 'MC4_GF1_OUT', 'GAS_GF2_OUT', 'NC3_GF2_OUT', 'RC3_GF2_OUT', 'MC4_GF2_OUT', 'MTL', 
        'MTB', 'GAS_SSUL_OUT', 'GAS_PSA_OUT', 'PH2_PSA_OUT', 'ZBB', 'EBN', 'SMF', 'EBW', 'TOL', 'BYY', 'THW', 'ALK', 
        'ALL', 'ALS', 'MX1_1', 'MX1_2', 'GAS_CUT_IN', 'WN2_CUT_IN', 'IV2_FC1_IN', 'GW1', 'H6O_FC1_IN', 'HCO_FC1_IN',
        'GV1', 'H6O_FC2_IN', 'HCO_FC2_IN', 'BHS', 'WN1_TP1_IN', 'WN2_TP1_IN', 'H2T_TP1_IN', 'IV1', 'IV2_HCU_IN', 
        'C2D_HCU_IN', 'H2T_HCU_IN', 'GVR', 'DAA_CK1_IN', 'VR1_SAU_IN', 'VR2_HF6_IN', 'IV2_HF6_IN', 'C2D_HF6_IN',
        'H2T_HF6_IN', 'LD2_HF1_IN', 'RFH_HF1_IN', 'C2D_HF2_IN', 'LD2_HF2_IN', 'HD2_HF2_IN', 'RFH_HF2_IN', 'KE1_HF4_IN',
        'HD2_HF4_IN', 'RFH_HF4_IN', 'KE1_HF5_IN', 'RFH_HF5_IN', 'H2T_HF7_IN', 'RFH_HF8_IN', 'C1L_GF1_IN', 'C1L_GF2_IN',
        'MC4_MTB_IN', 'RFH_PSA_IN', 'R3S_EBN_IN', 'BNZ_EBN_IN', 'MC4_ALK_IN', 'RFH_ALK_IN', 'R3S_PL1_IN', 'GAS_PL1_IN',
        'NC3_PL2_IN', 'PH2', 'V31_PL5_IN', 'GST', 'H2T', 'CKE', 'IV2', 'V31_GV1_IN', 'VR1_GVR_IN', 'VR2_GVR_IN',
        'GV1_FC2_IN', 'RC3', 'XYL_W92_IN', 'HCJ_JET_IN', 'HCJ_W00_IN', 'WN1_EEN_IN', 'DAA']
            
        # 源和产品子集
        self.Sou = ['MX1', 'BME', 'BHR', 'BH2', 'BRF', 'BMT', 'BEB']
        self.Sou_un = ['BME', 'BHR', 'BH2', 'BRF', 'BMT', 'BEB']
        
        self.Des = ['W92', 'W95', 'JET', 'W00', 'EEN', 'BNZ', 'XYL', 'PLG', 'PPP', 'CCK', 'SULH', 'NH3', 'STR', 'A60', 'PGS', 'PUL']
        self.Des_un = ['W92', 'W95', 'JET', 'W00', 'EEN', 'BNZ', 'XYL', 'PLG', 'PPP', 'CCK', 'SULH', 'NH3', 'STR']
        self.Des_ce = [item for item in self.Des if item not in self.Des_un]
        
        # 处理单元集合
        self.u = ['CDU1', 'CDU2', 'CUT', 'TP1', 'AEX', 'FC1', 'FC2', 'SVTP', 'HCU', 'CK1', 'SAU', 'HF6', 'HF1', 'HF2', 'HF3', 
             'HF4', 'HF5', 'HF7', 'HF8', 'GF1', 'GF2', 'SPPU', 'SMTB', 'SSUL', 'SPSA', 'SEBN', 'SSTR', 'SALK']
        
        # 处理单元子集
        self.u_process1 = ['CDU1', 'CDU2', 'CUT', 'FC1', 'FC2', 'AEX', 'SVTP', 'CK1', 'SAU', 'GF1', 'GF2', 'SPPU', 'SSUL', 'HF3']
        self.u_process2 = ['TP1', 'HCU', 'HF6', 'HF1', 'HF2', 'HF4', 'HF5', 'HF7', 'HF8']
        self.u_process3 = ['SMTB', 'SPSA', 'SEBN', 'SSTR', 'SALK']
        self.u_process4 = ['FC1', 'FC2', 'HCU']
        self.u_process5 = ['CDU1', 'CDU2', 'CUT', 'AEX', 'SVTP', 'CK1', 'SAU', 'GF1', 'GF2', 'SPPU', 'SSUL', 'TP1', 'HF6', 'HF1',
                      'HF2', 'HF3', 'HF4', 'HF5', 'HF7', 'HF8', 'SMTB', 'SPSA', 'SEBN', 'SSTR', 'SALK']
        
        # 混合器集合
        self.m = ['SPL1', 'SPL2', 'SPL3', 'SPL4', 'SPL5']
        self.is_mix = {
            'SPL1': ['R1S', 'R3S_PL1_IN', 'R5S', 'R6S', 'R7S', 'R8S', 'R9S', 'R0S', 'GAS_PL1_IN', 'CDG', 'EBW', 'THW', 'ALS'],
            'SPL2': ['RSL', 'R1L', 'K1L', 'NC3_PL2_IN', 'MTL', 'CDL', 'HCL', 'ALL'],
            'SPL3': ['BH2', 'PH2'],
            'SPL4': ['C1K', 'C2K', 'K1K'],
            'SPL5': ['V11', 'V21', 'V31_PL5_IN', 'V12', 'V22', 'V32']
        }
        self.os_mix = {
            'SPL1': ['GST', 'PGS'],
            'SPL2': ['PLG'],
            'SPL3': ['H2T'],
            'SPL4': ['CKE', 'CCK'],
            'SPL5': ['IV1', 'IV2']
        }
        
        # 储罐集合
        self.tank = ['SGW1', 'SGV1', 'SGVR']
        self.is_tank = {
            'SGW1': ['KLR'],
            'SGV1': ['V31_GV1_IN'],
            'SGVR': ['VR1_GVR_IN', 'VR2_GVR_IN']
        }
        self.os_tank = {
            'SGW1': ['GW1'],
            'SGV1': ['GV1'],
            'SGVR': ['GVR']
        }
        
        # 处理单元进出流股
        self.is_process = {
            'CDU1': ['MX1_1'],
            'CDU2': ['MX1_2'],
            'CUT': ['AG2', 'HCG', 'GAS_CUT_IN', 'WN2_CUT_IN', 'H6N'],
            'FC1': ['IV2_FC1_IN', 'GW1', 'H6O_FC1_IN', 'AG1', 'HCO_FC1_IN'],
            'FC2': ['GV1_FC2_IN', 'H6O_FC2_IN', 'HCO_FC2_IN'],
            'AEX': ['R1N', 'HR1'],
            'SVTP': ['C1S'],
            'CK1': ['GVR', 'C2S', 'DAA_CK1_IN'],
            'SAU': ['VR1_SAU_IN'],
            'GF1': ['C1L_GF1_IN'],
            'GF2': ['C1L_GF2_IN', 'C2L'],
            'SPPU': ['RC3'],
            'SSUL': ['R1G', 'HS3', 'HS4', 'HS5', 'HS6', 'HS7', 'HS9', 'SH7', 'H3S', 'HG9', 'BHS'],
            'TP1': ['CDT', 'WN1_TP1_IN', 'BRF', 'WN2_TP1_IN', 'H2T_TP1_IN', 'HHN'],
            'HCU': ['IV1', 'IV2_HCU_IN', 'C1D', 'C2D_HCU_IN', 'H2T_HCU_IN'],
            'HF6': ['VR2_HF6_IN', 'V41', 'IV2_HF6_IN', 'K1O', 'D1O', 'C2D_HF6_IN', 'H2T_HF6_IN'],
            'HF1': ['LD2_HF1_IN', 'RFH_HF1_IN'],
            'HF2': ['C2D_HF2_IN', 'LD2_HF2_IN', 'HD2_HF2_IN', 'K1D', 'K1N', 'RFH_HF2_IN'],
            'HF3': ['C1N', 'C2N'],
            'HF4': ['KE1_HF4_IN', 'LD1', 'HD1', 'HD2_HF4_IN', 'H6D', 'RFH_HF4_IN'],
            'HF5': ['KE1_HF5_IN', 'KE3', 'RFH_HF5_IN'],
            'HF7': ['H3N', 'H2T_HF7_IN'],
            'HF8': ['TOP', 'R1M', 'RFH_HF8_IN'],
            'SMTB': ['MC4_MTB_IN', 'BME'],
            'SPSA': ['RFH_PSA_IN', 'QT6', 'HLG'],
            'SEBN': ['R3S_EBN_IN', 'R4S', 'BNZ_EBN_IN'],
            'SSTR': ['EBN', 'SMF', 'BEB'],
            'SALK': ['MC4_ALK_IN', 'RFH_ALK_IN']
        }
        
        
        self.os_process = {
            'CDU1': ['AG1', 'WN1', 'KE1', 'LD1', 'HD1', 'V11', 'V21', 'V31', 'V41', 'VR1'],
            'CDU2': ['AG2', 'WN2', 'KE3', 'LD2', 'HD2', 'V12', 'V22', 'V32', 'VR2'],
            'CUT': ['HG9', 'CDG', 'CDT', 'CDL'],
            'FC1': ['HS3', 'R3S', 'C1L', 'C1N', 'C1D', 'C1S', 'C1K'],
            'FC2': ['HS4', 'R4S', 'C2L', 'C2N', 'C2D', 'C2S', 'C2K'],
            'AEX': ['BNZ_AEX_OUT', 'XYL_AEX_OUT', 'RF1', 'HAR', 'PUL'],
            'SVTP': ['KLR', 'PLS'],
            'CK1': ['HS5', 'R5S', 'K1L', 'K1N', 'K1D', 'K1O', 'K1K'],
            'SAU': ['D1O', 'DAA_SAU_OUT'],
            'GF1': ['GAS_GF1_OUT', 'NC3_GF1_OUT', 'RC3_GF1_OUT', 'MC4_GF1_OUT'],
            'GF2': ['GAS_GF2_OUT', 'NC3_GF2_OUT', 'RC3_GF2_OUT', 'MC4_GF2_OUT'],
            'SPPU': ['PPP'],
            'SSUL': ['GAS_SSUL_OUT', 'NH3', 'SULH'],
            'TP1': ['R1G', 'RSL', 'TOP', 'R1S', 'RFH', 'R1M', 'R1L', 'R1N', 'HR1'],
            'HCU': ['HCG', 'HLG', 'HCL', 'HLN', 'HHN', 'HCJ', 'HCD', 'HCO'],
            'HF6': ['H3S', 'QT6', 'H6N', 'H6D', 'H6O'],
            'HF1': ['HS6', 'R6S', 'H1J'],
            'HF2': ['HS7', 'R7S', 'H2N', 'H2E', 'H2D', 'H2K'],
            'HF3': ['R8S', 'H1N', 'H3N'],
            'HF4': ['HS9', 'R9S', 'H4N', 'H4D'],
            'HF5': ['R0S', 'H5J'],
            'HF7': ['GAS_HF7_OUT', 'SH7', 'SZB'],
            'HF8': ['PH2_HF8_OUT', 'H8N', 'H8O'],
            'SMTB': ['MTL', 'MTB'],
            'SPSA': ['GAS_PSA_OUT', 'PH2_PSA_OUT'],
            'SEBN': ['ZBB', 'EBN', 'SMF', 'EBW'],
            'SSTR': ['TOL', 'STR', 'BYY', 'THW'],
            'SALK': ['ALK', 'ALL', 'ALS']
        }

        self.os_process4 = {
            'FC1': ['HS3', 'R3S', 'C1L', 'C1N', 'C1D', 'C1S', 'C1K'],
            'FC2': ['HS4', 'R4S', 'C2L', 'C2N', 'C2D', 'C2S', 'C2K'],
            'HCU': ['HCG', 'HLG', 'HCL', 'HLN', 'HHN', 'HCJ', 'HCD', 'HCO']
        }

        # 加氢单元
        self.is_H2_process2 = {
            'TP1': ['H2T_TP1_IN'],
            'HCU': ['H2T_HCU_IN'],
            'HF6': ['H2T_HF6_IN'],
            'HF1': ['RFH_HF1_IN'],
            'HF2': ['RFH_HF2_IN'],
            'HF4': ['RFH_HF4_IN'],
            'HF5': ['RFH_HF5_IN'],
            'HF7': ['H2T_HF7_IN'],
            'HF8': ['RFH_HF8_IN']
        }
        self.os_process5 = {}
        for u_val in self.u_process5:
           self.os_process5[u_val] = self.os_process.get(u_val, [])

        # 产品单元集合
        self.os_process3 = {
            'SMTB': ['MTB'],
            'SPSA': ['PH2_PSA_OUT'],
            'SEBN': ['EBN'],
            'SSTR': ['STR'],
            'SALK': ['ALK']
        }
        # 添加产品库存储罐集合
        self.product_tanks = {product: f"{product}_TANK" for product in self.Des}
        
        # 质量属性集合
        self.q = ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE', 'CTI']
        self.QLinVol = ['SPG', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE', 'CTI']
        self.QLinWt = ['SUL']
        self.SpecificGravity = ['SPG']
        
        # 调和器集合
        self.b = ['gasoil_blender', 'kero_blender', 'diesel_blender', 'naphtha_blender']
        self.is_blender = {
            'gasoil_blender': ['BHR', 'BMT', 'TOL', 'XYL_W92_IN', 'MTB', 'ZBB', 'RF1', 'HAR', 'HLN', 'H1N', 'SZB', 'H8N', 'ALK'],
            'kero_blender': ['HCJ_JET_IN', 'H5J'],
            'diesel_blender': ['HCJ_W00_IN', 'HCD', 'H1J', 'H2E', 'H2D', 'H2K', 'H4D'],
            'naphtha_blender': ['WN1_EEN_IN', 'H4N', 'H2N']
        }
        self.os_blender = {
            'gasoil_blender': ['W92', 'W95'],
            'kero_blender': ['JET'],
            'diesel_blender': ['W00'],
            'naphtha_blender': ['EEN']
        }
        
        # 质量属性跟踪(只保留部分示例)
        self.SQB = {
            'BHR': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'BMT': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'TOL': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'XYL_W92_IN': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'MTB': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'ZBB': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'RF1': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'HAR': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'HLN': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'H1N': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'SZB': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'H8N': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'ALK': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'W92': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'W95': ['SPG', 'SUL', 'RON', 'DON', 'OLV', 'BNZ', 'ARW', 'MTE'],
            'HCJ_JET_IN': ['SPG', 'SUL', 'ARW'],
            'H5J': ['SPG', 'SUL', 'ARW'],
            'JET': ['SPG', 'SUL', 'ARW'],
            'HCJ_W00_IN': ['SPG', 'SUL', 'CTI'],
            'HCD': ['SPG', 'SUL', 'CTI'],
            'H1J': ['SPG', 'SUL', 'CTI'],
            'H2E': ['SPG', 'SUL', 'CTI'],
            'H2D': ['SPG', 'SUL', 'CTI'],
            'H2K': ['SPG', 'SUL', 'CTI'],
            'H4D': ['SPG', 'SUL', 'CTI'],
            'W00': ['SPG', 'SUL', 'CTI'],
            'WN1_EEN_IN': ['SPG', 'SUL', 'OLV', 'ARW'],
            'H4N': ['SPG', 'SUL', 'OLV', 'ARW'],
            'H2N': ['SPG', 'SUL', 'OLV', 'ARW'],
            'EEN': ['SPG', 'SUL', 'OLV', 'ARW']
        }
        
        # 添加碳排放单元集合
        self.carbon_units = ['FC1', 'FC2']  # 考虑碳排放的装置
        self.carbon_coefficient = {
            'FC1': 1024.2,  # FC1的碳排放系数 (tco2eq)
            'FC2': 1229.9   # FC2的碳排放系数 (tco2eq)
        }
    def load_parameters(self):
        """加载模型参数"""
        # 从文件加载基础参数
        txt_file = "Jiujiang_Refinery_DO_data.txt" 
        excel_file = "Jiujiang_Refinery_DO_data.xlsx"
        params = func_load_para.load_parameters(txt_file, excel_file)
        
        # 提取参数
        self.CS = params.get('CS', {})
        self.PS = params.get('PS', {})
        self.CU = params.get('CU', {})
        self.MU_min = params.get('MU_min', {})
        self.MU_max = params.get('MU_max', {})
        self.H2_fix_pro = params.get('H2_fix_pro', {})
        self.Y_fix_mix = params.get('Y_fix_mix', {})
        self.Y_fix_pro = params.get('Y_fix_pro', {})
        self.A_min = params.get('A_min', {})
        self.A_max = params.get('A_max', {})
        self.D_min = params.get('D_min', {})
        self.D_max = params.get('D_max', {})
        self.QS_fix = params.get('QS_fix', {})
        self.QSB_min = params.get('QSB_min', {})
        self.QSB_max = params.get('QSB_max', {})
        self.Y_pro_mo = params.get('Y_pro_mo', {})
        # 定义MX1原油基础成本
        self.base_MX1_cost = 481.4
        
        # 定义产品库存初始值、最小库存和最大库存
        self.initial_inventory = {product: max(self.D_min.get(product, 0)*1, 1000) for product in self.Des}
        self.min_inventory = {product: max(self.D_min.get(product, 0)*0.1, 0) for product in self.Des}
        self.max_inventory = {product: max(self.D_max.get(product, 0)*10, 10000) for product in self.Des}
    
    def generate_market_fluctuations(self):
        """生成市场波动因素：产品需求波动、产品价格波动和原油成本波动"""
        # 如果是多周期模型，则使用统一的随机种子逻辑
        if self.time_periods >= 1:
            # 确保使用相同的随机种子开始
            base_seed = self.random_seed
            
            # 初始化波动字典
            self.min_demand_fluctuation = {}
            self.max_demand_fluctuation = {}
            self.price_fluctuation = {}
            
            for t in range(self.time_periods):
                # 为每个周期设置可控的随机种子
                period_seed = base_seed + t
                random.seed(period_seed)
                np.random.seed(period_seed)
                
                # 为每个产品在当前周期生成波动
                for product in self.Des:
                    # 初始化字典
                    if product not in self.min_demand_fluctuation:
                        self.min_demand_fluctuation[product] = [0] * self.time_periods
                        self.max_demand_fluctuation[product] = [0] * self.time_periods
                        self.price_fluctuation[product] = [0] * self.time_periods
                    
                    base_demand_min = self.D_min.get(product, 0)
                    base_demand_max = self.D_max.get(product, 0)
                    base_price = self.PS.get(product, 0)
                    
                    if base_demand_min > 0:
                        # 需求波动
                        min_fluctuation = random.uniform(0.8, 1.0)
                        max_fluctuation = random.uniform(1.0, 1.2)
                        
                        # 季节性和趋势
                        seasonal = 1 + 0.1 * np.sin(2 * np.pi * t / (self.time_periods/2))
                        trend = 1 + 0.02 * t
                        
                        min_fluctuation *= seasonal * trend
                        max_fluctuation *= seasonal * trend
                        
                        self.min_demand_fluctuation[product][t] = base_demand_min * min_fluctuation
                        self.max_demand_fluctuation[product][t] = base_demand_max * max_fluctuation
                    else:
                        self.min_demand_fluctuation[product][t] = 0
                        self.max_demand_fluctuation[product][t] = base_demand_max * max_fluctuation
                    
                    # 价格波动
                    price_fluctuation = random.uniform(0.9, 1.1)
                    self.price_fluctuation[product][t] = base_price * price_fluctuation
            
            # 原油成本波动固定值
            self.MX1_cost_fluctuation = [self.base_MX1_cost] * self.time_periods
        else:
            # 单周期模式使用固定种子
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        # 原油成本波动固定值
        self.MX1_cost_fluctuation = [self.base_MX1_cost] * self.time_periods
    
    def create_variables(self):
        """创建模型变量"""
        # 目标函数变量
        self.z = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="z")
        self.Profit = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Profit")
        
        # 产量和流率变量 - 添加时间维度
        self.MS = self.model.addVars(self.s, self.periods, vtype=GRB.CONTINUOUS, lb=0, name="MS")
        self.MU = self.model.addVars(self.u, self.periods, vtype=GRB.CONTINUOUS, lb=0, name="MU")
        self.MM = self.model.addVars(self.m, self.periods, vtype=GRB.CONTINUOUS, lb=0, name="MM")
        self.MT = self.model.addVars(self.tank, self.periods, vtype=GRB.CONTINUOUS, lb=0, name="MT")
        
        # 调和相关变量 - 添加时间维度
        self.MB = self.model.addVars(self.Des, self.periods, vtype=GRB.CONTINUOUS, lb=0, name="MB")
        self.MBC = self.model.addVars(
            [(s1, s2, t) for b in self.b for s1 in self.is_blender.get(b, []) for s2 in self.os_blender.get(b, []) for t in self.periods],
            vtype=GRB.CONTINUOUS, lb=0, name="MBC"
        )
        
        # 质量相关变量 - 添加时间维度
        self.QS = self.model.addVars(self.s, self.q, self.periods, vtype=GRB.CONTINUOUS, lb=0, name="QS")
        self.QVBC = self.model.addVars(
            [(s1, s2, q, t) for b in self.b for s1 in self.is_blender.get(b, [])
             for s2 in self.os_blender.get(b, []) for q in self.q if q in self.SQB.get(s1, []) and q in self.SQB.get(s2, [])
             for t in self.periods],
            vtype=GRB.CONTINUOUS, lb=0, name="QVBC"
        )
        self.QMBC = self.model.addVars(
            [(s1, s2, q, t) for b in self.b for s1 in self.is_blender.get(b, [])
             for s2 in self.os_blender.get(b, []) for q in self.q if q in self.SQB.get(s1, []) and q in self.SQB.get(s2, [])
             for t in self.periods],
            vtype=GRB.CONTINUOUS, lb=0, name="QMBC"
        )
        
        # 运行模式选择变量 - 添加时间维度
        self.x = self.model.addVars(self.u_process4, self.mo, self.periods, vtype=GRB.BINARY, name="x")
        
        # 新增变量：产品库存变量
        self.inventory = self.model.addVars(self.Des, self.periods_extended, vtype=GRB.CONTINUOUS, lb=0, name="inventory")
        
        # 新增变量：产品销售量
        self.sales = self.model.addVars(self.Des, self.periods, vtype=GRB.CONTINUOUS, lb=0, name="sales")
        
        # 新增变量：库存持有成本
        self.holding_cost = self.model.addVars(self.periods, vtype=GRB.CONTINUOUS, lb=0, name="holding_cost")
        
        # 经济指标变量
        self.period_profit = self.model.addVars(self.periods, vtype=GRB.CONTINUOUS, name="period_profit")
        self.revenue = self.model.addVars(self.periods, vtype=GRB.CONTINUOUS, lb=0, name="revenue")
        self.material_cost = self.model.addVars(self.periods, vtype=GRB.CONTINUOUS, lb=0, name="material_cost")
        self.processing_cost = self.model.addVars(self.periods, vtype=GRB.CONTINUOUS, lb=0, name="processing_cost")
        self.total_profit = self.model.addVar(vtype=GRB.CONTINUOUS, name="total_profit")
        
        # 新增变量：碳排放变量
        self.carbon_emission = self.model.addVars(self.carbon_units, self.periods, vtype=GRB.CONTINUOUS, lb=0, name="carbon_emission")
        self.total_carbon_emission = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_carbon_emission")
        self.period_carbon_emission = self.model.addVars(self.periods, vtype=GRB.CONTINUOUS, lb=0, name="period_carbon_emission")
        
        # 双目标函数变量
        self.weighted_objective = self.model.addVar(vtype=GRB.CONTINUOUS, name="weighted_objective")
    
    def setup_objective_and_constraints(self):
        """设置模型的目标函数和约束条件"""
       
        # 1. 设置初始库存
        for product in self.Des:
            self.model.addConstr(self.inventory[product, 0] == self.initial_inventory[product], name=f"initial_inventory_{product}")
        
        # 2. 库存平衡约束
        for product in self.Des:
            for t in self.periods:
                self.model.addConstr(
                    self.inventory[product, t+1] == self.inventory[product, t] + self.MS[product, t] - self.sales[product, t],
                    name=f"inventory_balance_{product}_{t}"
                )
        
        # 3. 库存上下限约束
        for product in self.Des:
            for t in range(1, self.time_periods+1):
                self.model.addConstr(
                    self.inventory[product, t] >= self.min_inventory[product],
                    name=f"min_inventory_{product}_{t}"
                )
                self.model.addConstr(
                    self.inventory[product, t] <= self.max_inventory[product],
                    name=f"max_inventory_{product}_{t}"
                )
               
        # 4. 需求满足约束
        for product in self.Des:
            for t in self.periods:
                min_demand = self.min_demand_fluctuation.get(product, [0]*self.time_periods)[t]
                max_demand = self.max_demand_fluctuation.get(product, [0]*self.time_periods)[t]
                
                # 确保最小需求小于最大需求
                if min_demand > max_demand:
                    min_demand, max_demand = max_demand, min_demand
                
                # 添加需求范围约束
                self.model.addConstr(
                    self.sales[product, t] >= min_demand,
                    name=f"min_demand_{product}_{t}"
                )
                self.model.addConstr(
                    self.sales[product, t] <= max_demand,
                    name=f"max_demand_{product}_{t}"
                )
        
        # 5. 计算每期的利润
        for t in self.periods:
            # 收入 = 销售收入
            self.model.addConstr(
                self.revenue[t] == quicksum(self.price_fluctuation.get(product, [0]*self.time_periods)[t] * self.sales[product, t] 
                                        for product in self.Des),
                name=f"revenue_{t}"
            )
            
            # 原材料成本 = 原油成本 + 其他外购原材料成本
            self.model.addConstr(
                self.material_cost[t] == self.MX1_cost_fluctuation[t] * self.MS['MX1', t] + 
                                      quicksum(self.CS.get(s_val, 0) * self.MS[s_val, t] for s_val in self.Sou_un),
                name=f"material_cost_{t}"
            )
            
            # 加工成本
            self.model.addConstr(
                self.processing_cost[t] == quicksum(self.CU.get(u_val, 0) * self.MU[u_val, t] for u_val in self.u),
                name=f"processing_cost_{t}"
            )
            
            # 期间利润
            self.model.addConstr(
                self.period_profit[t] == self.revenue[t] - self.material_cost[t] - self.processing_cost[t],
                name=f"period_profit_{t}"
            )
        
        # 6. 计算总利润
        self.model.addConstr(
            self.total_profit == quicksum(self.period_profit[t] for t in self.periods),
            name="total_profit_calc"
        )
        
        # 7. 添加碳排放计算约束
        for t in self.periods:
            # 计算每个装置的碳排放
            for unit in self.carbon_units:
                self.model.addConstr(
                    self.carbon_emission[unit, t] == self.carbon_coefficient[unit] * self.MU[unit, t],
                    name=f"carbon_emission_{unit}_{t}"
                )
            
            # 计算每期的总碳排放
            self.model.addConstr(
                self.period_carbon_emission[t] == quicksum(self.carbon_emission[unit, t] for unit in self.carbon_units),
                name=f"period_carbon_emission_{t}"
            )
        
        # 计算所有周期的总碳排放
        self.model.addConstr(
            self.total_carbon_emission == quicksum(self.period_carbon_emission[t] for t in self.periods),
            name="total_carbon_emission_calc"
        )
        
        # 设置双目标函数：最大化利润-碳排放加权值
        self.model.addConstr(
            self.weighted_objective == self.total_profit - self.carbon_weight * self.total_carbon_emission,
            name="weighted_objective_calc"
        )
        
        # 8. 设置目标函数：最大化加权目标函数
        self.model.setObjective(self.weighted_objective, GRB.MAXIMIZE)
        
        # 9. 复制原始模型的约束（每个时期单独考虑）
        for t in self.periods:
            # 源约束
            for s_val in self.Sou:
                self.model.addConstr(self.MS[s_val, t] >= self.A_min.get(s_val, 0), name=f"SOU3_EQ1_{s_val}_{t}")
                self.model.addConstr(self.MS[s_val, t] <= self.A_max.get(s_val, 0), name=f"SOU3_EQ2_{s_val}_{t}")
            # 特定产品约束
            self.model.addConstr(self.MS['A60',t] == self.MS['PLS',t] + self.MS['DAA',t], name="DES1_EQ3")
            
            # 处理单元约束
            # PRO_EQ1: 定义装置处理量为所有输入流股的总和
            for u_val in self.u:
                process_inputs = self.is_process.get(u_val, [])
                self.model.addConstr(
                    self.MU[u_val, t] == quicksum(self.MS[s_val, t] for s_val in process_inputs),
                    name=f"PRO_EQ1_{u_val}_{t}"
                )
            
            # PRO_EQ2: 固定产出率的处理单元
            for u_val in self.u_process5:
                for s_val in self.os_process.get(u_val, []):
                    self.model.addConstr(
                        self.MS[s_val, t] == (self.Y_fix_pro.get((s_val, u_val), 0)/100) * self.MU[u_val, t], 
                        name=f"PRO_EQ2_{u_val}_{s_val}_{t}"
                    )
            
            # PRO_EQ3和PRO_EQ4: 处理能力限制
            for u_val in self.u_process1:
                self.model.addConstr(
                    self.MU[u_val, t] >= self.MU_min.get(u_val, 0), 
                    name=f"PRO_EQ3_{u_val}_{t}"
                )
                self.model.addConstr(
                    self.MU[u_val, t] <= self.MU_max.get(u_val, 0), 
                    name=f"PRO_EQ4_{u_val}_{t}"
                )
            # PRO_EQ7: 加氢单元氢气比例约束
            for u_val in self.u_process2:
                for s_val in self.is_H2_process2.get(u_val, []):
                    self.model.addConstr(self.MS[s_val,t] == self.MU[u_val,t] * (self.H2_fix_pro.get((s_val, u_val), 0)/100),
                                name=f"PRO_EQ7_{u_val}_{s_val}")

            # PRO_EQ8和PRO_EQ9: 加氢单元容量约束(不含氢气)
            for u_val in self.u_process2:
                for s_val in self.is_H2_process2.get(u_val, []):
                    self.model.addConstr(self.MU[u_val,t] - self.MS[s_val,t] >= self.MU_min.get(u_val, 0), 
                                name=f"PRO_EQ8_{u_val}_{s_val}")
                    self.model.addConstr(self.MU[u_val,t] - self.MS[s_val,t] <= self.MU_max.get(u_val, 0), 
                                name=f"PRO_EQ9_{u_val}_{s_val}")

            # PRO_EQ10和PRO_EQ11: 按产出定义的产能约束
            for u_val in self.u_process3:
                for s_val in self.os_process3.get(u_val, []):
                    self.model.addConstr(self.MS[s_val,t] >= self.MU_min.get(u_val, 0), name=f"PRO_EQ10_{u_val}_{s_val}_{t}")
                    self.model.addConstr(self.MS[s_val,t] <= self.MU_max.get(u_val, 0), name=f"PRO_EQ11_{u_val}_{s_val}_{t}")  
                        
            # PRO_EQ12: 不同操作模式下的产出约束
            for u_val in self.u_process4:
                for s_val in self.os_process.get(u_val, []):
                    self.model.addConstr(
                        self.MS[s_val, t] == quicksum((self.Y_pro_mo.get((s_val, u_val, mo_val), 0) * 
                                               self.x[u_val, mo_val, t] / 100) * self.MU[u_val, t] 
                                               for mo_val in self.mo), 
                        name=f"PRO_EQ12_{u_val}_{s_val}_{t}"
                    )
            
            # PRO_EQ13: 每个处理单元只能选择一种操作模式
            for u_val in self.u_process4:
                self.model.addConstr(
                    quicksum(self.x[u_val, mo_val, t] for mo_val in self.mo) == 1, 
                    name=f"PRO_EQ13_{u_val}_{t}"
                )
           
                       
            # # 调试观测装置运行模式为: FC1=A, FC2=A, HCU=B
            # # FC1设置为B模式
            # self.model.addConstr(
            #     self.x['FC1', 'B', t] == 0,
            #     name=f"fixed_mode_B_FC1_{t}"
            # )
            # self.model.addConstr(
            #     self.x['FC1', 'A', t] == 1,
            #     name=f"fixed_mode_not_A_FC1_{t}"
            # )
            
            # # FC2设置为A模式
            # self.model.addConstr(
            #     self.x['FC2', 'A', t] == 1,
            #     name=f"fixed_mode_A_FC2_{t}"
            # )
            # self.model.addConstr(
            #     self.x['FC2', 'B', t] == 0,
            #     name=f"fixed_mode_not_B_FC2_{t}"
            # )
            
            # # HCU设置为B模式
            # self.model.addConstr(
            #     self.x['HCU', 'B', t] == 1,
            #     name=f"fixed_mode_B_HCU_{t}"
            # )
            # self.model.addConstr(
            #     self.x['HCU', 'A', t] == 0,
            #     name=f"fixed_mode_not_A_HCU_{t}"
            # )

            # 混合器约束
            # MIX_EQ1: 混合器输入等于所有进入流股的总和
            for m_val in self.m:
                self.model.addConstr(self.MM[m_val,t] == quicksum(self.MS[s_val,t] for s_val in self.is_mix.get(m_val, [])), 
                            name=f"MIX_EQ1_{m_val}")

            # MIX_EQ2: 混合器产出约束
            for m_val in self.m:
                for s_val in self.os_mix.get(m_val, []):
                    self.model.addConstr(self.MS[s_val,t] == (self.Y_fix_mix.get((s_val, m_val), 0)/100) * self.MM[m_val,t], 
                                name=f"MIX_EQ2_{m_val}_{s_val}")

            # 物料平衡约束
            # MB_EQ1: MX1原料分配
            self.model.addConstr(
                self.MS['MX1', t] == self.MS['MX1_1', t] + self.MS['MX1_2', t], 
                name=f"MB_EQ1_{t}"
            )
            # MB_EQ2和MB_EQ3: 芳烃产品平衡
            self.model.addConstr(self.MS['BNZ',t] == self.MS['BNZ_AEX_OUT',t] - self.MS['BNZ_EBN_IN',t], name="MB_EQ2_{t}")
            self.model.addConstr(self.MS['XYL',t] == self.MS['XYL_AEX_OUT',t] - self.MS['XYL_W92_IN',t], name="MB_EQ3_{t}")

            # MB_EQ4至MB_EQ28: 中间产品流量平衡
            self.model.addConstr(self.MS['KE1',t] == self.MS['KE1_HF4_IN',t] + self.MS['KE1_HF5_IN',t], name="MB_EQ4")
            self.model.addConstr(self.MS['V31',t] == self.MS['V31_GV1_IN',t] + self.MS['V31_PL5_IN',t], name="MB_EQ5")
            self.model.addConstr(self.MS['VR1',t] == self.MS['VR1_SAU_IN',t] + self.MS['VR1_GVR_IN',t], name="MB_EQ6")
            self.model.addConstr(self.MS['VR2',t] == self.MS['VR2_HF6_IN',t] + self.MS['VR2_GVR_IN',t], name="MB_EQ7")
            self.model.addConstr(self.MS['LD2',t] == self.MS['LD2_HF1_IN',t] + self.MS['LD2_HF2_IN',t], name="MB_EQ8")
            self.model.addConstr(self.MS['HD2',t] == self.MS['HD2_HF2_IN',t] + self.MS['HD2_HF4_IN',t], name="MB_EQ9")
            
            # 气体平衡约束
            self.model.addConstr(
                self.MS['GAS_CUT_IN',t] + self.MS['GAS_PL1_IN',t] == self.MS['GAS_GF1_OUT',t] + self.MS['GAS_HF7_OUT',t] + 
                self.MS['GAS_GF2_OUT',t] + self.MS['GAS_PSA_OUT',t] + self.MS['GAS_SSUL_OUT',t], 
                name="MB_EQ10"
            )
            self.model.addConstr(self.MS['WN1_TP1_IN',t] + self.MS['WN1_EEN_IN',t] == self.MS['WN1',t], name="MB_EQ11")
            self.model.addConstr(self.MS['WN2_CUT_IN',t] + self.MS['WN2_TP1_IN',t] == self.MS['WN2',t], name="MB_EQ12")
            self.model.addConstr(self.MS['WN2_CUT_IN',t] >= 1000, name="MB_EQ13")
            self.model.addConstr(self.MS['GAS_CUT_IN',t] >= 200, name="MB_EQ14")

            # 储罐约束
            # TANK_EQ1: 储罐输入等于所有进入流股的总和
            for t_val in self.tank:
                self.model.addConstr(self.MT[t_val,t] == quicksum(self.MS[s_val,t] for s_val in self.is_tank.get(t_val, [])), 
                            name=f"TANK_EQ1_{t_val}")

            # TANK_EQ2: 储罐输出等于储罐容量
            for t_val in self.tank:
                for s_val in self.os_tank.get(t_val, []):
                    self.model.addConstr(self.MS[s_val,t] == self.MT[t_val,t], name=f"TANK_EQ2_{t_val}_{s_val}")

            # TANK_EQ3: 特定储罐约束
            self.model.addConstr(self.MS['GV1_FC2_IN',t] == self.MS['GV1',t] + 2000, name="TANK_EQ3")

            
            # 调和器约束
            # BLEND_EQ1: 调和产品输出等于所有输入组分的总和
            for b_val in self.b:
                for s_val in self.os_blender.get(b_val, []):
                    self.model.addConstr(
                        self.MB[s_val, t] == quicksum(self.MBC[s1_val, s_val, t] for s1_val in self.is_blender.get(b_val, [])),
                        name=f"BLEND_EQ1_{b_val}_{s_val}_{t}"
                    )
            
            # BLEND_EQ2: 调和器输入组分等于其分配到各产品的总和
            for b_val in self.b:
                for s1_val in self.is_blender.get(b_val, []):
                    self.model.addConstr(
                        self.MS[s1_val, t] == quicksum(self.MBC[s1_val, s_val, t] for s_val in self.os_blender.get(b_val, [])),
                        name=f"BLEND_EQ2_{b_val}_{s1_val}_{t}"
                    )
            
            # BLEND_EQ3: 产品产量等于其调和量
            for b_val in self.b:
                for s_val in self.os_blender.get(b_val, []):
                    self.model.addConstr(
                        self.MS[s_val, t] == self.MB[s_val, t], 
                        name=f"BLEND_EQ3_{b_val}_{s_val}_{t}"
                    )
            # 特定调和约束
            self.model.addConstr(self.MBC['ALK', 'W92',t] == 0, name="BLEND_EQ4")
            self.model.addConstr(self.MBC['BHR', 'W95',t] == 0, name="BLEND_EQ5")
            self.model.addConstr(self.MBC['BMT', 'W95',t] == 0, name="BLEND_EQ6")
            self.model.addConstr(self.MBC['TOL', 'W95',t] == 0, name="BLEND_EQ7")
            self.model.addConstr(self.MBC['XYL_W92_IN', 'W95',t] == 0, name="BLEND_EQ8")
            self.model.addConstr(self.MBC['MTB', 'W95',t] == 0, name="BLEND_EQ9")
            self.model.addConstr(self.MBC['ZBB', 'W95',t] == 0, name="BLEND_EQ10")
            self.model.addConstr(self.MBC['RF1', 'W95',t] == 0, name="BLEND_EQ11")
            self.model.addConstr(self.MBC['HLN', 'W95',t] == 0, name="BLEND_EQ12")
            self.model.addConstr(self.MBC['H1N', 'W95',t] == 0, name="BLEND_EQ13")
            self.model.addConstr(self.MBC['H8N', 'W95',t] == 0, name="BLEND_EQ14")
            
            # 质量约束 - 质量线性混合(重量基准)
            for b_val in self.b:
                for s_val in self.os_blender.get(b_val, []):
                    for q_val in self.q:
                        if q_val in self.QLinWt and any(q_val in self.SQB.get(s_val, []) for s_val in [s_val,t] + self.is_blender.get(b_val, [])):
                            self.model.addConstr(
                                quicksum(self.QMBC[s1_val, s_val, q_val,t] for s1_val in self.is_blender.get(b_val, []) 
                                        if q_val in self.SQB.get(s1_val, [])) >= self.QSB_min.get((s_val, q_val), 0) * self.MB[s_val,t],
                                name=f"BLEND_EQ15_{b_val}_{s_val}_{q_val}"
                            )
                            self.model.addConstr(
                                quicksum(self.QMBC[s1_val, s_val, q_val,t] for s1_val in self.is_blender.get(b_val, []) 
                                        if q_val in self.SQB.get(s1_val, [])) <= self.QSB_max.get((s_val, q_val), float('inf')) * self.MB[s_val,t],
                                name=f"BLEND_EQ16_{b_val}_{s_val}_{q_val}"
                            )

            # 质量约束 - 质量线性混合(体积基准)
            for b_val in self.b:
                for s_val in self.os_blender.get(b_val, []):
                    for q_val in self.q:
                        if q_val in self.QLinVol and any(q_val in self.SQB.get(s_val, []) for s_val in [s_val] + self.is_blender.get(b_val, [])):
                            for q1_val in self.SpecificGravity:
                                self.model.addConstr(
                                    quicksum(self.QVBC[s1_val, s_val, q_val,t] for s1_val in self.is_blender.get(b_val, []) 
                                            if q_val in self.SQB.get(s1_val, [])) == self.QS[s_val, q_val,t] * 
                                    quicksum(self.MBC[s1_val, s_val,t]/self.QS_fix.get((s1_val, q1_val), 1) for s1_val in self.is_blender.get(b_val, [])),
                                    name=f"BLEND_EQ18_{b_val}_{s_val}_{q_val}_{q1_val}"
                                )
                            self.model.addConstr(
                                self.QS[s_val, q_val,t] >= self.QSB_min.get((s_val, q_val), 0),
                                name=f"BLEND_EQ19_{b_val}_{s_val}_{q_val}"
                            )
                            self.model.addConstr(
                                self.QS[s_val, q_val,t] <= self.QSB_max.get((s_val, q_val), float('inf')),
                                name=f"BLEND_EQ20_{b_val}_{s_val}_{q_val}"
                            )

            # 质量属性计算约束
            for b_val in self.b:
                for s1_val in self.is_blender.get(b_val, []):
                    for s_val in self.os_blender.get(b_val, []):
                        for q_val in self.q:
                            if q_val in self.SQB.get(s1_val, []) and q_val in self.SQB.get(s_val, []):
                                self.model.addConstr(
                                    self.QMBC[s1_val, s_val, q_val,t] == self.MBC[s1_val, s_val,t] * self.QS_fix.get((s1_val, q_val), 0),
                                    name=f"BLEND_EQ21_{b_val}_{s1_val}_{s_val}_{q_val}"
                                )
                                if q_val in self.QLinVol:
                                    for q1_val in self.SpecificGravity:
                                        if q1_val in self.SQB.get(s1_val, []):
                                            self.model.addConstr(
                                                self.QMBC[s1_val, s_val, q_val,t] == self.QVBC[s1_val, s_val, q_val,t] * self.QS_fix.get((s1_val, q1_val), 1),
                                                name=f"BLEND_EQ22_{b_val}_{s1_val}_{s_val}_{q_val}_{q1_val}"
                                            )
            # 中间产品分流约束
            self.model.addConstr(self.MS['IV2_FC1_IN',t] + self.MS['IV2_HCU_IN',t] + self.MS['IV2_HF6_IN',t] == self.MS['IV2',t], name="MB_EQ15")
            self.model.addConstr(self.MS['H6O_FC1_IN',t] + self.MS['H6O_FC2_IN',t] == self.MS['H6O',t], name="MB_EQ16")
            self.model.addConstr(self.MS['HCO_FC1_IN',t] + self.MS['HCO_FC2_IN',t] == self.MS['HCO',t], name="MB_EQ17")
            self.model.addConstr(self.MS['HCJ_W00_IN',t] + self.MS['HCJ_JET_IN',t] == self.MS['HCJ',t], name="MB_EQ18")
            self.model.addConstr(self.MS['R3S_PL1_IN',t] + self.MS['R3S_EBN_IN',t] == self.MS['R3S',t], name="MB_EQ19")
            self.model.addConstr(self.MS['C1L_GF2_IN',t] + self.MS['C1L_GF1_IN',t] == self.MS['C1L',t], name="MB_EQ20")
            self.model.addConstr(self.MS['C2D_HF2_IN',t] + self.MS['C2D_HCU_IN',t] + self.MS['C2D_HF6_IN',t] == self.MS['C2D',t], name="MB_EQ21")

            # 氢气和氢资源平衡约束
            self.model.addConstr(
                self.MS['H2T_TP1_IN',t] + self.MS['H2T_HCU_IN',t] + self.MS['H2T_HF6_IN',t] + self.MS['H2T_HF7_IN',t] == self.MS['H2T',t], 
                name="MB_EQ22"
            )
            self.model.addConstr(
                self.MS['RFH_HF1_IN',t] + self.MS['RFH_PSA_IN',t] + self.MS['RFH_HF5_IN',t] + self.MS['RFH_HF8_IN',t] + 
                self.MS['RFH_ALK_IN',t] + self.MS['RFH_HF2_IN',t] + self.MS['RFH_HF4_IN',t] == self.MS['RFH',t],
                name="MB_EQ23"
            )

            # 其他物料平衡约束
            self.model.addConstr(self.MS['DAA_CK1_IN',t] + self.MS['DAA',t] == self.MS['DAA_SAU_OUT',t], name="MB_EQ24")
            self.model.addConstr(self.MS['DAA',t] == self.MS['PLS',t], name="MB_EQ25")
            self.model.addConstr(self.MS['PH2',t] == self.MS['PH2_PSA_OUT',t] + self.MS['PH2_HF8_OUT',t], name="MB_EQ26")
            self.model.addConstr(self.MS['NC3_PL2_IN',t] == self.MS['NC3_GF1_OUT',t] + self.MS['NC3_GF2_OUT',t], name="MB_EQ27")
            self.model.addConstr(self.MS['RC3',t] == self.MS['RC3_GF1_OUT',t] + self.MS['RC3_GF2_OUT',t], name="MB_EQ28")
            self.model.addConstr(
                self.MS['MC4_ALK_IN',t] + self.MS['MC4_MTB_IN',t] == self.MS['MC4_GF1_OUT',t] + self.MS['MC4_GF2_OUT',t], 
                name="MB_EQ29"
            )

            # BME购买平衡
            self.model.addConstr(self.MS['BME',t] == self.MS['MC4_MTB_IN',t] * 0.067, name="MB_EQ30")

            # 特定装置的额外约束
            self.model.addConstr(self.MS['RFH_ALK_IN',t] == self.MU['SALK',t] * 1.19/100, name="MB_EQ34")
            self.model.addConstr(self.MS['IV2_FC1_IN',t] == self.MU['FC1',t] * 11.93/100, name="MB_EQ35")
            self.model.addConstr(self.MS['HCO_FC1_IN',t] == self.MU['FC1',t] * 14.76/100, name="MB_EQ36")
            self.model.addConstr(self.MS['C2D_HCU_IN',t] == self.MU['HCU',t] * 0.18/100, name="MB_EQ37")
            self.model.addConstr(self.MS['HD2_HF4_IN',t] == self.MU['HF4',t] * 15/100, name="MB_EQ38")
            self.model.addConstr(self.MS['V31_GV1_IN',t] == 1800, name="MB_EQ39")
    
    def solve(self):
        """求解模型"""
        # 设置模型参数
        self.model.setParam('TimeLimit', 7200)  # 设置最大求解时间为7200秒
        self.model.setParam('MIPGap', 1e-3)     # 设置相对MIP Gap为1%
        self.model.setParam('DualReductions', 0)
        self.model.setParam('NonConvex', 2)  # 启用全局优化
        # 求解模型
        self.model.optimize()
        
        # 返回求解状态
        return self.model.status
    
    def print_operation_results(self, results):
        """打印设备加工模式和关键流量信息"""
        if results['status'] not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print(f"模型未能成功求解，状态：{results['status']}")
            return
        
        print("\n" + "="*50)
        print("设备加工模式和关键流量信息")
        print("="*50)
        
        # 1. 设备加工模式
        print("\n1. 设备加工模式:")
        print("-"*30)
        
        for t in self.periods:
            print(f"\n时期 {t+1}:")
            for u_val in self.u_process4:
                for m_val in self.mo:
                    if results['solution']['unit_mode'].get((u_val, m_val, t), 0) > 0.5:
                        print(f"  {u_val}: 模式 {m_val}")
        
        # 2. 关键流量信息
        print("\n2. 关键流量信息:")
        print("-"*30)
        
        # 主要产品产出量
        print("\n主要产品产出量:")
        for product in ['W92', 'W95', 'JET', 'W00']:
            print(f"\n  {product}:")
            for t in self.periods:
                print(f"    时期 {t+1}: 产量={results['solution']['production'].get((product, t), 0):.2f}, " 
                      f"销量={results['solution']['sales'].get((product, t), 0):.2f}")
        
        # 3. 新增：每个周期的详细利润信息
        print("\n3. 各周期利润明细:")
        print("-"*60)
        print(f"{'周期':<6} {'收入':<12} {'原材料成本':<12} {'加工成本':<12} {'净利润':<12}")
        print("-"*60)
        
        total_revenue = 0
        total_material_cost = 0
        total_processing_cost = 0
        total_profit = 0
        
        for t in self.periods:
            revenue = self.revenue[t].x
            material_cost = self.material_cost[t].x
            processing_cost = self.processing_cost[t].x
            profit = results['solution']['profit'].get(t, 0)
            
            total_revenue += revenue
            total_material_cost += material_cost
            total_processing_cost += processing_cost
            total_profit += profit
            
            print(f"{t+1:<6} {revenue:<12.2f} {material_cost:<12.2f} {processing_cost:<12.2f} {profit:<12.2f}")
        
        print("-"*60)
        print(f"{'总计':<6} {total_revenue:<12.2f} {total_material_cost:<12.2f} {total_processing_cost:<12.2f} {total_profit:<12.2f}")
        print("-"*60)
        
        # 4. 新增：碳排放信息
        print("\n4. 碳排放信息:")
        print("-"*60)
        print(f"{'周期':<6} {'FC1排放量':<12} {'FC2排放量':<12} {'总排放量':<12}")
        print("-"*60)
        
        total_fc1 = 0
        total_fc2 = 0
        total_emission = 0
        
        for t in self.periods:
            fc1_emission = results['solution']['carbon_emission'].get(('FC1', t), 0)
            fc2_emission = results['solution']['carbon_emission'].get(('FC2', t), 0)
            period_emission = results['solution']['period_carbon_emission'].get(t, 0)
            
            total_fc1 += fc1_emission
            total_fc2 += fc2_emission
            total_emission += period_emission
            
            print(f"{t+1:<6} {fc1_emission:<12.2f} {fc2_emission:<12.2f} {period_emission:<12.2f}")
        
        print("-"*60)
        print(f"{'总计':<6} {total_fc1:<12.2f} {total_fc2:<12.2f} {total_emission:<12.2f}")
        print("-"*60)
        
        # 5. 经济与环保的平衡信息
        print("\n5. 经济-环保平衡:")
        print("-"*40)
        print(f"总利润: {results['solution']['total_profit']:.2f}")
        print(f"总碳排放: {results['solution']['total_carbon_emission']:.2f}")
        print(f"碳排放权重: {self.carbon_weight}")
        print(f"加权目标值: {results['solution']['weighted_objective']:.2f}")
        print("-"*40)
    
    def plot_market_fluctuations(self):
        """可视化市场波动数据"""
        plt.figure(figsize=(12, 8))
    
        # 绘制价格波动
        plt.subplot(2, 1, 1)
        for product in self.Des:
            if product in self.price_fluctuation:
                plt.plot(range(self.time_periods), self.price_fluctuation[product], label=f"{product}Price Fluctuation")
        plt.xlabel("T")
        plt.ylabel("Price")
        plt.title("Product Price Fluctuation")
        plt.legend()
        plt.grid(True)
    
        # 绘制需求波动
        plt.subplot(2, 1, 2)
        for product in self.Des:
            if product in self.min_demand_fluctuation and product in self.max_demand_fluctuation:
                plt.fill_between(
                    range(self.time_periods),
                    self.min_demand_fluctuation[product],
                    self.max_demand_fluctuation[product],
                    alpha=0.3,
                    label=f"{product}Demand Fluctuation"
                )
        plt.xlabel("T")
        plt.ylabel("Demand")
        plt.title("Demand Fluctuation")
        plt.legend()
        plt.grid(True)
    
        plt.tight_layout()
        plt.show()
    
    def plot_carbon_emission(self, results):
        """可视化碳排放数据"""
        if 'solution' not in results or 'period_carbon_emission' not in results['solution']:
            print("没有可用的碳排放数据进行可视化")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制每期的碳排放
        periods = list(range(1, self.time_periods + 1))
        fc1_emissions = [results['solution']['carbon_emission'].get(('FC1', t), 0) for t in self.periods]
        fc2_emissions = [results['solution']['carbon_emission'].get(('FC2', t), 0) for t in self.periods]
        
        ax1.bar(periods, fc1_emissions, label='FC1Emissions', alpha=0.7)
        ax1.bar(periods, fc2_emissions, bottom=fc1_emissions, label='FC2Emissions', alpha=0.7)
        ax1.set_xlabel('T')
        ax1.set_ylabel('Carbon Emission(tcoal)')
        ax1.set_title('Each Period Carbon Emission')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 绘制利润与碳排放的关系
        profits = [results['solution']['profit'].get(t, 0) for t in self.periods]
        carbon_emissions = [results['solution']['period_carbon_emission'].get(t, 0) for t in self.periods]
        
        ax2.scatter(carbon_emissions, profits, s=80, c=periods, cmap='viridis', edgecolor='k')
        for i, period in enumerate(periods):
            ax2.annotate(f"T{period}", (carbon_emissions[i], profits[i]), 
                         xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Carbon Emission(tcoal)')
        ax2.set_ylabel('Profit')
        ax2.set_title('Profit-Carbon Emission Relationship')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.show()
    
    def get_results(self):
        """获取并返回求解结果"""
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            results = {
                'status': self.model.status,
                'objective_value': self.model.objVal,
                'runtime': self.model.Runtime,
                'gap': self.model.MIPGap,
                'solution': {
                    'inventory': {(p, t): self.inventory[p, t].x for p in self.Des for t in range(self.time_periods+1)},
                    'sales': {(p, t): self.sales[p, t].x for p in self.Des for t in self.periods},
                    'production': {(s, t): self.MS[s, t].x for s in self.s for t in self.periods},
                    'unit_throughput': {(u, t): self.MU[u, t].x for u in self.u for t in self.periods},
                    'unit_mode': {(u, m, t): self.x[u, m, t].x for u in self.u_process4 for m in self.mo for t in self.periods},
                    'profit': {t: self.period_profit[t].x for t in self.periods},
                    'revenue': {t: self.revenue[t].x for t in self.periods},            # 添加收入数据
                    'material_cost': {t: self.material_cost[t].x for t in self.periods}, # 添加原材料成本数据
                    'processing_cost': {t: self.processing_cost[t].x for t in self.periods}, # 添加加工成本数据
                    'total_profit': self.total_profit.x,
                    'carbon_emission': {(u, t): self.carbon_emission[u, t].x for u in self.carbon_units for t in self.periods},
                    'period_carbon_emission': {t: self.period_carbon_emission[t].x for t in self.periods},
                    'total_carbon_emission': self.total_carbon_emission.x,
                    'weighted_objective': self.weighted_objective.x
                },
                'market_data': {
                    'demand': {
                        'min': {p: self.min_demand_fluctuation.get(p, [0]*self.time_periods) for p in self.Des},
                        'max': {p: self.max_demand_fluctuation.get(p, [0]*self.time_periods) for p in self.Des}
                    },
                    'price': {p: self.price_fluctuation.get(p, [0]*self.time_periods) for p in self.Des},
                    'crude_cost': self.MX1_cost_fluctuation
                }
            }
            return results
        else:
            return {'status': self.model.status, 'message': 'Model not optimally solved'}

if __name__ == "__main__":
    # 设置随机种子
    custom_seed = 42  # 可以修改为任何整数
    random.seed(custom_seed)
    np.random.seed(custom_seed)
    
    # 创建模型实例并设置碳排放权重和随机种子
    model = RefineryInventoryModel(time_periods=20, carbon_weight=0.05, random_seed=custom_seed)
    
    # 求解模型
    status = model.solve()
    
    # 获取结果
    results = model.get_results()
    model.print_operation_results(results)
    
    # 可视化结果
    if status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # 打印关键指标
        print(f"加权目标值: {results['solution']['weighted_objective']:.2f}")
        print(f"总利润: {results['solution']['total_profit']:.2f}")
        print(f"总碳排放: {results['solution']['total_carbon_emission']:.2f}")
        print(f"求解时间: {results['runtime']:.2f}秒")
        print(f"MIP Gap: {results['gap']*100:.2f}%")
        
        # 可视化市场波动和碳排放
        model.plot_market_fluctuations()
        model.plot_carbon_emission(results)
        
    else:
        print(f"模型未能成功求解，状态：{status}")
        if status == GRB.INFEASIBLE:
            model.model.computeIIS()
            model.model.write("refinery_inventory_infeasible.ilp")
            print("不可行分析已保存到 refinery_inventory_infeasible.ilp 文件")
