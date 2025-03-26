# 导入必要的库和模块
import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import mplfinance as mpf  # 用于绘制金融图表
from perceptually_important import find_pips  # 导入感知重要点(PIP)识别函数
from rolling_window import rw_top, rw_bottom  # 导入滚动窗口极值识别函数
from trendline_automation import fit_trendlines_single  # 导入趋势线拟合函数
from dataclasses import dataclass  # 用于创建数据类


@dataclass
class FlagPattern:
    """
    这是一个使用Python的dataclass装饰器定义的类，用于表示旗形和三角旗形态。
    @dataclass是Python的一个装饰器，它会自动为类生成特殊方法，如__init__、__repr__等，
    简化了数据类的创建过程，使代码更加简洁。
    """

    # 旗形和三角旗形态的数据结构
    
    # 属性:
    base_x: int         # 趋势起点索引，旗杆的底部
    base_y: float       # 趋势起点价格
    
    tip_x: int   = -1       # 旗杆顶部/底部索引，旗帜开始点
    tip_y: float = -1.      # 旗杆顶部/底部价格
    
    conf_x: int   = -1      # 形态确认点索引（突破点）
    conf_y: float = -1.     # 形态确认点价格
    
    pennant: bool = False   # True表示三角旗，False表示旗形
    
    flag_width: int    = -1    # 旗帜宽度（时间跨度）
    flag_height: float = -1.   # 旗帜高度（价格跨度）
    
    pole_width: int    = -1    # 旗杆宽度（时间跨度）
    pole_height: float = -1.   # 旗杆高度（价格跨度）
    
    # 旗帜的上下趋势线，截距在旗杆顶部/底部
    support_intercept: float = -1.  # 支撑线截距
    support_slope: float = -1.      # 支撑线斜率
    resist_intercept: float = -1.   # 阻力线截距
    resist_slope: float = -1.       # 阻力线斜率


def check_bear_pattern_pips(pending: FlagPattern, data: np.array, i:int, order:int):
    """
    检查熊市旗形/三角旗形态（基于PIP点方法）
    
    参数:
    pending: FlagPattern - 待填充的旗形对象
    data: np.array - 价格数据数组
    i: int - 当前检查的索引位置
    order: int - 滚动窗口大小参数
    
    返回:
    bool - 如果识别到有效形态则返回True，否则返回False
    """
    
    # 找出自局部顶部以来的最低价格（旗杆底部）
    # 举例说明 data[pending.base_x: i + 1] 的含义:
    # 假设 pending.base_x = 10, i = 15
    # 那么 data[10:16] 表示从索引10到15的数据切片
    # 比如对于价格数组 data = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    # data[10:16] 就会得到 [110, 111, 112, 113, 114, 115]
    # 这个切片包含了从旗杆底部(base_x)到当前检查位置(i)的所有价格数据
    # pending 是一个待处理的形态对象
    # 在这里，pending 表示一个正在检查和构建的潜在熊市旗形或三角旗形态
    # 它包含了形态的各种属性，如起点、终点、趋势线等信息
    # 当完整的形态被确认后，pending 对象会被添加到相应的结果列表中(bear_flags 或 bear_pennants)
    data_slice = data[pending.base_x: i + 1]  # i + 1包含当前价格
    min_i = data_slice.argmin() + pending.base_x  # 自局部顶部以来的最低点索引
    
    # 确保从最低点到当前位置有足够的距离来形成旗帜
    if i - min_i < max(5, order * 0.5):  # 距离最低点足够远，可以绘制潜在的旗形/三角旗
        return False
    
    # 测试旗帜宽度/高度
    pole_width = min_i - pending.base_x  # 旗杆宽度
    flag_width = i - min_i  # 旗帜宽度
    # 旗帜宽度应小于旗杆宽度的一半
    if flag_width > pole_width * 0.5:
        return False

    pole_height = pending.base_y - data[min_i]  # 旗杆高度
    flag_height = data[min_i:i+1].max() - data[min_i]  # 旗帜高度
    # 旗帜高度应小于旗杆高度的一半
    if flag_height > pole_height * 0.5:
        return False

    # 到这里，宽度/高度检查通过
    
    # 找出旗帜部分的感知重要点(PIP)
    # 找出从最低点到当前索引之间的5个PIP点
    pips_x, pips_y = find_pips(data[min_i:i+1], 5, 3)

    # 检查中心PIP点是否低于相邻的两个点，形成/\/\形状
    if not (pips_y[2] < pips_y[1] and pips_y[2] < pips_y[3]):
        return False
    
    # 计算旗帜的支撑线和阻力线
    # 支撑线：连接第1个和第3个PIP点
    support_rise = pips_y[2] - pips_y[0]  # 支撑线上升高度
    support_run = pips_x[2] - pips_x[0]  # 支撑线水平距离
    support_slope = support_rise / support_run  # 支撑线斜率
    support_intercept = pips_y[0]  # 支撑线截距
    
    # 阻力线：连接第2个和第4个PIP点
    resist_rise = pips_y[3] - pips_y[1]  # 阻力线上升高度
    resist_run = pips_x[3] - pips_x[1]  # 阻力线水平距离
    resist_slope = resist_rise / resist_run  # 阻力线斜率
    resist_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * resist_slope  # 阻力线截距

    # 计算两条线的交点
    if resist_slope != support_slope:  # 非平行线
        intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
    else:
        intersection = -flag_width * 100  # 平行线，设置一个远离旗帜区域的交点

    # 如果交点在旗帜区域内，则不是有效的旗形/三角旗
    if intersection <= pips_x[4] and intersection >= 0:
        return False

    # 检查当前点是否突破旗帜下边界（支撑线），确认形态
    support_endpoint = pips_y[0] + support_slope * pips_x[4]
    if pips_y[4] > support_endpoint:  # 如果价格高于支撑线，则未突破
        return False
    
    # 判断是旗形还是三角旗
    # 如果阻力线向下倾斜（斜率为负），则为三角旗
    if resist_slope < 0:
        pending.pennant = True
    else:
        pending.pennant = False
    
    # 过滤严重发散的线（交点太近）
    if intersection < 0 and intersection > -flag_width:
        return False

    # 形态确认，填充旗形对象的属性
    pending.tip_x = min_i  # 旗杆底部索引
    pending.tip_y = data[min_i]  # 旗杆底部价格
    pending.conf_x = i  # 确认点索引
    pending.conf_y = data[i]  # 确认点价格
    pending.flag_width = flag_width  # 旗帜宽度
    pending.flag_height = flag_height  # 旗帜高度
    pending.pole_width = pole_width  # 旗杆宽度
    pending.pole_height = pole_height  # 旗杆高度
    pending.support_slope = support_slope  # 支撑线斜率
    pending.support_intercept = support_intercept  # 支撑线截距
    pending.resist_slope = resist_slope  # 阻力线斜率
    pending.resist_intercept = resist_intercept  # 阻力线截距
    
    return True  # 返回True表示识别到有效形态
    

def check_bull_pattern_pips(pending: FlagPattern, data: np.array, i:int, order:int):
    """
    检查牛市旗形/三角旗形态（基于PIP点方法）
    
    参数:
    pending: FlagPattern - 待填充的旗形对象
    data: np.array - 价格数据数组
    i: int - 当前检查的索引位置
    order: int - 滚动窗口大小参数
    
    返回:
    bool - 如果识别到有效形态则返回True，否则返回False
    """
    
    # 找出自局部底部以来的最高价格（旗杆顶部）
    data_slice = data[pending.base_x: i + 1]  # i + 1包含当前价格
    max_i = data_slice.argmax() + pending.base_x  # 自局部底部以来的最高点索引
    pole_width = max_i - pending.base_x  # 旗杆宽度
    
    # 确保从最高点到当前位置有足够的距离来形成旗帜
    if i - max_i < max(5, order * 0.5):
        return False

    # 测试旗帜宽度/高度
    flag_width = i - max_i  # 旗帜宽度
    # 旗帜宽度应小于旗杆宽度的一半
    if flag_width > pole_width * 0.5:
        return False

    pole_height = data[max_i] - pending.base_y  # 旗杆高度
    flag_height = data[max_i] - data[max_i:i+1].min()  # 旗帜高度
    # 旗帜高度应小于旗杆高度的一半
    if flag_height > pole_height * 0.5:
        return False

    # 找出旗帜部分的感知重要点(PIP)
    pips_x, pips_y = find_pips(data[max_i:i+1], 5, 3)  # 找出从最高点到当前索引之间的5个PIP点

    # 检查中心PIP点是否高于相邻的两个点，形成\/\/形状
    if not (pips_y[2] > pips_y[1] and pips_y[2] > pips_y[3]):
        return False
        
    # 计算旗帜的阻力线和支撑线
    # 阻力线：连接第1个和第3个PIP点
    resist_rise = pips_y[2] - pips_y[0]  # 阻力线上升高度
    resist_run = pips_x[2] - pips_x[0]  # 阻力线水平距离
    resist_slope = resist_rise / resist_run  # 阻力线斜率
    resist_intercept = pips_y[0]  # 阻力线截距

    # 支撑线：连接第2个和第4个PIP点
    support_rise = pips_y[3] - pips_y[1]  # 支撑线上升高度
    support_run = pips_x[3] - pips_x[1]  # 支撑线水平距离
    support_slope = support_rise / support_run  # 支撑线斜率
    support_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * support_slope  # 支撑线截距

    # 计算两条线的交点
    if resist_slope != support_slope:  # 非平行线
        intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
    else:
        intersection = -flag_width * 100  # 平行线，设置一个远离旗帜区域的交点

    # 如果交点在旗帜区域内，则不是有效的旗形/三角旗
    if intersection <= pips_x[4] and intersection >= 0:
        return False
    
    # 过滤严重发散的线（交点太近）
    if intersection < 0 and intersection > -1.0 * flag_width:
        return False

    # 检查当前点是否突破旗帜上边界（阻力线），确认形态
    resist_endpoint = pips_y[0] + resist_slope * pips_x[4]
    if pips_y[4] < resist_endpoint:  # 如果价格低于阻力线，则未突破
        return False

    # 判断是旗形还是三角旗
    # 如果支撑线向上倾斜（斜率为正），则为三角旗
    if support_slope > 0:
        pending.pennant = True
    else:
        pending.pennant = False

    # 形态确认，填充旗形对象的属性
    pending.tip_x = max_i  # 旗杆顶部索引
    pending.tip_y = data[max_i]  # 旗杆顶部价格
    pending.conf_x = i  # 确认点索引
    pending.conf_y = data[i]  # 确认点价格
    pending.flag_width = flag_width  # 旗帜宽度
    pending.flag_height = flag_height  # 旗帜高度
    pending.pole_width = pole_width  # 旗杆宽度
    pending.pole_height = pole_height  # 旗杆高度
    
    pending.support_slope = support_slope  # 支撑线斜率
    pending.support_intercept = support_intercept  # 支撑线截距

    pending.resist_slope = resist_slope  # 阻力线斜率
    pending.resist_intercept = resist_intercept  # 阻力线截距
    
    return True  # 返回True表示识别到有效形态


def find_flags_pennants_pips(data: np.array, order:int):
    """
    基于PIP点方法识别旗形和三角旗形态
    
    参数:
    data: np.array - 价格数据数组
    order: int - 滚动窗口大小参数，用于识别局部极值
    
    返回:
    bull_flags: list - 牛市旗形列表
    bear_flags: list - 熊市旗形列表
    bull_pennants: list - 牛市三角旗列表
    bear_pennants: list - 熊市三角旗列表
    """
    assert(order >= 3)  # 确保窗口大小参数至少为3
    pending_bull = None  # 待处理的牛市形态
    pending_bear = None  # 待处理的熊市形态

    # 初始化结果列表
    bull_pennants = []  # 牛市三角旗列表
    bear_pennants = []  # 熊市三角旗列表
    bull_flags = []     # 牛市旗形列表
    bear_flags = []     # 熊市旗形列表
    
    # 遍历价格数据
    for i in range(len(data)):

        # 识别局部极值点作为形态起点
        if rw_top(data, i, order):  # 如果是局部高点
            # 创建新的熊市形态对象，以当前高点为起点
            pending_bear = FlagPattern(i - order, data[i - order])
        
        if rw_bottom(data, i, order):  # 如果是局部低点
            # 创建新的牛市形态对象，以当前低点为起点
            pending_bull = FlagPattern(i - order, data[i - order])

        # 检查并处理待处理的熊市形态
        if pending_bear is not None:
            # 检查是否形成熊市旗形/三角旗
            if check_bear_pattern_pips(pending_bear, data, i, order):
                # 根据形态类型添加到相应列表
                if pending_bear.pennant:
                    bear_pennants.append(pending_bear)  # 添加熊市三角旗
                else:
                    bear_flags.append(pending_bear)     # 添加熊市旗形
                pending_bear = None  # 重置待处理形态

        # 检查并处理待处理的牛市形态
        if pending_bull is not None:
            # 检查是否形成牛市旗形/三角旗
            if check_bull_pattern_pips(pending_bull, data, i, order):
                # 根据形态类型添加到相应列表
                if pending_bull.pennant:
                    bull_pennants.append(pending_bull)  # 添加牛市三角旗
                else:
                    bull_flags.append(pending_bull)     # 添加牛市旗形
                pending_bull = None  # 重置待处理形态

    # 返回识别结果
    return bull_flags, bear_flags, bull_pennants, bear_pennants

def check_bull_pattern_trendline(pending: FlagPattern, data: np.array, i:int, order:int):
    """
    检查牛市旗形/三角旗形态（基于趋势线方法）
    
    参数:
    pending: FlagPattern - 待填充的旗形对象
    data: np.array - 价格数据数组
    i: int - 当前检查的索引位置
    order: int - 滚动窗口大小参数
    
    返回:
    bool - 如果识别到有效形态则返回True，否则返回False
    """
    
    # 检查旗杆顶部之后的价格是否超过旗杆顶部价格
    if data[pending.tip_x + 1 : i].max() > pending.tip_y:
        return False

    # 找出旗帜部分的最低价格
    flag_min = data[pending.tip_x:i].min()

    # 计算旗杆和旗帜的高度和宽度
    pole_height = pending.tip_y - pending.base_y  # 旗杆高度
    pole_width = pending.tip_x - pending.base_x   # 旗杆宽度
    
    flag_height = pending.tip_y - flag_min  # 旗帜高度
    flag_width = i - pending.tip_x          # 旗帜宽度

    # 旗帜宽度应小于旗杆宽度的一半
    if flag_width > pole_width * 0.5:
        return False

    # 旗帜高度应小于旗杆高度的75%
    if flag_height > pole_height * 0.75:
        return False

    # 使用趋势线拟合算法找出旗帜部分的支撑线和阻力线
    support_coefs, resist_coefs = fit_trendlines_single(data[pending.tip_x:i])
    support_slope, support_intercept = support_coefs[0], support_coefs[1]  # 支撑线系数
    resist_slope, resist_intercept = resist_coefs[0], resist_coefs[1]      # 阻力线系数

    # 检查当前价格是否突破上趋势线（阻力线），确认形态
    current_resist = resist_intercept + resist_slope * (flag_width + 1)
    if data[i] <= current_resist:  # 如果价格未突破阻力线
        return False

    # 判断是旗形还是三角旗
    # 如果支撑线向上倾斜（斜率为正），则为三角旗
    if support_slope > 0:
        pending.pennant = True
    else:
        pending.pennant = False

    # 形态确认，填充旗形对象的属性
    pending.conf_x = i  # 确认点索引
    pending.conf_y = data[i]  # 确认点价格
    pending.flag_width = flag_width  # 旗帜宽度
    pending.flag_height = flag_height  # 旗帜高度
    pending.pole_width = pole_width  # 旗杆宽度
    pending.pole_height = pole_height  # 旗杆高度
    
    pending.support_slope = support_slope  # 支撑线斜率
    pending.support_intercept = support_intercept  # 支撑线截距
    pending.resist_slope = resist_slope  # 阻力线斜率
    pending.resist_intercept = resist_intercept  # 阻力线截距

    return True  # 返回True表示识别到有效形态

def check_bear_pattern_trendline(pending: FlagPattern, data: np.array, i:int, order:int):
    """
    检查熊市旗形/三角旗形态（基于趋势线方法）
    
    参数:
    pending: FlagPattern - 待填充的旗形对象
    data: np.array - 价格数据数组
    i: int - 当前检查的索引位置
    order: int - 滚动窗口大小参数
    
    返回:
    bool - 如果识别到有效形态则返回True，否则返回False
    """
    
    # 检查旗杆底部之后的价格是否低于旗杆底部价格
    if data[pending.tip_x + 1 : i].min() < pending.tip_y:
        return False

    # 找出旗帜部分的最高价格
    flag_max = data[pending.tip_x:i].max()

    # 计算旗杆和旗帜的高度和宽度
    pole_height = pending.base_y - pending.tip_y  # 旗杆高度
    pole_width = pending.tip_x - pending.base_x   # 旗杆宽度
    
    flag_height = flag_max - pending.tip_y  # 旗帜高度
    flag_width = i - pending.tip_x          # 旗帜宽度

    # 旗帜宽度应小于旗杆宽度的一半
    if flag_width > pole_width * 0.5:
        return False

    # 旗帜高度应小于旗杆高度的75%
    if flag_height > pole_height * 0.75:
        return False

    # 使用趋势线拟合算法找出旗帜部分的支撑线和阻力线
    support_coefs, resist_coefs = fit_trendlines_single(data[pending.tip_x:i])
    support_slope, support_intercept = support_coefs[0], support_coefs[1]  # 支撑线系数
    resist_slope, resist_intercept = resist_coefs[0], resist_coefs[1]      # 阻力线系数

    # 检查当前价格是否突破下趋势线（支撑线），确认形态
    current_support = support_intercept + support_slope * (flag_width + 1)
    if data[i] >= current_support:  # 如果价格未突破支撑线
        return False

    # 判断是旗形还是三角旗
    # 如果阻力线向下倾斜（斜率为负），则为三角旗
    if resist_slope < 0:
        pending.pennant = True
    else:
        pending.pennant = False

    # 形态确认，填充旗形对象的属性
    pending.conf_x = i  # 确认点索引
    pending.conf_y = data[i]  # 确认点价格
    pending.flag_width = flag_width  # 旗帜宽度
    pending.flag_height = flag_height  # 旗帜高度
    pending.pole_width = pole_width  # 旗杆宽度
    pending.pole_height = pole_height  # 旗杆高度
    
    pending.support_slope = support_slope  # 支撑线斜率
    pending.support_intercept = support_intercept  # 支撑线截距
    pending.resist_slope = resist_slope  # 阻力线斜率
    pending.resist_intercept = resist_intercept  # 阻力线截距

    return True  # 返回True表示识别到有效形态

def find_flags_pennants_trendline(data: np.array, order:int):
    """
    基于趋势线方法识别旗形和三角旗形态
    
    参数:
    data: np.array - 价格数据数组
    order: int - 滚动窗口大小参数，用于识别局部极值
    
    返回:
    bull_flags: list - 牛市旗形列表
    bear_flags: list - 熊市旗形列表
    bull_pennants: list - 牛市三角旗列表
    bear_pennants: list - 熊市三角旗列表
    """
    last_bottom = -1  # 最近的局部底部索引
    last_top = -1     # 最近的局部顶部索引
    pending_bull = None  # 待处理的牛市形态
    pending_bear = None  # 待处理的熊市形态

    # 初始化结果列表
    bull_pennants = []  # 牛市三角旗列表
    bear_pennants = []  # 熊市三角旗列表
    bull_flags = []     # 牛市旗形列表
    bear_flags = []     # 熊市旗形列表
    
    # 遍历价格数据
    for i in range(len(data)):

        # 识别局部极值点
        if rw_top(data, i, order):  # 如果是局部高点
            last_top = i - order  # 更新最近的局部顶部索引
            if last_bottom != -1:  # 如果已有局部底部
                # 创建新的牛市形态对象，从底部到顶部
                pending = FlagPattern(last_bottom, data[last_bottom])
                pending.tip_x = last_top  # 设置旗杆顶部
                pending.tip_y = data[last_top]
                pending_bull = pending
        
        if rw_bottom(data, i, order):  # 如果是局部低点
            last_bottom = i - order  # 更新最近的局部底部索引
            if last_top != -1:  # 如果已有局部顶部
                # 创建新的熊市形态对象，从顶部到底部
                pending = FlagPattern(last_top, data[last_top])
                pending.tip_x = last_bottom  # 设置旗杆底部
                pending.tip_y = data[last_bottom]
                pending_bear = pending

        # 检查并处理待处理的熊市形态
        if pending_bear is not None:
            # 检查是否形成熊市旗形/三角旗
            if check_bear_pattern_trendline(pending_bear, data, i, order):
                # 根据形态类型添加到相应列表
                if pending_bear.pennant:
                    bear_pennants.append(pending_bear)  # 添加熊市三角旗
                else:
                    bear_flags.append(pending_bear)     # 添加熊市旗形
                pending_bear = None  # 重置待处理形态
        
        # 检查并处理待处理的牛市形态
        if pending_bull is not None:
            # 检查是否形成牛市旗形/三角旗
            if check_bull_pattern_trendline(pending_bull, data, i, order):
                # 根据形态类型添加到相应列表
                if pending_bull.pennant:
                    bull_pennants.append(pending_bull)  # 添加牛市三角旗
                else:
                    bull_flags.append(pending_bull)     # 添加牛市旗形
                pending_bull = None  # 重置待处理形态

    # 返回识别结果
    return bull_flags, bear_flags, bull_pennants, bear_pennants

def plot_flag(candle_data: pd.DataFrame, pattern: FlagPattern, pad=2):
    """
    绘制旗形/三角旗形态
    
    参数:
    candle_data: pd.DataFrame - K线数据
    pattern: FlagPattern - 旗形/三角旗对象
    pad: int - 图表两侧的额外空间
    """
    if pad < 0:
        pad = 0

    # 截取需要显示的数据范围
    start_i = pattern.base_x - pad
    end_i = pattern.conf_x + 1 + pad
    dat = candle_data.iloc[start_i:end_i]
    idx = dat.index
    
    # 设置绘图风格
    plt.style.use('dark_background')
    fig = plt.gcf()
    ax = fig.gca()

    # 获取关键点的索引
    tip_idx = idx[pattern.tip_x - start_i]  # 旗杆顶部/底部索引
    conf_idx = idx[pattern.conf_x - start_i]  # 确认点索引

    # 定义要绘制的线
    pole_line = [(idx[pattern.base_x - start_i], pattern.base_y), (tip_idx, pattern.tip_y)]  # 旗杆线
    upper_line = [(tip_idx, pattern.resist_intercept), (conf_idx, pattern.resist_intercept + pattern.resist_slope * pattern.flag_width)]  # 上趋势线
    lower_line = [(tip_idx, pattern.support_intercept), (conf_idx, pattern.support_intercept + pattern.support_slope * pattern.flag_width)]  # 下趋势线

    # 绘制K线图和趋势线
    mpf.plot(dat, alines=dict(alines=[pole_line, upper_line, lower_line], colors=['w', 'b', 'b']), type='candle', style='charles', ax=ax)
    plt.show()

# 主程序（当直接运行此文件时执行）
if __name__ == '__main__':
    # 加载数据
    data = pd.read_excel('000001.xlsx')
    data['Date'] = data['Date'].astype('datetime64[s]')
    data = data.set_index('Date')


    # 对除Change列外的所有列取对数
    data.loc[:, data.columns != 'Change'] = np.log(data.loc[:, data.columns != 'Change']).copy()

    # 提取收盘价数据
    dat_slice = data['Close'].to_numpy()
    
    # 识别旗形和三角旗
    bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_pips(dat_slice, 12)  # 使用PIP点方法
    #bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(dat_slice, 10)  # 使用趋势线方法

    # 创建数据框来存储形态属性
    bull_flag_df = pd.DataFrame()
    bull_pennant_df = pd.DataFrame()
    bear_flag_df = pd.DataFrame()
    bear_pennant_df = pd.DataFrame()

    # 将形态数据组织到数据框中
    hold_mult = 1.0  # 持有期乘数（持有时间 = 旗帜宽度 * 乘数）
    
    # 处理牛市旗形
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    # 这里 i 是索引(从0开始)，flag 是 bull_flags 中的每个元素
    for i, flag in enumerate(bull_flags):
        # 记录形态属性
        bull_flag_df.loc[i, 'flag_width'] = flag.flag_width
        bull_flag_df.loc[i, 'flag_height'] = flag.flag_height
        bull_flag_df.loc[i, 'pole_width'] = flag.pole_width
        bull_flag_df.loc[i, 'pole_height'] = flag.pole_height
        bull_flag_df.loc[i, 'slope'] = flag.resist_slope

        # 计算持有期收益
        hp = int(flag.flag_width * hold_mult)
        if flag.conf_x + hp >= len(data):
            bull_flag_df.loc[i, 'return'] = np.nan
        else:
            ret = dat_slice[flag.conf_x + hp] - dat_slice[flag.conf_x]
            bull_flag_df.loc[i, 'return'] = ret


                # 处理熊市旗形
    for i, flag in enumerate(bear_flags):
        # 记录形态属性
        bear_flag_df.loc[i, 'flag_width'] = flag.flag_width
        bear_flag_df.loc[i, 'flag_height'] = flag.flag_height
        bear_flag_df.loc[i, 'pole_width'] = flag.pole_width
        bear_flag_df.loc[i, 'pole_height'] = flag.pole_height
        bear_flag_df.loc[i, 'slope'] = flag.support_slope

        # 计算持有期收益（注意熊市形态是做空，所以收益取负）
        hp = int(flag.flag_width * hold_mult)
        if flag.conf_x + hp >= len(data):
            bear_flag_df.loc[i, 'return'] = np.nan
        else:
            ret = -1 * (dat_slice[flag.conf_x + hp] - dat_slice[flag.conf_x])
            bear_flag_df.loc[i, 'return'] = ret 

    # 处理牛市三角旗
    for i, pennant in enumerate(bull_pennants):
        # 记录形态属性
        bull_pennant_df.loc[i, 'pennant_width'] = pennant.flag_width
        bull_pennant_df.loc[i, 'pennant_height'] = pennant.flag_height
        bull_pennant_df.loc[i, 'pole_width'] = pennant.pole_width
        bull_pennant_df.loc[i, 'pole_height'] = pennant.pole_height

        # 计算持有期收益
        hp = int(pennant.flag_width * hold_mult)gi
        if pennant.conf_x + hp >= len(data):
            bull_pennant_df.loc[i, 'return'] = np.nan
        else:
            ret = dat_slice[pennant.conf_x + hp] - dat_slice[pennant.conf_x]
            bull_pennant_df.loc[i, 'return'] = ret 

    # 处理熊市三角旗
    for i, pennant in enumerate(bear_pennants):
        # 记录形态属性
        bear_pennant_df.loc[i, 'pennant_width'] = pennant.flag_width
        bear_pennant_df.loc[i, 'pennant_height'] = pennant.flag_height
        bear_pennant_df.loc[i, 'pole_width'] = pennant.pole_width
        bear_pennant_df.loc[i, 'pole_height'] = pennant.pole_height

        # 计算持有期收益（注意熊市形态是做空，所以收益取负）
        hp = int(pennant.flag_width * hold_mult)
        if pennant.conf_x + hp >= len(data):
            bear_pennant_df.loc[i, 'return'] = np.nan
        else:
            ret = -1 * (dat_slice[pennant.conf_x + hp] - dat_slice[pennant.conf_x])
            bear_pennant_df.loc[i, 'return'] = ret 












