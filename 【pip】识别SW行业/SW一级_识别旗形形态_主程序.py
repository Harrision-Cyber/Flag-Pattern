import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt  # 用于数据可视化
import mplfinance as mpf  # 用于绘制金融图表
from matplotlib import style
from dataclasses import dataclass
from important_point_algorithm import rw_top,rw_bottom,rw_extremes,directional_change,get_extremes,find_pips


'''====================读取数据======================================='''
# 读取Excel文件中的收盘价和成交量数据
def load_stock_data(file_path):
    """
    读取Excel文件中的收盘价和成交量数据，创建字典结构
    
    参数:
    file_path: Excel文件路径
    
    返回:
    dict: 包含收盘价和成交量数据的字典

    种数据结构既保留了原始DataFrame便于按日期查询，又创建了按股票代码组织的字典便于单只股票的历史数据分析。您可以:
    通过 data['stock_data']['000001.SZ'] 获取单只股票的所有数据
    通过 data['close_df'].loc['2023-01-05'] 获取特定日期所有股票的收盘价
    通过上面提供的辅助函数快速获取所需数据
    后续识别旗形形态时，可以遍历每只股票的历史数据进行分析。

    """
    print(f"正在读取 {file_path} 文件...")
    
    # 读取收盘价sheet
    close_df = pd.read_excel(file_path, sheet_name='收盘价', index_col=0)
    # 读取成交量sheet
    volume_df = pd.read_excel(file_path, sheet_name='成交量', index_col=0)

    high_df = pd.read_excel(file_path, sheet_name='最高价', index_col=0)

    low_df = pd.read_excel(file_path, sheet_name='最低价', index_col=0)



    
    print(f"数据读取完成，收盘价数据形状: {close_df.shape}, 成交量数据形状: {volume_df.shape}")
    
    # 获取所有股票代码
    stock_codes = close_df.columns.tolist()
    
    # 创建字典结构
    stock_data = {}
    
    # 计算对数收盘价
    ln_close_df = np.log(close_df)
    ln_high_df = np.log(high_df)
    ln_low_df = np.log(low_df)

    
    for code in stock_codes:
        # 提取当前股票的收盘价和成交量数据
        close_series = close_df[code].dropna()
        ln_close_series = ln_close_df[code].dropna()
        volume_series = volume_df[code].dropna()
        high_series = high_df[code].dropna()
        ln_high_series = ln_high_df[code].dropna()
        ln_low_series = ln_low_df[code].dropna()    
        low_series = low_df[code].dropna()


        # 确保所有序列有相同的索引
        common_dates = close_series.index.intersection(volume_series.index).intersection(ln_close_series.index)
        common_dates = common_dates.intersection(high_series.index).intersection(low_series.index)
        
        # 创建当前股票的数据字典
        stock_data[code] = {
            'dates': common_dates,
            'close': close_series[common_dates].values,
            'ln_close': ln_close_series[common_dates].values,
            'volume': volume_series[common_dates].values,
            'high': high_series[common_dates].values,
            'ln_high': ln_high_series[common_dates].values,
            'low': low_series[common_dates].values,
            'ln_low': ln_low_series[common_dates].values
        }
    
    print(f"成功创建 {len(stock_data)} 只股票的数据字典")
    
    return {
        'stock_data': stock_data,          # 股票数据字典
        'close_df': close_df,              # 原始收盘价DataFrame
        'ln_close_df': ln_close_df,        # 对数收盘价DataFrame
        'high_df': high_df,                # 原始最高价DataFrame
        'ln_high_df': ln_high_df, 
        'low_df': low_df,                  # 原始最低价DataFrame
        'ln_low_df': ln_low_df, 
        'volume_df': volume_df,            # 原始成交量DataFrame
        'dates': close_df.index.tolist(),  # 所有日期列表
        'stocks': stock_codes              # 所有股票代码列表
    }


'''====================计算平均真实波幅(ATR)==========================='''
def calculate_atr(high_data, low_data, close_data, index, period):
    """
    计算特定时间点的平均真实波幅(ATR)
    
    参数:
    high_data: 高价数据数组
    low_data: 低价数据数组
    close_data: 收盘价数据数组
    index: 当前检查的索引位置（ATR 计算的终点）
    period: ATR计算周期，默认14，但将由旗帜宽度动态设置
    
    返回:
    atr_value: 指定位置的ATR值
    """
    # 确保有足够的数据计算ATR
    start_index = max(1, index - period + 1)
    
    if start_index >= index:
        return 0.0  # 数据不足，返回默认值
    
    # 计算真实波幅(TR)
    tr_values = []
    for i in range(start_index, index + 1):
        if i >= len(close_data) or i <= 0:
            continue
            
        high = high_data[i]
        low = low_data[i]
        prev_close = close_data[i-1]
        
        # 真实波幅计算
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr_values.append(max(tr1, tr2, tr3))
    
    # 如果没有足够的TR值，返回默认值
    if not tr_values:
        return 0.0
    
    # 计算ATR - 使用简单移动平均
    atr_value = np.mean(tr_values)
    
    return atr_value



'''====================定义class FlagPattern==========================='''
# 这是Python的一个装饰器,用来简化类的定义。它会自动帮我们生成__init__()等基础方法,让我们只需要定义类的属性就可以了,不用写很多重复的代码。
#__init__()是类的构造函数,当我们创建类的实例时会自动调用它来初始化实例的属性。比如定义一个普通的类需要写构造函数,而用@dataclass就不用写了。
@dataclass  
class FlagPattern:  # 定义一个旗形模式类,用于存储和表示股票价格中的旗形形态特征
    """
    这是一个使用Python的dataclass装饰器定义的类，用于表示旗形和三角旗形态。
    @dataclass是Python的一个装饰器，它会自动为类生成特殊方法，如__init__、__repr__等，
    简化了数据类的创建过程，使代码更加简洁。
    """

    # 旗形和三角旗形态的数据结构
    
    # 属性:
    base_x: int         # 趋势起点索引，旗杆的底部
    base_y: float       # 趋势起点价格
    
    tip_x: int   = -1       # 旗杆顶部/底部索引，旗帜开始点。初始化为-1表示尚未找到有效的旗杆顶部/底部点
    tip_y: float = -1.      # 旗杆顶部/底部价格。初始化为-1表示尚未找到有效的价格点
    
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

    # 新增成交量相关属性
    avg_flag_volume: float = -1.  # 旗帜区间的平均成交量
    breakout_volume: float = -1.  # 突破时的成交量
    volume_ratio: float = -1.     # 突破成交量/平均成交量的比率

    # 新增ATR相关属性
    current_atr: float = -1.  # 当前点的ATR值
    breakout_magnitude: float = -1.  # 突破幅度


'''====================定义旗形形态识别==========================='''

def check_bear_pattern_pips(pending: FlagPattern, data: np.array, i: int, order: int, 
                           volume: np.array = None, high_data: np.array = None, 
                           low_data: np.array = None):
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
    # 从旗杆底部(pending.base_x)到当前检查位置(i)的数据切片
    # data[pending.base_x: i + 1]表示取数据从旗杆底部到当前位置的子集
    # i+1是因为在Python中切片的右边界是开区间,即不包含i+1这个位置
    
    # pending.base_x是旗杆底部的索引位置
    # 这行代码从旗杆底部(pending.base_x)到当前检查位置(i)提取了一段数据
    # i+1是因为切片右边界是开区间,所以需要+1才能包含i这个位置
    data_slice = data[pending.base_x: i + 1]  
    
    # 切片(slice)是从数组中提取一段连续数据的操作
    # 例如data_slice = data[pending.base_x: i + 1]就是一个切片操作
    # 它从原始数组data中提取了从pending.base_x到i的这一段数据
    # argmin()返回切片中最小值的索引位置
    # 由于切片是原始数组的一部分,所以需要加上切片起点pending.base_x才是在原始数组中的实际位置
    min_i = data_slice.argmin() + pending.base_x  # 自局部顶部以来的最低点索引
    
    # 确保从最低点到当前位置有足够的距离来形成旗帜
    if i - min_i < max(5, order * 0.5):  # 这行代码检查当前位置i到最低点min_i的距离是否小于两个值中的较大值:
                                         # 1. 固定值5
                                         # 2. order参数的一半
                                         # 如果距离太小,说明还没有形成足够宽的旗形形态,返回False
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
    # 5表示要找出5个重要点位(PIP点)

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
    

    '''====================计算成交量相关属性==========================='''
    flag_start = min_i  # 旗帜开始点
    flag_volumes = volume[flag_start:i+1]
    avg_flag_volume = np.mean(flag_volumes)
    breakout_volume = volume[i]
    volume_ratio = breakout_volume / avg_flag_volume
    
    # # 确认点的成交量应大于旗帜区域的平均成交量
    # if breakout_volume < avg_flag_volume:
    #     return False  # 成交量不足，不确认突破
    

    '''====================计算ATR相关属性==========================='''
    # 使用旗帜宽度作为ATR计算周期
    atr_period = max(5, flag_width)  # 确保至少有5个周期
    
    # 计算当前点的ATR值，使用旗帜宽度作为周期
    current_atr = calculate_atr(high_data, low_data, data, i, period=atr_period)
        
    # 设置突破确认参数
    # min_breakout_multiple = 0.5  # 突破至少要达到0.5倍ATR
    min_breakout_multiple = 0.3 + (0.3 * min(1.0, flag_width / 20))  # 0.3 到 0.6 之间
    min_breakout_percent = 0.01  # 或者突破至少要达到1%
        
    # 计算实际突破幅度 (注意方向是向下的)
    breakout_magnitude = support_endpoint - pips_y[4]
        
        # 检查突破幅度是否足够
    if breakout_magnitude < min(current_atr * min_breakout_multiple, data[i] * min_breakout_percent):
        return False  # 突破幅度不足，不确认突破
    



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
    

    # # 保存成交量信息到旗形对象
    pending.avg_flag_volume = avg_flag_volume
    pending.breakout_volume = breakout_volume
    pending.volume_ratio = volume_ratio

    # 保存ATR信息到旗形对象
    pending.current_atr = current_atr
    pending.breakout_magnitude = breakout_magnitude
    

    pending.flag_width = flag_width  # 旗帜宽度
    pending.flag_height = flag_height  # 旗帜高度
    pending.pole_width = pole_width  # 旗杆宽度
    pending.pole_height = pole_height  # 旗杆高度
    pending.support_slope = support_slope  # 支撑线斜率
    pending.support_intercept = support_intercept  # 支撑线截距
    pending.resist_slope = resist_slope  # 阻力线斜率
    pending.resist_intercept = resist_intercept  # 阻力线截距
    
    return True  # 返回True表示识别到有效形态
    

def check_bull_pattern_pips(pending: FlagPattern, data: np.array, i: int, order: int, 
                           volume: np.array = None, high_data: np.array = None, 
                           low_data: np.array = None):
    """
    检查牛市旗形/三角旗形态（基于PIP点方法），可选结合成交量确认
    
    参数:
    pending: FlagPattern - 待填充的旗形对象
    data: np.array - 价格数据数组(一维)
    i: int - 当前检查的索引位置
    order: int - 滚动窗口大小参数
    volume_data: np.array - 可选的成交量数据数组(一维)
    
    返回:
    bool - 如果识别到有效形态则返回True，否则返回False
    """

    # 找出自局部底部以来的最高价格（旗杆顶部）
    # 这行代码从价格数组data中提取了从pending.base_x（局部底部）到i+1（当前位置）的一段数据。i+1是为了包含当前价格点，因为Python切片是左闭右开的。
    data_slice = data[pending.base_x: i + 1]  # i + 1包含当前价格

    # 这行代码在寻找从局部底部到当前位置之间的最高价格点的索引。
    # data_slice.argmax()找到切片中最高价格的位置，加上pending.base_x是为了将这个相对位置转换为在原始数组中的绝对位置。这个索引将用于确定旗杆的顶部位置。
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
    # 找出从最高点到当前索引之间的5个PIP点
    # pips_y[0]是第一个PIP点的价格,代表旗帜区域的起始点
    # pips_y[4]是最后一个PIP点的价格,代表当前价格点
    pips_x, pips_y = find_pips(data[max_i:i+1], 5, 3)  

    # 检查中心PIP点是否高于相邻的两个点，形成\/\/形状
    if not (pips_y[2] > pips_y[1] and pips_y[2] > pips_y[3]):
        return False
        
    # 计算旗帜的阻力线和支撑线
    # 阻力线：连接第1个和第3个PIP点
    # 计算阻力线的上升高度，即第3个PIP点(pips_y[2])与第1个PIP点(pips_y[0])的垂直距离
    resist_rise = pips_y[2] - pips_y[0]  # 阻力线上升高度
    
    # 计算阻力线的水平距离，即第3个PIP点(pips_x[2])与第1个PIP点(pips_x[0])的水平距离
    resist_run = pips_x[2] - pips_x[0]  # 阻力线水平距离
    
    # 计算阻力线的斜率，使用上升高度除以水平距离
    # 斜率为正表示向上倾斜，为负表示向下倾斜
    resist_slope = resist_rise / resist_run  # 阻力线斜率
    
    # 计算阻力线的截距，即阻力线与y轴的交点
    # 这里直接使用第1个PIP点的y值作为截距
    resist_intercept = pips_y[0]  # 阻力线截距

    # 支撑线：连接第2个和第4个PIP点
    support_rise = pips_y[3] - pips_y[1]  # 支撑线上升高度
    support_run = pips_x[3] - pips_x[1]  # 支撑线水平距离
    support_slope = support_rise / support_run  # 支撑线斜率
    support_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * support_slope  # 支撑线截距

    # 计算两条线的交点
    if resist_slope != support_slope:  # 非平行线
        # 计算支撑线和阻力线的交点的x坐标
        # 使用两条直线方程联立求解:
        # y = resist_slope * x + resist_intercept
        # y = support_slope * x + support_intercept
        # 解出x坐标(intersection)
        intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
    else:
        # 当支撑线和阻力线平行时,将交点设置在旗帜区域左侧很远的位置
        # 这样做是为了确保交点不会落在旗帜区域内
        # flag_width是旗帜的宽度,乘以100是为了将交点设得足够远
        intersection = -flag_width * 100  # 平行线，设置一个远离旗帜区域的交点

    # 如果交点在旗帜区域内，则不是有效的旗形/三角旗
    # 因为有效的旗形/三角旗的支撑线和阻力线应该在旗帜区域外相交
    # 如果在旗帜区域内相交，说明两条趋势线收敛太快，形成的是一个楔形形态而不是旗形
    # 楔形形态通常代表趋势的延续或反转，而旗形则代表趋势的暂时休整
    if intersection <= pips_x[4] and intersection >= 0:
        return False
    
    # 过滤严重发散的线（交点太近）
    # 如果交点在旗帜宽度的负1倍范围内,说明两条趋势线发散得太快,不是有效形态
    # 例如:如果旗帜宽度为10,那么交点应该在x<-10的位置,否则说明趋势线发散太快
    if intersection < 0 and intersection > -1.0 * flag_width:
        return False

    # 检查当前点是否突破旗帜上边界（阻力线），确认形态
    resist_endpoint = pips_y[0] + resist_slope * pips_x[4]
    if pips_y[4] < resist_endpoint:  # 如果价格低于阻力线，则未突破
        return False
    

    '''====================计算成交量相关属性==========================='''
    flag_start = max_i  # 旗帜开始点
    flag_volumes = volume[flag_start:i+1]
    avg_flag_volume = np.mean(flag_volumes)
    breakout_volume = volume[i]
    volume_ratio = breakout_volume / avg_flag_volume
    
    # # 确认点的成交量应大于旗帜区域的平均成交量
    # if breakout_volume < avg_flag_volume:
    #     return False  # 成交量不足，不确认突破
    

    '''====================计算ATR相关属性==========================='''

    # 使用旗帜宽度作为ATR计算周期
    atr_period = max(5, flag_width)  # 确保至少有5个周期
 
     # 计算当前点的ATR值，使用旗帜宽度作为周期
    current_atr = calculate_atr(high_data, low_data, data, i, period=atr_period)
        
    # 设置突破确认参数
    # min_breakout_multiple = 0.5  # 突破至少要达到0.5倍ATR
    min_breakout_multiple = 0.3 + (0.3 * min(1.0, flag_width / 20))  # 0.3 到 0.6 之间
    min_breakout_percent = 0.01  # 或者突破至少要达到1%
        
    # 计算实际突破幅度
    breakout_magnitude = pips_y[4] - resist_endpoint
        
    # 检查突破幅度是否足够
    if breakout_magnitude < min(current_atr * min_breakout_multiple, data[i] * min_breakout_percent):
        return False  # 突破幅度不足，不确认突破


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


    # # 保存成交量信息到旗形对象
    pending.avg_flag_volume = avg_flag_volume
    pending.breakout_volume = breakout_volume
    pending.volume_ratio = volume_ratio

    # 保存ATR信息到旗形对象
    pending.current_atr = current_atr
    pending.breakout_magnitude = breakout_magnitude

    
    pending.flag_width = flag_width  # 旗帜宽度
    pending.flag_height = flag_height  # 旗帜高度
    pending.pole_width = pole_width  # 旗杆宽度
    pending.pole_height = pole_height  # 旗杆高度
    
    pending.support_slope = support_slope  # 支撑线斜率
    pending.support_intercept = support_intercept  # 支撑线截距

    pending.resist_slope = resist_slope  # 阻力线斜率
    pending.resist_intercept = resist_intercept  # 阻力线截距
    
    return True  # 返回True表示识别到有效形态



def find_flags_pennants_pips(data: np.array, order: int, volume: np.array = None, 
                            high_data: np.array = None, low_data: np.array = None):
    """
    基于PIP点方法识别旗形和三角旗形态，可选结合成交量确认
    
    参数:
    data: np.array - 价格数据数组(一维)
    order: int - 滚动窗口大小参数
    volume: np.array - 可选的成交量数据数组(一维)
    
    返回:
    bull_flags, bear_flags, bull_pennants, bear_pennants
    """
    assert(order >= 3)  # 确保窗口大小参数至少为3
    # assert(data.ndim == 1), "价格数据必须是一维数组"
    # if volume is not None:
    #     assert(volume.ndim == 1), "成交量数据必须是一维数组"
    #     assert(len(volume) == len(data)), "成交量数据长度必须与价格数据相同"
    
    
    pending_bull = None  # 待处理的牛市形态
    pending_bear = None  # 待处理的熊市形态

    # 初始化结果列表
    bull_pennants = []  # 牛市三角旗列表
    bear_pennants = []  # 熊市三角旗列表
    bull_flags = []     # 牛市旗形列表
    bear_flags = []     # 熊市旗形列表
    
    '''
    因为：
    熊旗形态是从高点开始向下运动，所以需要从局部高点开始寻找
    牛旗形态是从低点开始向上运动，所以需要从局部低点开始寻找
    这符合市场趋势的基本原理 - 熊市从高点下跌，牛市从低点上涨。
        
    '''

    # 遍历价格数据 len(data)返回7320，而range()函数生成的序列是从0开始到结束值-1，所以i的取值范围是0到7319。这与数据帧的7320行数据相对应。
    for i in range(len(data)):# range(len(data))的取值范围是从0到7319，因为根据上下文中的数据帧输出显示总共有7320行数据（[7320 rows x 6 columns]）。

        # 识别局部极值点作为形态起点
        '''
        rw_top(data, i, order)函数会检查i-order位置的点是否是在[i-2*order, i]这个窗口范围内的局部高点。
        也就是说，它需要等待后续order个点的数据才能确认i-order位置是否真的是局部高点。
        这样设计是为了避免在实时分析中的"提前预知"问题。
        '''
        if rw_top(data, i, order):   # 如果是局部高点，i是当前遍历到的数据点的索引，order参数为12，表示在前后各12个点(共25个点，包括当前点)的范围内是最高点
        # 创建新的熊市形态对象，以当前高点为起点
        # 代码在每次检测到局部高点时(rw_top返回True)，就会创建一个新的熊市旗形对象(pending_bear)。这是因为每个高点都可能是潜在的熊市旗形或三角旗形态的起点。
        # FlagPattern是一个类，这里创建了该类的实例，传入两个参数：
        # 第一个参数i - order：形态的基准点位置（索引）
        # 第二个参数data[i - order]：该位置对应的价格值
        # 这两个值会被存储为对象的base_x和base_y属性，用于后续旗形形态的识别和分析。
        # 构造函数只初始化了base_x和base_y这两个属性，其余13个属性此时都是空值。这些属性会在后续的check_bull_pattern_pips或check_bear_pattern_pips函数中被赋值。

        # 代码在每次检测到局部高点时(rw_top返回True)，就会创建一个新的熊市旗形对象(pending_bear)。这是因为每个高点都可能是潜在的熊市旗形或三角旗形态的起点。
            pending_bear = FlagPattern(i - order, data[i - order])
        if rw_bottom(data, i, order):  # 如果是局部低点
            # 创建新的牛市形态对象，以当前低点为起点
            pending_bull = FlagPattern(i - order, data[i - order])

        # 检查并处理待处理的熊市形态
        if pending_bear is not None:
            # 检查是否形成熊市旗形/三角旗
            if check_bear_pattern_pips(pending_bear, data, i, order, volume, high_data, low_data):
                # 根据形态类型添加到相应列表
                if pending_bear.pennant:
                    bear_pennants.append(pending_bear)  # 添加熊市三角旗
                else:
                    bear_flags.append(pending_bear)     # 添加熊市旗形
                pending_bear = None  # 重置待处理形态

        # 检查并处理待处理的牛市形态
        if pending_bull is not None:
            # 检查是否形成牛市旗形/三角旗
            if check_bull_pattern_pips(pending_bull, data, i, order, volume, high_data, low_data):
                # 根据形态类型添加到相应列表
                if pending_bull.pennant:
                    bull_pennants.append(pending_bull)  # 添加牛市三角旗
                else:
                    bull_flags.append(pending_bull)     # 添加牛市旗形
                pending_bull = None  # 重置待处理形态

    # 返回识别结果
    return bull_flags, bear_flags, bull_pennants, bear_pennants



'''====================主程序==========================='''


if __name__ == "__main__":
    # 定义要测试的窗口大小参数范围（从5到50）
    orders = list(range(5, 61))

    # 创建一个字典来存储所有股票的结果
    all_stocks_results = {}

    # 创建汇总结果的数据框
    summary_results_df = pd.DataFrame(index=orders)

    # 初始化汇总数据存储列表
    all_bull_flag_count = [0] * len(orders)
    all_bull_pennant_count = [0] * len(orders)
    all_bear_flag_count = [0] * len(orders)
    all_bear_pennant_count = [0] * len(orders)

    all_bull_flag_total_ret = [0] * len(orders)
    all_bull_pennant_total_ret = [0] * len(orders)
    all_bear_flag_total_ret = [0] * len(orders)
    all_bear_pennant_total_ret = [0] * len(orders)

    all_bull_flag_wins = [0] * len(orders)
    all_bull_pennant_wins = [0] * len(orders)
    all_bear_flag_wins = [0] * len(orders)
    all_bear_pennant_wins = [0] * len(orders)

    # 读取数据
    file_path = 'SW一级数据.xlsx'
    data_raw = load_stock_data(file_path)
    data = data_raw.copy()
    stocks_list = data['stocks']

    # 遍历每只股票
    # for stock_code in stocks_list[:2]: 这是获取0、1两只股票
    for stock_code in stocks_list:
        print(f"处理股票: {stock_code}")


        # 获取特定股票数据
        stock_df = pd.DataFrame({
            'ln_close': np.log(data['close_df'][stock_code]),
            'ln_high': np.log(data['high_df'][stock_code]),
            'ln_low': np.log(data['low_df'][stock_code]),
            'volume': data['volume_df'][stock_code]
        }, index=data['close_df'].index)
        
        # # 使用Series保留原始时间索引信息，而不是使用values属性转换为numpy数组
        # price_data = stock_df['ln_close']  # price_data保留了原始的DatetimeIndex时间索引
        # volume_data = stock_df['volume'] 

        # 转换为numpy数组格式
        price_data = stock_df['ln_close'].values  
        volume_data = stock_df['volume'].values 
        high_data = stock_df['ln_high'].values  # 转换为np数组
        low_data = stock_df['ln_low'].values    # 转换为np数组

        # 初始化结果存储列表
        # 胜率（Win Rate）列表
        bull_flag_wr = []  # 牛市旗形胜率
        bull_pennant_wr = []  # 牛市三角旗胜率
        bear_flag_wr = []  # 熊市旗形胜率
        bear_pennant_wr = []  # 熊市三角旗胜率

        # 平均收益列表
        bull_flag_avg = []  # 牛市旗形平均收益
        bull_pennant_avg = []  # 牛市三角旗平均收益
        bear_flag_avg = []  # 熊市旗形平均收益
        bear_pennant_avg = []  # 熊市三角旗平均收益

        # 形态数量列表
        bull_flag_count = []  # 牛市旗形数量
        bull_pennant_count = []  # 牛市三角旗数量
        bear_flag_count = []  # 熊市旗形数量
        bear_pennant_count = []  # 熊市三角旗数量

        # 总收益列表
        bull_flag_total_ret = []  # 牛市旗形总收益
        bull_pennant_total_ret = []  # 牛市三角旗总收益
        bear_flag_total_ret = []  # 熊市旗形总收益
        bear_pennant_total_ret = []  # 熊市三角旗总收益

        # 创建一个字典来存储每个order参数下的形态详细信息
        pattern_details = {}

        # 遍历每个窗口大小参数进行测试
        for i, order in enumerate(orders):
            
            # 使用PIP点方法识别旗形和三角旗
            bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_pips(price_data, order, volume_data, high_data, low_data)
            # 也可以使用趋势线方法（取消下面的注释即可）
            # bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(price_data, order)

            # 创建数据框来存储形态属性和收益
            bull_flag_df = pd.DataFrame()  # 牛市旗形数据框
            bull_pennant_df = pd.DataFrame()  # 牛市三角旗数据框
            bear_flag_df = pd.DataFrame()  # 熊市旗形数据框
            bear_pennant_df = pd.DataFrame()  # 熊市三角旗数据框

            # 设置持有期乘数（持有时间 = 旗帜宽度 * 乘数）
            hold_mult = 1.0  # 默认持有时间等于旗帜宽度
            
            # 处理牛市旗形数据
            for j, flag in enumerate(bull_flags):
                # 记录形态属性
                bull_flag_df.loc[j, 'flag_width'] = flag.flag_width  # 旗帜宽度
                bull_flag_df.loc[j, 'flag_height'] = flag.flag_height  # 旗帜高度
                bull_flag_df.loc[j, 'pole_width'] = flag.pole_width  # 旗杆宽度
                bull_flag_df.loc[j, 'pole_height'] = flag.pole_height  # 旗杆高度
                bull_flag_df.loc[j, 'slope'] = flag.resist_slope  # 阻力线斜率
                
                # 记录关键点位索引
                bull_flag_df.loc[j, 'start_x'] = flag.base_x  # 起始点索引
                bull_flag_df.loc[j, 'end_x'] = flag.tip_x  # 旗杆顶部索引
                bull_flag_df.loc[j, 'conf_x'] = flag.conf_x  # 确认点索引
                
                # 记录日期信息
                bull_flag_df.loc[j, 'start_date'] = stock_df.index[flag.base_x]  # 起始日期
                bull_flag_df.loc[j, 'end_date'] = stock_df.index[flag.tip_x]  # 旗杆顶部日期
                bull_flag_df.loc[j, 'conf_date'] = stock_df.index[flag.conf_x]  # 确认日期

                # 计算持有期收益
                hp = int(flag.flag_width * hold_mult)  # 持有期长度
                if flag.conf_x + hp >= len(price_data):  # 如果持有期超出数据范围
                    bull_flag_df.loc[j, 'return'] = np.nan  # 设置收益为NaN
                    bull_flag_df.loc[j, 'exit_date'] = np.nan  # 退出日期为NaN
                else:
                    # 计算对数收益率（确认点到持有期结束的价格变化）
                    ret = price_data[flag.conf_x + hp] - price_data[flag.conf_x]
                    bull_flag_df.loc[j, 'return'] = ret 
                    bull_flag_df.loc[j, 'exit_date'] = stock_df.index[flag.conf_x + hp]  # 退出日期

                # 如果有成交量数据，添加成交量相关统计
                if volume_data is not None and flag.volume_ratio > 0:
                    bull_flag_df.loc[j, 'avg_volume'] = flag.avg_flag_volume
                    bull_flag_df.loc[j, 'breakout_volume'] = flag.breakout_volume
                    bull_flag_df.loc[j, 'volume_ratio'] = flag.volume_ratio

            # 处理熊市旗形数据
            for j, flag in enumerate(bear_flags):
                # 记录形态属性
                bear_flag_df.loc[j, 'flag_width'] = flag.flag_width  # 旗帜宽度
                bear_flag_df.loc[j, 'flag_height'] = flag.flag_height  # 旗帜高度
                bear_flag_df.loc[j, 'pole_width'] = flag.pole_width  # 旗杆宽度
                bear_flag_df.loc[j, 'pole_height'] = flag.pole_height  # 旗杆高度
                bear_flag_df.loc[j, 'slope'] = flag.support_slope  # 支撑线斜率
                
                # 记录关键点位索引
                bear_flag_df.loc[j, 'start_x'] = flag.base_x  # 起始点索引
                bear_flag_df.loc[j, 'end_x'] = flag.tip_x  # 结束点索引
                bear_flag_df.loc[j, 'conf_x'] = flag.conf_x  # 确认点索引
                
                # 记录日期信息
                bear_flag_df.loc[j, 'start_date'] = stock_df.index[flag.base_x]  # 起始日期
                bear_flag_df.loc[j, 'end_date'] = stock_df.index[flag.tip_x]  # 旗杆顶部日期
                bear_flag_df.loc[j, 'conf_date'] = stock_df.index[flag.conf_x]  # 确认日期

                # 计算持有期收益（注意熊市形态是做空，所以收益取负）
                hp = int(flag.flag_width * hold_mult)  # 持有期长度
                if flag.conf_x + hp >= len(price_data):  # 如果持有期超出数据范围
                    bear_flag_df.loc[j, 'return'] = np.nan  # 设置收益为NaN
                    bear_flag_df.loc[j, 'exit_date'] = np.nan  # 退出日期为NaN
                else:
                    # 计算对数收益率（确认点到持有期结束的价格变化的负值）
                    ret = -1 * (price_data[flag.conf_x + hp] - price_data[flag.conf_x])
                    bear_flag_df.loc[j, 'return'] = ret 
                    bear_flag_df.loc[j, 'exit_date'] = stock_df.index[flag.conf_x + hp]  # 退出日期

                # 如果有成交量数据，添加成交量相关统计
                if volume_data is not None and flag.volume_ratio > 0:
                    bear_flag_df.loc[j, 'avg_volume'] = flag.avg_flag_volume
                    bear_flag_df.loc[j, 'breakout_volume'] = flag.breakout_volume
                    bear_flag_df.loc[j, 'volume_ratio'] = flag.volume_ratio

            # 处理牛市三角旗数据
            for j, pennant in enumerate(bull_pennants):
                # 记录形态属性
                bull_pennant_df.loc[j, 'pennant_width'] = pennant.flag_width  # 三角旗宽度
                bull_pennant_df.loc[j, 'pennant_height'] = pennant.flag_height  # 三角旗高度
                bull_pennant_df.loc[j, 'pole_width'] = pennant.pole_width  # 旗杆宽度
                bull_pennant_df.loc[j, 'pole_height'] = pennant.pole_height  # 旗杆高度
                
                # 记录关键点位索引
                bull_pennant_df.loc[j, 'start_x'] = pennant.base_x  # 起始点索引
                bull_pennant_df.loc[j, 'end_x'] = pennant.tip_x  # 结束点索引
                bull_pennant_df.loc[j, 'conf_x'] = pennant.conf_x  # 确认点索引
                
                # 记录日期信息
                bull_pennant_df.loc[j, 'start_date'] = stock_df.index[pennant.base_x]  # 起始日期
                bull_pennant_df.loc[j, 'end_date'] = stock_df.index[pennant.tip_x]  # 旗杆顶部日期
                bull_pennant_df.loc[j, 'conf_date'] = stock_df.index[pennant.conf_x]  # 确认日期

                # 计算持有期收益
                hp = int(pennant.flag_width * hold_mult)  # 持有期长度
                if pennant.conf_x + hp >= len(price_data):  # 如果持有期超出数据范围
                    bull_pennant_df.loc[j, 'return'] = np.nan  # 设置收益为NaN
                    bull_pennant_df.loc[j, 'exit_date'] = np.nan  # 退出日期为NaN
                else:
                    # 计算对数收益率（确认点到持有期结束的价格变化）
                    ret = price_data[pennant.conf_x + hp] - price_data[pennant.conf_x]
                    bull_pennant_df.loc[j, 'return'] = ret 
                    bull_pennant_df.loc[j, 'exit_date'] = stock_df.index[pennant.conf_x + hp]  # 退出日期

                # 如果有成交量数据，添加成交量相关统计
                if volume_data is not None and flag.volume_ratio > 0:
                    bull_pennant_df.loc[j, 'avg_volume'] = pennant.avg_flag_volume
                    bull_pennant_df.loc[j, 'breakout_volume'] = pennant.breakout_volume
                    bull_pennant_df.loc[j, 'volume_ratio'] = pennant.volume_ratio

            # 处理熊市三角旗数据
            for j, pennant in enumerate(bear_pennants):
                # 记录形态属性
                bear_pennant_df.loc[j, 'pennant_width'] = pennant.flag_width  # 三角旗宽度
                bear_pennant_df.loc[j, 'pennant_height'] = pennant.flag_height  # 三角旗高度
                bear_pennant_df.loc[j, 'pole_width'] = pennant.pole_width  # 旗杆宽度
                bear_pennant_df.loc[j, 'pole_height'] = pennant.pole_height  # 旗杆高度
                
                # 记录关键点位索引
                bear_pennant_df.loc[j, 'start_x'] = pennant.base_x  # 起始点索引
                bear_pennant_df.loc[j, 'end_x'] = pennant.tip_x  # 结束点索引
                bear_pennant_df.loc[j, 'conf_x'] = pennant.conf_x  # 确认点索引
                
                # 记录日期信息
                bear_pennant_df.loc[j, 'start_date'] = stock_df.index[pennant.base_x]  # 起始日期
                bear_pennant_df.loc[j, 'end_date'] = stock_df.index[pennant.tip_x]  # 旗杆顶部日期
                bear_pennant_df.loc[j, 'conf_date'] = stock_df.index[pennant.conf_x]  # 确认日期

                # 计算持有期收益（注意熊市形态是做空，所以收益取负）
                hp = int(pennant.flag_width * hold_mult)  # 持有期长度
                if pennant.conf_x + hp >= len(price_data):  # 如果持有期超出数据范围
                    bear_pennant_df.loc[j, 'return'] = np.nan  # 设置收益为NaN
                    bear_pennant_df.loc[j, 'exit_date'] = np.nan  # 退出日期为NaN
                else:
                    # 计算对数收益率（确认点到持有期结束的价格变化的负值）
                    ret = -1 * (price_data[pennant.conf_x + hp] - price_data[pennant.conf_x])
                    bear_pennant_df.loc[j, 'return'] = ret 
                    bear_pennant_df.loc[j, 'exit_date'] = stock_df.index[pennant.conf_x + hp]  # 退出日期

                # 如果有成交量数据，添加成交量相关统计
                if volume_data is not None and flag.volume_ratio > 0:
                    bear_pennant_df.loc[j, 'avg_volume'] = pennant.avg_flag_volume
                    bear_pennant_df.loc[j, 'breakout_volume'] = pennant.breakout_volume
                    bear_pennant_df.loc[j, 'volume_ratio'] = pennant.volume_ratio

            # 保存每个order参数下的形态详细信息
            pattern_details[order] = {
                'bull_flag': bull_flag_df.copy() if not bull_flag_df.empty else None,
                'bear_flag': bear_flag_df.copy() if not bear_flag_df.empty else None,
                'bull_pennant': bull_pennant_df.copy() if not bull_pennant_df.empty else None,
                'bear_pennant': bear_pennant_df.copy() if not bear_pennant_df.empty else None
            }

            # 计算牛市旗形的统计数据
            if not bull_flag_df.empty:
                bull_flag_count.append(len(bull_flag_df))
                valid_returns = bull_flag_df['return'].dropna()
                if not valid_returns.empty:
                    bull_flag_avg.append(valid_returns.mean())
                    bull_flag_wr.append(len(valid_returns[valid_returns > 0]) / len(valid_returns))
                    bull_flag_total_ret.append(valid_returns.sum())
                    
                    # 更新全局统计
                    all_bull_flag_count[i] += len(bull_flag_df)
                    all_bull_flag_total_ret[i] += valid_returns.sum()
                    all_bull_flag_wins[i] += len(valid_returns[valid_returns > 0])
                else:
                    bull_flag_avg.append(np.nan)
                    bull_flag_wr.append(np.nan)
                    bull_flag_total_ret.append(0)
            else:
                bull_flag_count.append(0)
                bull_flag_avg.append(np.nan)
                bull_flag_wr.append(np.nan)
                bull_flag_total_ret.append(0)
            
            # 计算熊市旗形的统计数据
            if not bear_flag_df.empty:
                bear_flag_count.append(len(bear_flag_df))
                valid_returns = bear_flag_df['return'].dropna()
                if not valid_returns.empty:
                    bear_flag_avg.append(valid_returns.mean())
                    bear_flag_wr.append(len(valid_returns[valid_returns > 0]) / len(valid_returns))
                    bear_flag_total_ret.append(valid_returns.sum())
                    
                    # 更新全局统计
                    all_bear_flag_count[i] += len(bear_flag_df)
                    all_bear_flag_total_ret[i] += valid_returns.sum()
                    all_bear_flag_wins[i] += len(valid_returns[valid_returns > 0])
                else:
                    bear_flag_avg.append(np.nan)
                    bear_flag_wr.append(np.nan)
                    bear_flag_total_ret.append(0)
            else:
                bear_flag_count.append(0)
                bear_flag_avg.append(np.nan)
                bear_flag_wr.append(np.nan)
                bear_flag_total_ret.append(0)
            
            # 计算牛市三角旗的统计数据
            if not bull_pennant_df.empty:
                bull_pennant_count.append(len(bull_pennant_df))
                valid_returns = bull_pennant_df['return'].dropna()
                if not valid_returns.empty:
                    bull_pennant_avg.append(valid_returns.mean())
                    bull_pennant_wr.append(len(valid_returns[valid_returns > 0]) / len(valid_returns))
                    bull_pennant_total_ret.append(valid_returns.sum())
                    
                    # 更新全局统计
                    all_bull_pennant_count[i] += len(bull_pennant_df)
                    all_bull_pennant_total_ret[i] += valid_returns.sum()
                    all_bull_pennant_wins[i] += len(valid_returns[valid_returns > 0])
                else:
                    bull_pennant_avg.append(np.nan)
                    bull_pennant_wr.append(np.nan)
                    bull_pennant_total_ret.append(0)
            else:
                bull_pennant_count.append(0)
                bull_pennant_avg.append(np.nan)
                bull_pennant_wr.append(np.nan)
                bull_pennant_total_ret.append(0)
            
            # 计算熊市三角旗的统计数据
            if not bear_pennant_df.empty:
                bear_pennant_count.append(len(bear_pennant_df))
                valid_returns = bear_pennant_df['return'].dropna()
                if not valid_returns.empty:
                    bear_pennant_avg.append(valid_returns.mean())
                    bear_pennant_wr.append(len(valid_returns[valid_returns > 0]) / len(valid_returns))
                    bear_pennant_total_ret.append(valid_returns.sum())
                    
                    # 更新全局统计
                    all_bear_pennant_count[i] += len(bear_pennant_df)
                    all_bear_pennant_total_ret[i] += valid_returns.sum()
                    all_bear_pennant_wins[i] += len(valid_returns[valid_returns > 0])
                else:
                    bear_pennant_avg.append(np.nan)
                    bear_pennant_wr.append(np.nan)
                    bear_pennant_total_ret.append(0)
            else:
                bear_pennant_count.append(0)
                bear_pennant_avg.append(np.nan)
                bear_pennant_wr.append(np.nan)
                bear_pennant_total_ret.append(0)
        
        # 创建结果数据框，以窗口大小参数为索引
        results_df = pd.DataFrame(index=orders)

        # 添加牛市旗形的统计数据
        results_df['bull_flag_count'] = bull_flag_count  # 形态数量
        results_df['bull_flag_avg'] = bull_flag_avg  # 平均收益
        results_df['bull_flag_wr'] = bull_flag_wr  # 胜率
        results_df['bull_flag_total'] = bull_flag_total_ret  # 总收益

        # 添加熊市旗形的统计数据
        results_df['bear_flag_count'] = bear_flag_count  # 形态数量
        results_df['bear_flag_avg'] = bear_flag_avg  # 平均收益
        results_df['bear_flag_wr'] = bear_flag_wr  # 胜率
        results_df['bear_flag_total'] = bear_flag_total_ret  # 总收益

        # 添加牛市三角旗的统计数据
        results_df['bull_pennant_count'] = bull_pennant_count  # 形态数量
        results_df['bull_pennant_avg'] = bull_pennant_avg  # 平均收益
        results_df['bull_pennant_wr'] = bull_pennant_wr  # 胜率
        results_df['bull_pennant_total'] = bull_pennant_total_ret  # 总收益

        # 添加熊市三角旗的统计数据
        results_df['bear_pennant_count'] = bear_pennant_count  # 形态数量
        results_df['bear_pennant_avg'] = bear_pennant_avg  # 平均收益
        results_df['bear_pennant_wr'] = bear_pennant_wr  # 胜率
        results_df['bear_pennant_total'] = bear_pennant_total_ret  # 总收益
        
        # 保存当前股票的结果
        all_stocks_results[stock_code] = {
            'results_df': results_df.copy(),
            'pattern_details': pattern_details.copy()
        }

    # 计算所有股票的汇总统计
    for i, order in enumerate(orders):
        # 计算胜率
        if all_bull_flag_count[i] > 0:
            summary_results_df.loc[order, 'bull_flag_wr'] = all_bull_flag_wins[i] / all_bull_flag_count[i]
        else:
            summary_results_df.loc[order, 'bull_flag_wr'] = np.nan
            
        if all_bull_pennant_count[i] > 0:
            summary_results_df.loc[order, 'bull_pennant_wr'] = all_bull_pennant_wins[i] / all_bull_pennant_count[i]
        else:
            summary_results_df.loc[order, 'bull_pennant_wr'] = np.nan
            
        if all_bear_flag_count[i] > 0:
            summary_results_df.loc[order, 'bear_flag_wr'] = all_bear_flag_wins[i] / all_bear_flag_count[i]
        else:
            summary_results_df.loc[order, 'bear_flag_wr'] = np.nan
            
        if all_bear_pennant_count[i] > 0:
            summary_results_df.loc[order, 'bear_pennant_wr'] = all_bear_pennant_wins[i] / all_bear_pennant_count[i]
        else:
            summary_results_df.loc[order, 'bear_pennant_wr'] = np.nan
        
        # 记录形态数量
        summary_results_df.loc[order, 'bull_flag_count'] = all_bull_flag_count[i]
        summary_results_df.loc[order, 'bull_pennant_count'] = all_bull_pennant_count[i]
        summary_results_df.loc[order, 'bear_flag_count'] = all_bear_flag_count[i]
        summary_results_df.loc[order, 'bear_pennant_count'] = all_bear_pennant_count[i]
        
        # 记录总收益
        summary_results_df.loc[order, 'bull_flag_total'] = all_bull_flag_total_ret[i]
        summary_results_df.loc[order, 'bull_pennant_total'] = all_bull_pennant_total_ret[i]
        summary_results_df.loc[order, 'bear_flag_total'] = all_bear_flag_total_ret[i]
        summary_results_df.loc[order, 'bear_pennant_total'] = all_bear_pennant_total_ret[i]
        
        # 计算平均收益
        if all_bull_flag_count[i] > 0:
            summary_results_df.loc[order, 'bull_flag_avg'] = all_bull_flag_total_ret[i] / all_bull_flag_count[i]
        else:
            summary_results_df.loc[order, 'bull_flag_avg'] = np.nan
            
        if all_bull_pennant_count[i] > 0:
            summary_results_df.loc[order, 'bull_pennant_avg'] = all_bull_pennant_total_ret[i] / all_bull_pennant_count[i]
        else:
            summary_results_df.loc[order, 'bull_pennant_avg'] = np.nan
            
        if all_bear_flag_count[i] > 0:
            summary_results_df.loc[order, 'bear_flag_avg'] = all_bear_flag_total_ret[i] / all_bear_flag_count[i]
        else:
            summary_results_df.loc[order, 'bear_flag_avg'] = np.nan
            
        if all_bear_pennant_count[i] > 0:
            summary_results_df.loc[order, 'bear_pennant_avg'] = all_bear_pennant_total_ret[i] / all_bear_pennant_count[i]
        else:
            summary_results_df.loc[order, 'bear_pennant_avg'] = np.nan

            
    print("统计结果:")
    # 只输出胜率(_wr)和数量(_count)列
    wr_count_columns = [col for col in summary_results_df.columns if '_wr' in col or '_count' in col]
    print(summary_results_df[wr_count_columns])
    # 保存汇总结果到Excel文件SW一级_旗形形态统计结果
    output_file = '【close+atr】SW一级_旗形形态统计结果.xlsx'
    summary_results_df.to_excel(output_file)
    print(f"汇总结果已保存到 {output_file}")

    


'''====================保存order详细结果==========================='''

    # 使用ExcelWriter创建Excel文件
with pd.ExcelWriter('【close+atr】SW一级_旗形形态_详细结果.xlsx', engine='openpyxl') as writer:
    # 首先保存汇总结果
    summary_results_df.to_excel(writer, sheet_name='汇总结果')
    
    # 保存每个order参数下的形态详细信息
    for order in orders:
        # 获取当前order的形态详情
        patterns = pattern_details[order]
        
        # 创建order索引页
        order_sheet_name = f'Order_{order}_索引'
        order_index_data = []
        
        # 处理牛市旗形
        if patterns['bull_flag'] is not None and not patterns['bull_flag'].empty:
            bull_flag_df = patterns['bull_flag']
            bull_flag_df.to_excel(writer, sheet_name=f'Order_{order}_牛旗')
            order_index_data.append({
                '形态类型': '牛旗',
                '数量': len(bull_flag_df),
                '平均收益': bull_flag_df['return'].mean() if 'return' in bull_flag_df.columns else np.nan,
                '胜率': len(bull_flag_df[bull_flag_df['return'] > 0]) / len(bull_flag_df) if 'return' in bull_flag_df.columns else np.nan
            })
        
        # 处理熊市旗形
        if patterns['bear_flag'] is not None and not patterns['bear_flag'].empty:
            bear_flag_df = patterns['bear_flag']
            bear_flag_df.to_excel(writer, sheet_name=f'Order_{order}_熊旗')
            order_index_data.append({
                '形态类型': '熊旗',
                '数量': len(bear_flag_df),
                '平均收益': bear_flag_df['return'].mean() if 'return' in bear_flag_df.columns else np.nan,
                '胜率': len(bear_flag_df[bear_flag_df['return'] > 0]) / len(bear_flag_df) if 'return' in bear_flag_df.columns else np.nan
            })
        
        # 处理牛市三角旗
        if patterns['bull_pennant'] is not None and not patterns['bull_pennant'].empty:
            bull_pennant_df = patterns['bull_pennant']
            bull_pennant_df.to_excel(writer, sheet_name=f'Order_{order}_牛三角旗')
            order_index_data.append({
                '形态类型': '牛三角旗',
                '数量': len(bull_pennant_df),
                '平均收益': bull_pennant_df['return'].mean() if 'return' in bull_pennant_df.columns else np.nan,
                '胜率': len(bull_pennant_df[bull_pennant_df['return'] > 0]) / len(bull_pennant_df) if 'return' in bull_pennant_df.columns else np.nan
            })
        
        # 处理熊市三角旗
        if patterns['bear_pennant'] is not None and not patterns['bear_pennant'].empty:
            bear_pennant_df = patterns['bear_pennant']
            bear_pennant_df.to_excel(writer, sheet_name=f'Order_{order}_熊三角旗')
            order_index_data.append({
                '形态类型': '熊三角旗',
                '数量': len(bear_pennant_df),
                '平均收益': bear_pennant_df['return'].mean() if 'return' in bear_pennant_df.columns else np.nan,
                '胜率': len(bear_pennant_df[bear_pennant_df['return'] > 0]) / len(bear_pennant_df) if 'return' in bear_pennant_df.columns else np.nan
            })
        
        # 创建并保存order索引页
        if order_index_data:
            order_index_df = pd.DataFrame(order_index_data)
            order_index_df.to_excel(writer, sheet_name=order_sheet_name)
    
    print(f"结果已保存至SW一级_旗形形态_详细结果.xlsx")


'''====================绘制统计图表==========================='''

# 导入seaborn库和matplotlib库
import seaborn as sns
import matplotlib.pyplot as plt

# 设置样式
sns.set_theme(style="darkgrid")
# sns.set_theme() 函数可以接受以下参数：
# context：控制绘图元素的比例，可选值有 "paper"、"notebook"、"talk"、"poster"，默认为 "notebook"
# style：控制绘图美学风格，可选值有 "darkgrid"、"whitegrid"、"dark"、"white"、"ticks"
# palette：控制颜色方案，可使用 Seaborn 内置调色板如 "deep"、"muted"、"bright" 等
# font：设置字体系列
# font_scale：字体大小的缩放因子
# color_codes：是否启用简短颜色代码
# rc：可传递字典覆盖默认的 matplotlib 参数

# sns.set_theme()与plt.style.use的区别：
# 1. sns.set_theme()是seaborn的新API，设置默认主题，包括颜色、网格等
# 2. plt.style.use直接使用Matplotlib的样式表，'seaborn-v0_8-bright'是保留的旧版seaborn风格
# sns.set_style("whitegrid")  # 设置网格样式

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 定义一个函数来绘制形态性能图表
def plot_pattern_performance(pattern_type, pattern_name):
    """
    绘制特定形态的性能图表
    
    参数:
    pattern_type: str - 形态类型前缀 (如 'bull_flag', 'bear_pennant' 等)
    pattern_name: str - 图表标题中显示的形态名称
    """
    # 创建2x2的子图
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"{pattern_name}表现结果", fontsize=20)  # 设置总标题
    
    # 绘制四个指标
    summary_results_df[f'{pattern_type}_count'].plot.bar(ax=ax[0,0])  # 形态数量柱状图
    summary_results_df[f'{pattern_type}_avg'].plot.bar(ax=ax[0,1])  # 平均收益柱状图
    summary_results_df[f'{pattern_type}_total'].plot.bar(ax=ax[1,0])  # 总收益柱状图
    summary_results_df[f'{pattern_type}_wr'].plot.bar(ax=ax[1,1])  # 胜率柱状图
    
    # 添加参考线
    ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='gray')  # 平均收益为0的参考线
    ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='gray')  # 总收益为0的参考线
    ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='gray')  # 胜率为50%的参考线
    
    # 设置子图标题和标签
    ax[0,0].set_title('形态数量')  # 形态数量子图标题
    ax[0,0].set_xlabel('Order 参数')  # x轴标签
    ax[0,0].set_ylabel('数量')  # y轴标签
    ax[0,1].set_title('平均收益率(ln)')  # 平均收益子图标题
    ax[0,1].set_xlabel('Order 参数')  # x轴标签
    ax[0,1].set_ylabel('平均收益率(ln)')  # y轴标签
    ax[1,0].set_title('总收益率(ln)')  # 总收益子图标题
    ax[1,0].set_xlabel('Order 参数')  # x轴标签
    ax[1,0].set_ylabel('总收益率(ln)')  # y轴标签
    ax[1,1].set_title('胜率')  # 胜率子图标题
    ax[1,1].set_xlabel('Order 参数')  # x轴标签
    ax[1,1].set_ylabel('胜率')  # y轴标签
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，避免标题重叠
    plt.show()  # 显示图表

# 绘制四种形态的性能图表
plot_pattern_performance('bull_flag', '牛旗')
plot_pattern_performance('bear_flag', '熊旗')
plot_pattern_performance('bull_pennant', '牛市三角旗')
plot_pattern_performance('bear_pennant', '熊市三角旗')








