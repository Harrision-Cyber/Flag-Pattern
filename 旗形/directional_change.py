# Directional Change 算法

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def dc_top(data: np.array, high: np.array, low: np.array, curr_index: int, sigma: float, last_extreme_index: int, last_extreme_price: float, is_last_extreme_bottom: bool) -> bool:
    """
    检测是否形成了一个顶部
    
    参数:
    data: 收盘价数组
    high: 最高价数组
    low: 最低价数组
    curr_index: 当前检查的索引位置
    sigma: 价格变化阈值（百分比）
    last_extreme_index: 上一个极值点的索引
    last_extreme_price: 上一个极值点的价格
    is_last_extreme_bottom: 上一个极值点是否为底部
    
    返回:
    bool: 是否形成顶部
    """
    # 如果上一个极值点不是底部，则不可能形成顶部
    if not is_last_extreme_bottom:
        return False
    
    # 如果当前价格回撤超过了最高价的sigma百分比，则确认形成了顶部
    if data[curr_index] < last_extreme_price - last_extreme_price * sigma:
        return True
    
    return False

def dc_bottom(data: np.array, high: np.array, low: np.array, curr_index: int, sigma: float, last_extreme_index: int, last_extreme_price: float, is_last_extreme_bottom: bool) -> bool:
    """
    检测是否形成了一个底部
    
    参数:
    data: 收盘价数组
    high: 最高价数组
    low: 最低价数组
    curr_index: 当前检查的索引位置
    sigma: 价格变化阈值（百分比）
    last_extreme_index: 上一个极值点的索引
    last_extreme_price: 上一个极值点的价格
    is_last_extreme_bottom: 上一个极值点是否为底部
    
    返回:
    bool: 是否形成底部
    """
    # 如果上一个极值点是底部，则不可能形成底部
    if is_last_extreme_bottom:
        return False
    
    # 如果当前价格反弹超过了最低价的sigma百分比，则确认形成了底部
    if data[curr_index] > last_extreme_price + last_extreme_price * sigma:
        return True
    
    return False

def dc_extremes(close: np.array, high: np.array, low: np.array, sigma: float):
    """
    使用Directional Change算法找出所有极值点
    
    参数:
    close: 收盘价数组
    high: 最高价数组
    low: 最低价数组
    sigma: 价格变化阈值（百分比）
    
    返回:
    tops: 顶部列表，每个元素包含[确认索引, 极值索引, 极值价格]
    bottoms: 底部列表，每个元素包含[确认索引, 极值索引, 极值价格]
    """
    up_zig = True  # 上一个极值是底部，下一个将是顶部
    tmp_max = high[0]  # 当前最高价
    tmp_min = low[0]   # 当前最低价
    tmp_max_i = 0      # 当前最高价的索引
    tmp_min_i = 0      # 当前最低价的索引
    
    tops = []      # 存储顶部
    bottoms = []   # 存储底部
    
    for i in range(len(close)):
        if up_zig:  # 上一个极值是底部，寻找顶部
            if high[i] > tmp_max:
                # 更新最高价
                tmp_max = high[i]
                tmp_max_i = i
            elif dc_top(close, high, low, i, sigma, tmp_max_i, tmp_max, True):
                # 确认形成顶部
                # top[0] = 确认索引
                # top[1] = 顶部索引
                # top[2] = 顶部价格
                top = [i, tmp_max_i, tmp_max]
                tops.append(top)
                
                # 准备寻找下一个底部
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = i
        else:  # 上一个极值是顶部，寻找底部
            if low[i] < tmp_min:
                # 更新最低价
                tmp_min = low[i]
                tmp_min_i = i
            elif dc_bottom(close, high, low, i, sigma, tmp_min_i, tmp_min, False):
                # 确认形成底部
                # bottom[0] = 确认索引
                # bottom[1] = 底部索引
                # bottom[2] = 底部价格
                bottom = [i, tmp_min_i, tmp_min]
                bottoms.append(bottom)
                
                # 准备寻找下一个顶部
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i
    
    return tops, bottoms

def plot_dc_extremes(data, sigma=0.03):
    """
    使用Directional Change算法检测极值点并绘制图表
    
    参数:
    data: 包含OHLC数据的DataFrame
    sigma: 价格变化阈值（百分比）
    """
    # 调用dc_extremes函数检测极值点
    tops, bottoms = dc_extremes(data['Close'].to_numpy(), data['High'].to_numpy(), data['Low'].to_numpy(), sigma)
    
    # 打印检测到的顶部信息
    print(f"检测到 {len(tops)} 个顶部:")
    for i, top in enumerate(tops[:5]):  # 只打印前5个顶部
        conf_date = data.index[top[0]]  # 确认日期
        top_date = data.index[top[1]]   # 顶部日期
        print(f"顶部 {i+1}: 确认日期={conf_date}, 顶部日期={top_date}, 价格={top[2]}")
    
    # 打印检测到的底部信息
    print(f"\n检测到 {len(bottoms)} 个底部:")
    for i, bottom in enumerate(bottoms[:5]):  # 只打印前5个底部
        conf_date = data.index[bottom[0]]  # 确认日期
        bottom_date = data.index[bottom[1]]  # 底部日期
        print(f"底部 {i+1}: 确认日期={conf_date}, 底部日期={bottom_date}, 价格={bottom[2]}")
    
    # 使用plotly创建图表
    fig = go.Figure()
    
    # 添加收盘价线图
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='收盘价',
        line=dict(color='royalblue', width=2)
    ))
    
    # 添加顶部标记
    fig.add_trace(go.Scatter(
        x=[data.index[top[1]] for top in tops],
        y=[top[2] for top in tops],
        mode='markers',
        name='顶部',
        marker=dict(color='rgba(192,0,0,0.7)', size=8, symbol='circle')
    ))
    
    # 添加底部标记
    fig.add_trace(go.Scatter(
        x=[data.index[bottom[1]] for bottom in bottoms],
        y=[bottom[2] for bottom in bottoms],
        mode='markers',
        name='底部',
        marker=dict(color='rgba(78,167,46,0.7)', size=8, symbol='circle')
    ))
    
    # 设置图表布局
    fig.update_layout(
        title={
            'text': f"上证指数价格与Directional Change检测的极值点 (sigma={sigma*100}%)",
            'x': 0.5,  # 标题居中
            'xanchor': 'center'
        },
        xaxis_title="日期",
        yaxis_title="收盘价",
        template="plotly_white",
        width=1200,
        height=600,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        ),
        plot_bgcolor='rgba(240, 245, 250, 1)',
        paper_bgcolor='rgba(240, 245, 250, 1)',
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='white',
        yaxis_gridcolor='white',
        xaxis_gridwidth=1,
        yaxis_gridwidth=1
    )
    
    fig.show()
    
    return tops, bottoms

if __name__ == "__main__":
    # 这里可以添加测试代码
    pass
