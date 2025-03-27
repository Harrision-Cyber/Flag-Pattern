# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import mplfinance as mpf  # 用于绘制金融图表
from important_point_algorithm import rw_top, rw_bottom, rw_extremes,directional_change, get_extremes, find_pips # 导入重要点算法函数
from flag_pattern_algorithm import find_flags_pennants_pips, find_flags_pennants_trendline, plot_flag # 导入旗形和三角旗识别函数
# from get_stock_data import get_stock_data # 导入获取股票数据函数
import plotly.graph_objects as go

# data = get_stock_data("000001.SH", "1995-01-01", "2025-02-28")
# 从Excel文件中读取上证指数数据
data = pd.read_excel('上证指数数据.xlsx')
data = data.set_index('Date')  # 将Date列设置为索引

# 对价格取对数
# 对除Change列外的所有列取对数
# 将日期索引转换为DatetimeIndex格式
data.index = pd.to_datetime(data.index).copy()

# 对价格数据取对数
data.loc[:, data.columns != 'Change'] = np.log(data.loc[:, data.columns != 'Change']).copy()

# 提取收盘价数据
dat_slice = data['Close'].to_numpy().copy()
# 识别旗形和三角旗
bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_pips(dat_slice, 10)  # 使用PIP点方法
#bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(dat_slice, 10)  # 使用趋势线方法

# 创建数据框来存储形态属性
bull_flag_df = pd.DataFrame()
bull_pennant_df = pd.DataFrame()
bear_flag_df = pd.DataFrame()
bear_pennant_df = pd.DataFrame()

# 将形态数据组织到数据框中
hold_mult = 1.0  # 持有期乘数（持有时间 = 旗帜宽度 * 乘数）

print('\n====================单独绘制所有图形===========================\n')
# 打印牛市旗形形态统计信息
print("\n=== 牛市旗形形态统计 ===")
print(f"共发现牛市旗形数量: {len(bull_flags)}")

# 处理牛市旗形

# 分析这段代码:
# enumerate是Python内置函数,用于遍历序列时同时获取索引和值
# bull_flags是一个列表,包含了所有牛市旗形对象
# i是索引号(从0开始),flag是每个旗形对象
# 这是一个典型的Python遍历模式
# </claudeThinking>
# 这行代码的意思是:遍历bull_flags列表中的所有牛市旗形对象,其中i是每个旗形的序号(从0开始),flag是对应的旗形对象。enumerate()函数让我们能同时获取到索引和值。
for i, flag in enumerate(bull_flags):
    # 记录形态属性
    # 这是pandas DataFrame的赋值语法,用于将 flag.flag_width 的值存储到 bull_flag_df 数据框的第 i 行、flag_width 列的位置。.loc[] 是pandas用来精确定位行列位置的索引器。
    # 这里flag.表示访问FlagPattern类实例flag中的属性。    
    bull_flag_df.loc[i, 'flag_width'] = flag.flag_width
    bull_flag_df.loc[i, 'flag_height'] = flag.flag_height
    bull_flag_df.loc[i, 'pole_width'] = flag.pole_width
    bull_flag_df.loc[i, 'pole_height'] = flag.pole_height
    bull_flag_df.loc[i, 'slope'] = flag.resist_slope

    # 计算持有期收益
    # 持有期长度 = 旗形宽度 * 持有期乘数
    hp = int(flag.flag_width * hold_mult)  # hp是holding period(持有期)的缩写
    
    # 检查持有期结束点是否超出数据范围
    if flag.conf_x + hp >= len(data):  # 如果确认点位置加上持有期超过了数据长度
        bull_flag_df.loc[i, 'return'] = np.nan  # 将收益率设为缺失值NaN
    else:
        # 计算收益率 = 持有期结束时的价格 - 确认点的价格
        # dat_slice[flag.conf_x + hp]是持有期结束时的价格
        # dat_slice[flag.conf_x]是确认点的价格
        ret = dat_slice[flag.conf_x + hp] - dat_slice[flag.conf_x]
        bull_flag_df.loc[i, 'return'] = ret  # 将计算得到的收益率存入数据框

    # 绘制牛市旗形
    plot_flag(data, flag)

# 计算牛市旗形的胜率和平均收益率
if len(bull_flags) > 0:
    # 计算胜率 - 正收益的比例
    win_rate = (bull_flag_df['return'] > 0).mean() * 100
    # 计算平均收益率
    avg_return = bull_flag_df['return'].mean()
    
    print("\n牛市旗形绩效统计:")
    print(f"胜率: {win_rate:.2f}%")
    print(f"平均收益率: {avg_return:.4f}")

# if len(bull_flags) > 0:
#     print("\n旗形特征统计:")
#     print(f"旗形宽度均值: {bull_flag_df['flag_width'].mean():.2f}")
#     print(f"旗形高度均值: {bull_flag_df['flag_height'].mean():.2f}")
#     print(f"旗杆宽度均值: {bull_flag_df['pole_width'].mean():.2f}")
#     print(f"旗杆高度均值: {bull_flag_df['pole_height'].mean():.2f}")
#     print(f"旗形斜率均值: {bull_flag_df['slope'].mean():.4f}")
    # print(f"\n持有期收益均值: {bull_flag_df['return'].mean():.4f}")
    # print(f"持有期收益标准差: {bull_flag_df['return'].std():.4f}")

# 处理熊市旗形
# 打印熊市旗形形态统计信息
print("\n=== 熊市旗形形态统计 ===")
print(f"共发现熊市旗形数量: {len(bear_flags)}")



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

    # 绘制熊市旗形
    plot_flag(data, flag)

# if len(bear_flags) > 0:
#     print("\n旗形特征统计:")
#     print(f"旗形宽度均值: {bear_flag_df['flag_width'].mean():.2f}")
#     print(f"旗形高度均值: {bear_flag_df['flag_height'].mean():.2f}")
#     print(f"旗杆宽度均值: {bear_flag_df['pole_width'].mean():.2f}")
#     print(f"旗杆高度均值: {bear_flag_df['pole_height'].mean():.2f}")
#     print(f"旗形斜率均值: {bear_flag_df['slope'].mean():.4f}")
    # print(f"\n持有期收益均值: {bear_flag_df['return'].mean():.4f}")
    # print(f"持有期收益标准差: {bear_flag_df['return'].std():.4f}")

# 计算熊市旗形的胜率和平均收益率
if len(bear_flags) > 0:
    # 计算胜率 - 正收益的比例
    win_rate = (bear_flag_df['return'] > 0).mean() * 100
    # 计算平均收益率
    avg_return = bear_flag_df['return'].mean()
    
    print("\n熊市旗形绩效统计:")
    print(f"胜率: {win_rate:.2f}%")
    print(f"平均收益率: {avg_return:.4f}")


# 打印牛市三角旗形态统计信息
print("\n=== 牛市三角旗形态统计 ===")
print(f"共发现牛市三角旗数量: {len(bull_pennants)}")



# 处理牛市三角旗
for i, pennant in enumerate(bull_pennants):
    # 记录形态属性
    bull_pennant_df.loc[i, 'pennant_width'] = pennant.flag_width
    bull_pennant_df.loc[i, 'pennant_height'] = pennant.flag_height
    bull_pennant_df.loc[i, 'pole_width'] = pennant.pole_width
    bull_pennant_df.loc[i, 'pole_height'] = pennant.pole_height

    # 计算持有期收益
    hp = int(pennant.flag_width * hold_mult)
    if pennant.conf_x + hp >= len(data):
        bull_pennant_df.loc[i, 'return'] = np.nan
    else:
        ret = dat_slice[pennant.conf_x + hp] - dat_slice[pennant.conf_x]
        bull_pennant_df.loc[i, 'return'] = ret 

    # 绘制牛市三角旗
    plot_flag(data, pennant)

# if len(bull_pennants) > 0:
#     print("\n三角旗特征统计:")
#     print(f"三角旗宽度均值: {bull_pennant_df['pennant_width'].mean():.2f}")
#     print(f"三角旗高度均值: {bull_pennant_df['pennant_height'].mean():.2f}")
#     print(f"旗杆宽度均值: {bull_pennant_df['pole_width'].mean():.2f}")
#     print(f"旗杆高度均值: {bull_pennant_df['pole_height'].mean():.2f}")
    # print(f"\n持有期收益均值: {bull_pennant_df['return'].mean():.4f}")
    # print(f"持有期收益标准差: {bull_pennant_df['return'].std():.4f}")

# 计算牛市三角旗的胜率和平均收益率
if len(bull_pennants) > 0:
    # 计算胜率 - 正收益的比例
    win_rate = (bull_pennant_df['return'] > 0).mean() * 100
    # 计算平均收益率
    avg_return = bull_pennant_df['return'].mean()
    
    print("\n牛市三角旗形绩效统计:")
    print(f"胜率: {win_rate:.2f}%")
    print(f"平均收益率: {avg_return:.4f}")


# 打印熊市三角旗形态统计信息
print("\n=== 熊市三角旗形态统计 ===")
print(f"共发现熊市三角旗数量: {len(bear_pennants)}")



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

    # 绘制熊市三角旗
    plot_flag(data, pennant)

# if len(bear_pennants) > 0:
#     print("\n三角旗特征统计:")
#     print(f"三角旗宽度均值: {bear_pennant_df['pennant_width'].mean():.2f}")
#     print(f"三角旗高度均值: {bear_pennant_df['pennant_height'].mean():.2f}")
#     print(f"旗杆宽度均值: {bear_pennant_df['pole_width'].mean():.2f}")
#     print(f"旗杆高度均值: {bear_pennant_df['pole_height'].mean():.2f}")
    # print(f"\n持有期收益均值: {bear_pennant_df['return'].mean():.4f}")
    # print(f"持有期收益标准差: {bear_pennant_df['return'].std():.4f}")

# 计算熊市三角旗的胜率和平均收益率
if len(bear_pennants) > 0:
    # 计算胜率 - 正收益的比例
    win_rate = (bear_pennant_df['return'] > 0).mean() * 100
    # 计算平均收益率
    avg_return = bear_pennant_df['return'].mean()
    
    print("\n熊市三角旗形绩效统计:")
    print(f"胜率: {win_rate:.2f}%")
    print(f"平均收益率: {avg_return:.4f}")



print('\n====================将旗形绘制到同一坐标系下===========================\n')

def plot_all_flags(candle_data: pd.DataFrame, patterns_list, pattern_names, pad=2):
    """
    在同一个图中绘制所有旗形
    
    参数:
    candle_data: pd.DataFrame - K线数据
    patterns_list: list - 包含所有形态的列表 [bull_flags, bear_flags, bull_pennants, bear_pennants]
    pattern_names: list - 形态名称列表
    pad: int - 图表两侧的额外空间
    """
    # 创建图表对象
    fig = go.Figure()
    
    # 找出所有形态的最早和最晚时间点
    min_base_x = float('inf')
    max_conf_x = 0
    for patterns in patterns_list:
        for pattern in patterns:
            min_base_x = min(min_base_x, pattern.base_x)
            max_conf_x = max(max_conf_x, pattern.conf_x)
    
    # 使用原始数据
    dat = candle_data
    
    # 添加K线图
    fig.add_trace(go.Candlestick(
        x=dat.index,
        open=dat['Open'],
        high=dat['High'],
        low=dat['Low'],
        close=dat['Close'],
        name='K线'
    ))
    
    # 定义不同形态的颜色
    colors = ['red', 'green', 'blue', 'orange']
    
    # 为每种形态添加线条
    for patterns, pattern_name, color in zip(patterns_list, pattern_names, colors):
        # 用于存储所有同类型的线条
        pole_lines = []
        resist_lines = []
        support_lines = []
        key_points = []
        
        for pattern in patterns:
            # 获取相对于显示范围的索引
            base_idx = candle_data.index[pattern.base_x]
            tip_idx = candle_data.index[pattern.tip_x]
            conf_idx = candle_data.index[pattern.conf_x]
            
            # 收集旗杆线数据
            pole_lines.extend([
                [base_idx, pattern.base_y],
                [tip_idx, pattern.tip_y],
                [None, None]  # 用于分隔不同的线段
            ])
            
            # 收集阻力线数据
            resist_y = [pattern.resist_intercept, 
                       pattern.resist_intercept + pattern.resist_slope * pattern.flag_width]
            resist_lines.extend([
                [tip_idx, resist_y[0]],
                [conf_idx, resist_y[1]],
                [None, None]
            ])
            
            # 收集支撑线数据
            support_y = [pattern.support_intercept,
                        pattern.support_intercept + pattern.support_slope * pattern.flag_width]
            support_lines.extend([
                [tip_idx, support_y[0]],
                [conf_idx, support_y[1]],
                [None, None]
            ])
            
            # 收集关键点数据
            key_points.extend([
                [base_idx, pattern.base_y],
                [tip_idx, pattern.tip_y],
                [conf_idx, pattern.conf_y],
                [None, None]
            ])
        
        # 添加所有旗杆线
        if pole_lines:
            x_poles, y_poles = zip(*pole_lines)
            fig.add_trace(go.Scatter(
                x=x_poles, y=y_poles,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'{pattern_name}-旗杆'
            ))
        
        # 添加所有阻力线
        if resist_lines:
            x_resist, y_resist = zip(*resist_lines)
            fig.add_trace(go.Scatter(
                x=x_resist, y=y_resist,
                mode='lines',
                line=dict(color=color, dash='dash'),
                name=f'{pattern_name}-趋势线'
            ))
        
        # 添加所有支撑线
        if support_lines:
            x_support, y_support = zip(*support_lines)
            fig.add_trace(go.Scatter(
                x=x_support, y=y_support,
                mode='lines',
                line=dict(color=color, dash='dash'),
                showlegend=False  # 不显示支撑线的图例（与阻力线共用一个）
            ))
        
        # 添加所有关键点
        if key_points:
            x_points, y_points = zip(*key_points)
            fig.add_trace(go.Scatter(
                x=x_points, y=y_points,
                mode='markers',
                marker=dict(size=8, color=color),
                name=f'{pattern_name}-关键点'
            ))
    
    # 更新图表布局
    fig.update_layout(
        template='plotly',
        xaxis_title='日期',
        yaxis_title='价格',
        showlegend=True,
        height=800,
        title='旗形与三角旗形态识别'
    )
    
    # 添加范围滑块
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig

# 使用示例：
pattern_names = ['牛市旗形', '熊市旗形', '牛市三角旗', '熊市三角旗']
patterns_list = [bull_flags, bear_flags, bull_pennants, bear_pennants]


# 绘制图形
fig = plot_all_flags(data, patterns_list, pattern_names)
# fig.show()

# 计算各种形态的数量、胜率和平均收益率
def calculate_pattern_statistics(patterns_list, pattern_names):
    """
    计算各种形态的数量、胜率和平均收益率
    
    参数:
    patterns_list: list - 包含所有形态的列表 [bull_flags, bear_flags, bull_pennants, bear_pennants]
    pattern_names: list - 形态名称列表
    
    返回:
    stats_df: pd.DataFrame - 包含各种形态统计信息的数据框
    """
    # 初始化结果存储
    stats = {
        '形态': pattern_names,
        '数量': [],
        '胜率': [],
        '平均收益率': []
    }
    
    # 对应的数据框列表
    dfs = [bull_flag_df, bear_flag_df, bull_pennant_df, bear_pennant_df]
    
    # 计算每种形态的统计数据
    for i, patterns in enumerate(patterns_list):
        # 计算数量
        count = len(patterns)
        stats['数量'].append(count)
        
        # 计算胜率和平均收益率
        if count > 0:
            # 计算胜率 - 正收益的比例
            win_rate = (dfs[i]['return'] > 0).mean() * 100
            # 计算平均收益率
            avg_return = dfs[i]['return'].mean()
            
            stats['胜率'].append(f"{win_rate:.2f}%")
            stats['平均收益率'].append(f"{avg_return:.4f}")
        else:
            stats['胜率'].append("0.00%")
            stats['平均收益率'].append("0.00")
    
    # 创建数据框
    stats_df = pd.DataFrame(stats)
    
    return stats_df

# 计算并显示统计信息
stats_df = calculate_pattern_statistics(patterns_list, pattern_names)
print("\n旗形与三角旗形态统计信息:")
print(stats_df.to_string(index=False))
print("\n")

# 保存为HTML文件,可以在浏览器中打开查看
fig.write_html("将旗形绘制到同一坐标系下.html")

# 如果想要保存为图片格式
fig.write_image("将旗形绘制到同一坐标系下.png")

print("将旗形绘制到同一坐标系下.html和将旗形绘制到同一坐标系下.png")


print('\n====================将旗形绘制到同一坐标系下,只使用收盘价===========================\n')

def plot_all_flags(candle_data: pd.DataFrame, patterns_list, pattern_names, pad=2):
    """
    在同一个图中绘制所有旗形,只使用收盘价
    
    参数:
    candle_data: pd.DataFrame - K线数据
    patterns_list: list - 包含所有形态的列表 [bull_flags, bear_flags, bull_pennants, bear_pennants]
    pattern_names: list - 形态名称列表
    pad: int - 图表两侧的额外空间
    """
    # 创建图表对象
    fig = go.Figure()
    
    # 找出所有形态的最早和最晚时间点
    min_base_x = float('inf')
    max_conf_x = 0
    for patterns in patterns_list:
        for pattern in patterns:
            min_base_x = min(min_base_x, pattern.base_x)
            max_conf_x = max(max_conf_x, pattern.conf_x)
    
    # 使用原始数据
    dat = candle_data
    
    # 添加收盘价线图
    fig.add_trace(go.Scatter(
        x=dat.index,
        y=dat['Close'],
        mode='lines',
        name='收盘价',
        line=dict(color='black', width=1)
    ))
    
    # 定义不同形态的颜色
    colors = ['red', 'green', 'blue', 'orange']
    
    # 为每种形态添加线条
    for patterns, pattern_name, color in zip(patterns_list, pattern_names, colors):
        # 用于存储所有同类型的线条
        pole_lines = []
        resist_lines = []
        support_lines = []
        key_points = []
        
        for pattern in patterns:
            # 获取相对于显示范围的索引
            base_idx = candle_data.index[pattern.base_x]
            tip_idx = candle_data.index[pattern.tip_x]
            conf_idx = candle_data.index[pattern.conf_x]
            
            # 收集旗杆线数据
            pole_lines.extend([
                [base_idx, pattern.base_y],
                [tip_idx, pattern.tip_y],
                [None, None]  # 用于分隔不同的线段
            ])
            
            # 收集阻力线数据
            resist_y = [pattern.resist_intercept, 
                       pattern.resist_intercept + pattern.resist_slope * pattern.flag_width]
            resist_lines.extend([
                [tip_idx, resist_y[0]],
                [conf_idx, resist_y[1]],
                [None, None]
            ])
            
            # 收集支撑线数据
            support_y = [pattern.support_intercept,
                        pattern.support_intercept + pattern.support_slope * pattern.flag_width]
            support_lines.extend([
                [tip_idx, support_y[0]],
                [conf_idx, support_y[1]],
                [None, None]
            ])
            
            # 收集关键点数据
            key_points.extend([
                [base_idx, pattern.base_y],
                [tip_idx, pattern.tip_y],
                [conf_idx, pattern.conf_y],
                [None, None]
            ])
        
        # 添加所有旗杆线
        if pole_lines:
            x_poles, y_poles = zip(*pole_lines)
            fig.add_trace(go.Scatter(
                x=x_poles, y=y_poles,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'{pattern_name}-旗杆'
            ))
        
        # 添加所有阻力线
        if resist_lines:
            x_resist, y_resist = zip(*resist_lines)
            fig.add_trace(go.Scatter(
                x=x_resist, y=y_resist,
                mode='lines',
                line=dict(color=color, dash='dash'),
                name=f'{pattern_name}-趋势线'
            ))
        
        # 添加所有支撑线
        if support_lines:
            x_support, y_support = zip(*support_lines)
            fig.add_trace(go.Scatter(
                x=x_support, y=y_support,
                mode='lines',
                line=dict(color=color, dash='dash'),
                showlegend=False  # 不显示支撑线的图例（与阻力线共用一个）
            ))
        
        # 添加所有关键点
        if key_points:
            x_points, y_points = zip(*key_points)
            fig.add_trace(go.Scatter(
                x=x_points, y=y_points,
                mode='markers',
                marker=dict(size=8, color=color),
                name=f'{pattern_name}-关键点'
            ))
    
    # 更新图表布局
    fig.update_layout(
        template='plotly',
        xaxis_title='日期',
        yaxis_title='价格',
        showlegend=True,
        height=800,
        title='旗形与三角旗形态识别'
    )
    
    # 添加范围滑块
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig

# 使用示例：
pattern_names = ['牛市旗形', '熊市旗形', '牛市三角旗', '熊市三角旗']
patterns_list = [bull_flags, bear_flags, bull_pennants, bear_pennants]

# 绘制图形
fig = plot_all_flags(data, patterns_list, pattern_names)
# fig.show()

# 计算并显示统计信息
stats_df = calculate_pattern_statistics(patterns_list, pattern_names)
print("\n旗形与三角旗形态统计信息:")
print(stats_df.to_string(index=False))
print("\n")

# 保存为HTML文件,可以在浏览器中打开查看
fig.write_html("将旗形绘制到同一坐标系下[收盘价].html")

# 如果想要保存为图片格式
fig.write_image("将旗形绘制到同一坐标系下[收盘价].png")

print("将旗形绘制到同一坐标系下[收盘价].html和将旗形绘制到同一坐标系下[收盘价].png")