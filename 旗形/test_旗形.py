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
    
    # 确定数据显示范围
    start_i = max(0, min_base_x - pad)
    end_i = min(len(candle_data), max_conf_x + 1 + pad)
    dat = candle_data.iloc[start_i:end_i]
    
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
fig.show()



# 主程序（当直接运行此文件时执行）
if __name__ == '__main__':
    # 加载数据
  
    # 对价格取对数
    # 对除Change列外的所有列取对数
    # 将日期索引转换为DatetimeIndex格式
    data.index = pd.to_datetime(data.index)
    
    # 对价格数据取对数
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

        # 绘制牛市旗形
        # plot_flag(data, flag)

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

        # 绘制熊市旗形
        # plot_flag(data, flag)

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
        # plot_flag(data, pennant)

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
        # plot_flag(data, pennant)
