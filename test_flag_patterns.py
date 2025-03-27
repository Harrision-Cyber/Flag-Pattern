# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import mplfinance as mpf  # 用于绘制金融图表
from flag_pattern_algorithm import find_flags_pennants_pips, find_flags_pennants_trendline  # 导入旗形和三角旗识别函数
    
# 加载数据
data = pd.read_excel('C:\\Users\\Amber\\Desktop\\2025年策略Task\\PA量化\\数据\\000001.xlsx')
data['date'] = data['日期'].astype('datetime64[s]')  # 将日期列转换为datetime格式
data = data.set_index('date')  # 将日期列设置为索引

# 对价格取对数，使收益率更符合正态分布
# 先将数值列转换为数值类型，确保可以应用对数函数
numeric_columns = ['收盘价(元)', '开盘价(元)', '最高价(元)', '最低价(元)']
for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# 对数值列取对数
data_log = data.copy()
for col in numeric_columns:
    if col in data.columns:
        data_log[col] = np.log(data[col])

dat_slice = data_log['收盘价(元)'].to_numpy()  # 提取对数转换后的收盘价数据

# 定义要测试的窗口大小参数范围（从3到48）
orders = list(range(3, 49))

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
for order in orders:
    # 使用PIP点方法识别旗形和三角旗
    # bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_pips(dat_slice, order)
    # 也可以使用趋势线方法（取消下面的注释即可）
    bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_trendline(dat_slice, order)

    # 创建数据框来存储形态属性和收益
    bull_flag_df = pd.DataFrame()  # 牛市旗形数据框
    bull_pennant_df = pd.DataFrame()  # 牛市三角旗数据框
    bear_flag_df = pd.DataFrame()  # 熊市旗形数据框
    bear_pennant_df = pd.DataFrame()  # 熊市三角旗数据框

    # 设置持有期乘数（持有时间 = 旗帜宽度 * 乘数）
    hold_mult = 1.0  # 默认持有时间等于旗帜宽度
    
    # 处理牛市旗形数据
    for i, flag in enumerate(bull_flags):
        # 记录形态属性
        bull_flag_df.loc[i, 'flag_width'] = flag.flag_width  # 旗帜宽度
        bull_flag_df.loc[i, 'flag_height'] = flag.flag_height  # 旗帜高度
        bull_flag_df.loc[i, 'pole_width'] = flag.pole_width  # 旗杆宽度
        bull_flag_df.loc[i, 'pole_height'] = flag.pole_height  # 旗杆高度
        bull_flag_df.loc[i, 'slope'] = flag.resist_slope  # 阻力线斜率
        
        # 记录关键点位索引
        bull_flag_df.loc[i, 'start_x'] = flag.base_x  # 起始点索引
        bull_flag_df.loc[i, 'end_x'] = flag.tip_x  # 旗杆顶部索引
        bull_flag_df.loc[i, 'conf_x'] = flag.conf_x  # 确认点索引
        
        # 记录日期信息
        bull_flag_df.loc[i, 'start_date'] = data.index[flag.base_x]  # 起始日期
        bull_flag_df.loc[i, 'end_date'] = data.index[flag.tip_x]  # 旗杆顶部日期
        bull_flag_df.loc[i, 'conf_date'] = data.index[flag.conf_x]  # 确认日期

        # 计算持有期收益
        hp = int(flag.flag_width * hold_mult)  # 持有期长度
        if flag.conf_x + hp >= len(data):  # 如果持有期超出数据范围
            bull_flag_df.loc[i, 'return'] = np.nan  # 设置收益为NaN
            bull_flag_df.loc[i, 'exit_date'] = np.nan  # 退出日期为NaN
        else:
            # 计算对数收益率（确认点到持有期结束的价格变化）
            ret = dat_slice[flag.conf_x + hp] - dat_slice[flag.conf_x]
            bull_flag_df.loc[i, 'return'] = ret 
            bull_flag_df.loc[i, 'exit_date'] = data.index[flag.conf_x + hp]  # 退出日期

    # 处理熊市旗形数据
    for i, flag in enumerate(bear_flags):
        # 记录形态属性
        bear_flag_df.loc[i, 'flag_width'] = flag.flag_width  # 旗帜宽度
        bear_flag_df.loc[i, 'flag_height'] = flag.flag_height  # 旗帜高度
        bear_flag_df.loc[i, 'pole_width'] = flag.pole_width  # 旗杆宽度
        bear_flag_df.loc[i, 'pole_height'] = flag.pole_height  # 旗杆高度
        bear_flag_df.loc[i, 'slope'] = flag.support_slope  # 支撑线斜率
        
        # 记录关键点位索引
        bear_flag_df.loc[i, 'start_x'] = flag.base_x  # 起始点索引
        bear_flag_df.loc[i, 'end_x'] = flag.tip_x  # 结束点索引
        bear_flag_df.loc[i, 'conf_x'] = flag.conf_x  # 确认点索引
        
        # 记录日期信息
        bear_flag_df.loc[i, 'start_date'] = data.index[flag.base_x]  # 起始日期
        bear_flag_df.loc[i, 'end_date'] = data.index[flag.tip_x]  # 旗杆顶部日期
        bear_flag_df.loc[i, 'conf_date'] = data.index[flag.conf_x]  # 确认日期

        # 计算持有期收益（注意熊市形态是做空，所以收益取负）
        hp = int(flag.flag_width * hold_mult)  # 持有期长度
        if flag.conf_x + hp >= len(data):  # 如果持有期超出数据范围
            bear_flag_df.loc[i, 'return'] = np.nan  # 设置收益为NaN
            bear_flag_df.loc[i, 'exit_date'] = np.nan  # 退出日期为NaN
        else:
            # 计算对数收益率（确认点到持有期结束的价格变化的负值）
            ret = -1 * (dat_slice[flag.conf_x + hp] - dat_slice[flag.conf_x])
            bear_flag_df.loc[i, 'return'] = ret 
            bear_flag_df.loc[i, 'exit_date'] = data.index[flag.conf_x + hp]  # 退出日期

    # 处理牛市三角旗数据
    for i, pennant in enumerate(bull_pennants):
        # 记录形态属性
        bull_pennant_df.loc[i, 'pennant_width'] = pennant.flag_width  # 三角旗宽度
        bull_pennant_df.loc[i, 'pennant_height'] = pennant.flag_height  # 三角旗高度
        bull_pennant_df.loc[i, 'pole_width'] = pennant.pole_width  # 旗杆宽度
        bull_pennant_df.loc[i, 'pole_height'] = pennant.pole_height  # 旗杆高度
        
        # 记录关键点位索引
        bull_pennant_df.loc[i, 'start_x'] = pennant.base_x  # 起始点索引
        bull_pennant_df.loc[i, 'end_x'] = pennant.tip_x  # 结束点索引
        bull_pennant_df.loc[i, 'conf_x'] = pennant.conf_x  # 确认点索引
        
        # 记录日期信息
        bull_pennant_df.loc[i, 'start_date'] = data.index[pennant.base_x]  # 起始日期
        bull_pennant_df.loc[i, 'end_date'] = data.index[pennant.tip_x]  # 旗杆顶部日期
        bull_pennant_df.loc[i, 'conf_date'] = data.index[pennant.conf_x]  # 确认日期

        # 计算持有期收益
        hp = int(pennant.flag_width * hold_mult)  # 持有期长度
        if pennant.conf_x + hp >= len(data):  # 如果持有期超出数据范围
            bull_pennant_df.loc[i, 'return'] = np.nan  # 设置收益为NaN
            bull_pennant_df.loc[i, 'exit_date'] = np.nan  # 退出日期为NaN
        else:
            # 计算对数收益率（确认点到持有期结束的价格变化）
            ret = dat_slice[pennant.conf_x + hp] - dat_slice[pennant.conf_x]
            bull_pennant_df.loc[i, 'return'] = ret 
            bull_pennant_df.loc[i, 'exit_date'] = data.index[pennant.conf_x + hp]  # 退出日期

    # 处理熊市三角旗数据
    for i, pennant in enumerate(bear_pennants):
        # 记录形态属性
        bear_pennant_df.loc[i, 'pennant_width'] = pennant.flag_width  # 三角旗宽度
        bear_pennant_df.loc[i, 'pennant_height'] = pennant.flag_height  # 三角旗高度
        bear_pennant_df.loc[i, 'pole_width'] = pennant.pole_width  # 旗杆宽度
        bear_pennant_df.loc[i, 'pole_height'] = pennant.pole_height  # 旗杆高度
        
        # 记录关键点位索引
        bear_pennant_df.loc[i, 'start_x'] = pennant.base_x  # 起始点索引
        bear_pennant_df.loc[i, 'end_x'] = pennant.tip_x  # 结束点索引
        bear_pennant_df.loc[i, 'conf_x'] = pennant.conf_x  # 确认点索引
        
        # 记录日期信息
        bear_pennant_df.loc[i, 'start_date'] = data.index[pennant.base_x]  # 起始日期
        bear_pennant_df.loc[i, 'end_date'] = data.index[pennant.tip_x]  # 旗杆顶部日期
        bear_pennant_df.loc[i, 'conf_date'] = data.index[pennant.conf_x]  # 确认日期

        # 计算持有期收益（注意熊市形态是做空，所以收益取负）
        hp = int(pennant.flag_width * hold_mult)  # 持有期长度
        if pennant.conf_x + hp >= len(data):  # 如果持有期超出数据范围
            bear_pennant_df.loc[i, 'return'] = np.nan  # 设置收益为NaN
            bear_pennant_df.loc[i, 'exit_date'] = np.nan  # 退出日期为NaN
        else:
            # 计算对数收益率（确认点到持有期结束的价格变化的负值）
            ret = -1 * (dat_slice[pennant.conf_x + hp] - dat_slice[pennant.conf_x])
            bear_pennant_df.loc[i, 'return'] = ret 
            bear_pennant_df.loc[i, 'exit_date'] = data.index[pennant.conf_x + hp]  # 退出日期

    # 保存每个order参数下的形态详细信息
    pattern_details[order] = {
        'bull_flag': bull_flag_df.copy() if not bull_flag_df.empty else None,
        'bear_flag': bear_flag_df.copy() if not bear_flag_df.empty else None,
        'bull_pennant': bull_pennant_df.copy() if not bull_pennant_df.empty else None,
        'bear_pennant': bear_pennant_df.copy() if not bear_pennant_df.empty else None
    }

    # 计算牛市旗形的统计数据
    if len(bull_flag_df) > 0:  # 如果找到了牛市旗形
        bull_flag_count.append(len(bull_flag_df))  # 记录形态数量
        bull_flag_avg.append(bull_flag_df['return'].mean())  # 计算平均收益
        # 计算胜率（收益为正的比例）
        bull_flag_wr.append(len(bull_flag_df[bull_flag_df['return'] > 0]) / len(bull_flag_df))
        bull_flag_total_ret.append(bull_flag_df['return'].sum())  # 计算总收益
    else:  # 如果没有找到牛市旗形
        bull_flag_count.append(0)  # 形态数量为0
        bull_flag_avg.append(np.nan)  # 平均收益为NaN
        bull_flag_wr.append(np.nan)  # 胜率为NaN
        bull_flag_total_ret.append(0)  # 总收益为0
    
    # 计算熊市旗形的统计数据
    if len(bear_flag_df) > 0:  # 如果找到了熊市旗形
        bear_flag_count.append(len(bear_flag_df))  # 记录形态数量
        bear_flag_avg.append(bear_flag_df['return'].mean())  # 计算平均收益
        # 计算胜率（收益为正的比例）
        bear_flag_wr.append(len(bear_flag_df[bear_flag_df['return'] > 0]) / len(bear_flag_df))
        bear_flag_total_ret.append(bear_flag_df['return'].sum())  # 计算总收益
    else:  # 如果没有找到熊市旗形
        bear_flag_count.append(0)  # 形态数量为0
        bear_flag_avg.append(np.nan)  # 平均收益为NaN
        bear_flag_wr.append(np.nan)  # 胜率为NaN
        bear_flag_total_ret.append(0)  # 总收益为0
    
    # 计算牛市三角旗的统计数据
    if len(bull_pennant_df) > 0:  # 如果找到了牛市三角旗
        bull_pennant_count.append(len(bull_pennant_df))  # 记录形态数量
        bull_pennant_avg.append(bull_pennant_df['return'].mean())  # 计算平均收益
        # 计算胜率（收益为正的比例）
        bull_pennant_wr.append(len(bull_pennant_df[bull_pennant_df['return'] > 0]) / len(bull_pennant_df))
        bull_pennant_total_ret.append(bull_pennant_df['return'].sum())  # 计算总收益
    else:  # 如果没有找到牛市三角旗
        bull_pennant_count.append(0)  # 形态数量为0
        bull_pennant_avg.append(np.nan)  # 平均收益为NaN
        bull_pennant_wr.append(np.nan)  # 胜率为NaN
        bull_pennant_total_ret.append(0)  # 总收益为0
    
    # 计算熊市三角旗的统计数据
    if len(bear_pennant_df) > 0:  # 如果找到了熊市三角旗
        bear_pennant_count.append(len(bear_pennant_df))  # 记录形态数量
        bear_pennant_avg.append(bear_pennant_df['return'].mean())  # 计算平均收益
        # 计算胜率（收益为正的比例）
        bear_pennant_wr.append(len(bear_pennant_df[bear_pennant_df['return'] > 0]) / len(bear_pennant_df))
        bear_pennant_total_ret.append(bear_pennant_df['return'].sum())  # 计算总收益
    else:  # 如果没有找到熊市三角旗
        bear_pennant_count.append(0)  # 形态数量为0
        bear_pennant_avg.append(np.nan)  # 平均收益为NaN
        bear_pennant_wr.append(np.nan)  # 胜率为NaN
        bear_pennant_total_ret.append(0)  # 总收益为0
    
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

# 将结果保存到Excel文件
results_df.to_excel('pattern_performance_summary.xlsx')

# 将每个order参数下的形态详细信息保存到Excel文件
with pd.ExcelWriter('pattern_details.xlsx') as writer:
    # 遍历每个order参数
    for order in orders:
        # 获取当前order参数下的形态详细信息
        details = pattern_details[order]
        
        # 保存牛市旗形详细信息
        if details['bull_flag'] is not None:
            details['bull_flag'].to_excel(writer, sheet_name=f'Bull_Flag_Order_{order}')
        
        # 保存熊市旗形详细信息
        if details['bear_flag'] is not None:
            details['bear_flag'].to_excel(writer, sheet_name=f'Bear_Flag_Order_{order}')
        
        # 保存牛市三角旗详细信息
        if details['bull_pennant'] is not None:
            details['bull_pennant'].to_excel(writer, sheet_name=f'Bull_Pennant_Order_{order}')
        
        # 保存熊市三角旗详细信息
        if details['bear_pennant'] is not None:
            details['bear_pennant'].to_excel(writer, sheet_name=f'Bear_Pennant_Order_{order}')

# 绘制牛市旗形的性能图表
plt.style.use('dark_background')  # 使用深色背景
fig, ax = plt.subplots(2, 2)  # 创建2x2的子图
fig.suptitle("Bull Flag Performance", fontsize=20)  # 设置总标题

# 绘制牛市旗形的四个指标
results_df['bull_flag_count'].plot.bar(ax=ax[0,0])  # 形态数量柱状图
results_df['bull_flag_avg'].plot.bar(ax=ax[0,1], color='yellow')  # 平均收益柱状图
results_df['bull_flag_total'].plot.bar(ax=ax[1,0], color='green')  # 总收益柱状图
results_df['bull_flag_wr'].plot.bar(ax=ax[1,1], color='orange')  # 胜率柱状图

# 添加参考线
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')  # 平均收益为0的参考线
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')  # 总收益为0的参考线
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')  # 胜率为50%的参考线

# 设置子图标题和标签
ax[0,0].set_title('Number of Patterns Found')  # 形态数量子图标题
ax[0,0].set_xlabel('Order Parameter')  # x轴标签
ax[0,0].set_ylabel('Number of Patterns')  # y轴标签
ax[0,1].set_title('Average Pattern Return')  # 平均收益子图标题
ax[0,1].set_xlabel('Order Parameter')  # x轴标签
ax[0,1].set_ylabel('Average Log Return')  # y轴标签
ax[1,0].set_title('Sum of Returns')  # 总收益子图标题
ax[1,0].set_xlabel('Order Parameter')  # x轴标签
ax[1,0].set_ylabel('Total Log Return')  # y轴标签
ax[1,1].set_title('Win Rate')  # 胜率子图标题
ax[1,1].set_xlabel('Order Parameter')  # x轴标签
ax[1,1].set_ylabel('Win Rate Percentage')  # y轴标签

plt.show()  # 显示图表

# 绘制熊市旗形的性能图表
fig, ax = plt.subplots(2, 2)  # 创建2x2的子图
fig.suptitle("Bear Flag Performance", fontsize=20)  # 设置总标题

# 绘制熊市旗形的四个指标
results_df['bear_flag_count'].plot.bar(ax=ax[0,0])  # 形态数量柱状图
results_df['bear_flag_avg'].plot.bar(ax=ax[0,1], color='yellow')  # 平均收益柱状图
results_df['bear_flag_total'].plot.bar(ax=ax[1,0], color='green')  # 总收益柱状图
results_df['bear_flag_wr'].plot.bar(ax=ax[1,1], color='orange')  # 胜率柱状图

# 添加参考线
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')  # 平均收益为0的参考线
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')  # 总收益为0的参考线
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')  # 胜率为50%的参考线

# 设置子图标题和标签
ax[0,0].set_title('Number of Patterns Found')  # 形态数量子图标题
ax[0,0].set_xlabel('Order Parameter')  # x轴标签
ax[0,0].set_ylabel('Number of Patterns')  # y轴标签
ax[0,1].set_title('Average Pattern Return')  # 平均收益子图标题
ax[0,1].set_xlabel('Order Parameter')  # x轴标签
ax[0,1].set_ylabel('Average Log Return')  # y轴标签
ax[1,0].set_title('Sum of Returns')  # 总收益子图标题
ax[1,0].set_xlabel('Order Parameter')  # x轴标签
ax[1,0].set_ylabel('Total Log Return')  # y轴标签
ax[1,1].set_title('Win Rate')  # 胜率子图标题
ax[1,1].set_xlabel('Order Parameter')  # x轴标签
ax[1,1].set_ylabel('Win Rate Percentage')  # y轴标签

plt.show()  # 显示图表

# 绘制牛市三角旗的性能图表
fig, ax = plt.subplots(2, 2)  # 创建2x2的子图
fig.suptitle("Bull Pennant Performance", fontsize=20)  # 设置总标题

# 绘制牛市三角旗的四个指标
results_df['bull_pennant_count'].plot.bar(ax=ax[0,0])  # 形态数量柱状图
results_df['bull_pennant_avg'].plot.bar(ax=ax[0,1], color='yellow')  # 平均收益柱状图
results_df['bull_pennant_total'].plot.bar(ax=ax[1,0], color='green')  # 总收益柱状图
results_df['bull_pennant_wr'].plot.bar(ax=ax[1,1], color='orange')  # 胜率柱状图

# 添加参考线
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')  # 平均收益为0的参考线
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')  # 总收益为0的参考线
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')  # 胜率为50%的参考线

# 设置子图标题和标签
ax[0,0].set_title('Number of Patterns Found')  # 形态数量子图标题
ax[0,0].set_xlabel('Order Parameter')  # x轴标签
ax[0,0].set_ylabel('Number of Patterns')  # y轴标签
ax[0,1].set_title('Average Pattern Return')  # 平均收益子图标题
ax[0,1].set_xlabel('Order Parameter')  # x轴标签
ax[0,1].set_ylabel('Average Log Return')  # y轴标签
ax[1,0].set_title('Sum of Returns')  # 总收益子图标题
ax[1,0].set_xlabel('Order Parameter')  # x轴标签
ax[1,0].set_ylabel('Total Log Return')  # y轴标签
ax[1,1].set_title('Win Rate')  # 胜率子图标题
ax[1,1].set_xlabel('Order Parameter')  # x轴标签
ax[1,1].set_ylabel('Win Rate Percentage')  # y轴标签

plt.show()  # 显示图表

# 绘制熊市三角旗的性能图表
fig, ax = plt.subplots(2, 2)  # 创建2x2的子图
fig.suptitle("Bear Pennant Performance", fontsize=20)  # 设置总标题

# 绘制熊市三角旗的四个指标
results_df['bear_pennant_count'].plot.bar(ax=ax[0,0])  # 形态数量柱状图
results_df['bear_pennant_avg'].plot.bar(ax=ax[0,1], color='yellow')  # 平均收益柱状图
results_df['bear_pennant_total'].plot.bar(ax=ax[1,0], color='green')  # 总收益柱状图
results_df['bear_pennant_wr'].plot.bar(ax=ax[1,1], color='orange')  # 胜率柱状图

# 添加参考线
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')  # 平均收益为0的参考线
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')  # 总收益为0的参考线
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')  # 胜率为50%的参考线

# 设置子图标题和标签
ax[0,0].set_title('Number of Patterns Found')  # 形态数量子图标题
ax[0,0].set_xlabel('Order Parameter')  # x轴标签
ax[0,0].set_ylabel('Number of Patterns')  # y轴标签
ax[0,1].set_title('Average Pattern Return')  # 平均收益子图标题
ax[0,1].set_xlabel('Order Parameter')  # x轴标签
ax[0,1].set_ylabel('Average Log Return')  # y轴标签
ax[1,0].set_title('Sum of Returns')  # 总收益子图标题
ax[1,0].set_xlabel('Order Parameter')  # x轴标签
ax[1,0].set_ylabel('Total Log Return')  # y轴标签
ax[1,1].set_title('Win Rate')  # 胜率子图标题
ax[1,1].set_xlabel('Order Parameter')  # x轴标签
ax[1,1].set_ylabel('Win Rate Percentage')  # y轴标签

plt.show()  # 显示图表