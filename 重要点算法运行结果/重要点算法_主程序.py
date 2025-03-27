# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import mplfinance as mpf  # 用于绘制金融图表
from important_point_algorithm import rw_top, rw_bottom, rw_extremes,directional_change, get_extremes, find_pips # 导入重要点算法函数
# from get_stock_data import get_stock_data # 导入获取股票数据函数
import plotly.graph_objects as go

# data = get_stock_data("000001.SH", "1995-01-01", "2025-02-28")
# 从Excel文件中读取上证指数数据
data = pd.read_excel('上证指数数据.xlsx')
data = data.set_index('Date')  # 将Date列设置为索引


print('\n====================1.Rolling Window 算法===========================\n')

# 设置窗口大小参数order=10，这意味着每个窗口包含21个点（中心点前后各10个点）
order = 10
# 调用rw_extremes函数检测极值点
tops, bottoms = rw_extremes(data['Close'].to_numpy(), order)

# 打印检测到的顶部信息
print(f"检测到 {len(tops)} 个顶部:")
for i, top in enumerate(tops[:5]):  # 只打印前5个顶部
    conf_date = data.index[top[0]]  # 确认日期
    top_date = data.index[top[1]]   # 顶部日期
    # print(f"顶部 {i+1}: 确认日期={conf_date}, 顶部日期={top_date}, 价格={top[2]}")

# 打印检测到的底部信息
print(f"\n检测到 {len(bottoms)} 个底部:")
for i, bottom in enumerate(bottoms[:5]):  # 只打印前5个底部
    conf_date = data.index[bottom[0]]  # 确认日期
    bottom_date = data.index[bottom[1]]  # 底部日期
    # print(f"底部 {i+1}: 确认日期={conf_date}, 底部日期={bottom_date}, 价格={bottom[2]}")

# 创建一个DataFrame来存储顶部和底部点的信息
tops_df = pd.DataFrame(tops, columns=['确认索引', '顶部索引', '价格'])
bottoms_df = pd.DataFrame(bottoms, columns=['确认索引', '底部索引', '价格'])

# 添加日期信息
tops_df['确认日期'] = tops_df['确认索引'].apply(lambda x: data.index[x])
tops_df['顶部日期'] = tops_df['顶部索引'].apply(lambda x: data.index[x])
bottoms_df['确认日期'] = bottoms_df['确认索引'].apply(lambda x: data.index[x])
bottoms_df['底部日期'] = bottoms_df['底部索引'].apply(lambda x: data.index[x])

# 重新排列列顺序
tops_df = tops_df[['确认日期', '顶部日期', '价格', '确认索引', '顶部索引']]
bottoms_df = bottoms_df[['确认日期', '底部日期', '价格', '确认索引', '底部索引']]

# # 创建Excel写入器
# with pd.ExcelWriter('【Rolling Window】极值点数据.xlsx') as writer:
#     tops_df.to_excel(writer, sheet_name='顶部点', index=False)
#     bottoms_df.to_excel(writer, sheet_name='底部点', index=False)

# print("极值点数据已保存到'【Rolling Window】极值点数据.xlsx'文件中")

# 创建图表
# 使用plotly创建图表
fig = go.Figure()

# 添加收盘价线图
fig.add_trace(go.Scatter(
    x=data.index,  # data.index包含了DataFrame的索引,在这里是日期数据,用作x轴的时间坐标
    y=data['Close'],  # data['Close']是DataFrame的'Close'列,包含了股票的收盘价数据,用作y轴的数值坐标
    mode='lines',  # 设置绘图模式为'lines'表示用线条连接各个数据点
    name='收盘价',  # 设置图例名称为'收盘价'
    line=dict(width=2)  # 设置线条宽度为2个像素
))

# 创建顶部和底部点的字典
# 转折点字典 (Turning Points Dictionary)
Turning_points = {
    "顶部": [[top[1], top[2], top[0]] for top in tops],
    "底部": [[bottom[1], bottom[2], bottom[0]] for bottom in bottoms]
}

# 转折点
for Turning_type, points in Turning_points.items():
    points_x, points_y, _ = zip(*points)
    points_x = [data.index[x] for x in points_x]  # 直接使用datetime索引,不进行格式化
    # 使用字典直接映射转折点类型到对应颜色
    colors = {
        "顶部": "rgba(192,0,0,0.5)",  # 红色，透明度0.5
        "底部": "rgba(78,167,46,0.5)"   # 绿色，透明度0.5
    }
    color = colors[Turning_type]
    fig.add_trace(go.Scatter(x=points_x, 
                            y=points_y,
                            mode='markers',
                            name=Turning_type,
                            marker=dict(
                                color=color,
                                size=7,
                                line=dict(width=1)
                            )))

# 设置图表布局
fig.update_layout(
    title={
        'text': f"上证指数价格与Rolling Window检测的极值点 (order={order})",
        'x': 0.5,  # 标题居中
        'xanchor': 'center'
    },
    xaxis_title="日期",
    yaxis_title="收盘价",
    # 这是设置图表的样式模板为"plotly_white"，它是Plotly库中预定义的一种白色主题样式。Plotly还提供了其他样式如"plotly_dark"(深色主题)、"plotly"(默认样式)、"seaborn"(类似Seaborn库风格)、"ggplot2"(类似R的ggplot2风格)、"simple_white"(简洁白色)等多种模板
    template="plotly",  
    width=1200,
    height=600,
    xaxis_rangeslider_visible = True
)

# 保存为HTML文件,可以在浏览器中打开查看
fig.write_html("rolling_window_extremes.html")

# 如果想要保存为图片格式
fig.write_image("rolling_window_extremes.png")

print("图表已保存为rolling_window_extremes.html和rolling_window_extremes.png")

# 显示图表
# fig.show()

print('\n====================2.Directional Change 算法===========================\n')
sigma = 0.05
# 调用directional_change函数检测极值点
tops, bottoms = directional_change(data['Close'].to_numpy(), data['High'].to_numpy(), data['Low'].to_numpy(), sigma)

# 打印检测到的顶部信息
print(f"检测到 {len(tops)} 个顶部:")
# for i, top in enumerate(tops[:5]):  # 只打印前5个顶部
#     conf_date = data.index[top[0]]  # 确认日期
#     top_date = data.index[top[1]]   # 顶部日期
#     conf_price = data['Close'][top[0]]  # 确认日期价格
#     top_price = top[2]  # 顶部价格
#     amplitude = (top_price - conf_price) / conf_price * 100  # 振幅
#     print(f"顶部 {i+1}: 确认日期={conf_date}, 确认价格={conf_price:.2f}, 顶部日期={top_date}, 顶部价格={top_price:.2f}, 振幅={amplitude:.2f}%")

# 打印检测到的底部信息
print(f"\n检测到 {len(bottoms)} 个底部:")
# for i, bottom in enumerate(bottoms[:5]):  # 只打印前5个底部
#     conf_date = data.index[bottom[0]]  # 确认日期
#     bottom_date = data.index[bottom[1]]  # 底部日期
#     conf_price = data['Close'][bottom[0]]  # 确认日期价格
#     bottom_price = bottom[2]  # 底部价格
#     amplitude = (conf_price - bottom_price) / bottom_price * 100  # 振幅
#     print(f"底部 {i+1}: 确认日期={conf_date}, 确认价格={conf_price:.2f}, 底部日期={bottom_date}, 底部价格={bottom_price:.2f}, 振幅={amplitude:.2f}%")

# 创建一个DataFrame来存储顶部和底部点的信息
tops_df = pd.DataFrame(tops, columns=['确认索引', '顶部索引', '顶部价格'])
bottoms_df = pd.DataFrame(bottoms, columns=['确认索引', '底部索引', '底部价格'])

# 添加日期和价格信息
# 这行代码的作用是:
# 1. tops_df['确认日期'] 创建一个新列叫"确认日期"
# 2. apply()方法对"确认索引"列的每个值x执行lambda函数
# 3. lambda x: data.index[x] 是一个匿名函数,用x作为索引从data.index中获取对应的日期
# 4. 最终将每个确认索引对应的日期填入"确认日期"列
tops_df['确认日期'] = tops_df['确认索引'].apply(lambda x: data.index[x])
tops_df['确认价格'] = tops_df['确认索引'].apply(lambda x: data['Close'][x])
tops_df['顶部日期'] = tops_df['顶部索引'].apply(lambda x: data.index[x])
tops_df['振幅'] = (tops_df['顶部价格'] - tops_df['确认价格']) / tops_df['确认价格'] * 100

bottoms_df['确认日期'] = bottoms_df['确认索引'].apply(lambda x: data.index[x])
bottoms_df['确认价格'] = bottoms_df['确认索引'].apply(lambda x: data['Close'][x])
bottoms_df['底部日期'] = bottoms_df['底部索引'].apply(lambda x: data.index[x])
bottoms_df['振幅'] = (bottoms_df['确认价格'] - bottoms_df['底部价格']) / bottoms_df['底部价格'] * 100

# 重新排列列顺序
tops_df = tops_df[['确认日期', '确认价格', '顶部日期', '顶部价格', '振幅', '确认索引', '顶部索引']]
bottoms_df = bottoms_df[['确认日期', '确认价格', '底部日期', '底部价格', '振幅', '确认索引', '底部索引']]

# 创建Excel写入器
with pd.ExcelWriter('【Directional Change】极值点数据.xlsx') as writer:
    tops_df.to_excel(writer, sheet_name='顶部点', index=False)
    bottoms_df.to_excel(writer, sheet_name='底部点', index=False)

print("极值点数据已保存到'【Directional Change】极值点数据.xlsx'文件中")

# 创建图表对象
# go.Figure()是plotly库中的一个函数,用于创建一个空的图形对象
# 它提供了一个画布,我们可以在上面添加各种图表元素,如线图、散点图等
# 创建一个空的图形对象,后续可以通过add_trace方法添加图表
fig = go.Figure()

# 添加收盘价线图,显示为连续线条
fig.add_trace(go.Scatter(
    x=data.index,  # x轴为日期索引
    y=data['Close'],  # y轴为收盘价
    mode='lines',  # 显示模式为线条
    name='收盘价',  # 图例名称
    line=dict(width=2)  # 线条宽度为2
))

# 注释掉的K线图代码,如需显示K线可取消注释
# fig.add_trace(go.Candlestick(
#         x=data.index,  # x轴为日期索引
#         open=data['Open'],  # 开盘价
#         high=data['High'],  # 最高价
#         low=data['Low'],  # 最低价
#         close=data['Close'],  # 收盘价
#         name='K线'  # 图例名称
# ))

# 创建顶部和底部点的字典,存储转折点信息
Turning_points = {
    "顶部": [[top[1], top[2], top[0]] for top in tops],  # 顶部点列表:[顶部索引,顶部价格,确认索引]
    "底部": [[bottom[1], bottom[2], bottom[0]] for bottom in bottoms]  # 底部点列表:[底部索引,底部价格,确认索引]
}

# 添加转折点标记
# Turning_points是一个字典数据结构,用于存储和组织数据
# 使用字典的原因:
# 1. 可以通过"顶部"和"底部"这两个键快速访问对应的转折点数据
# 2. 结构清晰,便于分别处理顶部点和底部点
# 3. 代码可读性好,后续遍历处理数据时逻辑简单
# Turning_type是字典的键("顶部"或"底部"),points是对应的值(转折点列表)
# points中每个元素是[索引,价格,确认索引]格式的列表
# zip(*points)将points中的每个列表解压成3个元组,分别对应索引、价格和确认索引
# 我们只需要前两个值(索引和价格)作为x,y坐标,所以用_忽略第三个值
for Turning_type, points in Turning_points.items():
    points_x, points_y, _ = zip(*points)  # 解压获取索引和价格,忽略确认索引
    points_x = [data.index[x] for x in points_x]  # 将索引转换为实际日期
    colors = {
        "顶部": "rgba(192,0,0,0.5)",  # 顶部点为红色,透明度0.5
        "底部": "rgba(78,167,46,0.5)"  # 底部点为绿色,透明度0.5
    }
    color = colors[Turning_type]
    # 添加散点标记
    fig.add_trace(go.Scatter(x=points_x, 
                            y=points_y,
                            mode='markers',  # 显示模式为标记点
                            name=Turning_type,  # 图例名称
                            marker=dict(
                                color=color,  # 标记点颜色
                                size=7,  # 标记点大小
                                line=dict(width=1)  # 标记点边框宽度
                            )))

# 设置图表布局
fig.update_layout(
    title={
        'text': f"上证指数价格与Directional Change检测的极值点 (sigma={sigma*100}%)",  # 图表标题
        'x': 0.5,  # 标题水平位置
        'xanchor': 'center'  # 标题对齐方式
    },
    xaxis_title="日期",  # x轴标题
    yaxis_title="收盘价",  # y轴标题
    template="plotly",  # 使用plotly默认模板
    width=1200,  # 图表宽度
    height=600,  # 图表高度
    xaxis_rangeslider_visible = True  # 显示时间范围选择器
)

# 保存为HTML文件,可以在浏览器中打开查看
fig.write_html("directional_change_extremes.html")

# 如果想要保存为图片格式
fig.write_image("directional_change_extremes.png")

print("图表已保存为directional_change_extremes.html和directional_change_extremes.png")

# 显示图表
# fig.show()


print('\n====================3.Perceptually Important Points 算法===========================\n')

# 设置参数
n_pips = 100  # 要识别的重要点数量
dist_measure = 3  # 使用垂直距离度量

# 获取收盘价数据
close_prices = data['Close'].to_numpy()

# 调用PIP算法找出重要点
pips_x, pips_y = find_pips(close_prices, n_pips, dist_measure)

# 打印检测到的重要点信息
print(f"检测到 {len(pips_x)} 个重要点:")
for i, (x, y) in enumerate(zip(pips_x[:5], pips_y[:5])):  # 只打印前5个点
    point_date = data.index[x]
    print(f"重要点 {i+1}: 日期={point_date}, 价格={y:.2f}")

# 创建DataFrame来存储重要点信息
pips_df = pd.DataFrame({
    '日期': [data.index[x] for x in pips_x],
    '价格': pips_y,
    '索引': pips_x
})

# 保存到Excel
with pd.ExcelWriter('【PIP算法】重要点数据.xlsx') as writer:
    pips_df.to_excel(writer, sheet_name='重要点', index=False)

print("\n重要点数据已保存到'【PIP算法】重要点数据.xlsx'文件中")

# 创建不同参数组合的对比图
n_pips_list = [10, 20, 50,100]  # 不同数量的重要点用于比较效果

# # 使用不同的距离度量方式
# dist_measures = {
#     1: "欧几里得距离",
#     2: "垂直距离", 
#     3: "垂直距离(简化)"
# }

# 获取收盘价数据
# price_data = data['Close'].to_numpy()

# # 创建一个大图表
# plt.figure(figsize=(15, 12))

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# # 为每种距离度量方式和不同数量的重要点创建子图
# for i, dist_measure in enumerate(dist_measures.keys()):
#     for j, n_pips in enumerate(n_pips_list):
#         # 计算子图位置
#         subplot_idx = i * len(n_pips_list) + j + 1
        
#         # 创建子图
#         plt.subplot(len(dist_measures), len(n_pips_list), subplot_idx)
        
#         # 找出重要点
#         pips_x, pips_y = find_pips(price_data, n_pips, dist_measure)
        
#         # 绘制原始价格数据
#         plt.plot(range(len(price_data)), price_data, label='价格', alpha=0.5)
        
#         # 绘制重要点
#         plt.plot(pips_x, pips_y, 'ro-', label='重要点')
        
#         # 设置标题和标签
#         plt.title(f"{dist_measures[dist_measure]}, {n_pips}个重要点")
#         plt.grid(True)
        
#         # 只在最下面一行显示x轴标签
#         if i == len(dist_measures) - 1:
#             plt.xlabel('时间索引')
        
#         # 只在最左边一列显示y轴标签
#         if j == 0:
#             plt.ylabel('价格')
        
#         # 添加图例
#         plt.legend()

# plt.tight_layout()
# plt.show()

# 创建图表
fig = go.Figure()

# 添加收盘价线图
fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='收盘价',
    line=dict(width=2)
))

# 添加重要点标记
fig.add_trace(go.Scatter(
    x=[data.index[x] for x in pips_x],
    y=pips_y,
    mode='markers',
    name='重要点',
    marker=dict(
        color='red',
        size=8,
        line=dict(width=1)
    )
))

# 设置图表布局
fig.update_layout(
    title={
        'text': f"上证指数价格与PIP算法检测的重要点 (n_pips={n_pips})",
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis_title="日期",
    yaxis_title="收盘价",
    template="plotly",
    width=1200,
    height=600,
    xaxis_rangeslider_visible=True
)

# 保存为HTML文件,可以在浏览器中打开查看
fig.write_html("pip_extremes.html")

# 如果想要保存为图片格式
fig.write_image("pip_extremes.png")

print("图表已保存为pip_extremes.html和pip_extremes.png")

# fig.show()





