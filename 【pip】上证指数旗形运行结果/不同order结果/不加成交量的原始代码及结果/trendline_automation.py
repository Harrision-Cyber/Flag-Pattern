import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    """
    检查趋势线是否有效并计算误差
    
    参数:
    support: bool - True表示支撑线，False表示阻力线
    pivot: int - 枢轴点的索引位置
    slope: float - 趋势线的斜率
    y: np.array - 价格数据数组
    
    返回:
    float - 如果趋势线有效，返回误差值；如果无效，返回-1.0
    """
    # 通过枢轴点和给定斜率计算截距
    intercept = -slope * pivot + y[pivot]
    # 计算趋势线上所有点的值
    # 结果是一个包含[0, 1, 2, ..., len(y)-1]的NumPy数组    
    line_vals = slope * np.arange(len(y)) + intercept
     
    # 计算趋势线与实际价格之间的差值

    # 支撑线位于价格点的下方
    # 因此支撑线的值 line_vals 应小于实际价格值 y
    # 所以 line_vals - y 的结果为负
    diffs = line_vals - y
    
    # 验证趋势线的有效性
    # 对于支撑线，所有价格点应该在线上方（允许很小的误差1e-5）
    # 对于阻力线，所有价格点应该在线下方（允许很小的误差1e-5）
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0
    

    # 首先检查趋势线是否有效：
    # 如果是支撑线(support=True)，那么所有价格点应该在线上方（或者最多只有很小的误差1e-5）。
    # --如果有任何价格点明显低于支撑线（diffs.max() > 1e-5，即最大差值大于阈值），则支撑线无效，返回-1.0。
    # 如果是阻力线(support=False)，那么所有价格点应该在线下方（或者最多只有很小的误差1e-5）。
    # --如果有任何价格点明显高于阻力线（diffs.min() < -1e-5，即最小差值小于负阈值），则阻力线无效，返回-1.0。
    # 如果趋势线有效，则计算价格与趋势线之间的误差平方和（均方误差），并返回这个误差值。这个误差值可以用来评估趋势线的拟合质量。

    # 计算价格与趋势线之间的误差平方和
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    """
    优化趋势线的斜率以获得最佳拟合
    
    参数:
    support: bool - True表示支撑线，False表示阻力线
    pivot: int - 枢轴点的索引位置
    init_slope: float - 初始斜率
    y: np.array - 价格数据数组
    
    返回:
    tuple - (最优斜率, 对应的截距)
    """
    # 计算斜率调整单位，基于价格范围和数据长度
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # 优化参数设置
    opt_step = 1.0        # 初始步长
    min_step = 0.0001     # 最小步长
    curr_step = opt_step  # 当前步长
    
    # 使用最小二乘法得到的斜率作为起始点
    best_slope = init_slope #初始斜率

    # 函数接收四个参数：support(是否为支撑线)、pivot(枢轴点索引)、init_slope(初始斜率)和y(价格数据)
    # 返回的误差值表示趋势线与实际价格点之间的拟合程度(误差平方和)
    # 如果趋势线无效(不满足支撑/阻力线条件)，则返回-1.0
    # 这个初始误差值将作为后续优化过程的基准点
    # 后面的断言语句(assert(best_err >= 0.0))确保初始斜率是有效的，这样优化算法才能正常进行。
    best_err = check_trend_line(support, pivot, init_slope, y)
    
    # 断言语句(assertion)
    # 这是Python中的断言(assertion)语句。断言用于检查一个条件是否为真，如果条件为假，则会引发AssertionError异常。
    # 在这段代码中，断言确保best_err变量的值必须大于或等于0.0，即初始斜率是有效的。
    # 如果条件不满足，程序会立即停止执行并抛出异常，这有助于在开发过程中快速发现问题。
    assert(best_err >= 0.0)  # 确保初始斜率是有效的
    # 在 check_trend_line 函数中，当趋势线无效（不满足支撑/阻力线条件）时会返回 -1.0 作为特殊标记。
    # 而当趋势线有效时，返回的是误差平方和（必然≥0）。

    # 用于控制是否需要重新计算导数
    get_derivative = True
    derivative = None
    
    # 优化循环，直到步长小于最小步长
    while curr_step > min_step:
        if get_derivative:
            # 通过数值微分计算误差对斜率的导数
            # 通过很小的斜率变化来估计误差的变化方向
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            
            # 如果增加斜率导致无效解，尝试减小斜率
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:  # 如果仍然失败，说明出现问题
                raise Exception("导数计算失败，请检查数据。")

            get_derivative = False

        # 根据导数决定斜率调整方向
        if derivative > 0.0:  # 如果增加斜率会增加误差，则减小斜率
            test_slope = best_slope - slope_unit * curr_step
        else:  # 如果增加斜率会减小误差，则增加斜率
            test_slope = best_slope + slope_unit * curr_step
        
        # 测试新斜率
        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # 如果新斜率无效或没有改善，减小步长
            curr_step *= 0.5
        else:  # 如果新斜率更好，更新最佳值
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True  # 需要重新计算导数
    
    # 循环条件：while curr_step > min_step
    # 只要当前步长大于最小步长，循环就会继续执行
    # 初始时，curr_step = opt_step = 1.0，min_step = 0.0001
    # 导数计算部分：
    # 如果需要计算导数（get_derivative为True），则通过数值微分计算误差对斜率的导数
    # 尝试增加一个很小的斜率变化，计算新的误差，并与当前最佳误差比较得到导数
    # 如果增加斜率导致无效解（test_err < 0.0），则尝试减小斜率
    # 如果减小斜率仍然导致无效解，则抛出异常
    # 计算完导数后，将get_derivative设为False，避免重复计算
    # 斜率调整部分：
    # 根据导数的符号决定斜率调整方向
    # 如果导数为正（增加斜率会增加误差），则减小斜率
    # 如果导数为负（增加斜率会减小误差），则增加斜率
    # 调整的幅度为slope_unit * curr_step
    # 测试新斜率：
    # 使用新斜率计算误差
    # 如果新斜率无效（test_err < 0）或没有改善（test_err >= best_err），则减小步长（curr_step = 0.5）
    # 如果新斜率更好，则更新最佳误差和最佳斜率，并设置get_derivative为True，表示需要重新计算导数
    # 循环终止条件：
    # 当curr_step减小到小于min_step时，循环结束
    # 此时，best_slope应该是找到的最优斜率
    # 这个循环实际上是一个梯度下降的变种，通过数值方法估计梯度（导数），然后沿着梯度方向调整参数（斜率），以最小化误差函数。它使用了自适应步长策略，当无法找到更好的解时，会减小步长以进行更精细的搜索。


    # 返回最优斜率和对应的截距
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.array):
    """
    为单一价格序列拟合支撑线和阻力线
    
    参数:
    data: np.array - 价格数据数组
    
    返回:
    tuple - ((支撑线斜率,截距), (阻力线斜率,截距))
    """
    # 使用最小二乘法计算初始趋势线
    # 创建一个从0到data长度-1的整数数组，作为x轴坐标
    # 这样每个价格点都对应一个索引位置，用于后续的线性拟合
    # np.arange()创建一个等差数列，从0开始，步长为1
    # 结果是一个形如[0, 1, 2, ..., len(data)-1]的数组
    x = np.arange(len(data))
    # np.polyfit()函数用于拟合数据点到一个多项式曲线上
    # 这里使用1次多项式（即线性拟合）来拟合数据
    # 返回的coefs是一个包含两个元素的数组，coefs[0]是斜率，coefs[1]是截距
    coefs = np.polyfit(x, data, 1)  # coefs[0]=斜率, coefs[1]=截距

    # 计算趋势线上的点
    # 使用拟合的斜率和截距计算趋势线上的所有点
    # 结果是一个与data长度相同的数组，表示趋势线在每个价格点上的值
    line_points = coefs[0] * x + coefs[1]

    # 找出价格与趋势线偏差最大的点作为枢轴点
    # 计算data与line_points之间的差值
    # 使用.argmax()方法找到最大差值的索引位置
    # 返回的upper_pivot是data中与line_points偏差最大的点
    upper_pivot = (data - line_points).argmax()  # 上方偏差最大点（返回索引位置）
    # 使用.argmin()方法找到最小差值的索引位置
    # 返回的lower_pivot是data中与line_points偏差最小的点
    lower_pivot = (data - line_points).argmin()  # 下方偏差最大点
   
    # 优化支撑线和阻力线的斜率
    # optimize_slope()函数用于优化趋势线的斜率
    # True表示支撑线，False表示阻力线
    # lower_pivot是data中与line_points偏差最小的点
    # coefs[0]是拟合的斜率
    # data是价格数据数组
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs)


def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    """
    使用最高价和最低价数据拟合支撑线和阻力线
    
    参数:
    high: np.array - 最高价数据
    low: np.array - 最低价数据
    close: np.array - 收盘价数据
    
    返回:
    tuple - ((支撑线斜率,截距), (阻力线斜率,截距))
    """
    # 使用收盘价计算初始趋势线
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]
    
    # 使用最高价和最低价找出枢轴点
    upper_pivot = (high - line_points).argmax()  # 最高价与趋势线最大偏差点
    lower_pivot = (low - line_points).argmin()   # 最低价与趋势线最大偏差点
    
    # 分别优化支撑线和阻力线
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


# if __name__ == '__main__':
#     # 加载数据

#     # 加载数据
#     data = pd.read_excel('C:\\Users\\Amber\\Desktop\\2025年策略Task\\PA量化\\数据\\000001.xlsx')
#     data['date'] = data['日期'].astype('datetime64[s]')  # 将日期列转换为datetime格式
#     data = data.set_index('date')  # 将日期列设置为索引


#     # data = pd.read_csv('BTCUSDT86400.csv')
#     # data['date'] = data['date'].astype('datetime64[s]')
#     # data = data.set_index('date')

#     # 对数据取自然对数，解决价格尺度问题
#     data = np.log(data)
    
#     # 设置回溯期长度
#     lookback = 30

#     # 初始化斜率数组
#     support_slope = [np.nan] * len(data)
#     resist_slope = [np.nan] * len(data)
    
#     # 循环计算每个时间窗口的支撑线和阻力线斜率
#     for i in range(lookback - 1, len(data)):
#         candles = data.iloc[i - lookback + 1: i + 1]
#         support_coefs, resist_coefs = fit_trendlines_high_low(candles['high'], 
#                                                             candles['low'], 
#                                                             candles['close'])
#         support_slope[i] = support_coefs[0]
#         resist_slope[i] = resist_coefs[0]

#     # 将计算结果添加到数据框中
#     data['support_slope'] = support_slope
#     data['resist_slope'] = resist_slope

#     # 绘制结果图表
#     plt.style.use('dark_background')
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#     data['close'].plot(ax=ax1)
#     data['support_slope'].plot(ax=ax2, label='支撑线斜率', color='green')
#     data['resist_slope'].plot(ax=ax2, label='阻力线斜率', color='red')
#     plt.title("BTC-USDT 日线趋势线斜率")
#     plt.legend()
#     plt.show()