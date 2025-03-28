import pandas as pd
import numpy as np


'''====================1.Rolling Window 算法==========================='''
# 检测局部顶部的函数
# data: 价格数据数组
# curr_index: 当前检查的索引位置
# order: 窗口大小的一半（窗口总大小 = 2*order + 1）
# 这个函数确实是在检查 k 点（即 curr_index - order）是否是局部最大值点。函数的逻辑是：
# 计算窗口中心点 k = curr_index - order
# 获取中心点的价格值 v = data[k]
# 检查中心点前后各 order 个点的价格
# 如果窗口内有任何一个点的价格高于中心点，则 k 点不是顶部
# 只有当 k 点的价格高于或等于窗口内所有其他点时，才认为它是局部顶部（最大值点）
# 这是一个典型的滚动窗口方法来识别价格数据中的局部极值点。
def rw_top(data: np.array, curr_index: int, order: int) -> bool:
    # 如果当前索引小于窗口大小，无法形成完整窗口，返回False
    if curr_index < order * 2 + 1:  # 加1是因为窗口总大小为2*order+1,中心点需要前后各order个点,总共需要2*order+1个点
        return False

    top = True  # 假设是顶部
    k = curr_index - order  # 计算中心点索引 
    v = data[k]  # 中心点的价格值
    
    # 检查中心点前后各order个点的价格
    # 如果有任何一个点的价格高于中心点，则不是顶部
    #range(1, order + 1) 的取值范围是从 1 到 order（包含 order）的整数序列。例如，如果 order = 3，则取值为 1, 2, 3。
    
    # 从0开始遍历会导致k+i和k-i的索引超出范围
    # 因为k是中心点,需要前后各order个点进行比较
    # 所以i必须从1开始,这样k±i才能正确访问窗口内的点
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            top = False
            break
    
    return top

# 检测局部底部的函数
# data: 价格数据数组
# curr_index: 当前检查的索引位置
# order: 窗口大小的一半（窗口总大小 = 2*order + 1）
def rw_bottom(data: np.array, curr_index: int, order: int) -> bool:
    # 如果当前索引小于窗口大小，无法形成完整窗口，返回False
    if curr_index < order * 2 + 1:
        return False

    bottom = True  # 假设是底部
    k = curr_index - order  # 计算中心点索引
    v = data[k]  # 中心点的价格值
    
    # 检查中心点前后各order个点的价格
    # 如果有任何一个点的价格低于中心点，则不是底部
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            bottom = False
            break
    
    return bottom

# 找出所有极值点的函数
# data: 价格数据数组
# order: 窗口大小的一半
def rw_extremes(data: np.array, order:int):
    # 初始化存储顶部和底部的列表
    tops = []
    bottoms = []
    
    # 遍历整个数据集，i的取值范围是从0到data数组长度减1。根据上下文可以看到，data是一个包含7320行的价格数据数组，所以i的取值范围是0到7319。
    for i in range(len(data)):
        # 检查是否是顶部
        if rw_top(data, i, order):
            # 记录顶部信息：
            # top[0] = 确认索引（当前索引i）
            # top[1] = 顶部索引（i - order，即窗口中心）
            # top[2] = 顶部价格
            # 创建一个包含顶部信息的列表:
            # i: 当前确认索引位置
            # i - order: 顶部实际位置(窗口中心)
            # data[i - order]: 顶部价格值
            top = [i, i - order, data[i - order]]
            tops.append(top)  # 将找到的顶部点信息添加到tops列表中
        
        # 检查是否是底部
        if rw_bottom(data, i, order):
            # 记录底部信息：
            # bottom[0] = 确认索引（当前索引i）
            # bottom[1] = 底部索引（i - order，即窗口中心）
            # bottom[2] = 底部价格
            bottom = [i, i - order, data[i - order]]
            bottoms.append(bottom)
    
    return tops, bottoms


'''====================2.Directional Change 算法==========================='''

def directional_change(close: np.array, high: np.array, low: np.array, sigma: float):
    """
    方向性变化算法 - 根据价格回撤幅度检测市场转折点
    
    参数:
    close: 收盘价数组
    high: 最高价数组
    low: 最低价数组
    sigma: 回撤阈值，例如0.02表示2%的回撤
    
    返回:
    tops: 检测到的顶部列表，每个顶部包含[确认索引, 顶部索引, 顶部价格]
    bottoms: 检测到的底部列表，每个底部包含[确认索引, 底部索引, 底部价格]
    """
    
    # 初始状态：假设最后一个极值是底部，下一个将是顶部
    up_zig = True 
    
    # 初始化临时变量，用于跟踪当前的最高点和最低点
    tmp_max = high[0]  # 当前最高价
    tmp_min = low[0]   # 当前最低价
    tmp_max_i = 0      # 当前最高价的索引
    tmp_min_i = 0      # 当前最低价的索引

    # 存储检测到的顶部和底部
    tops = []
    bottoms = []

    # 遍历价格数据
    for i in range(len(close)):
        if up_zig:  # 如果最后一个极值是底部，我们正在寻找顶部
            if high[i] > tmp_max:
                # 发现新的最高价，更新临时最高点
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max - tmp_max * sigma: 
                # 价格从最高点回落超过sigma%，确认一个顶部
                # top[0] = 确认索引（当前索引i）
                # top[1] = 顶部索引（tmp_max_i）
                # top[2] = 顶部价格（tmp_max）
                top = [i, tmp_max_i, tmp_max]
                tops.append(top)

                # 设置寻找下一个底部的初始状态
                up_zig = False  # 切换状态，现在寻找底部
                tmp_min = low[i]  # 初始化当前最低价
                tmp_min_i = i     # 初始化当前最低价的索引
        else:  # 如果最后一个极值是顶部，我们正在寻找底部
            if low[i] < tmp_min:
                # 发现新的最低价，更新临时最低点
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min + tmp_min * sigma: 
                # 价格从最低点上涨超过sigma%，确认一个底部
                # bottom[0] = 确认索引（当前索引i）
                # bottom[1] = 底部索引（tmp_min_i）
                # bottom[2] = 底部价格（tmp_min）
                bottom = [i, tmp_min_i, tmp_min]
                bottoms.append(bottom)

                # 设置寻找下一个顶部的初始状态
                up_zig = True  # 切换状态，现在寻找顶部
                tmp_max = high[i]  # 初始化当前最高价
                tmp_max_i = i      # 初始化当前最高价的索引

    return tops, bottoms

def get_extremes(ohlc: pd.DataFrame, sigma: float):
    """
    将directional_change函数的结果转换为DataFrame格式
    
    参数:
    ohlc: 包含'close', 'high', 'low'列的DataFrame
    sigma: 回撤阈值
    
    返回:
    extremes: 包含所有极值点的DataFrame，按确认索引排序
    """
    # 调用directional_change函数获取顶部和底部
    tops, bottoms = directional_change(ohlc['close'], ohlc['high'], ohlc['low'], sigma)
    
    # 将顶部和底部转换为DataFrame
    tops = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
    
    # 添加类型标识：1表示顶部，-1表示底部
    tops['type'] = 1
    bottoms['type'] = -1
    
    # 合并顶部和底部
    extremes = pd.concat([tops, bottoms])
    
    # 按确认索引排序
    extremes = extremes.set_index('conf_i')
    extremes = extremes.sort_index()
    
    return extremes
    


'''====================3.Perceptually Important Points 算法==========================='''

def find_pips(data: np.array, n_pips: int, dist_measure: int):
    """
    感知重要点(Perceptually Important Points, PIP)算法
    
    参数:
    data: 价格数据数组
    n_pips: 要识别的重要点数量
    dist_measure: 距离度量方式
        1 = 欧几里得距离(Euclidean Distance)
        2 = 垂直距离(Perpendicular Distance)
        3 = 垂直距离(Vertical Distance)
    
    返回:
    pips_x: 重要点的索引
    pips_y: 重要点的价格值
    """
    # 初始化，将起点和终点作为第一批重要点
    # len(data)-1 是因为数组索引从0开始,最后一个元素的索引是长度减1
    pips_x = [0, len(data) - 1]  # 索引,0表示第一个点,len(data)-1表示最后一个点
    pips_y = [data[0], data[-1]]  # 价格,data[0]是第一个价格,data[-1]是最后一个价格

    # 迭代添加n_pips-2个重要点（因为已经有起点和终点两个点）
    for curr_point in range(2, n_pips):
        md = 0.0  # 最大距离
        md_i = -1  # 最大距离对应的索引,初始化为-1表示还未找到最大距离点
        insert_index = -1  # 插入位置,初始化为-1表示还未确定插入位置

        # 遍历当前已有的重要点之间的所有区间
        for k in range(0, curr_point - 1):
            # 获取左右相邻重要点的索引
            # 获取相邻两个重要点的索引,用于构建线段
            # left_adj表示左侧重要点在pips数组中的位置
            # right_adj表示右侧重要点在pips数组中的位置
            left_adj = k  
            right_adj = k + 1

            # 计算这两个重要点之间的直线方程 y = slope * x + intercept
            # 用于后续计算其他点到这条线段的距离
            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff  # 计算斜率
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope  # 计算截距

            # 遍历两个重要点之间的所有点
            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                d = 0.0  # 距离
                
                # 根据选择的距离度量方式计算距离
                if dist_measure == 1:  # 欧几里得距离
                    # 计算点到左右两个重要点的欧几里得距离之和
                    d = ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2) ** 0.5
                    d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2) ** 0.5
                elif dist_measure == 2:  # 垂直距离（点到直线的垂直距离）
                    # 计算点到直线的垂直距离
                    d = abs((slope * i + intercept) - data[i]) / (slope ** 2 + 1) ** 0.5
                else:  # 垂直距离（点到直线的垂直距离，不考虑斜率）
                    # 计算点到直线的垂直距离（简化版）
                    d = abs((slope * i + intercept) - data[i])

                # 如果找到更大的距离，更新最大距离和对应的索引
                if d > md:
                    md = d
                    # 记录当前找到的最大距离点的索引i
                    md_i = i
                    # right_adj是当前区间右端点的位置,将新点插入到right_adj位置
                    insert_index = right_adj

        # 将新找到的重要点插入到重要点列表中
        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    return pips_x, pips_y