from WindPy import w 
import pandas as pd
import numpy as np

def get_stock_data(code, start_date, end_date):
    """
    从Wind获取股票数据
    
    参数:
    code: 股票代码
    start_date: 开始日期，格式'YYYY-MM-DD'
    end_date: 结束日期，格式'YYYY-MM-DD'
    
    返回:
    DataFrame: 包含股票数据的DataFrame
    """
    w.start()
    wind_data = w.wsd(code, "close,open,high,low,volume,pct_chg", start_date, end_date, "PriceAdj=F")
    
    if wind_data.ErrorCode != 0:
        print(f"获取数据出错: {wind_data.Data}")
        return None
        
    # 转换为DataFrame
    df = pd.DataFrame(data=wind_data.Data, 
                     index=wind_data.Fields, 
                     columns=wind_data.Times).T
    
    # 重命名列以匹配yfinance格式
    df.columns = ['Close', 'Open', 'High', 'Low', 'Volume', 'Change']
    df.index.name = 'Date'
    
    print(f"成功获取{code}的数据，共{len(df)}条记录")
    return df


data = get_stock_data("000001.SH", "1995-01-01", "2025-02-28")
# 将数据保存到Excel文件
excel_path = '上证指数数据.xlsx'
data.to_excel(excel_path)
print(f"数据已保存至 {excel_path}")

print(data)