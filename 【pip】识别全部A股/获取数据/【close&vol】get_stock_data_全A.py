from WindPy import w 
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime



def get_stock_database(stock_codes, stock_names, start_date, end_date):
    """建立完整股票数据库"""
    
    # 初始化WindPy
    w.start()
    if not w.isconnected():
        print("Wind未连接，请检查Wind终端")
        return False
    
    # 获取股票名称
    # stock_names = []
    # for code in stock_codes:
    #     name_data = w.wss(code, "sec_name")
    #     if name_data.ErrorCode != 0:
    #         print(f"获取{code}名称出错: {name_data.ErrorCode}")
    #         stock_names.append("未知")
    #     else:
    #         stock_names.append(name_data.Data[0][0])
    
    # print(f"共获取到{len(stock_codes)}只股票")
    
    # 每100只股票批量获取，避免API限制
    batch_size = 100
    success_count = 0
    all_data = {}
    
    # 创建不同指标的数据字典
    # high_data = {}
    # low_data = {}
    close_data = {}
    volume_data = {}
    
    for i in range(0, len(stock_codes), batch_size):
        batch_codes = stock_codes[i:i+batch_size]
        batch_names = stock_names[i:i+batch_size]
        
        print(f"正在获取第{i+1}-{i+len(batch_codes)}只股票数据...")
        
        for j, code in enumerate(batch_codes):
            
            try:
                # 获取股票数据
                # wind_data = w.wsd(code, "high,low,close,volume", 
                #                      start_date, end_date, 
                #                      "Currency=CNY;PriceAdj=F")

                wind_data = w.wsd(code, "close,volume", 
                                 start_date, end_date, 
                                 "Currency=CNY;PriceAdj=F")
                
                if wind_data.ErrorCode != 0:
                    print(f"获取{code} {batch_names[j]}数据出错: {wind_data.ErrorCode}")
                    continue
                
                # 转换为DataFrame
                df = pd.DataFrame(data=np.array(wind_data.Data).T,
                                 index=wind_data.Times,
                                 columns=wind_data.Fields)
                
                # 添加代码和名称
                df['code'] = code
                df['name'] = batch_names[j]
                
                success_count += 1
                
                # 将数据添加到返回字典中
                all_data[code] = df
                
                # 分别存储不同指标的数据
                for date_idx, date in enumerate(wind_data.Times):
                    # 高价数据
                    # if date not in high_data:
                    #     high_data[date] = {}
                    # high_data[date][code] = wind_data.Data[0][date_idx]
                    
                    # # 低价数据
                    # if date not in low_data:
                    #     low_data[date] = {}
                    # low_data[date][code] = wind_data.Data[1][date_idx]
                    
                    # 收盘价数据
                    if date not in close_data:
                        close_data[date] = {}
                    close_data[date][code] = wind_data.Data[0][date_idx]
                    
                    # 成交量数据
                    if date not in volume_data:
                        volume_data[date] = {}
                    volume_data[date][code] = wind_data.Data[1][date_idx]
                
                # 避免API限制
                time.sleep(0.2)
                
            except Exception as e:
                print(f"处理{code}时发生错误: {str(e)}")
        
        print(f"当前批次完成，休息5秒...")
        time.sleep(5)  # 批次间休息
    
    print(f"数据库构建完成，成功获取{success_count}只股票数据")
    
    # 将不同指标数据转换为DataFrame
    # high_df = pd.DataFrame(high_data).T
    # low_df = pd.DataFrame(low_data).T
    close_df = pd.DataFrame(close_data).T
    volume_df = pd.DataFrame(volume_data).T
    
    # 添加股票名称作为列名
    for i, code in enumerate(stock_codes):
        if i < len(stock_names):
            # high_df.rename(columns={code: f"{code}_{stock_names[i]}"}, inplace=True) # 000004.SZ_国华网安
            # low_df.rename(columns={code: f"{code}_{stock_names[i]}"}, inplace=True)
            # close_df.rename(columns={code: f"{code}_{stock_names[i]}"}, inplace=True)
            # volume_df.rename(columns={code: f"{code}_{stock_names[i]}"}, inplace=True)

            # high_df.rename(columns={code: f"{code}"}, inplace=True) # 000004.SZ
            # low_df.rename(columns={code: f"{code}"}, inplace=True)
            close_df.rename(columns={code: f"{code}"}, inplace=True)
            volume_df.rename(columns={code: f"{code}"}, inplace=True)
    
    # 将整合后的数据添加到返回字典中
    # all_data['high_df'] = high_df
    # all_data['low_df'] = low_df
    all_data['close_df'] = close_df
    all_data['volume_df'] = volume_df
    
    return all_data



df = pd.read_excel("stocklist.xlsx", sheet_name="list")
# 使用 tolist() 将结果转换为 Python 列表
stock_codes = df['Wind代码'].dropna().tolist()
stock_names = df['证券名称'].dropna().tolist()

# 获取指定股票列表的数据
all_data = get_stock_database(stock_codes, stock_names, "2018-01-01", "2025-03-28")

# 创建保存Excel文件的目录
excel_dir = '全部A股数据'
if not os.path.exists(excel_dir):
    os.makedirs(excel_dir)

# 将所有股票数据保存到一个Excel文件，不同指标保存在不同sheet
excel_path = os.path.join(excel_dir, "【close&vol】全部A股数据.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # all_data['high_df'].to_excel(writer, sheet_name='高价')
    # all_data['low_df'].to_excel(writer, sheet_name='低价')
    all_data['close_df'].to_excel(writer, sheet_name='收盘价')
    all_data['volume_df'].to_excel(writer, sheet_name='成交量')

print(f"全部A股数据已保存至 {excel_path}")





