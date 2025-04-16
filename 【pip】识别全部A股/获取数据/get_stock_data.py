stock_codes = all_stocks.Data[1]  # 股票代码列表
stock_names = all_stocks.Data[2]  # 股票名称列表

# 添加长度校验
if len(stock_codes) != len(stock_names):
    print(f"股票代码数量({len(stock_codes)})与股票名称数量({len(stock_names)})不一致")
    return None

print(f"共获取到{len(stock_codes)}只A股股票")

# 修改循环方式
for code, name in zip(stock_codes, stock_names):
    print(f"正在获取 {code} {name} 的数据...")
    
    # ... existing code ...
    
    # 使用当前循环中的name变量
    df['Name'] = name
# ... existing code ... 