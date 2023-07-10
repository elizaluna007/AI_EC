import csv
import json
# 对原始数据进行初步处理，生成用户-商品评分矩阵
user_info = [[0 for j in range(36)]for i in range(3001)]  # 3000个用户，35个商品
for i in range(36):
    user_info[0][i] = i
for i in range(3001):
    user_info[i][0] = i


# 购物车全量数据，如果添加该商品，就加1，如果下单，就加1
with open(r'...\..\data\mysql\gmall_ods_cart_info_full.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行

    for row in reader:
        user_id = int(row[1])
        sku_id = int(row[2])
        is_ordered = int(row[10])

        user_info[user_id][sku_id] = user_info[user_id][sku_id]+1

        if is_ordered == 1:
            user_info[user_id][sku_id] = user_info[user_id][sku_id] + 1
            
    
# 购物车增量数据，如果添加该商品，就加1，如果下单，就加1
with open(r'...\..\data\mysql\gmall_ods_cart_info_inc.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        
        data = json.loads(row[2])
        user_id = data.get('user_id')
        if user_id is None or user_id == 'null':  # 跳过 user_id 为 null 的行
            continue
        
        if(row[3] !=''):  # 跳过 sku_id 为 null 的行
            if(row[3] == '{"is_ordered":"0","order_time":null}'):
                11
            elif(row[3][2:12] == 'order_time'):
                22
            elif(row[3][2:9] == 'sku_num'):
                continue
            else:
                33

        user_id = int(user_id)
        sku_id = int(data.get('sku_id'))
        is_ordered = data.get('is_ordered')
        if is_ordered == '1':
            user_info[user_id][sku_id] = user_info[user_id][sku_id] + 2
        else:
            user_info[user_id][sku_id] = user_info[user_id][sku_id] + 1

    
# 购物车增量数据，如果添加该商品，就加1，如果下单，就加1
with open(r'...\..\data\mysql\gmall_ods_favor_info_inc.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        
        data = json.loads(row[2])
        user_id = data.get('user_id')
        is_cancer=data.get('is_cancel')
        if user_id is None or user_id == 'null':  # 跳过 user_id 为 null 的行
            continue

        if(is_cancer == "1"):
            continue
            

        user_id = int(user_id)
        sku_id = int(data.get('sku_id'))
        user_info[user_id][sku_id] = user_info[user_id][sku_id] + 1


#将数据写入文件
with open(r'...\..\src\用户-商品评分.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入数据行
    for row in user_info:
        writer.writerow(row)