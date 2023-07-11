from sklearn.linear_model import LinearRegression

import csv

traffic_stats_by_channel = {}

# 读取CSV文件
with open(r'...\..\src\result.csv', 'r') as file:
    reader = csv.reader(file)

    # 跳过CSV文件的标题行
    next(reader)

    # 遍历每一行数据
    for row in reader:
        # 获取渠道名称和数据字典字符串
        channel = row[0]
        data_dict_str = row[1]

        # 将数据字典字符串转换为字典类型
        data_dict = eval(data_dict_str)

        # 将数据存储到traffic_stats_by_channel字典中
        traffic_stats_by_channel[channel] = data_dict
print("traffic_stats_by_channel:")  
print(traffic_stats_by_channel)
l = 0
m = 0
result = [[]for i in range(8)]
for i, j in traffic_stats_by_channel.items():
    print(i)
    for k, v in j.items():
        result[l].append(v[0])
        m = m+1
        print(k, v)
    l = l+1

print(result)



# 给定的数据序列
data = result[0]  # 选择其中一个渠道的数据

# 创建输入特征和目标变量
X = [[i] for i in range(len(data))]
y = data

# 创建线性回归模型并进行训练
model = LinearRegression()
model.fit(X, y)

# 预测下一条数据
next_value = model.predict([[len(data)]])

print("预测的下一条数据为:", int(next_value[0]))

# 给定的数据序列
data = result[1]  # 选择其中一个渠道的数据

# 创建输入特征和目标变量
X = [[i] for i in range(len(data))]
y = data

# 创建线性回归模型并进行训练
model = LinearRegression()
model.fit(X, y)

# 预测下一条数据
next_value = model.predict([[len(data)]])

print("预测的下一条数据为:", int(next_value[0]))