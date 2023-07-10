import json
import csv
from sklearn.linear_model import LinearRegression
# common：常规信息，包含设备信息、用户信息等。

# "ar": 区域代码（例如："230000"）。
# "ba": 品牌（例如："Redmi"）。
# "ch": 渠道（例如："vivo"）。
# "is_new": 是否为新用户（例如："0"表示不是新用户，"1"表示新用户）。
# "md": 设备型号（例如："Redmi k30"）。
# "mid": 设备ID（例如："mid_21352"）。
# "os": 操作系统（例如："Android 11.0"）。
# "uid": 用户ID（例如："606"）。
# "vc": 应用版本（例如："v2.1.134"）。
# page：页面信息。

# "during_time": 在该页面停留的时间（例如："7443"）。
# "item": 与页面相关的项（例如："26"）。
# "item_type": 与页面相关项的类型（例如："sku_id"）。
# "last_page_id": 上一个页面的ID。
# "page_id": 当前页面的ID（例如："home"表示主页，"good_detail"表示商品详情页）。
# "source_type": 来源类型（例如："activity"表示活动）。
# actions：用户操作信息。

# "action_id": 操作ID（例如："get_coupon"表示领取优惠券）。
# "item": 与操作相关的项（例如："3"）。
# "item_type": 与操作相关项的类型（例如："coupon_id"）。
# "ts": 操作时间戳。
# displays：商品展示信息。

# "display_type": 展示类型（例如："activity"表示活动，"query"表示搜索，"promotion"表示促销，"recommend"表示推荐）。
# "item": 展示的商品项（例如："1"）。
# "item_type": 展示的商品项的类型（例如："activity_id"表示活动ID，"sku_id"表示商品ID）。
# "order": 展示的排序顺序。
# "pos_id": 展示位置ID。
# start：应用启动信息。

# "entry": 启动入口（例如："icon"表示图标启动）。
# "loading_time": 应用加载时间。
# "open_ad_id": 打开的广告ID。
# "open_ad_ms": 打开广告的显示时间。
# "open_ad_skip_ms": 跳过广告的时间。
# err：错误信息。

# ts：时间戳。

# dt：日期。

# dt,recent_days,channel,uv_count,avg_duration_sec,avg_page_count,sv_count,bounce_rate
# data = {}
# with open(r'...\..\data\mysql\gmall_ads_traffic_stats_by_channel.csv', 'r', encoding='utf-8') as file:
#     reader = csv.reader(file)
#     next(reader)  # 跳过标题行
#     for row in reader:
#         if (row[2] not in data):
#             data[row[2]] = {}
#         data[row[2]][row[1]] = [row[3], row[4], row[5], row[6], row[7]]

# for i, j in data.items():
#     print(i)
#     for k, v in j.items():
#         print(k, v)


# xiaomi
# 1 ['100', '64', '6', '100', '0.08']
# 7 ['100', '64', '6', '100', '0.08']
# 30 ['100', '64', '6', '100', '0.08']
# oppo
# 7 ['69', '69', '6', '69', '0.09']
# 30 ['69', '69', '6', '69', '0.09']
# 1 ['69', '69', '6', '69', '0.09']
# wandoujia
# 7 ['33', '56', '5', '33', '0.09']
# 1 ['33', '56', '5', '33', '0.09']
# 30 ['33', '56', '5', '33', '0.09']
# vivo
# 1 ['16', '65', '6', '16', '0.06']
# 7 ['16', '65', '6', '16', '0.06']
# 30 ['16', '65', '6', '16', '0.06']
# Appstore
# 7 ['122', '69', '6', '122', '0.05']
# 1 ['122', '69', '6', '122', '0.05']
# 30 ['122', '69', '6', '122', '0.05']
# web
# 7 ['26', '60', '6', '26', '0.12']
# 30 ['26', '60', '6', '26', '0.12']
# 1 ['26', '60', '6', '26', '0.12']
# 360
# 1 ['15', '83', '8', '15', '0.00']
# 30 ['15', '83', '8', '15', '0.00']
# 7 ['15', '83', '8', '15', '0.00']
# huawei
# 30 ['19', '72', '6', '19', '0.05']
# 1 ['19', '72', '6', '19', '0.05']
# 7 ['19', '72', '6', '19', '0.05']


traffic_stats_by_channel = {}

date_0 = [[]for i in range(15)]

# 购物车全量数据，如果添加该商品，就加1，如果下单，就加1
with open(r'...\..\data\gmall_ods_log_inc.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行

    for row in reader:
        data = json.loads(row[0])
        mid = data.get('mid')
        ch = data.get('ch')
        dt = int(row[7].split('/')[-1])

        if (ch not in traffic_stats_by_channel):
            traffic_stats_by_channel[ch] = {}
        if (row[7] not in traffic_stats_by_channel[ch]):
            traffic_stats_by_channel[ch][row[7]] = [0, 0, 0, 0, 0]

        for i in range(len(date_0[dt])):
            if (date_0[dt][i] == mid):
                continue
        date_0[dt].append(mid)
        traffic_stats_by_channel[ch][row[7]
                                     ][0] = traffic_stats_by_channel[ch][row[7]][0]+1
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
