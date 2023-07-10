#基于商品的协同过滤算法
import csv
import numpy as np

# 读取CSV文件并生成评分矩阵
user_info = []
with open(r'...\..\src\用户-商品评分.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        user_info.append([int(cell) for cell in row[1:]])  # 跳过第一列的用户ID

user_info = np.array(user_info)  # 将评分数据转换为NumPy数组

def collaborative_filtering(item_id, user_info, num_recommendations):
    item_id=item_id-1
    # 获取给定商品的评分向量
    item_ratings = user_info[:, item_id]

    # 计算商品之间的相似度
    similarities = np.dot(user_info.T, item_ratings) / (np.linalg.norm(user_info, axis=0) * np.linalg.norm(item_ratings))

    # 根据相似度排序，获取最相似的商品索引
    similar_items = np.argsort(similarities)[::-1][1:]  # 排除自身商品

    # 统计最相似商品被用户评分的数量
    user_ratings = user_info[:, similar_items]
    item_scores = np.sum(user_ratings > 0, axis=0)

    # 获取前 num_recommendations 个评分最高的商品索引
    recommended_items = similar_items[np.argsort(item_scores)[::-1][:num_recommendations]]

    return recommended_items

# 假设要为商品ID为3的商品推荐10个相关商品
item_id = 3
num_recommendations = 10
recommended_items = collaborative_filtering(item_id, user_info, num_recommendations)

print(f"Recommended items for item {item_id}: {recommended_items}")

# 假设要为商品ID为3的商品推荐10个相关商品
item_id = 1
num_recommendations = 20
recommended_items = collaborative_filtering(item_id, user_info, num_recommendations)

print(f"Recommended items for item {item_id}: {recommended_items}")
