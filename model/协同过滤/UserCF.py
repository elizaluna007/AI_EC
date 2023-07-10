#基于用户的协同过滤算法
import csv
import numpy as np
import sys
import random
import math
from operator import itemgetter
# 读取CSV文件并生成评分矩阵
user_info = []
with open(r'...\..\src\用户-商品评分.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        user_info.append([int(cell) for cell in row[1:]])  # 跳过第一列的用户ID

user_info = np.array(user_info)  # 将评分数据转换为NumPy数组

def collaborative_filtering(user_id, user_info, num_recommendations):
    user_id=user_id-1
    # 获取给定用户的评分向量
    user_ratings = user_info[user_id]

    # 计算用户之间的相似度
    similarities = np.dot(user_info, user_ratings) / (np.linalg.norm(user_info, axis=1) * np.linalg.norm(user_ratings))

    # 根据相似度排序，获取最相似的用户索引
    similar_users = np.argsort(similarities)[::-1][1:]  # 排除自身用户

    # 统计最相似用户对商品的评分
    item_ratings = np.sum(user_info[similar_users], axis=0)

    # 过滤掉已经被用户评分过的商品
    item_ratings[user_ratings > 0] = 0

    # 获取前 num_recommendations 个评分最高的商品索引
    recommended_items = np.argsort(item_ratings)[::-1][:num_recommendations]

    return recommended_items

# 假设要为用户ID为5的用户推荐10个商品
user_id = 5
num_recommendations = 10
recommended_items = collaborative_filtering(user_id, user_info, num_recommendations)

print(f"Recommended items for user {user_id}: {recommended_items}")

# 假设要为用户ID为5的用户推荐20个商品
user_id = 1
num_recommendations = 20
recommended_items = collaborative_filtering(user_id, user_info, num_recommendations)

print(f"Recommended items for user {user_id}: {recommended_items}")