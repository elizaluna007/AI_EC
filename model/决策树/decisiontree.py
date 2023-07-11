import math
import numpy
import numpy as np
from typing import Union
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # 导入决策树
from sklearn.tree import DecisionTreeRegressor

# 定义决策树结构
class Node(object):
    def __init__(self, f_idx, threshold, value=None, L=None, R=None):
        self.f_idx = f_idx  # 属性的下标，表示通过下标为f_idx的属性来划分样本
        self.threshold = threshold  # 下标 `f_idx` 对应属性的阈值
        self.value = value  # 如果该节点是叶子节点，对应的是被划分到这个节点的类别
        self.L = L          # 左子树
        self.R = R          # 右子树

# 属性划分
def ThresholdClass(dataset: np.ndarray, f_idx: int, split_choice: str):  # dataset:numpy.ndarray (n,m+1) x<-[x,y]  f_idx:feature index
    best_gain = -math.inf  # 先设置 best_gain 为无穷小
    best_gini = math.inf   # 先设置 best_gini 为无穷大
    best_threshold = None
    dataset_sorted = sorted(list(set(dataset[:, f_idx].reshape(-1))))  # 去重
    T = []  # 划分点

    for i in range(len(dataset_sorted) - 1):
        T.append(round((dataset_sorted[i] + dataset_sorted[i + 1]) / 2.0, 2))

    for threshold in T:
        L, R = Split(dataset, f_idx, threshold)   # 根据阈值分割数据集，小于阈值
        gain = None
        if split_choice == "gain":
            gain = Gain(dataset, L, R)  # 根据数据集和分割之后的数
            if gain > best_gain:    # 如果增益大于最大增益，则更换最大增益和最大阈值
                best_gain = gain
                best_threshold = threshold
        if split_choice == "gain_ratio":
            gain = GainRatio(dataset, L, R)
            if gain > best_gain:    # 如果增益大于最大增益，则更换最大增益和最大阈值
                best_gain = gain
                best_threshold = threshold
        if split_choice == "gini":
            gini = Gini_index(dataset, L, R)
            if gini < best_gini:    # gini指数越小越好
                best_gini = gini
                best_threshold = threshold
    if split_choice == "gini":
        gain_return=best_gini
    else:
        gain_return=best_gain

    return best_threshold, gain_return  # 返回最佳划分点，及信息增益等值

# 划分选择
# 信息熵
def Entropy(dataset: np.ndarray):
    scale = dataset.shape[0]  # 有多少条数据
    sum = {}                  # 每个类别对应的样本数量
    for data in dataset:
        i = data[-1]
        if i in sum:
            sum[i] += 1   # 之前存在这个标签
        else:
            sum[i] = 1   # 之前不存在这个标签
    entropy = 0.0
    for i in sum.keys():  # 计算信息熵
        p = sum[i] / scale
        entropy -= p * math.log2(p)
    return entropy

# 信息增益
def Gain(dataset, left, right):
    g1 = Entropy(dataset)
    g2 = len(left) / len(dataset) * Entropy(left) + len(right) / len(dataset) * Entropy(right)
    gain = g1 - g2
    return gain

# 增益率
def GainRatio(dataset, left, right):
    gain = Gain(dataset, left, right)
    p1 = len(left) / len(dataset)
    p2 = len(right) / len(dataset)
    # 可能出现样本全被划分到一边的情况
    if p1 == 0:
        IV = p2 * math.log2(p2)  # IV属性固有值
    elif p2 == 0:
        IV = p1 * math.log2(p1)
    else:
        IV = - p1 * math.log2(p1) - p2 * math.log2(p2)
    if IV == 0:
        gain_ratio = math.inf
    else:
        gain_ratio = gain / IV
    return gain_ratio

# 基尼指数
def Gini(dataset: np.ndarray):
    scale = dataset.shape[0]  # 有多少条数据
    sum = {}
    for data in dataset:
        key = data[-1]
        if key in sum:
            sum[key] += 1
        else:
            sum[key] = 1
    gini = 1.0
    for key in sum.keys():
        p = sum[key] / scale
        gini -= p * p
    return gini

# 属性a的基尼指数
def Gini_index(dataset, left, right):
    gini_index = len(left) / len(dataset) * Gini(left) + len(right) / len(dataset) * Gini(right)
    return gini_index

# 样本划分
def Split(X: np.ndarray, f_idx: int, threshold: float):
    L = []
    R = []
    for (idx, d) in enumerate(X[:, f_idx]):  # idx:索引, d:值
        if d < threshold:
            L.append(idx)
        else:
            R.append(idx)
    return X[L], X[R]

# 找出类别最多的
def CommonClass(dataset):
    class_list = [data[-1] for data in dataset]
    return collections.Counter(class_list).most_common(1)[0][0]

# 生成分类决策树
def BuildClassTree(dataset: np.ndarray, f_idx_list: list, split_choice: str):   # return Node 递归
    class_list = [data[-1] for data in dataset]  # 类别
    # 全属于同一类别
    if class_list.count(class_list[0]) == len(class_list):
        return Node(None, None, value=class_list[0])
    # 若属性都用完, 标记为类别最多的那一类
    elif len(f_idx_list) == 0:
        value = collections.Counter(class_list).most_common(1)[0][0]
        return Node(None, None, value=value)
    else:
        # 找到划分 增益最大的属性
        best_gain = -math.inf
        best_gini = math.inf
        best_threshold = None
        best_f_idx = None

        for i in f_idx_list:
            threshold, gain = ThresholdClass(dataset, i, split_choice)
            if split_choice == "gini":
                if gain < best_gini:
                    best_gini = gain
                    best_threshold = threshold
                    best_f_idx = i
            if split_choice == "gain" or split_choice == "gain_ratio" :
                if gain > best_gain:  # 如果增益大于最大增益，则更换最大增益和最大
                    best_gain = gain
                    best_threshold = threshold    # 阈值s
                    best_f_idx = i   # 最大增益对应的属性为当前最佳属性j

        son_f_idx_list = f_idx_list.copy()
        son_f_idx_list.remove(best_f_idx)
        # 创建分支
        L, R = Split(dataset, best_f_idx, best_threshold)
        if len(L) == 0:
            L_tree = Node(None, None, CommonClass(dataset))  # 叶子节点
        else:
            L_tree = BuildClassTree(L, son_f_idx_list, split_choice)  # return DecisionNode

        if len(R) == 0:
            R_tree = Node(None, None, CommonClass(dataset))  # 叶子节点
        else:
            R_tree = BuildClassTree(R, son_f_idx_list, split_choice)  # return DecisionNode
        return Node(best_f_idx, best_threshold, value=None, L=L_tree, R=R_tree)

# 预测
def Predict(model: Node, data):
    if model.value is not None:
        return model.value
    else:
        feature_one = data[model.f_idx]
        if feature_one >= model.threshold:
            branch = model.R  # 走右边
        else:
            branch = model.L   # 走左边
        return Predict(branch, data)

# 求分类决策数的准确率
def Accuracy(y_predict, y_test):
    y_predict = y_predict.tolist()
    y_test = y_test.tolist()
    count = 0
    # count = np.sum(y_predict == y_test)
    for i in range(len(y_predict)):
        if int(y_predict[i]) == y_test[i]:
            count = count + 1
    accuracy = count / len(y_predict)
    return accuracy

# 据不同选择调用生成决策树的函数
class SimpleDecisionTree(object):
    def __init__(self, split_choice, min_samples: int = 1, min_gain: float = 0, max_depth: Union[int, None] = None,
                 max_leaves: Union[int, None] = None):
        self.split_choice = split_choice

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        dataset_in = np.c_[X, y]
        if X.shape[1]==1:
            f_idx_list=[0]
        else:
            f_idx_list = [i for i in range(X.shape[1])]
        self.my_tree = BuildClassTree(dataset_in, f_idx_list, self.split_choice)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predict_list = []
        for data in X:
            predict_list.append(Predict(self.my_tree, data))

        return np.array(predict_list)

if __name__ == "__main__":

    for i in range(1):     # 可设置训练多次
        pd = pd.read_excel("user_buy.xlsx")
        x = pd.values[:,0:5]
        y = pd.values[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        print("预测属性数据\n",X_test)
        print("我的手写决策树")
        split_choice_list = ["gain", "gain_ratio", "gini"]  # 增益，增益率，基尼指数
        for split_choice in split_choice_list:
            m = SimpleDecisionTree(split_choice) # 划分选择
            m.fit(X_train, y_train)              # 训练
            y_predict = m.predict(X_test)        # 预测
            y_preaccu = Accuracy(y_predict, y_test.reshape(-1))  # 计算准确率
            if split_choice=="gain":
                print("采用信息增益时：")
                print("预测值", y_predict)
                print("真实值", y_test)
                print("精度为:{:.3f}".format(y_preaccu))

            elif split_choice=="gain_ratio":
                print("采用增益率时：")
                print("预测值", y_predict)
                print("真实值", y_test)
                print("精度为:{:.3f}".format(y_preaccu))
            else:
                print("采用基尼指数时：")
                print("预测值", y_predict)
                print("真实值", y_test)
                print("精度为:{:.3f}".format(y_preaccu))

        # 对比sklearn分类决策树
        print("sklearn")
        clf_gain = DecisionTreeClassifier(criterion='entropy')
        clf_gain.fit(X_train, y_train)  # 使用训练集训练模型
        predicted_gain = clf_gain.predict(X_test)
        print("采用信息熵时的精度是:{:.3f}".format(clf_gain.score(X_test, y_test)))
        clf_gini = DecisionTreeClassifier(criterion='gini')
        clf_gini.fit(X_train, y_train)  # 使用训练集训练模型
        predicted = clf_gini.predict(X_test)
        print("采用基尼指数时的精度是:{:.3f}".format(clf_gini.score(X_test, y_test)))
