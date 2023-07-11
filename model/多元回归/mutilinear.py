from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split   #切分训练集和验证集
import pandas as pd
import os
# import seaborn as sns

#1.获得正确的数据
path = os.getcwd().replace('\\','/') + str('/user_trade.csv')
data = pd.read_csv(path)
X = data[['user_id','order_original_amount_30d','category1_id','favor_level1','favor_level2','add_level1','add_level2']]
Y = data[['level']]  #功能是一样的，将一维改成二维数组。
# 数据可视化
# sns.pairplot(data,x_vars = ['user_id','order_original_amount_30d','category1_id',
#           'favor_level1','favor_level2','add_level1','add_level2'],y_vars = 'level',height = 4,aspect = 0.8,kind='reg')

#2.生成linalg regression模型---拟合函数
linear_regressor = LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)   #则表示7:3进行划分数据集
linear_regressor.fit(x_train,y_train)  #拟合函数

# 3.使用测试集进行验证
Y_pred = linear_regressor.predict(x_test)   #预测y   使用训练集train拟合得到拟合函数后，再使用测试集test得到预测值
print('a权重：',linear_regressor.coef_)
print('b截距：',linear_regressor.intercept_)

# 4.模型评估--这些任意可以选择，根据模型的还会却确定取舍
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mes_test = mean_squared_error(y_test,Y_pred)

print("平均方误差（MSE）：",mean_squared_error(y_test,Y_pred))
print("根均方误差（RMSE）：",mean_absolute_error(y_test,Y_pred))
print("平均绝对值误差（MAE）：",r2_score(y_test,Y_pred))

RR = linear_regressor.score(x_test,y_test)
print("决定系数:",RR)  #模型越接近1，表示该模型越好

# 补充---验证该模型是否过拟合（通过训练集train进行验证）
Y_train_pred = linear_regressor.predict(x_train)   #预测y
mse_train = mean_squared_error(y_train,Y_train_pred)
# print(mse_train)
# 当mes_train和mse_test误差很小，则证明该模型未发生过拟合现象。
