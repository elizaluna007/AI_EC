import numpy as np
import pandas as pd
from RNN import RNN
import matplotlib.pyplot as plt

# 数据处理
def data_process(file):
    df = pd.read_csv(file)
    df.sort_values('dt', inplace=True)
    X = df.drop(columns=['user_id', 'payment_amount_1d']).values
    y = df['payment_amount_1d'].values
    return X, y

# 训练模型
def train_model(X, y, epochs=10):
    input_size = X.shape[1]
    rnn = RNN(input_size, 1)

    losses = []
    for epoch in range(epochs):
        loss = 0
        for i in range(X.shape[0]):
            inputs = X[i]
            target = y[i]

            out, _ = rnn.forward(inputs)
            loss += (out - target) ** 2

            d_y = 2 * (out - target)
            rnn.backprop(d_y)

        losses.append(loss)
        print(f'Epoch: {epoch+1}, Loss: {loss}')

    return rnn, losses

# 画出损失函数
def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over time')
    plt.show()

# 主函数
def main():
    X, y = data_process('gmall_dws_trade_user_payment_1d.csv')
    rnn, losses = train_model(X, y)
    plot_losses(losses)

if __name__ == '__main__':
    main()
