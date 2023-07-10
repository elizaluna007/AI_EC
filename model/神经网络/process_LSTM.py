import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LSTM import LSTM

# Assuming your CSVs have been loaded into pandas DataFrames
df1 = pd.read_csv('gmall_dws_trade_user_payment_1d.csv')
df2 = pd.read_csv('gmall_dws_trade_user_payment_nd.csv')

# Merge the two DataFrames on user_id and dt
df = pd.merge(df1, df2, on=['user_id', 'dt'])

# Drop non-numeric columns if necessary
df = df.select_dtypes(include=[np.number])

# Normalizing the data
scaler = StandardScaler()
df = scaler.fit_transform(df)

# Split the data into features and target
X = df[:, :-1]
y = df[:, -1]

# Reshape input data into 3D array as required by LSTM model
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LSTM
lstm = LSTM(input_dim=X_train.shape[2], hidden_dim=10, output_dim=1)

# Specify the learning rate and the number of epochs
learning_rate = 0.01
epochs = 100

# Start training
for epoch in range(epochs):
    h_prev = np.zeros((lstm.hidden_dim, 1))  # reset the hidden state
    c_prev = np.zeros((lstm.hidden_dim, 1))  # reset the cell state

    for i in range(X_train.shape[0]):
        # Forward pass
        y_pred, h_prev, c_prev = lstm.forward(X_train[i], h_prev, c_prev)

        # Compute the loss
        loss = np.square(y_pred - y_train[i]).sum() / 2

        # Backward pass
        d_h_next = y_pred - y_train[i]
        d_c_next = np.zeros_like(c_prev)

        d_h_prev, d_c_prev, d_Wo, d_Wc, d_Wi, d_Wf, d_bf, d_bi, d_bc, d_bo = \
            lstm.backward(X_train[i], h_prev, c_prev, d_h_next, d_c_next)

        # Update the parameters
        lstm.update(d_Wo, d_Wc, d_Wi, d_Wf, d_bf, d_bi, d_bc, d_bo, learning_rate)

    # Print the loss for this epoch
    print('Epoch', epoch, 'Loss', loss)

# Evaluate the model on the test set
h_prev = np.zeros((lstm.hidden_dim, 1))  # reset the hidden state
c_prev = np.zeros((lstm.hidden_dim, 1))  # reset the cell state

test_loss = 0

for i in range(X_test.shape[0]):
    y_pred, h_prev, c_prev = lstm.forward(X_test[i], h_prev, c_prev)
    test_loss += np.square(y_pred - y_test[i]).sum() / 2

print('Test Loss:', test_loss / X_test.shape[0])
