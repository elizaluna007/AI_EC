import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from BP import NeuralNetwork

# Load the data
df1 = pd.read_csv('gmall_dws_trade_user_payment_1d.csv')
df2 = pd.read_csv('gmall_dws_trade_user_payment_nd.csv')
df = pd.merge(df1, df2, on='user_id')

# Specify the columns for input and output
input_columns = ['payment_count_1d', 'payment_num_1d', 'payment_amount_1d', 
                 'payment_count_7d', 'payment_num_7d', 'payment_amount_7d', 
                 'payment_count_30d', 'payment_num_30d', 'payment_amount_30d']
output_column = ['dt']

# Preprocess the data
scaler = StandardScaler()
df[input_columns] = scaler.fit_transform(df[input_columns])

X = df[input_columns].values
y = df[output_column].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Neural Network with number of nodes in each layer
nn = NeuralNetwork([9, 5, 1])

# Train the network
nn.fit(X_train, y_train)

# Test the network
prediction = nn.predict(X_test)
