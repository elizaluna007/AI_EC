import numpy as np

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.Wf = np.random.randn(hidden_dim, hidden_dim + input_dim)
        self.Wi = np.random.randn(hidden_dim, hidden_dim + input_dim)
        self.Wc = np.random.randn(hidden_dim, hidden_dim + input_dim)
        self.Wo = np.random.randn(hidden_dim, hidden_dim + input_dim)
        self.Wy = np.random.randn(output_dim, hidden_dim)

        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1 - np.tanh(x)**2

    def forward(self, x, h_prev, c_prev):
        z = np.row_stack((h_prev, x))
        f = self.sigmoid(self.Wf.dot(z) + self.bf)
        i = self.sigmoid(self.Wi.dot(z) + self.bi)
        c_bar = self.tanh(self.Wc.dot(z) + self.bc)

        c = f * c_prev + i * c_bar
        o = self.sigmoid(self.Wo.dot(z) + self.bo)
        h = o * self.tanh(c)

        y = self.Wy.dot(h) + self.by

        return y, h, c

    def backward(self, x, h_prev, c_prev, f, i, c_bar, c, o, h, y, d_h_next, d_c_next):
        d_y = np.copy(d_h_next)
        d_c = np.copy(d_c_next)
        d_c += (o * d_h_next * self.dtanh(c))

        d_o = d_h_next * self.tanh(c) * self.dsigmoid(o)
        d_c_bar = d_c * i * self.dtanh(c_bar)
        d_i = d_c * c_bar * self.dsigmoid(i)
        d_f = d_c * c_prev * self.dsigmoid(f)

        d_Wo = d_o.dot(np.row_stack((h_prev, x)).T)
        d_Wc = d_c_bar.dot(np.row_stack((h_prev, x)).T)
        d_Wi = d_i.dot(np.row_stack((h_prev, x)).T)
        d_Wf = d_f.dot(np.row_stack((h_prev, x)).T)

        d_bf = d_f.sum(axis=1)
        d_bi = d_i.sum(axis=1)
        d_bc = d_c_bar.sum(axis=1)
        d_bo = d_o.sum(axis=1)

        d_z = (self.Wf.T.dot(d_f)
               + self.Wi.T.dot(d_i)
               + self.Wc.T.dot(d_c_bar)
               + self.Wo.T.dot(d_o))

        d_h_prev = d_z[:self.hidden_dim, :]
        d_c_prev = f * d_c

        return d_h_prev, d_c_prev, d_Wo, d_Wc, d_Wi, d_Wf, d_bf, d_bi, d_bc, d_bo

    def update(self, d_Wo, d_Wc, d_Wi, d_Wf, d_bf, d_bi, d_bc, d_bo, learning_rate):
        self.Wo -= learning_rate * d_Wo
        self.Wc -= learning_rate * d_Wc
        self.Wi -= learning_rate * d_Wi
        self.Wf -= learning_rate * d_Wf
        self.bo -= learning_rate * d_bo
        self.bc -= learning_rate * d_bc
        self.bi -= learning_rate * d_bi
        self.bf -= learning_rate * d_bf
