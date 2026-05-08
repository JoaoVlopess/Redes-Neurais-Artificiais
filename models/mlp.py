import numpy as np

import numpy as np

class MLP:
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01, n_epochs=100):
        self.eta = learning_rate
        self.epoch = n_epochs
        
        self.w_hidden = np.random.randn(n_input, n_hidden) * 0.01
        self.w_output = np.random.randn(n_hidden, n_output) * 0.01
        
        self.b_hidden = np.zeros((1, n_hidden))
        self.b_output = np.zeros((1, n_output))
        
        self.errors = []

    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        return 1.0 - np.square(x)
    
    def fit(self, X, d):
        n_samples = X.shape[0]

        for epoch in range(self.epoch):
            u_hidden = np.dot(X, self.w_hidden) + self.b_hidden
            y_hidden = self._tanh(u_hidden)

            u_output = np.dot(y_hidden, self.w_output) + self.b_output
            y_output = self._tanh(u_output) 

            error = d - y_output
            mse = np.mean(np.square(error))
            self.errors.append(mse)

            d_output = error * self._tanh_derivative(y_output)
            error_hidden = np.dot(d_output, self.w_output.T)
            d_hidden = error_hidden * self._tanh_derivative(y_hidden)

            self.w_output += np.dot(y_hidden.T, d_output) * self.eta
            self.b_output += np.sum(d_output, axis=0, keepdims=True) * self.eta

            self.w_hidden += np.dot(X.T, d_hidden) * self.eta
            self.b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.eta

            if epoch % 100 == 0:
                print(f"Época {epoch} - Erro (MSE): {mse}")
        
    def predict(self, X):
        u_hidden = np.dot(X, self.w_hidden) + self.b_hidden
        y_hidden = self._tanh(u_hidden)

        u_output = np.dot(y_hidden, self.w_output) + self.b_output
        y_output = self._tanh(u_output)
            
        return y_output

