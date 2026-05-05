import numpy as np

class adaline:
    def __init__(self, learning_rate=1e-5, n_epochs=20):
        self.eta = learning_rate
        self.epoch = n_epochs
        self.w = None
        self.errors = []

    def fit(self,X,d, prec=1e-5):
        N,p = X.shape
        X_bias = np.insert(X, 0, -1, axis=1)
        self.w = np.zeros(X_bias.shape[1])
        
        EQM = float('inf')

        for i in range(self.epoch):
            soma_erros_quadrados = 0
            for j in range(X_bias.shape[0]):
                u = X_bias[j] @ self.w
                e = d[j] - u
                soma_erros_quadrados += e**2

                if e != 0:
                    self.w = self.w + self.eta * e * X_bias[j]

            EQM2 = (1 / (2 * N)) * soma_erros_quadrados
            self.errors.append(EQM2)

            if abs(EQM - EQM2) <= prec:
                print(f"Treinamento convergido na época {i}")
                break

            EQM = EQM2
            
    def predict(self, X):
        X_bias = np.insert(X, 0, -1, axis=1)

        u =  X_bias @ self.w
        y = np.where(u >= 0, 1, -1)

        return y

