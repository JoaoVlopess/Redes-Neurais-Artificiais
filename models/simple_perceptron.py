import numpy as np

class simples_perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=20):
        self.eta = learning_rate
        self.epoch = n_epochs
        self.w = None
        self.errors = []

    def fit(self, X, d, patience = 5):
        X_bias = np.insert(X, 0, -1, axis=1)
        self.w = np.zeros(X_bias.shape[1])

        contador = 0
        best_err = float('inf')

        for i in range(self.epoch):
            total_error_epoch = 0

            for j in range(X_bias.shape[0]):
                u =  X_bias[j] @ self.w
                y = 1 if u >= 0 else -1
                e = d[j] - y

                if e != 0:
                    self.w = self.w + self.eta * e * X_bias[j]
                    total_error_epoch += abs(e)

            self.errors.append(total_error_epoch)
            if total_error_epoch < best_err:
                best_err = total_error_epoch
                contador = 0
            else:
                contador += 1

            if total_error_epoch == 0:
                print(f"Treinamento concluído na época {i} (Erro zero atingido).")
                break


            if contador >= patience:
                print(f"Early Stopping na época {i}. O erro não melhora há {patience} épocas.")
                break

    def predict(self, X):
        X_bias = np.insert(X, 0, -1, axis=1)

        u =  X_bias @ self.w
        y = np.where(u >= 0, 1, -1)

        return y




               

