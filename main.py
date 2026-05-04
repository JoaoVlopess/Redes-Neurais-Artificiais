import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from simple_perceptron import simples_perceptron

data = np.loadtxt('data/spiral_d (1).csv', delimiter=',')
X = data[:,:-1]
y = data[:, -1]

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y.flatten(), palette='viridis')
plt.title("Visualização dos Dados de Entrada")
plt.show()

perceptron_simples = simples_perceptron(learning_rate=0.00005, n_epochs=100)
perceptron_simples.fit(X, y, patience=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', ax=ax1)
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()

w = perceptron_simples.w
if w[2] != 0: # Evitar divisão por zero
        x_vals = np.array([x1_min, x1_max])
        # Lembre-se: w[0] é o bias (entrada -1), w[1] é X1, w[2] é X2
        y_vals = (w[0] - w[1] * x_vals) / w[2]
        ax1.plot(x_vals, y_vals, '--r', label='Fronteira de Decisão')
    
ax1.set_title("Perceptron: Espaço de Decisão")
ax1.legend()

    # --- Subplot 2: Evolução do Erro ---
ax2.plot(perceptron_simples.errors, color='red')
ax2.set_title("Histórico de Erros (Soma por Época)")
ax2.set_xlabel("Época")
ax2.set_ylabel("Erro Absoluto")

plt.tight_layout()
plt.show()