import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from simple_perceptron import simples_perceptron
from adaline import adaline

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
if w[2] != 0: 
        x_vals = np.array([x1_min, x1_max])
        y_vals = (w[0] - w[1] * x_vals) / w[2]
        ax1.plot(x_vals, y_vals, '--r', label='Fronteira de Decisão')
    
ax1.set_title("Perceptron: Espaço de Decisão")
ax1.legend()


ax2.plot(perceptron_simples.errors, color='red')
ax2.set_title("Histórico de Erros (Soma por Época)")
ax2.set_xlabel("Época")
ax2.set_ylabel("Erro Absoluto")

plt.tight_layout()
plt.show()

# ---------------------------------------

adaline = adaline(learning_rate=0.0001, n_epochs=500)
adaline.fit(X, y, prec=1e-6)

    # 3. Visualização
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Subplot 1: Fronteira de Decisão ---
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', ax=ax1)
    
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
w = adaline.w
if w[2] != 0:
        x_vals = np.linspace(x1_min, x1_max, 100)
        # Equação: w0*(-1) + w1*x1 + w2*x2 = 0 -> x2 = (w0 - w1*x1) / w2
        # (Ajuste o sinal do w0 conforme sua inserção de bias -1)
        y_vals = (w[0] - w[1] * x_vals) / w[2]
        ax1.plot(x_vals, y_vals, '--r', label='Fronteira Adaline')
    
ax1.set_title("Adaline: Espaço de Decisão")
ax1.set_ylim(X[:, 1].min()-1, X[:, 1].max()+1) # Para não fugir da escala
ax1.legend()

    # --- Subplot 2: Evolução do EQM ---
ax2.plot(adaline.errors, color='blue')
ax2.set_title("Evolução do Erro Quadrático Médio (EQM)")
ax2.set_xlabel("Época")
ax2.set_ylabel("EQM")

plt.tight_layout()
plt.show()