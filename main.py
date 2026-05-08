import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from models.simple_perceptron import simples_perceptron
from models.adaline import adaline
from models.mlp import MLP

data = np.loadtxt('Redes-Neurais-Artificiais\data\spiral_d (1).csv', delimiter=',')
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

#-----------------------------------------------

def matriz_confusao_manual(y_real, y_pred):
    # Garante que os dados estão no formato de classes (-1 e 1)
    y_pred_class = np.where(y_pred > 0, 1, -1).flatten()
    y_real = y_real.flatten()
    
    # Identifica as classes únicas (ex: [-1, 1])
    classes = np.unique(y_real)
    n_classes = len(classes)
    
    # Cria uma matriz vazia (n_classes x n_classes)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Preenche a matriz
    # Linha i = Classe Real | Coluna j = Classe Predita
    for i in range(n_classes):
        for j in range(n_classes):
            # Conta quantos da classe real i foram classificados como j
            conf_matrix[i, j] = np.sum((y_real == classes[i]) & (y_pred_class == classes[j]))
            
    return conf_matrix, classes

# 1. Preparação dos dados para o MLP
# O MLP com Tanh funciona melhor com labels entre -1 e 1 ou 0 e 1. 
# Se seus dados forem 0 e 1, d_mlp deve ter o shape (n_samples, 1)
d_mlp = y.reshape(-1, 1)

# 2. Instanciando e Treinando o MLP
# n_input=2 (x e y), n_hidden=15 (pode ajustar), n_output=1
modelo_mlp = MLP(n_input=2, n_hidden=65, n_output=1, learning_rate=0.00022222222222, n_epochs=70000)
modelo_mlp.fit(X, d_mlp)



# 3. Visualização dos Resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Gráfico 1: Fronteira de Decisão Complexa ---
# Criamos um grid de pontos para pintar o fundo do gráfico
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predizemos para cada ponto do grid
grid_points = np.c_[xx.ravel(), yy.ravel()]

y_pred_dataset = modelo_mlp.predict(X)
cm, labels = matriz_confusao_manual(d_mlp, y_pred_dataset)

# 2. Predição para o GRÁFICO (usando o grid de pontos)
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z_grid = modelo_mlp.predict(grid_points)

if Z_grid.shape[1] > 1:
    Z_grid = Z_grid[:, 0] 

Z_class = np.where(Z_grid > 0, 1, -1) 
Z_class = Z_class.reshape(xx.shape)

# Agora use o Z_class no contourf
ax1.contourf(xx, yy, Z_class, alpha=0.3, cmap='viridis')
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', ax=ax1, edgecolor='k')
ax1.set_title("MLP: Fronteira de Decisão Não-Linear")

# --- Gráfico 2: Histórico de Erros (MSE) ---
ax2.plot(modelo_mlp.errors, color='blue')
ax2.set_title("MLP: Histórico de Erro (MSE)")
ax2.set_xlabel("Época")
ax2.set_ylabel("Mean Squared Error")



plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusão (NumPy Puro)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()


# =====================================================================
# Questão 4: MLP — underfitting / overfitting (acrescentado abaixo;
# não altera o fluxo anterior do relatório)
# =====================================================================
from pathlib import Path


def train_test_split_np(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_test = int(X.shape[0] * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def minmax_scale_bipolar(X):
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    return 2 * ((X - x_min) / (x_max - x_min + 1e-12)) - 1


def confusion_matrix_binary(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def binary_metrics_from_cm(cm):
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return acc, sens, spec


def plot_learning_curve(errors, title):
    plt.figure(figsize=(7, 4))
    plt.plot(errors, color="blue")
    plt.title(title)
    plt.xlabel("Epocas")
    plt.ylabel("Erro")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_q4(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[-1, 1],
        yticklabels=[-1, 1],
    )
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()


def run_mlp_case(case_name, n_hidden, X_train, X_test, y_train, y_test):
    model = MLP(
        n_input=2,
        n_hidden=n_hidden,
        n_output=1,
        learning_rate=0.01,
        n_epochs=5000,
    )
    model.fit(X_train, y_train.reshape(-1, 1))
    y_pred_raw = model.predict(X_test).flatten()
    y_pred = np.where(y_pred_raw >= 0, 1, -1)
    cm = confusion_matrix_binary(y_test, y_pred)
    acc, sens, spec = binary_metrics_from_cm(cm)
    print(f"\n[{case_name}]")
    print(f"Acuracia: {acc:.4f}")
    print(f"Sensibilidade: {sens:.4f}")
    print(f"Especificidade: {spec:.4f}")
    plot_learning_curve(model.errors, f"{case_name} - Curva de aprendizado")
    plot_confusion_matrix_q4(cm, f"{case_name} - Matriz de confusao")


_csv_q4 = Path(__file__).resolve().parent / "data" / "spiral_d (1).csv"
_data_q4 = np.loadtxt(_csv_q4, delimiter=",")
X_q4 = minmax_scale_bipolar(_data_q4[:, :-1])
y_q4 = _data_q4[:, -1].astype(int)
X_train_q4, X_test_q4, y_train_q4, y_test_q4 = train_test_split_np(
    X_q4, y_q4, test_size=0.2, seed=42
)

run_mlp_case(
    "MLP Underfitting (3 neuronios)", 3, X_train_q4, X_test_q4, y_train_q4, y_test_q4
)
run_mlp_case(
    "MLP Overfitting (80 neuronios)", 80, X_train_q4, X_test_q4, y_train_q4, y_test_q4
)