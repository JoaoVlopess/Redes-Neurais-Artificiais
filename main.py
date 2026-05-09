import contextlib
import io
from collections import defaultdict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from models.simple_perceptron import simples_perceptron
from models.adaline import adaline as Adaline
from models.mlp import MLP

# Caminho do CSV (vale para relatório + questões juntos)
CSV_DATA_PATH = Path(__file__).resolve().parent / "data" / "spiral_d (1).csv"

# MLP principal (painel duplo + matriz): mesma seed = mesmos erros nas figuras entre execucoes
SEED_MLP_PRINCIPAL = 42

# ---------- Questão 5: validação Monte Carlo (R rodadas 80/20) ----------
RODADAS_Q5 = 500
SEED_Q5_MASTER = 20260508
# MLP na Q5: mais épocas + LR um pouco menor que 0.01 tende a reduzir variância nas 500 rodadas
N_EPOCHS_MLP_Q5 = 1200
LR_MLP_Q5 = 0.005

data = np.loadtxt(CSV_DATA_PATH, delimiter=",")
X = data[:,:-1]
y = data[:, -1]

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y.flatten(), palette='viridis')
plt.title("Visualização dos Dados de Entrada")
plt.show()

perceptron_simples = simples_perceptron(learning_rate=0.00005, n_epochs=100)
with contextlib.redirect_stdout(io.StringIO()):
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

modelo_adaline = Adaline(learning_rate=0.0001, n_epochs=500)
with contextlib.redirect_stdout(io.StringIO()):
    modelo_adaline.fit(X, y, prec=1e-6)

    # 3. Visualização
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Subplot 1: Fronteira de Decisão ---
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', ax=ax1)
    
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
w = modelo_adaline.w
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
ax2.plot(modelo_adaline.errors, color='blue')
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
# n_input=2 (x e y), n_hidden=65, n_output=1
np.random.seed(SEED_MLP_PRINCIPAL)
modelo_mlp = MLP(n_input=2, n_hidden=65, n_output=1, learning_rate=0.00022222222222, n_epochs=70000)
print("Treinando MLP principal (70000 epocas): aguarde; o modelo imprime a cada 100 epocas.")
modelo_mlp.fit(X, d_mlp)
print("MLP principal concluído.")



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
    with contextlib.redirect_stdout(io.StringIO()):
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


_data_q4 = np.loadtxt(CSV_DATA_PATH, delimiter=",")
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


# =====================================================================
# Questão 5: R rodadas, 80/20 aleatório, métricas no teste (Perceptron,
# Adaline, MLP — sem RBF). Executa sempre após a Questão 4 (pode demorar).
# =====================================================================


def split_train_test_8020(X, y, rng, test_frac=0.2):
    n = X.shape[0]
    n_test = max(1, int(round(n * test_frac)))
    idx = rng.permutation(n)
    tst = idx[:n_test]
    trn = idx[n_test:]
    return X[trn], X[tst], y[trn], y[tst]


def classification_metrics_plus_one_positive(y_true, y_pred):
    """Classe +1 como positiva; classe -1 como negativa (convenção Questão 5)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))
    denom = tp + tn + fp + fn
    acc = (tp + tn) / denom if denom else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    if sens == 0.0 and prec == 0.0:
        f1 = 0.0
    else:
        f1 = (2 * prec * sens) / (prec + sens)
    return acc, sens, spec, prec, f1


_Xt = np.loadtxt(CSV_DATA_PATH, delimiter=",")
X5 = minmax_scale_bipolar(_Xt[:, :-1])
y5 = _Xt[:, -1].astype(int)
R = int(RODADAS_Q5)
nomes_metricas = ("acuracia", "sensibilidade", "especificidade", "precisao", "f1")
resultado = {}
resultado["Perceptron"] = np.zeros((R, 5))
resultado["Adaline"] = np.zeros((R, 5))
resultado["MLP"] = np.zeros((R, 5))

for r in range(R):
    # Cada rodada treina Perceptron + Adaline + MLP (silenciado); pode levar bastante tempo.
    print(f"Questao 5 — treinando rodada {r + 1}/{R}...", flush=True)
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        rng = np.random.default_rng(SEED_Q5_MASTER + r)
        X_tr, X_te, y_tr, y_te = split_train_test_8020(X5, y5, rng)

        ps = simples_perceptron(learning_rate=0.00005, n_epochs=100)
        ps.fit(X_tr, y_tr, patience=10)
        resultado["Perceptron"][r, :] = classification_metrics_plus_one_positive(
            y_te, ps.predict(X_te)
        )

        ada = Adaline(learning_rate=0.0001, n_epochs=500)
        ada.fit(X_tr, y_tr, prec=1e-6)
        resultado["Adaline"][r, :] = classification_metrics_plus_one_positive(
            y_te, ada.predict(X_te)
        )

        seed_mlp = int(SEED_Q5_MASTER + 100000 + r)
        np.random.seed(seed_mlp % (2**32 - 1))
        mlp = MLP(
            n_input=2,
            n_hidden=65,
            n_output=1,
            learning_rate=LR_MLP_Q5,
            n_epochs=N_EPOCHS_MLP_Q5,
        )
        mlp.fit(X_tr, y_tr.reshape(-1, 1))
        y_hat = np.where(mlp.predict(X_te).ravel() >= 0.0, 1, -1)
        resultado["MLP"][r, :] = classification_metrics_plus_one_positive(y_te, y_hat)

    print(f"Questao 5 — rodada {r + 1}/{R} concluida.", flush=True)

print("\n=== Questão 5: resumo (média ± desvio, teste apenas) ===")
for nome, arr in resultado.items():
    print(f"\n[{nome}]")
    for j, nome_m in enumerate(nomes_metricas):
        col = arr[:, j]
        print(f"  {nome_m}: media={np.mean(col):.4f}, std={np.std(col):.4f}, "
              f"min={np.min(col):.4f}, max={np.max(col):.4f}")

# --- Questão 7 (PDF): uma TABELA por métrica — linhas = modelos; colunas = média, desvio, min, max ---
print(
    "\n"
    + "*" * 60
    + "\n>>> AGORA: QUESTAO 7 — tabelas (media, desvio, min, max) por METRICA\n"
    + "    (apos as 500 rodadas; antes do Item 1.6 / figuras)\n"
    + "*" * 60
    + "\n",
    flush=True,
)
_ordem_q7 = ["Perceptron", "Adaline", "MLP"]
_linhas_q7 = []
for j, nome_m in enumerate(nomes_metricas):
    print(f"\n--- {nome_m.upper()} ---")
    hdr = f"{'Modelo':<14} {'Media':>10} {'Desvio':>10} {'Min':>10} {'Max':>10}"
    print(hdr)
    _linhas_q7.append("")
    _linhas_q7.append(f"=== {nome_m.upper()} ===")
    _linhas_q7.append(hdr)
    for nome in _ordem_q7:
        col = resultado[nome][:, j]
        linha = (
            f"{nome:<14} {np.mean(col):10.4f} {np.std(col):10.4f} "
            f"{np.min(col):10.4f} {np.max(col):10.4f}"
        )
        print(linha)
        _linhas_q7.append(linha)
_questao7_txt = CSV_DATA_PATH.parent / "questao7_tabelas_metricas.txt"
_questao7_txt.write_text("\n".join(_linhas_q7), encoding="utf-8")
print(f"\n(Também salvo em {_questao7_txt} para copiar no relatório.)", flush=True)
print(
    "*" * 60
    + "\n>>> FIM DA QUESTAO 7 — em seguida: salvar .npz e Item 1.6 (figuras)\n"
    + "*" * 60
    + "\n",
    flush=True,
)

resultado_path = CSV_DATA_PATH.parent / "metricas_questao5.npz"
np.savez(
    resultado_path,
    perceptron=resultado["Perceptron"],
    adaline=resultado["Adaline"],
    mlp=resultado["MLP"],
)
print(f"\nResultados salvos em: {resultado_path} (perceptron, adaline, mlp; colunas = 5 metricas)")

# Questão 5 — Item 6: sns.heatmap (API oficial) sobre matrizes NumPy montadas pela equipe;
# confusão 2x2 manual + uma folha só com todas as rodadas relevantes.


def replicar_split_e_treinos_q5(numero_rodada):
    rng = np.random.default_rng(SEED_Q5_MASTER + int(numero_rodada))
    X_tr, X_te, y_tr, y_te = split_train_test_8020(X5, y5, rng)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ps = simples_perceptron(learning_rate=0.00005, n_epochs=100)
        ps.fit(X_tr, y_tr, patience=10)
        y_hat_p = ps.predict(X_te)

        ada = Adaline(learning_rate=0.0001, n_epochs=500)
        ada.fit(X_tr, y_tr, prec=1e-6)
        y_hat_a = ada.predict(X_te)

        seed_mlp = int(SEED_Q5_MASTER + 100000 + int(numero_rodada))
        np.random.seed(seed_mlp % (2**32 - 1))
        mlp = MLP(
            n_input=2,
            n_hidden=65,
            n_output=1,
            learning_rate=LR_MLP_Q5,
            n_epochs=N_EPOCHS_MLP_Q5,
        )
        mlp.fit(X_tr, y_tr.reshape(-1, 1))
        y_hat_m = np.where(mlp.predict(X_te).ravel() >= 0.0, 1, -1)

    cms = {
        "Perceptron": confusion_matrix_binary(y_te, y_hat_p),
        "Adaline": confusion_matrix_binary(y_te, y_hat_a),
        "MLP": confusion_matrix_binary(y_te, y_hat_m),
    }
    curvas = {
        "Perceptron": list(ps.errors),
        "Adaline": list(ada.errors),
        "MLP": list(mlp.errors),
    }
    return y_te, cms, curvas


print("\n=== Questão 5 — Item 6: extremos, heatmaps seaborn.confusion + resumo modelo×métrica ===")

NOMES_MODELOS_ITEM6 = ["Perceptron", "Adaline", "MLP"]
ROT_METRICAS_LEGIVEL = [
    "Acuracia",
    "Sensibilidade",
    "Especificidade",
    "Precisao",
    "F1",
]

M_max = np.zeros((3, 5))
M_min = np.zeros((3, 5))
I_max = np.zeros((3, 5), dtype=int)
I_min = np.zeros((3, 5), dtype=int)
for mi, modelo in enumerate(NOMES_MODELOS_ITEM6):
    tbl = resultado[modelo]
    for j in range(5):
        M_max[mi, j] = np.max(tbl[:, j])
        M_min[mi, j] = np.min(tbl[:, j])
        I_max[mi, j] = int(np.argmax(tbl[:, j]))
        I_min[mi, j] = int(np.argmin(tbl[:, j]))

fig_res, axes_res = plt.subplots(2, 2, figsize=(13, 9))
sns.heatmap(
    M_max,
    annot=True,
    fmt=".4f",
    cmap="rocket",
    xticklabels=ROT_METRICAS_LEGIVEL,
    yticklabels=NOMES_MODELOS_ITEM6,
    linewidths=0.5,
    linecolor="white",
    ax=axes_res[0, 0],
    cbar_kws={"label": "Valor"},
)
axes_res[0, 0].set_title("Maior valor por modelo e métrica (500 rodadas)")

sns.heatmap(
    M_min,
    annot=True,
    fmt=".4f",
    cmap="mako_r",
    xticklabels=ROT_METRICAS_LEGIVEL,
    yticklabels=NOMES_MODELOS_ITEM6,
    linewidths=0.5,
    linecolor="white",
    ax=axes_res[0, 1],
    cbar_kws={"label": "Valor"},
)
axes_res[0, 1].set_title("Menor valor por modelo e métrica (500 rodadas)")

sns.heatmap(
    I_max.astype(float),
    annot=I_max,
    fmt="d",
    cmap="viridis",
    xticklabels=ROT_METRICAS_LEGIVEL,
    yticklabels=NOMES_MODELOS_ITEM6,
    linewidths=0.5,
    linecolor="white",
    ax=axes_res[1, 0],
    cbar_kws={"label": "Índice da rodada"},
)
axes_res[1, 0].set_title("Rodada onde ocorreu o MÁximo (mesma seed Monte Carlo da Q5)")

sns.heatmap(
    I_min.astype(float),
    annot=I_min,
    fmt="d",
    cmap="viridis",
    xticklabels=ROT_METRICAS_LEGIVEL,
    yticklabels=NOMES_MODELOS_ITEM6,
    linewidths=0.5,
    linecolor="white",
    ax=axes_res[1, 1],
    cbar_kws={"label": "Índice da rodada"},
)
axes_res[1, 1].set_title("Rodada onde ocorreu o MÍnimo")

fig_res.suptitle(
    "Resumo Questão 5 — Item 6 (matplotlib + seaborn.heatmap conforme documentação oficial)",
    fontsize=11,
    y=1.02,
)
plt.tight_layout()
plt.show()

casos_item6_por_rodada = defaultdict(list)
for modelo_ref, mat in resultado.items():
    for indice_metrica, nome_m in enumerate(nomes_metricas):
        valor_max = float(np.max(mat[:, indice_metrica]))
        valor_min = float(np.min(mat[:, indice_metrica]))
        rodada_max = int(np.argmax(mat[:, indice_metrica]))
        rodada_min = int(np.argmin(mat[:, indice_metrica]))

        casos_item6_por_rodada[rodada_max].append(
            f"{modelo_ref} | {nome_m} | max={valor_max:.4f}"
        )
        casos_item6_por_rodada[rodada_min].append(
            f"{modelo_ref} | {nome_m} | min={valor_min:.4f}"
        )

rodadas_item6 = sorted(casos_item6_por_rodada.keys())
print(
    "Item 6 — rodadas únicas onde algum extreme cai:",
    rodadas_item6,
)
cache_item6 = {rnd: replicar_split_e_treinos_q5(rnd) for rnd in rodadas_item6}

altura_linha_cm = max(3.0, min(4.8, 100.0 / max(len(rodadas_item6), 1)))
fig_cm, axes_cm = plt.subplots(
    len(rodadas_item6),
    3,
    figsize=(11, len(rodadas_item6) * altura_linha_cm),
    squeeze=False,
)
etiq_bin = [-1, 1]
for ri, rnd in enumerate(rodadas_item6):
    _, cms, _ = cache_item6[rnd]
    for ci, nome in enumerate(NOMES_MODELOS_ITEM6):
        sns.heatmap(
            cms[nome],
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=etiq_bin,
            yticklabels=etiq_bin,
            linewidths=0.5,
            linecolor="white",
            ax=axes_cm[ri, ci],
            cbar=ci == 2,
            annot_kws={"size": 9},
        )
        axes_cm[ri, ci].set_xlabel("Predito")
        axes_cm[ri, ci].set_ylabel("Real")
        axes_cm[0, ci].set_title(nome)
    axes_cm[ri, 1].annotate(
        f"particao idx {rnd}",
        xy=(0.5, 1.06),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="dimgray",
    )

fig_cm.suptitle(
    "Item 6 — matrizes de confusão (TN/FP/FN/TP construídas em NumPy; plot com sns.heatmap)",
    fontsize=11,
)
plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()

altura_linha_lc = max(2.6, min(5.5, 90.0 / max(len(rodadas_item6), 1)))
fig_lc, axes_lc = plt.subplots(
    len(rodadas_item6),
    3,
    figsize=(13, len(rodadas_item6) * altura_linha_lc),
    squeeze=False,
)
ylabel_lc = ("Soma |erro|/epoca", "EQM", "MSE")
for ri, rnd in enumerate(rodadas_item6):
    _, _, curvas = cache_item6[rnd]
    for ci, nome in enumerate(NOMES_MODELOS_ITEM6):
        axes_lc[ri, ci].plot(curvas[nome], lw=1.0)
        axes_lc[0, ci].set_title(nome)
        axes_lc[ri, ci].set_ylabel(ylabel_lc[ci])
        axes_lc[ri, ci].set_xlabel("Epoca")
    axes_lc[ri, 1].annotate(
        f"particao idx {rnd}",
        xy=(0.5, 1.04),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="dimgray",
    )

fig_lc.suptitle("Item 6 — curvas de aprendizado (mesmo split das linhas da figura anterior)", fontsize=11)
plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.show()