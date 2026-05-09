GUIA RÁPIDO — O QUE FOI ACRESCENTADO NO main.py (QUESTÃO 4 ATÉ QUESTÃO 5)
=======================================================================
  
Este arquivo explica, de forma simples, o que foi ADICIONADO no `main.py`
para cumprir a Questão 4 e a Questão 5 (sem RBF), e o que cada função faz.

IMPORTANTE
---------
- A parte "principal" do `main.py` (plots do Perceptron, Adaline e MLP grande)
  continua existindo como base do relatório.
- Depois desses plots, o código entra nos blocos:
  (1) QUESTÃO 4 e depois (2) QUESTÃO 5.


=========================
QUESTÃO 4 (Under/Over MLP)
=========================
Objetivo: mostrar underfitting e overfitting/superdimensionamento no MLP.
O código roda DOIS casos (ex.: poucos neurônios e muitos neurônios) e, para
cada caso, calcula:
- Curva de aprendizado (erro ao longo das épocas)
- Matriz de confusão (no conjunto de teste 20%)
- Acurácia, sensibilidade, especificidade (no teste)

Funções criadas na Questão 4:

1) train_test_split_np(X, y, test_size=0.2, seed=42)
   - Embaralha os índices com uma seed fixa e separa em:
     treino (80%) e teste (20%).
   - Por que existe: para avaliar o modelo em dados que ele não viu.

2) minmax_scale_bipolar(X)
   - Normaliza as features para aproximadamente [-1, +1] (escala bipolar).
   - Por que existe: ajuda o treinamento do MLP/Adaline a ficar mais estável.

3) confusion_matrix_binary(y_true, y_pred)
   - Monta a matriz 2x2 usando classes -1 e +1:
       [[TN, FP],
        [FN, TP]]
   - Por que existe: base para as métricas e para o heatmap.

4) binary_metrics_from_cm(cm)
   - Calcula (no teste):
     - acurácia = (TP + TN) / total
     - sensibilidade = TP / (TP + FN)  (classe +1 como positiva)
     - especificidade = TN / (TN + FP) (classe -1 como negativa)

5) plot_learning_curve(errors, title)
   - Plota a curva de erro (lista `errors` guardada pelo MLP durante o fit).

6) plot_confusion_matrix_q4(cm, title)
   - Plota a matriz de confusão (heatmap) da Questão 4.

7) run_mlp_case(case_name, n_hidden, X_train, X_test, y_train, y_test)
   - Executa um “caso” do MLP:
     (a) cria MLP com n_hidden neurônios,
     (b) treina no treino,
     (c) prediz no teste e aplica limiar (>=0 vira +1; <0 vira -1),
     (d) calcula matriz de confusão e métricas,
     (e) plota curva + heatmap.
   - Por que existe: evita duplicar código para o caso under e para o over.

Execução da Questão 4:
- O código carrega o CSV novamente, normaliza em [-1,1], faz split 80/20 e roda:
  - MLP Underfitting (poucos neurônios)
  - MLP Overfitting/superdimensionado (muitos neurônios)


=========================================
QUESTÃO 5 (Monte Carlo: 500 rodadas 80/20)
=========================================
Objetivo: validar os modelos com muitas partições aleatórias.
Aqui a simulação é repetida R = 500 vezes. Em cada rodada:
- Sorteia um split 80/20 diferente (treino/teste)
- Treina Perceptron, Adaline e MLP (sem RBF)
- Avalia no teste com 5 métricas:
  acurácia, sensibilidade, especificidade, precisão e F1

Funções criadas/ usadas na Questão 5:

1) split_train_test_8020(X, y, rng, test_frac=0.2)
   - Similar ao split da Q4, mas aqui o RNG muda a cada rodada.
   - Por que existe: cada rodada precisa de um conjunto de teste diferente.

2) classification_metrics_plus_one_positive(y_true, y_pred)
   - Calcula 5 métricas no teste (classe +1 como positiva):
     - acurácia
     - sensibilidade (TP/(TP+FN))
     - especificidade (TN/(TN+FP))
     - precisão (TP/(TP+FP))
     - F1 (média harmônica de precisão e sensibilidade)
   - Por que existe: o enunciado pede exatamente essas métricas.

Execução da Questão 5 (parte principal):
- O código normaliza X em [-1,1] (bipolar).
- Cria matrizes para armazenar 500x5 métricas para cada modelo:
  resultado["Perceptron"], resultado["Adaline"], resultado["MLP"].
- Loop: for r in range(RODADAS_Q5):
  a) cria um RNG por rodada (seed base + r),
  b) faz split 80/20,
  c) treina e avalia:
     - Perceptron  -> predict -> métricas no teste
     - Adaline     -> predict -> métricas no teste
     - MLP         -> predict -> limiar -> métricas no teste
  d) imprime progresso a cada 50 rodadas.

Depois do loop:
- Imprime resumo por modelo (média, desvio, min, max) para cada métrica.
- Salva os arrays em: data/metricas_questao5.npz
  com as chaves: perceptron, adaline, mlp
  (cada um com shape 500x5).


NOTAS IMPORTANTES PARA O RELATÓRIO
---------------------------------
- “max = 1.0” em alguma rodada NÃO prova overfitting sozinho: pode ser um teste
  particularmente “fácil” naquela partição aleatória.
- Para discutir overfitting de forma formal, o ideal é comparar treino vs teste
  e/ou usar validação adicional.
- O tempo de execução pode ser alto (500 rodadas). Isso é esperado.

