================================================================================
GUIA COMPLETO — main.py, MODELS/ E ITENS DO TRABALHO (ESP Q5 + Q4 + 1.6 + 1.7)
================================================================================
Este arquivo resume o que o projeto faz para um colega estudar sem ficar perdido
no código. Bibliotecas externas usadas: numpy, matplotlib, seaborn (+ pathlib,
contextlib, io, collections — biblioteca padrão do Python). Os modelos em
models/* são NumPy puro; RBF não está implementado (conforme orientação).

--------------------------------------------------------------------------------
1. ESTRUTURA DE PASTAS
--------------------------------------------------------------------------------
Redes-Neurais-Artificiais/
  main.py              → script principal: dados, relatório visual, Q4, Q5, Q7, item 1.6
  models/
    simple_perceptron.py → classe simples_perceptron (fit/predict, bias -1 na coluna)
    adaline.py           → classe adaline (EQM, fit/predict)
    mlp.py               → classe MLP (tanh, MSE, backprop; errors = MSE por época)
  data/
    spiral_d (1).csv   → CSV usado (caminho em CSV_DATA_PATH no main)
    metricas_questao5.npz → gerado após as 500 rodadas (arrays 500×5)
    questao7_tabelas_metricas.txt → gerado após o bloco “Questão 7” (tabelas texto)

--------------------------------------------------------------------------------
2. ORDEM DE EXECUÇÃO NO main.py (DE CIMA PARA BAIXO)
--------------------------------------------------------------------------------
A) Demos no conjunto COMPLETO (sem split) — base do relatório / questão 1.2–1.3
   - Carrega CSV → X, y
   - Scatter inicial (seaborn)
   - Perceptron em todo X,y → figuras fronteira + errors (stdout do fit pode ser
     silenciado com redirect_stdout)
   - Adaline idem
   - MLP grande (n_epochs alto, ex.: 70000) → fronteira no grid + errors + heatmap
     com matriz_confusao_manual (matriz genérica por classes)

B) QUESTÃO 4 — Underfitting / Overfitting só no MLP
   - Recarrega CSV, minmax_scale_bipolar, split FIXO 80/20 (seed=42)
   - run_mlp_case → dois MLPs (poucos vs muitos neurônios ocultos)

C) QUESTÃO 5 — Monte Carlo R=500
   - X5,y5 bipolar + loop r=0..499
   - Preenche resultado[modelo][r, :] com 5 métricas no TESTE
   - Prints “treinando rodada” / “concluida” + resumo por modelo

D) QUESTÃO 7 (item do PDF após as 500 rodadas — tabelas estatísticas)
   - Banner no terminal “>>> AGORA: QUESTAO 7”
   - Cinco “mini-tabelas” (uma por métrica): linhas = modelos; colunas =
     média, desvio, min, max sobre as 500 rodadas
   - Salva data/questao7_tabelas_metricas.txt
   - Banner “FIM DA QUESTAO 7”

E) Salvar NPZ
   - data/metricas_questao5.npz (perceptron, adaline, mlp)

F) ITEM 1.6 (enunciato: extremos + confusão + curvas)
   - Painel 2×2 heatmaps: máx/mín por modelo×métrica + índices de rodada
   - Lista rodadas únicas “extremas” → cache retreina cada índice uma vez
   - Figura grade: confusão 2×2 × 3 modelos × N rodadas
   - Figura grade: curvas errors × 3 modelos × N rodadas

Observação: no PDF a ordem dos itens é 1.6 antes de 1.7; no código a parte
numérica da “Questão 7” roda ANTES do item 1.6 para já gravar tabelas/.npz
antes da parte pesada de figuras. Os dados são os mesmos (resultado já cheio).

--------------------------------------------------------------------------------
3. CONSTANTES IMPORTANTES (TOPO DO main.py)
--------------------------------------------------------------------------------
CSV_DATA_PATH     → caminho do spiral_d
SEED_MLP_PRINCIPAL → reprodutibilidade do MLP “demo” inicial
RODADAS_Q5 = 500
SEED_Q5_MASTER    → base do RNG: rodada r usa default_rng(SEED_Q5_MASTER + r)
N_EPOCHS_MLP_Q5, LR_MLP_Q5 → hiperparâmetros do MLP dentro do Monte Carlo
nomes_metricas    → ordem das 5 colunas em resultado:
  acuracia, sensibilidade, especificidade, precisao, f1

--------------------------------------------------------------------------------
4. FUNÇÕES — O QUE FAZEM E QUEM USA QUEM
--------------------------------------------------------------------------------

--- Parte “demo” / relatório inicial ---
matriz_confusao_manual(y_real, y_pred)
  → Matriz de confusão genérica (classes em linhas/colunas), por contagem.
  → Usada no heatmap do MLP treinado em TODO o dataset (antes da Q4).

--- Questão 4 ---
train_test_split_np(X, y, test_size, seed)
  → Split 80/20 com seed fixa (reprodutível).
  → Usada só no bloco _data_q4 / X_train_q4, X_test_q4, ...

minmax_scale_bipolar(X)
  → Features em [-1, 1].
  → Usada em Q4 (X_q4) e em Q5 (X5).

confusion_matrix_binary(y_true, y_pred)
  → Matriz 2×2 com convenção [[TN, FP],[FN, TP]] para classes -1 e +1.
  → Usada em: run_mlp_case (Q4); replicar_split_e_treinos_q5 (item 1.6).

binary_metrics_from_cm(cm)
  → acc, sens, spec a partir da matriz 2×2.
  → Usada em run_mlp_case.

plot_learning_curve(errors, title)
  → plt.plot(errors, ...) para o MLP da Q4.

plot_confusion_matrix_q4(cm, title)
  → sns.heatmap da matriz Q4.

run_mlp_case(case_name, n_hidden, ..., y_train, y_test)
  → Instancia MLP(n_hidden), fit no treino, predição no teste com limiar ≥0 → ±1,
     confusion_matrix_binary, binary_metrics_from_cm, depois plot_learning_curve e
     plot_confusion_matrix_q4.
  → Chamada duas vezes: under (ex.: 3 neurônios) e over (ex.: 80 neurônios).

--- Questão 5 + item 1.6 ---
split_train_test_8020(X, y, rng, test_frac=0.2)
  → Split 80/20 com numpy Generator passado de fora (cada rodada = RNG novo).
  → Usada no loop das 500 rodadas e dentro de replicar_split_e_treinos_q5.

classification_metrics_plus_one_positive(y_true, y_pred)
  → Retorna tupla (acc, sens, spec, prec, f1) com classe +1 como positiva.
  → Usada no loop Q5 para preencher cada linha de resultado[...].

replicar_split_e_treinos_q5(numero_rodada)
  → Recria EXATAMENTE o split da rodada `numero_rodada`:
     rng = default_rng(SEED_Q5_MASTER + numero_rodada).
  → Treina Perceptron, Adaline, MLP com os MESMOS hiperparâmetros do loop Q5
     (seed do MLP = SEED_Q5_MASTER + 100000 + numero_rodada).
  → Retorna: y_te (não usado depois), dict cms com confusion_matrix_binary por
     modelo, dict curvas com list(ps.errors), list(ada.errors), list(mlp.errors).
  → Os prints dos fits podem ir para buffer (redirect_stdout) para não poluir.

--- Item 1.6 (bloco principal, não é uma função nomeada) ---
  - Calcula M_max, M_min, I_max, I_min a partir de resultado[modelo][:, j].
  - Monta fig_res com 4 sns.heatmap (valores máx/mín e índices de rodada).
  - defaultdict agrupa todos os (rodada_max, rodada_min) por índice de rodada.
  - cache_item6[rnd] = replicar_split_e_treinos_q5(rnd) — uma vez por rodada.
  - fig_cm: para cada rnd e cada modelo, sns.heatmap da matriz 2×2.
  - fig_lc: plt.plot das curvas em cache_item6[rnd]["Perceptron"/...].

--------------------------------------------------------------------------------
5. O DICIONÁRIO "resultado" (CORAÇÃO DA Q5, Q7 E ITEM 1.6)
--------------------------------------------------------------------------------
resultado["Perceptron"], ["Adaline"], ["MLP"]  → cada um shape (500, 5).
  Linha r = métricas no teste da rodada r.
  Coluna j = nomes_metricas[j].

A Questão 7 (tabelas no terminal + .txt) faz, para cada coluna j:
  np.mean/ std/ min/ max sobre resultado[nome][:, j].

O item 1.6 usa o mesmo resultado para argmax/argmin por modelo e métrica.

--------------------------------------------------------------------------------
6. MODELOS (models/) — PAPEL RÁPIDO
--------------------------------------------------------------------------------
simples_perceptron: insere coluna -1 (bias), atualiza pesos se erro ≠ 0,
  errors.append(soma |erro| na época).

adaline: EQM por época em errors; predict com limiar em u≥0 → ±1.

MLP: uma camada oculta tanh; errors.append(MSE médio por época); predict tanh na saída.

Nenhuma alteração obrigatória nesses arquivos para rodar o main; o main só importa
e instancia.

--------------------------------------------------------------------------------
7. ARQUIVOS GERADOS AO RODAR ATÉ O FIM
--------------------------------------------------------------------------------
data/metricas_questao5.npz
  Chaves: perceptron, adaline, mlp — arrays 500×5.

data/questao7_tabelas_metricas.txt
  Texto com as cinco seções (uma por métrica) para copiar no relatório.

--------------------------------------------------------------------------------
8. DICAS DE EXECUÇÃO E DEPURAÇÃO
--------------------------------------------------------------------------------
- plt.show() bloqueia até fechar a janela; o script “para” até você fechar.
- O MLP demo com muitas épocas demora; entre demos e Q5 há treinos longos.
- Loop Q5: cada rodada imprime início/fim; dentro usa redirect_stdout nos fits para
  não encher o terminal de Early Stopping / épocas do MLP.
- Item 1.6 retreina só nas rodadas “extremas” distintas — ainda assim pode demorar.

--------------------------------------------------------------------------------
9. MAPEAMENTO ENUNCIATO (PRIMEIRA ETAPA) ↔ CÓDIGO
--------------------------------------------------------------------------------
1.2 Scatter inicial       → início do main (sns.scatterplot)
1.3 Perceptron / Adaline   → blocos com fronteira + errors
1.3 MLP                    → treino global + contourf + errors + heatmap
1.4 Under/over MLP         → Questão 4 (run_mlp_case × 2)
1.5 Monte Carlo 500        → loop resultado + métricas no teste
1.6 Extremos + confusão + curvas → item 1.6 (heatmaps + grades + replicar_*)
1.7 Tabelas média/desvio/min/max → bloco “Questão 7” + questao7_tabelas_metricas.txt

================================================================================
FIM DO GUIA
================================================================================
