# Checkpoint 5 – Previsão de cancelamento de clientes (churn)

**Data Science & Statistical Computing – FIAP 2026 – Prof. Jones Egydio**

Comparação de Random Forest, XGBoost e LightGBM sob um mesmo protocolo experimental, com tuning por Grid Search e Optuna, escolha do modelo final, avaliação em teste isolado e aplicação Streamlit.

## Integrante

| Nome | RM |
|---|---|
| Pedro Henrique Vasco Antonieti | 570815 |

## Links

- **Repositório GitHub:** <https://github.com/P3terHenry/cp5-churn-telco>
- **Aplicação Streamlit:** _a preencher_

## Problema

Uma operadora de telefonia e internet quer saber, a partir do perfil, dos serviços contratados e dos dados da conta, quais clientes têm maior risco de cancelar o serviço, para priorizar ações de retenção.

- **Tipo:** classificação binária.
- **Variável-alvo:** `cancelou` (1 = cancelou no último mês, 0 = permaneceu). 26,5% dos clientes cancelaram.
- **Métrica principal:** ROC-AUC. Auxiliares: recall, precisão, F1 e acurácia no limiar 0,5.

## Dados

*Telco Customer Churn*, base de exemplo publicada pela IBM (licença Apache-2.0), com 7.043 clientes e 21 colunas.

- Fonte: <https://github.com/IBM/telco-customer-churn-on-icp4d> (também no Kaggle: `blastchar/telco-customer-churn`).
- O arquivo está versionado em [`data/Telco-Customer-Churn.csv`](data/Telco-Customer-Churn.csv); não é preciso baixar nada.
- Os dados descrevem uma operadora fictícia e não contêm informação pessoal identificável.

No notebook, colunas e categorias são traduzidas para português durante o data wrangling.

## Estrutura do repositório

| Arquivo | Conteúdo |
|---|---|
| `Checkpoint05_RF_XGBoost_LightGBM.ipynb` | Notebook do checkpoint, com os sete exercícios respondidos |
| `app.py` | Aplicação Streamlit |
| `modelo_final.joblib` | Pipeline final (one-hot + XGBoost) gerado pelo notebook |
| `exemplos_paridade.csv` | Cinco clientes do teste de consistência, com a previsão do notebook |
| `data/Telco-Customer-Churn.csv` | Base de dados |
| `requirements.txt` | Dependências com as versões usadas |

## Instalação e execução

Desenvolvido com Python 3.14.

```bash
git clone https://github.com/P3terHenry/cp5-churn-telco.git
cd cp5-churn-telco
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Aplicação:**

```bash
streamlit run app.py
```

**Notebook** (regenera `modelo_final.joblib` e `exemplos_paridade.csv`; leva cerca de 3 minutos):

```bash
jupyter notebook Checkpoint05_RF_XGBoost_LightGBM.ipynb
```

Execute com *Run All*. As sementes são fixas, então as métricas se repetem a cada execução com as versões do `requirements.txt`; só os tempos de treino e de busca variam.

## Protocolo experimental

- Separação 80/20 estratificada (`random_state=42`); o teste só é usado no Exercício 7, uma única vez.
- Validação cruzada estratificada de 5 folds dentro do treino, com o mesmo objeto de folds nos baselines, no Grid Search, no Optuna e na curva de aprendizado.
- Mesma preparação para os três algoritmos, dentro de um `Pipeline`: one-hot nas variáveis categóricas, ajustado só no treino de cada fold.
- Grid Search (18 a 24 combinações por modelo) e Optuna (30 trials por modelo, amostrador TPE com semente fixa).

## Resumo dos resultados

ROC-AUC nos 5 folds de validação cruzada (média ± desvio) e gap entre treino e validação:

| Configuração | AUC treino | AUC validação | Gap |
|---|---|---|---|
| RF – baseline | 1,000 | 0,825 ± 0,012 | 0,175 |
| XGBoost – baseline | 0,989 | 0,821 ± 0,006 | 0,169 |
| LightGBM – baseline | 0,963 | 0,836 ± 0,009 | 0,127 |
| RF – Grid Search | 0,877 | 0,848 ± 0,010 | 0,029 |
| RF – Optuna | 0,879 | 0,848 ± 0,011 | 0,031 |
| **XGBoost – Grid Search (escolhida)** | **0,863** | **0,849 ± 0,012** | **0,014** |
| XGBoost – Optuna | 0,865 | 0,850 ± 0,012 | 0,016 |
| LightGBM – Grid Search | 0,865 | 0,849 ± 0,011 | 0,016 |
| LightGBM – Optuna | 0,866 | 0,849 ± 0,012 | 0,017 |

- Os três baselines sobreajustam. Depois do tuning, as seis configurações ficam empatadas dentro da variação entre folds.
- Grid Search e Optuna chegam ao mesmo patamar; o Optuna custa mais tempo e ganha na quarta casa decimal.
- **Modelo final:** XGBoost com 100 árvores, profundidade 2 e taxa de aprendizado 0,1. Entre as configurações empatadas, é a mais simples e a de menor gap.
- **Teste final:** ROC-AUC de 0,845 (0,849 ± 0,012 na validação), F1 de 0,573, recall de 0,505, precisão de 0,661 e acurácia de 0,800.
- Variáveis mais importantes: contrato mensal, internet por fibra óptica, pagamento por cheque eletrônico e tempo como cliente.

## Teste de paridade notebook–Streamlit

A aplicação carrega `modelo_final.joblib`, o mesmo pipeline avaliado no notebook, e não refaz nenhuma preparação por conta própria. Para conferir:

1. Abra a aplicação e, no primeiro campo, selecione o cliente de índice **3380**.
2. A interface preenche as entradas desse cliente e exibe probabilidade de cancelamento de **0,9017**, igual à do Exercício 7 do notebook, com a confirmação de que as previsões coincidem.

Os outros quatro clientes do teste de consistência também podem ser selecionados.
