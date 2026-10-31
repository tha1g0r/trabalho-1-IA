# Trabalho 1
**Disciplina:** Inteligência Artificial  
**Atividade:** Trabalho Prático - Análise Comparativa de Algoritmos de Classificação  
**Modalidade:** Em grupo (até 3 integrantes)  

---

## 1. Objetivo
O objetivo deste trabalho prático é aplicar os conceitos fundamentais de Aprendizado de Máquina Supervisionado na resolução de problemas reais de classificação. Os estudantes deverão realizar o ciclo completo de experimentação: seleção e pré-processamento de dados, treinamento de modelos, avaliação quantitativa por meio de métricas de desempenho e análise crítica comparativa dos resultados obtidos.

---

## 2. Descrição da Atividade
Cada grupo deverá desenvolver um estudo experimental completo seguindo as etapas abaixo:

### 2.1. Seleção das Bases de Dados
- Escolher 2 (duas) bases de dados distintas voltadas para problemas de classificação (binária ou multiclasse).
- As bases podem ser obtidas em repositórios públicos reconhecidos (como UCI Machine Learning Repository, Kaggle ou OpenML).
- Recomenda-se a escolha de bases com características diferentes para enriquecer a discussão no relatório.

### 2.2. Algoritmos de Aprendizado
- Selecionar e treinar pelo menos 2 (dois) algoritmos de classificação distintos em ambas as bases de dados.
- **Requisito Obrigatório:** Um dos algoritmos escolhidos deve ser, obrigatoriamente, uma **Rede Neural Artificial** (ex.: Multilayer Perceptron - MLP).
- O segundo algoritmo fica a critério do grupo (ex.: Árvores de Decisão, Random Forest, SVM, Naive Bayes, k-NN, Regressão Logística, etc.).

### 2.3. Metodologia Experimental e Avaliação
- Avaliar e comparar os modelos utilizando métricas adequadas ao problema de classificação, tais como: Acurácia, Precisão, Revocação (Recall), F1-Score e Matriz de Confusão.
- Investigar o impacto da variação de hiperparâmetros (especialmente a arquitetura e parâmetros de treinamento da Rede Neural) no desempenho final.

---

## 3. Requisitos Técnicos e Restrições de Implementação

### 3.1. Linguagens Permitidas
O código-fonte do projeto poderá ser desenvolvido em uma das seguintes linguagens:
- Python
- C
- C++

### 3.2. Restrições de Bibliotecas (Exclusivo para Python)
Caso o grupo opte por desenvolver o trabalho em Python, o uso de bibliotecas externas está estritamente limitado a:
- **Manipulação e estruturação de dados:** `numpy` e `pandas`
- **Aprendizado de máquina e avaliação:** `scikit-learn`
- **Visualização e plotagem gráfica:** `matplotlib`, `seaborn` (ou bibliotecas equivalentes exclusivamente para geração de gráficos)

> **Atenção:** Não é permitido o uso de frameworks de aprendizado profundo ou bibliotecas de alto nível fora da lista acima (como TensorFlow, Keras, PyTorch, XGBoost, LightGBM, CatBoost, etc.). Em C ou C++, a implementação pode ser feita de forma nativa ou utilizando bibliotecas padrão da linguagem.

---

## 4. Formato de Entrega e Submissão
A entrega deverá ser realizada por apenas um integrante do grupo no ambiente virtual da disciplina, dentro do prazo estipulado no cronograma, em um único arquivo compactado no formato `.zip`.

### 4.1. Estrutura do Arquivo .zip
O arquivo compactado deve ser nomeado no padrão `IA_Trabalho_nome1_nome2_nome3.zip` e conter obrigatoriamente:

1. **Relatório Técnico (.pdf):**
   - Deve ser redigido estritamente de acordo com o modelo canônico ensinado em aula, contendo:
     - Contextualização e descrição detalhada das duas bases de dados escolhidas;
     - Metodologia de pré-processamento e configuração dos experimentos (incluindo hiperparâmetros testados);
     - Apresentação clara dos resultados (tabelas comparativas e gráficos);
     - Discussão crítica comparando o desempenho da Rede Neural com o(s) outro(s) algoritmo(s) em cada cenário.
2. **Código-Fonte:**
   - Scripts em `.py` (ou arquivos `.c`/`.cpp`/`.h` acompanhados de instruções de compilação ou Makefile).
   - Um arquivo `README.md` ou `README.txt` sucinto explicando como executar o código e como carregar as bases de dados para reproduzir os experimentos.

---

## 5. Critérios de Avaliação

| Critério | Peso | Descrição |
| :--- | :---: | :--- |
| **Adequação ao Modelo Canônico e Escrita** | 25% | Estruturação lógica do relatório conforme ensinado em aula, clareza textual, qualidade das figuras/tabelas e rigor técnico. |
| **Metodologia Experimental** | 25% | Correto pré-processamento, divisão adequada de treino/teste, ajuste de hiperparâmetros e ausência de *data leakage*. |
| **Análise Crítica e Comparação** | 30% | Profundidade na discussão das métricas obtidas, interpretação das matrizes de confusão e comparação fundamentada entre a Rede Neural e o(s) demais método(s). |
| **Qualidade e Reprodutibilidade do Código** | 20% | Organização, legibilidade, comentários pertinentes e cumprimento estrito das regras de linguagens e bibliotecas permitidas. |

> **Observação:** Trabalhos que utilizarem bibliotecas não autorizadas, que não incluírem uma Rede Neural entre os algoritmos ou que apresentarem indícios de plágio (no código ou no relatório) serão penalizados ou zerados conforme as normas da UFABC.