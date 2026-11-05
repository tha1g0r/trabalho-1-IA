# Resultados — Perceptron Multicamadas (MLP)

## 1. Resultados no Dataset MNIST

**Ajuste de Hiperparâmetros:** Durante o refinamento da arquitetura com 1000 épocas e uma taxa de aprendizado fixada em 0,10, a configuração com 128 neurônios ocultos obteve um F1-Score macro de 92,89% e acurácia de 92,95%. O aumento para 256 neurônios ocasionou uma levíssima queda (92,83% em F1 macro), justificando a escolha por 128 neurônios.

Posteriormente, expandir a duração do treinamento demonstrou uma melhoria contínua na capacidade preditiva do modelo até a marca de 3000 épocas, onde se apurou um F1-Score macro de validação de 95,48%.

**[INSERIR GRÁFICO: Impacto do número de épocas – MNIST]**

Na validação, essa configuração alcançou:

- **Acurácia:** 95,51%
- **F1-Score Macro:** 95,48%

**Desempenho Final:** Ao submeter a arquitetura ótima (128 neurônios, taxa de aprendizado de 0,10 e 3000 épocas) retreinada ao conjunto de teste, a rede apresentou os seguintes indicadores:

- **Acurácia:** 95,73%
- **Precisão Macro:**  95,69%
- **Recall Macro:** 95,69%
- **F1-Score Macro:** 95,69%

Observa-se um equilíbrio excepcional no reconhecimento global, onde as métricas compostas acompanham a Acurácia. A análise dos F1-scores separados por classe (e da matriz de confusão) aponta que a rede atinge índices superiores a 97% na identificação dos dígitos "1", "6" e "0", enquanto apresenta os maiores gargalos preditivos nos dígitos "9" e "8" (abaixo de 94%). Os falsos positivos tendem a ocorrer em pares de dígitos visualmente semelhantes, especialmente envolvendo as classes "3", "5", "8", "9" e "4".

**[INSERIR GRÁFICO: Matriz de confusão – MNIST (notebook final)]**

## 2. Resultados no Dataset Breast Cancer Wisconsin

Nos testes iniciais de validação usando F1-Score macro, uma arquitetura simples com 8 neurônios e taxa de aprendizado de 0,05 apresentou um excelente resultado (F1 macro de 97,65% em 500 épocas). Em seguida, os experimentos que variaram as épocas obtiveram desempenho igual em 750 épocas e pior em 1000 épocas, sugerindo possível início de degradação da generalização. Assim, o treinamento além de 500 épocas não apresentava benefícios substanciais.

**[INSERIR GRÁFICO: Impacto do número de épocas – Breast Cancer]**

Na validação, a configuração selecionada alcançou:

- **Acurácia:** 97,80%
- **F1-Score Macro:** 97,65%

**Desempenho Final:** A avaliação final, valendo-se do conjunto de teste (114 instâncias), registrou as seguintes métricas após retreinamento com os dados combinados de treino e validação:

- **Acurácia:** 99,12%
- **Precisão Macro:** 99,32%
- **Recall Macro:** 98,81%
- **F1-Score Macro:** 99,05%

| Classe | Precisão | Recall | F1-score | Suporte |
| --- | --- | --- | --- | --- |
| Benigno | 98,63% | 100,00% | 99,31% | 72 |
| Maligno | 100,00% | 97,62% | 98,80% | 42 |

A Rede Neural obteve um sucesso notável, classificando corretamente 113 dos 114 casos no conjunto de teste (matriz composta por 72 casos benignos todos identificados e 41 dos 42 casos malignos corretamente diagnosticados). O desempenho do *Recall* para tumores malignos situou-se em 97,62%, significando apenas a ocorrência de um falso negativo isolado. É um erro singular, mas criticamente relevante em um sistema preditivo para a área da saúde.

**[INSERIR GRÁFICO: Matriz de confusão – Breast Cancer (notebook final)]**
