# Metodologia — Perceptron Multicamadas (MLP)

Esta seção detalha o método de implementação, a preparação das bases de dados, a divisão dos conjuntos e o processo de investigação de hiperparâmetros para os experimentos com a Rede Neural (MLP).

## 1. Organização e Implementação do Modelo

A arquitetura do Multilayer Perceptron (MLP) foi implementada nativamente em Python, utilizando exclusivamente a biblioteca *NumPy* para as operações matriciais e matemáticas. A classe `NeuralNetwork` centraliza o algoritmo, contemplando:

- **Inicialização de pesos:** Utilização do método de He, adequado para a função de ativação ReLU, visando preservar a escala das ativações e gradientes no início do treinamento.
- **Propagação (Feedforward):** Implementação de uma camada oculta com função de ativação ReLU e uma camada de saída com função Softmax para gerar as distribuições de probabilidade das classes.
- **Função de perda e Otimização:** Utilização de entropia cruzada categórica como função de custo e o algoritmo de backpropagation acoplado ao gradiente descendente para atualização dos parâmetros em lote completo (*full batch*).

Para manter o modelo independente dos dados, as etapas de pré-processamento, divisão de conjuntos e avaliação foram isoladas em scripts e notebooks de investigação, utilizando *pandas* para manipulação de dados e *scikit-learn* para transformações e métricas.

## 2. Preparação dos Dados

Foram selecionadas duas bases de dados distintas para avaliação:

- **MNIST (Digit Recognizer):** Composta por imagens de dígitos manuscritos (0 a 9). Os dados já foram obtidos em matrizes numéricas, onde os 784 atributos (pixels achatados) de cada imagem foram normalizados pela divisão por 255,0, escalonando os valores para o intervalo [0, 1].

- **Breast Cancer Wisconsin:** Problema de classificação binária (tumores benignos e malignos). A variável alvo foi mapeada numericamente (0 para benigno e 1 para maligno). Os atributos numéricos foram padronizados via `StandardScaler`. O ajuste estatístico (`fit`) ocorreu estritamente no conjunto de treinamento, e as transformações foram aplicadas aos conjuntos de validação e teste para prevenir o vazamento de dados (*data leakage*).

## 3. Protocolo de Divisão e Avaliação

Para garantir a confiabilidade da avaliação, utilizou-se uma técnica de amostragem estratificada (`random_state=42`) para particionar ambas as bases de dados em três conjuntos disjuntos:

1. **Treinamento (aprox. 64%):** Utilizado para o ajuste iterativo dos pesos e vieses da rede.
2. **Validação (aprox. 16%):** Utilizado para guiar a seleção da arquitetura e hiperparâmetros (como o número de épocas e a taxa de aprendizado) sem expor o modelo aos dados finais.
3. **Teste (20%):** Reservado para a etapa final, simulando dados inéditos para a avaliação definitiva de desempenho, após o modelo ser retreinado combinando as partições de treinamento e validação sob a melhor combinação de hiperparâmetros.

As métricas adotadas para a análise dos modelos foram: Acurácia, Precisão Macro, Recall Macro e o F1-Score Macro. O F1-Score Macro guiou primariamente a seleção de modelos nos conjuntos de validação, oferecendo um balanço justo para todas as classes.

## 4. Investigação de Hiperparâmetros

O processo investigativo procurou a melhor configuração para cada contexto de base de dados:

- **Para o conjunto MNIST:**
    - **Busca em grade inicial:** Testaram-se camadas ocultas com 10, 32 e 64 neurônios; taxas de aprendizado de 0,01, 0,05 e 0,10, ao longo de 500 épocas.
    - **Refinamento de arquitetura:** Focado em arquiteturas mais robustas, testando 64, 128 e 256 neurônios (com taxa de aprendizado de 0,10 por 1000 épocas), onde 128 neurônios demonstrou o melhor custo-benefício.
    - **Otimização do treinamento:** Fixada a estrutura, variou-se o número de épocas (de 1000 a 3000), resultando na seleção do limite superior investigado.

- **Para o conjunto Breast Cancer:**
    - **Busca em grade inicial:** Exploraram-se camadas com 8, 16 e 32 neurônios; taxas de aprendizado de 0,001, 0,01 e 0,05 em 500 épocas.
    - **Análise do número de épocas:** Fixada a melhor configuração preliminar, observou-se a variação de 100 a 1000 épocas.
    - **Verificação de estabilidade:** Analisou-se o impacto da inicialização aleatória dos pesos utilizando cinco sementes distintas.
