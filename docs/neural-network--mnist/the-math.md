# Plano matemático: rede neural para MNIST

Ponto de partida: **MLP** (784 → camada oculta → 10).

## 1. Representação dos dados
- Cada imagem 28×28 vira um vetor $x \in \mathbb{R^{784 \times 1}}$, com pixels normalizados para [0,1] (divisão por 255).

- O rótulo vira um vetor **one-hot $y \in \mathbb{R^{10 \times 1}}$**.

- Para o subconjunto de treino com $m$ amostras: $X \in \mathbb{R^{784 \times m}}$.

## 2. Feedforward propagation
Para uma camada oculta com $h$ neurônios:

- $Z^{[1]} = W^{[1]} X + b^{[1]}$
    - $W^{[1]} \in \mathbb{R^{h \times 784}}$
    - $b^{[1]} \in \mathbb{R^{h \times 1}}$
    - $Z^{[1]} \in \mathbb{R^{h \times m}}$

- $A^{[1]} = g_1(Z^{[1]}) = \text{ReLU}(Z^{[1]}) = \max(0, Z^{[1]})$
    - $A^{[1]} \in \mathbb{R^{h \times m}}$

- $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$
    - $W^{[2]} \in \mathbb{R^{10 \times h}}$
    - $b^{[2]} \in \mathbb{R^{10 \times 1}}$
    - $Z^{[2]} \in \mathbb{R^{10 \times m}}$

- $ A^{[2]} = \hat{Y} = g_2(Z^{[2]}) = \text{softmax}(Z^{[2]})$, onde $\hat{y}_k = \dfrac{e^{z^{[2]}_k}}{\sum_j{e^{Z^{[2]}_j}}}$

*obs:* $Z$ é a **pré-ativação**, o que chega ao neurônio (combinação linear das saídas da camada anterior). $A$ é a **ativação**, o que sai do neurônio depois da função de ativação.

## 3. Backpropagation
Atualização dos parâmetros:

- $W^{[2]} := W^{[2]} - \eta \mathrm{d}W^{[2]}$

- $b^{[2]} := b^{[2]} - \eta \mathrm{d}b^{[2]}$

- $W^{[1]} := W^{[1]} - \eta \mathrm{d}W^{[1]}$

- $b^{[1]} := b^{[1]} - \eta \mathrm{d}b^{[1]}$

Onde

- $\mathrm{d}Z^{[2]} = A^{[2]} - Y$

- $\mathrm{d}W^{[2]} = \frac{1}{m} \mathrm{d}Z^{[2]} {A^{[1]}}^T$

- $\mathrm{d}b^{[2]} = \frac{1}{m} \sum{\mathrm{d}Z^{[2]}}$

- $\mathrm{d}Z^{[1]} = ({W^{[2]}}^T \mathrm{d}Z^{[2]}) \odot g_1'(Z^{[1]})$, com $g_1'(z) = 1$ se $z>0$, senão $0$

- $\mathrm{d}W^{[1]} = \frac{1}{m} \mathrm{d}Z^{[1]} X^T$

- $\mathrm{d}b^{[1]} = \frac{1}{m} \sum{\mathrm{d}Z^{[1]}}$

## 4. Avaliação
Acurácia, matriz de confusão e precisão/recall/F1 por classe.

## 5. Protocolo experimental

Os dados são separados em conjuntos de treino, validação e teste. O conjunto
de treino é usado para atualizar os parâmetros da rede. O conjunto de
validação é usado para escolher hiperparâmetros e acompanhar a generalização
durante o treinamento. O conjunto de teste permanece isolado até a avaliação
final.

Neste projeto, o treinamento usa um número máximo fixo de épocas. A validação
é executada no notebook após cada configuração ser treinada, para comparar
hiperparâmetros sem utilizar o conjunto de teste.

Para o MNIST, os pixels são divididos por 255 para ficarem no intervalo
$[0,1]$. Para o Breast Cancer, as variáveis são padronizadas usando média e
desvio-padrão calculados exclusivamente no conjunto de treino; os mesmos
parâmetros são aplicados à validação e ao teste. Assim, informações dos
conjuntos de avaliação não vazam para o treinamento.

A inicialização dos pesos usa a variância $2/n_{\text{in}}$, adequada à ReLU,
para reduzir a chance de escalas inadequadas nas ativações e nos gradientes.
Os hiperparâmetros, como tamanho da camada oculta e taxa de aprendizado, são
escolhidos pelo desempenho de validação. O teste é usado somente depois dessa
escolha.
