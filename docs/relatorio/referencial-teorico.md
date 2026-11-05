# Referencial Teórico — Perceptron Multicamadas (MLP)

## 1. Classificação supervisionada e partição dos dados

Na classificação supervisionada, o modelo recebe exemplos formados por atributos de entrada e um rótulo conhecido. Sejam m amostras $(x_i, y_i)$, com $x_i ∈ \mathbb{R}^n$ ($n$ atributos) e $y_i ∈ {0, ..., C−1}$ ($C$ classes). O treinamento ajusta os parâmetros do modelo para reduzir a diferença entre as classes reais e as previstas. Depois, o modelo é aplicado a amostras que não participaram do ajuste, o que permite medir sua capacidade de generalização.

Os dados são divididos em três conjuntos com funções distintas: o treinamento ajusta os pesos; a validação orienta a escolha de hiperparâmetros; o teste é reservado para a estimativa final de desempenho. Usar o teste na escolha da configuração produz uma estimativa otimista, pois o modelo e as decisões experimentais passam a depender indiretamente desse conjunto.

## 2. Perceptron multicamadas

O modelo é uma rede neural artificial do tipo Multilayer Perceptron (MLP) com uma camada de entrada, uma camada oculta com h neurônios e uma camada de saída com C neurônios (Rumelhart; Hinton; Williams, 1986). A camada oculta aprende combinações não lineares dos atributos e a de saída produz uma pontuação por classe. Para uma amostra x:

$$
z^{[1]} = W^{[1]}x + b^{[1]}, \qquad a^{[1]} = \operatorname{ReLU}(z^{[1]})
$$
$$
z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}, \qquad \hat{y} = \operatorname{Softmax}(z^{[2]})
$$

Nessas equações, $W^{[1]} ∈ \mathbb{R}^{(h×n)}$ e $W^{[2]} ∈ \mathbb{R}^{(C×h)}$ são os pesos, e $b^{[1]} ∈ \mathbb{R}^h$ e $b^{[2]} ∈ \mathbb{R}^C$ são os vieses. Pesos e vieses são os parâmetros, aprendidos por gradiente; o número de neurônios h, a taxa de aprendizado e o número de épocas são hiperparâmetros, definidos antes do treinamento e não aprendidos diretamente pelo gradiente.

O número de parâmetros treináveis é $h(n+1) + C(h+1)$. Para o MNIST ($n = 784$, $C = 10$) com $h = 128$, são $101.770$ parâmetros. Para o Breast Cancer ($n = 30$, $C = 2$) com $h = 8$, são $266$ parâmetros.

Em forma matricial, com $X ∈ \mathbb{R}^{(n×m)}$ (uma amostra por coluna), $Z^{[1]} = W^{[1]}X + b^{[1]}$ e as demais equações se aplicam coluna a coluna, com $b$ somado a cada coluna.

## 3. Função de ativação ReLU

A camada oculta usa a Rectified Linear Unit (Goodfellow; Bengio; Courville, 2016):

$$
\operatorname{ReLU}(z) = \max(0, z)
$$

Ela mantém valores positivos e zera os negativos. Sem uma função não linear entre as camadas, a composição de duas transformações lineares equivaleria a uma única transformação linear; a ReLU é o que permite à rede representar fronteiras de decisão não lineares. A derivada usada na retropropagação é:

$$
\operatorname{ReLU}'(z) = \begin{cases} 1, & z > 0 \\ 0, & z \leq 0 \end{cases}
$$

## 4. Saída Softmax

A Softmax converte os *scores* da camada de saída em probabilidades:

$$
\operatorname{Softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}
$$

As probabilidades de cada amostra somam 1 e a classe prevista é a de maior probabilidade. No MNIST a saída tem dez componentes (dígitos 0 a 9); no Breast Cancer, duas (benigno e maligno), de modo que a classificação binária é tratada como caso particular da multiclasse (Goodfellow; Bengio; Courville, 2016).

## 5. Entropia cruzada

O treinamento minimiza a entropia cruzada categórica. Com $m$ amostras, $y_{(k,i)}$ igual a $1$ se a classe real da amostra $i$ é $k$ (e $0$ caso contrário) e $ŷ_{(k,i)}$ a probabilidade prevista:

$$
L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{C} y_{k,i} \log\left(\hat{y}_{k,i}\right)
$$

A perda penaliza previsões que dão baixa probabilidade à classe correta. Na implementação, as probabilidades são limitadas inferiormente a $10^{−12}$ antes do logaritmo, para evitar $\log(0)$ (Goodfellow; Bengio; Courville, 2016).

## 6. Retropropagação e gradiente descendente

O gradiente da perda em relação a cada parâmetro é calculado pela regra da cadeia, a partir da saída em direção à entrada (retropropagação; Rumelhart; Hinton; Williams, 1986). Para a combinação Softmax com entropia cruzada, o erro da camada de saída é:

$$
dZ^{[2]} = A^{[2]} - Y
$$

Os gradientes da camada de saída são:

$$
dW^{[2]} = \frac{1}{m}\, dZ^{[2]} \left(A^{[1]}\right)^{T}, \qquad db^{[2]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[2]}_{i}
$$

E o erro propagado à camada oculta por:

$$
dZ^{[1]} = \left(W^{[2]}\right)^{T} dZ^{[2]} \odot \operatorname{ReLU}'\left(Z^{[1]}\right)
$$

Os gradientes ${dW}^{[1]}$ e ${db}^{[1]}$ saem de forma análoga, a partir de ${dZ}^{[1]}$ e $X$. Em seguida, os parâmetros são atualizados por gradiente descendente:

$$
W \leftarrow W - \eta\, dW, \qquad b \leftarrow b - \eta\, db
$$

Aqui $\eta$ é a taxa de aprendizado. O treinamento é em lote completo (batch): a cada época, todas as amostras de treinamento entram no cálculo do gradiente, que resulta em uma única atualização dos parâmetros.

## 7. Inicialização de He

Os pesos iniciais são sorteados de uma normal de média zero e variância proporcional ao inverso do número de entradas da camada (He et al., 2015):

$$
W^{[1]} \sim \mathcal{N}\left(0, \frac{2}{n}\right), \qquad W^{[2]} \sim \mathcal{N}\left(0, \frac{2}{h}\right)
$$

Essa escolha é adequada a redes com ReLU: busca manter a escala das ativações e dos gradientes em níveis úteis no início do treinamento. Os vieses começam em zero.

## 8. Hiperparâmetros


- **Neurônios na camada oculta ($h$):** controla a capacidade do modelo; poucos neurônios podem limitar o que a rede aprende, muitos aumentam o custo e o risco de ajuste excessivo ao treinamento.

- **Taxa de aprendizado ($\eta$):** valores muito pequenos tornam o treinamento lento; valores muito grandes podem causar oscilações ou impedir a convergência.

- **Número de épocas:** poucas épocas deixam o modelo subajustado; muitas aumentam o custo e podem levar ao ajuste excessivo. As curvas de perda e acurácia de treinamento descrevem a otimização, mas não medem generalização; para isso é preciso comparar com a validação.

Por isso, os três são tratados como hiperparâmetros e escolhidos com o conjunto de validação.

## 9. Pré-processamento e representação

Dois tratamentos são usados, conforme a natureza dos atributos:

- **Escala fixa:** no MNIST, cada imagem de 28 por 28 pixels vira um vetor de 784 atributos, e cada pixel, originalmente em [0, 255], é dividido por 255, ficando em [0, 1].

- **Padronização:** no Breast Cancer, cada atributo é transformado por x' = (x − μ)/σ, com μ e σ calculados somente no treinamento.

## 10. Métricas de avaliação

Para uma classe $k$ tratada como positiva, sejam TP (verdadeiros positivos), FP (falsos positivos) e FN (falsos negativos):

$$
\text{Acurácia} = \frac{\sum_k TP_k}{m}, \qquad \text{Precisão}_k = \frac{TP_k}{TP_k + FP_k}, \qquad \text{Recall}_k = \frac{TP_k}{TP_k + FN_k}
$$

$$
F1_k = \frac{2 \cdot \text{Precisão}_k \cdot \text{Recall}_k}{\text{Precisão}_k + \text{Recall}_k}, \qquad F1_{\text{macro}} = \frac{1}{C} \sum_{k=1}^{C} F1_k
$$

A matriz de confusão $M$ conta, em $M_{(ab)}$, as amostras de classe real $a$ previstas como $b$; a diagonal reúne os acertos. O F1 macro dá o mesmo peso a todas as classes, o que o torna útil quando elas têm tamanhos diferentes. No Breast Cancer, um falso negativo é uma amostra maligna classificada como benigna; por isso o recall da classe maligna é lido junto da acurácia, que isoladamente não mostra a natureza do erro.

## Referências

- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. (2016). *Deep Learning*. Cambridge, MA: MIT Press.
- HE, K. et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, p. 1026–1034. DOI: 10.1109/ICCV.2015.123.
- LECUN, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, v. 86, n. 11, p. 2278–2324. DOI: 10.1109/5.726791.
- RUMELHART, D. E.; HINTON, G. E.; WILLIAMS, R. J. (1986). Learning Representations by Back-Propagating Errors. *Nature*, v. 323, p. 533–536. DOI: 10.1038/323533a0.
