# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.neural_network import NeuralNetwork

# %%
df = pd.read_csv("data/mnist--digit-recognizer/train.csv")
X = df.drop(columns="label").to_numpy(dtype=float) / 255.0
y = df["label"].to_numpy(dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# %%
model = NeuralNetwork(
    hidden_layer_size=128,
    learning_rate=0.10,
    random_state=42,
)
model.fit(
    X_train,
    y_train,
    epochs=3000,
)

validation_predictions = model.predict(X_val)
print("Validação")
print("Acurácia:", accuracy_score(y_val, validation_predictions))
print("F1 macro:", f1_score(y_val, validation_predictions, average="macro"))

# %%
X_train_final = pd.concat(
    [pd.DataFrame(X_train), pd.DataFrame(X_val)]
).to_numpy()
y_train_final = pd.concat(
    [pd.Series(y_train), pd.Series(y_val)]
).to_numpy()

final_model = NeuralNetwork(
    hidden_layer_size=128,
    learning_rate=0.10,
    random_state=42,
)
final_model.fit(X_train_final, y_train_final, epochs=3000)

test_predictions = final_model.predict(X_test)
print("Teste")
print(
    classification_report(
        y_test, test_predictions, digits=4, zero_division=0
    )
)
print("Acurácia:", accuracy_score(y_test, test_predictions))
print("Precisão macro:", precision_score(
    y_test, test_predictions, average="macro", zero_division=0
))
print("Recall macro:", recall_score(
    y_test, test_predictions, average="macro", zero_division=0
))
print("F1 macro:", f1_score(
    y_test, test_predictions, average="macro", zero_division=0
))

# %%
confusion = confusion_matrix(y_test, test_predictions)
plt.imshow(confusion, cmap="Blues")
plt.title("Matriz de confusão - MNIST")
plt.xlabel("Predição")
plt.ylabel("Rótulo verdadeiro")
plt.xticks(range(10))
plt.yticks(range(10))
for i in range(10):
    for j in range(10):
        plt.text(j, i, confusion[i, j], ha="center", va="center")
plt.tight_layout()
plt.show()
