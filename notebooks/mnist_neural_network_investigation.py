# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split

from src.neural_network import NeuralNetwork

FIGURE_SIZE = (6.4, 4.8)
TWO_PANEL_FIGURE_SIZE = (12.8, 4.8)

# %%
df = pd.read_csv("data/mnist--digit-recognizer/train.csv")
X = df.drop(columns="label").to_numpy(dtype=float) / 255.0
y = df["label"].to_numpy(dtype=int)

class_counts = pd.Series(y).value_counts().sort_index()
class_counts.plot(
    kind="bar",
    title="Distribuição das classes - MNIST",
    figsize=FIGURE_SIZE,
)
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# %%
def train_configuration(params, epochs=500, seed=42):
    model = NeuralNetwork(random_state=seed, **params)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
    )
    predictions = model.predict(X_val)
    return model, {
        "hidden_layer_size": params["hidden_layer_size"],
        "learning_rate": params["learning_rate"],
        "epochs": model.n_epochs_,
        "val_accuracy": accuracy_score(y_val, predictions),
        "val_f1_macro": f1_score(y_val, predictions, average="macro"),
    }

results = []
models = {}
for params in ParameterGrid(
    {
        "hidden_layer_size": [10, 32, 64],
        "learning_rate": [0.01, 0.05, 0.10],
    }
):
    model, result = train_configuration(params)
    results.append(result)
    models[(params["hidden_layer_size"], params["learning_rate"])] = model

results_df = pd.DataFrame(results).sort_values(
    ["val_f1_macro", "val_accuracy"],
    ascending=False,
)
print(results_df.to_string(index=False))

# %%
results_df.plot(
    x="hidden_layer_size",
    y="val_f1_macro",
    kind="bar",
    title="F1 macro por configuração - MNIST",
    xlabel="Configuração",
    ylabel="F1 macro de validação",
    figsize=FIGURE_SIZE,
)
plt.xticks(
    range(len(results_df)),
    [
        f"h={row.hidden_layer_size}, lr={row.learning_rate}"
        for row in results_df.itertuples()
    ],
    rotation=45,
    ha="right",
)
plt.tight_layout()
plt.show()

# %%
refinement_results = []
refinement_models = {}
for hidden_layer_size in [64, 128, 256]:
    params = {
        "hidden_layer_size": hidden_layer_size,
        "learning_rate": 0.10,
    }
    model, result = train_configuration(params, epochs=1000)
    refinement_results.append(result)
    refinement_models[hidden_layer_size] = model

refinement_results_df = pd.DataFrame(refinement_results).sort_values(
    ["val_f1_macro", "val_accuracy"],
    ascending=False,
)
print(refinement_results_df.to_string(index=False))

# %%
refinement_results_df.plot(
    x="hidden_layer_size",
    y="val_f1_macro",
    kind="bar",
    title="Impacto do tamanho da camada oculta - MNIST",
    xlabel="Neurônios na camada oculta",
    ylabel="F1 macro de validação",
    figsize=FIGURE_SIZE,
)
plt.tight_layout()
plt.show()

# %%
best_hidden_layer_size = int(refinement_results_df.iloc[0]["hidden_layer_size"])
best_params = {
    "hidden_layer_size": best_hidden_layer_size,
    "learning_rate": 0.10,
}

epoch_results = []
epoch_models = {}
for epochs in [1000, 1250, 1500, 2000, 2500, 3000]:
    model, result = train_configuration(best_params, epochs=epochs)
    result["epochs"] = epochs
    epoch_results.append(result)
    epoch_models[epochs] = model

epoch_results_df = pd.DataFrame(epoch_results).sort_values(
    ["val_f1_macro", "val_accuracy"],
    ascending=False,
)
print(epoch_results_df.to_string(index=False))

best_epochs = int(epoch_results_df.iloc[0]["epochs"])

# %%
epoch_results_df.plot(
    x="epochs",
    y="val_f1_macro",
    marker="o",
    title="Impacto do número de épocas - MNIST",
    xlabel="Épocas",
    ylabel="F1 macro de validação",
    figsize=FIGURE_SIZE,
)
plt.tight_layout()
plt.show()

# %%
X_train_final = np.concatenate([X_train, X_val])
y_train_final = np.concatenate([y_train, y_val])
best_model = NeuralNetwork(random_state=42, **best_params)
best_model.fit(
    X_train_final,
    y_train_final,
    epochs=best_epochs,
)

# %%
test_predictions = best_model.predict(X_test)
test_report = classification_report(
    y_test,
    test_predictions,
    digits=4,
    zero_division=0,
    output_dict=True,
)
print(
    classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0,
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
confusion = confusion_matrix(y_test, test_predictions)
print("Matriz de confusão:\n", confusion)

# %%
fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL_FIGURE_SIZE)
axes[0].imshow(confusion, cmap="Blues")
axes[0].set_title("Matriz de confusão - MNIST")
axes[0].set_xlabel("Predição")
axes[0].set_ylabel("Rótulo verdadeiro")
axes[0].set_xticks(range(10))
axes[0].set_yticks(range(10))
for i in range(10):
    for j in range(10):
        axes[0].text(j, i, confusion[i, j], ha="center", va="center")

class_f1 = [
    test_report[str(label)]["f1-score"]
    for label in range(10)
]
axes[1].bar(range(10), class_f1)
axes[1].set_title("F1-score por classe - MNIST")
axes[1].set_xlabel("Classe")
axes[1].set_ylabel("F1-score")
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.show()

# %%
history = best_model.history
fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL_FIGURE_SIZE)
axes[0].plot(history["loss"], label="Treino")
axes[0].set_title("Perda de treinamento")
axes[0].set_xlabel("Época")
axes[0].legend()

axes[1].plot(history["accuracy"], label="Treino")
axes[1].set_title("Acurácia de treinamento")
axes[1].set_xlabel("Época")
axes[1].legend()
plt.tight_layout()
plt.show()
