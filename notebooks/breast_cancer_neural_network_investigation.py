# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler

from src.neural_network import NeuralNetwork

# %%
df = pd.read_csv("data/breast_cancer_winsconsin.csv")
df = df.drop(columns=["id", "Unnamed: 32"])
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

X = df.drop(columns="diagnosis").to_numpy(dtype=float)
y = df["diagnosis"].to_numpy(dtype=int)

class_counts = pd.Series(y).value_counts().sort_index()
class_counts.index = ["Benigno", "Maligno"]
class_counts.plot(kind="bar", title="Distribuição das classes - Breast Cancer")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Distribuição das classes:")
print(pd.Series(y).value_counts().sort_index())

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
        "hidden_layer_size": [8, 16, 32],
        "learning_rate": [0.001, 0.01, 0.05],
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
    title="F1 macro por configuração - Breast Cancer",
    xlabel="Configuração",
    ylabel="F1 macro de validação",
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
best = results_df.iloc[0]
best_params = {
    "hidden_layer_size": int(best["hidden_layer_size"]),
    "learning_rate": float(best["learning_rate"]),
}

epoch_results = []
epoch_models = {}
for epochs in [100, 250, 500, 750, 1000]:
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
best_model = epoch_models[best_epochs]

# %%
epoch_results_df.plot(
    x="epochs",
    y="val_f1_macro",
    marker="o",
    title="Impacto do número de épocas - Breast Cancer",
    xlabel="Épocas",
    ylabel="F1 macro de validação",
)
plt.tight_layout()
plt.show()

# %%
test_predictions = best_model.predict(X_test)
test_report = classification_report(
    y_test,
    test_predictions,
    target_names=["Benigno", "Maligno"],
    digits=4,
    zero_division=0,
    output_dict=True,
)
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Benigno", "Maligno"],
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
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(confusion, cmap="Blues")
axes[0].set_title("Matriz de confusão - Breast Cancer")
axes[0].set_xlabel("Predição")
axes[0].set_ylabel("Rótulo verdadeiro")
axes[0].set_xticks([0, 1], ["Benigno", "Maligno"])
axes[0].set_yticks([0, 1], ["Benigno", "Maligno"])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, confusion[i, j], ha="center", va="center")

class_f1 = [
    test_report["Benigno"]["f1-score"],
    test_report["Maligno"]["f1-score"],
]
axes[1].bar(["Benigno", "Maligno"], class_f1)
axes[1].set_title("F1-score por classe - Breast Cancer")
axes[1].set_ylabel("F1-score")
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.show()

# %%
history = best_model.history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
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

# %%
seed_results = []
for seed in [1, 2, 3, 4, 5]:
    model = NeuralNetwork(random_state=seed, **best_params)
    model.fit(
        X_train,
        y_train,
        epochs=500,
    )
    predictions = model.predict(X_test)
    seed_results.append(
        {
            "seed": seed,
            "accuracy": accuracy_score(y_test, predictions),
            "f1_macro": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "malignant_recall": recall_score(
                y_test, predictions, pos_label=1, zero_division=0
            ),
        }
    )

seed_results_df = pd.DataFrame(seed_results)
print(seed_results_df)
print(seed_results_df[["accuracy", "f1_macro", "malignant_recall"]].agg(
    ["mean", "std"]
))

# %%
