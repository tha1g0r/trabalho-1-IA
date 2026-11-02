# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.neural_network import NeuralNetwork
# %%
df = pd.read_csv("data/mnist--digit-recognizer/train.csv")
df.head()
# %%
data = np.array(df)
data[:5]
# %%
X = data[:, 1:] / 255.0
m, n_feat = X.shape

y = data[:, 0]
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# %%
model = NeuralNetwork(random_state=42)
model.fit(X_train, y_train, epochs=500)
# %%
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print(f"Acurácia de teste: {accuracy:.2%}")
# %%
