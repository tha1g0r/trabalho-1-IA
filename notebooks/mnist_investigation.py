# %%
import pandas as pd
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("../data/mnist--digit-recognizer/train.csv")

df.head()
# %%
label_distribution = df['label'].value_counts().sort_index()

label_distribution.plot(kind='bar')
plt.xlabel('Label')
plt.ylabel('Quantidade')
plt.title('Distribuição dos labels')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# %%
