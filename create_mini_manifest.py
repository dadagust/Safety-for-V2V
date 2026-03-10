import pandas as pd

df = pd.read_csv("manifests/librispeech_train.csv")
one = df.iloc[[0]].copy()
one.to_csv("manifests/one_train.csv", index=False)
one.to_csv("manifests/one_val.csv", index=False)