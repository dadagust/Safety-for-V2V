import pandas as pd

df = pd.read_csv("manifests/librispeech_train.csv")
df.iloc[:2000].to_csv("manifests/train_2k.csv", index=False)

dfv = pd.read_csv("manifests/librispeech_val.csv")
dfv.iloc[:200].to_csv("manifests/val_200.csv", index=False)