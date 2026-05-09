import pandas as pd
df = pd.read_csv('data/processed/train_qa.csv')  # or your raw file path
print(df['answer'].value_counts(dropna=False).head(50))