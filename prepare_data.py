import pandas as pd

jokes_df = pd.read_csv("jokes.csv", header=None)
jokes = jokes_df.iloc[:, 0].tolist()

with open("jokes.txt", "w", encoding="utf-8", errors="ignore") as f:
    for joke in jokes:
        f.write(joke + "\n")
