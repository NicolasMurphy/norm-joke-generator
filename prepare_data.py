import pandas as pd
import re

def clean_joke(joke):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', joke)

jokes_df = pd.read_csv("jokes.csv", header=None)
jokes = jokes_df.iloc[:, 0].tolist()

cleaned_jokes = [clean_joke(joke) for joke in jokes]

with open("jokes.txt", "w", encoding="utf-8", errors="ignore") as f:
    for joke in cleaned_jokes:
        f.write(joke + "\n")
