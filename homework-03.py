# Homework 3: Visualizing Data Using Matplotlib (Pareto Analysis)
# Task: Group anime into n equal parts using metric ranking and compute proportional contribution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import zipfile
import io

# =========================================================
# Download MyAnimeList Dataset
# =========================================================

url = 'https://github.com/Hernan4444/MyAnimeList-Database/archive/refs/heads/master.zip'
r = requests.get(url, stream=True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

# Load dataset
anime_data_raw = pd.read_csv('MyAnimeList-Database-master/data/anime.csv')

# =========================================================
# Preprocessing
# =========================================================

columns_to_keep = ['MAL_ID', 'Name', 'Score', 'Type', 'Episodes',
                   'Members', 'Completed', 'Watching', 'Dropped', 'Popularity']

anime_data = anime_data_raw[columns_to_keep].copy()

anime_data = anime_data.dropna(subset=['Members', 'Completed', 'Watching'])

anime_data = anime_data[
    (anime_data['Members'] > 0) &
    (anime_data['Completed'] > 0) &
    (anime_data['Watching'] > 0)
]

anime_data = anime_data.reset_index(drop=True)

# =========================================================
# Homework Function
# =========================================================

def homework(anime_data, metric_column, n):

    sorted_data = anime_data.sort_values(metric_column, ascending=False).reset_index(drop=True)

    ranks = np.arange(len(sorted_data))

    groups = pd.qcut(ranks, n, labels=[f"Group {i+1}" for i in range(n)])

    sorted_data = sorted_data.copy()
    sorted_data["Group"] = groups

    group_sum = sorted_data.groupby("Group")[metric_column].sum()

    result = group_sum / group_sum.sum()

    result = result.sort_values(ascending=False)

    return result

# =========================================================
# Test Run
# =========================================================

N = 10
data = homework(anime_data, 'Completed', N)

print(data)

# =========================================================
# Pareto Visualization
# =========================================================

fig, ax1 = plt.subplots(figsize=(8, 5))

data_num = len(data)
cum_per = np.cumsum(data)

ax1.bar(range(data_num), data, color='steelblue', edgecolor='white')
ax1.set_xticks(range(data_num))
ax1.set_xticklabels(data.index, rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.plot(range(data_num), cum_per, color='black', marker='o', linewidth=2)
ax2.set_ylim([0, 1.05])
ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)

ax1.set_xlabel('Groups')
ax1.set_ylabel('Proportion')
ax2.set_ylabel('Cumulative Proportion')

plt.title('Pareto Chart — Anime Distribution Analysis')
plt.tight_layout()
plt.show()