# Homework 2: Cleaning Data Using Pandas

import numpy as np
import pandas as pd

# Libraries for retrieving data from the web and handling zip files
import requests
import zipfile
import io

url = 'https://github.com/Hernan4444/MyAnimeList-Database/archive/refs/heads/master.zip'

r = requests.get(url, stream=True)

z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

anime_data = pd.read_csv('MyAnimeList-Database-master/data/anime.csv')

anime_data_extracted = anime_data[anime_data['Score'] != 'Unknown'].copy()
anime_data_extracted['Score'] = pd.to_numeric(anime_data_extracted['Score'])


def homework(anime_data_extracted):
    grouped_data = anime_data_extracted.groupby('Type')
    average_scores = grouped_data['Score'].mean()
    sorted_scores = average_scores.sort_values(ascending=False)
    return sorted_scores


result = homework(anime_data_extracted)
print(result)