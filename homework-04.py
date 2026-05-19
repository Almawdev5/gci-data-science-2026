# Homework 4: Supervised Learning
# Simple Linear Regression with R-squared

import numpy as np
import pandas as pd
from sklearn import linear_model
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


def homework(anime_data_extracted, X_column, Y_column):

    X = anime_data_extracted[[X_column]]

    Y = anime_data_extracted[Y_column]

    model = linear_model.LinearRegression()

    model.fit(X, Y)

    result = model.score(X, Y)

    return result


result = homework(
    anime_data_extracted,
    X_column='Members',
    Y_column='Completed'
)

print(result)