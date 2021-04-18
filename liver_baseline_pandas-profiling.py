# -*- coding: utf-8 -*-

# ライブラリの読み込み
import pandas as pd
from pandas_profiling import ProfileReport

# データの読み込み
train = pd.read_csv('./data/train.csv')

# idの削除
train.drop('id', axis = 1, inplace=True)

# pandas-profiling
profile = ProfileReport(train)
profile.to_file('profile_report.html')
