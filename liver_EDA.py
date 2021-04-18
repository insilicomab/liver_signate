# -*- coding: utf-8 -*-

"""
コメント：
baseline作成後のEDA
"""

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 学習データの読み込み
train = pd.read_csv('./data/train.csv')
print(train.info())

# numeric型の変数の取得
categories = train.columns[train.dtypes == 'object'] # カテゴリ変数の指定
train_num = train.drop(categories, axis=1)
train_num = train.drop(['id', 'Gender'], axis=1)


#グラフの表示
plt.figure(figsize=(12, 12))

for ncol, colname in enumerate(train_num.columns):
    plt.subplot(3, 3, ncol+1)
    sns.distplot(train_num.query("disease==0")[colname])
    sns.distplot(train_num.query("disease==1")[colname])
    plt.legend(labels=["non", "diseased"], loc='upper right')
plt.show()

'''
異常値
T_Bil（総ビリルビン）: high (>1.2 mg/dL)
D_Bil（直接ビリルビン）: high (>0.3 mg/dL)
ALP: high (>350 IU/mL)
ALT_GPT: high (>44 IU/L)
AST_GOT: high (>38 IU/L)
TP: low(<6.5 g/dL), high(>8.2 g/dL)
Alb: low(<3.9 g/dL)
AG_ratio: low(<1.3)
'''

# 異常値：1, 正常値：0
train['T_Bil_high'] = train['T_Bil'].apply(lambda x: 1 if x > 1.2 else 0)
train['D_Bil_high'] = train['D_Bil'].apply(lambda x: 1 if x > 0.3 else 0)
train['ALP_high'] = train['ALP'].apply(lambda x: 1 if x > 350 else 0)
train['ALT_GPT_high'] = train['ALT_GPT'].apply(lambda x: 1 if x > 44 else 0)
train['AST_GOT_high'] = train['AST_GOT'].apply(lambda x: 1 if x > 38 else 0)
train['TP_low'] = train['TP'].apply(lambda x: 1 if x < 6.5 else 0)
train['TP_high'] = train['TP'].apply(lambda x: 1 if x > 8.2 else 0)
train['Alb_low'] = train['Alb'].apply(lambda x: 1 if x < 3.9 else 0)
train['AG_ratio_low'] = train['AG_ratio'].apply(lambda x: 1 if x < 1.3 else 0)





'''
予測精度改善せず
'''

"""
コメント：
baseline ver2作成後のEDA
"""

# T_Bilを対数化
np.log(train['T_Bil'])

# 対数化したSalePriceをヒストグラムで可視化
plt.hist(np.log(train['T_Bil']), bins=20)

