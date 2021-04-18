# -*- coding: utf-8 -*-

'''
データの読み込みと確認
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)

# データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submit.csv', header=None)

# データの確認
print(train.head())
print(train.dtypes)

# 欠損値の確認
print(train.isnull().sum())
print(test.isnull().sum())

'''
特徴量エンジニアリング
'''

# 学習データとテストデータの連結
df = pd.concat([train, test], sort=False).reset_index(drop=True)

# 欠損値の補完（AG比=アルブミン/グロブリン　TP（総蛋白）=アルブミン+グロブリン）
df['AG_ratio'].fillna(df['Alb'] / (df['TP']-df['Alb']), inplace=True) 
print(df.isnull().sum())

# 性別を数値に変換（Male:0, Female:1）
df['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)

# T_Bil, ALT_GPTを対数化
df['T_Bil_log'] = np.log(df['T_Bil'])
df['ALT_GPT_log'] = np.log(df['ALT_GPT'])

# 不要なカラムを削除
df.drop(['T_Bil','ALT_GPT'],axis = 1, inplace=True)

# trainとtestに再分割
train = df[~df['disease'].isnull()]
test = df[df['disease'].isnull()]

'''
モデルの構築と評価
'''

# ライブラリのインポート
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
from statistics import mean
from sklearn.metrics import confusion_matrix

# K分割する
folds = 10
kf = KFold(n_splits=folds)

# ハイパーパラメータの設定
params = {
    'objective':'binary',
    'random_seed':1234    
}

# 説明変数と目的変数を指定
X_train = train.drop(['disease', 'id'], axis=1)
Y_train = train['disease']

# 各foldごとに作成したモデルごとの予測値を保存
models = []
aucs = []
oof = np.zeros(len(X_train))

for train_index, val_index in kf.split(X_train):
    x_train = X_train.iloc[train_index]
    x_valid = X_train.iloc[val_index]
    y_train = Y_train.iloc[train_index]
    y_valid = Y_train.iloc[val_index]
    
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)    
    
    model = lgb.train(params,
                      lgb_train, 
                      valid_sets=lgb_eval, 
                      num_boost_round=100, # 学習回数の実行回数
                      early_stopping_rounds=20, # early_stoppingの判定基準
                      verbose_eval=10)
    
    y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
    
    # AUCスコアの算出
    auc = roc_auc_score(y_valid, y_pred) 
    print(auc)
    
    # ROC曲線の要素（偽陽性率、真陽性率、閾値）の算出
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred)
    
    models.append(model)
    aucs.append(auc)    
    
    # ROC曲線の描画
    plt.plot(fpr, tpr, label='roc curve (area = %0.3f)' % auc)
    plt.plot([0, 1], [0, 1], linestyle=':', label='random')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', label='ideal')
    plt.legend()
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.show()
    
# 平均AUCを計算する
print(mean(aucs))

# 特徴量重要度の表示
for model in models:
    lgb.plot_importance(model, importance_type='gain',
                        max_num_features=15)

"""
予測精度：
0.9602850775753206
"""

'''
テストデータの予測
'''

# テストデータの説明変数を指定
X_test = test.drop(['disease', 'id'], axis=1)

# テストデータにおける予測
preds = []

for model in models:
    pred = model.predict(X_test)
    preds.append(pred)

# predsの平均を計算
preds_array = np.array(preds)
pred = np.mean(preds_array, axis = 0)

'''
提出
'''

# 提出用サンプルの読み込み
sub = pd.read_csv('./data/sample_submit.csv', header=None)

# カラム1の値を置き換え
sub[1] = pred

# CSVファイルの出力
sub.to_csv('./submit/liver_LightGBM.csv', header=None, index=False)

"""
スコア：
0.9380185
"""