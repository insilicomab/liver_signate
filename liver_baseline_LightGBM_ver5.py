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

# ライブラリのインポート
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# 学習データとテストデータの連結
df = pd.concat([train, test], sort=False).reset_index(drop=True)

# 欠損値の補完（AG比=アルブミン/グロブリン　TP（総蛋白）=アルブミン+グロブリン）
df['AG_ratio'].fillna(df['Alb'] / (df['TP']-df['Alb']), inplace=True) 
print(df.isnull().sum())

# 性別を数値に変換（Male:0, Female:1）
df['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)

# 交互作用特徴量の作成
polynomial = PolynomialFeatures(degree = 3, include_bias=False)
poly_features = df[['Age','T_Bil','D_Bil','ALP','ALT_GPT','AST_GOT','TP','Alb','AG_ratio']]
polynomial_arr = polynomial.fit_transform(poly_features)
X_polynomial = pd.DataFrame(polynomial_arr,
                            columns = polynomial.get_feature_names(['Age','T_Bil','D_Bil','ALP','ALT_GPT','AST_GOT','TP','Alb','AG_ratio'])) # polynomial_arrのデータフレーム化 （※カラムはshape[1]でpolynomial_arrの列数分だけ出力）

# 生成した多項式・交互作用特徴量の表示
print(X_polynomial.shape)
print(X_polynomial.head())

# 組み込み法のモデル、閾値の指定
fs_model = LogisticRegression(penalty='l1', solver='liblinear',random_state=0)
fs_threshold = "mean"

# 組み込み法モデルの初期化
selector = SelectFromModel(fs_model, threshold=fs_threshold)

# 特徴量選択の実行
x_polynomial = X_polynomial[:len(train)] # 多項式・交互作用特徴量の説明変数
y = train['disease']
selector.fit(x_polynomial, y)
mask = selector.get_support()

# 選択された特徴量だけのサンプル取得
X_polynomial_masked = X_polynomial.loc[:, mask]

# trainとtestに再分割
X_train = X_polynomial_masked[:len(train)]
X_test = X_polynomial_masked[len(train):]

'''
モデルの構築と評価
'''

# ライブラリのインポート
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
from statistics import mean

# K分割する
folds = 10
kf = KFold(n_splits=folds)

# ハイパーパラメータの設定
params = {
    'objective':'binary',
    'random_seed':1234    
}

# 説明変数と目的変数を指定
Y_train = train['disease']

# 各foldごとに作成したモデルごとの予測値を保存
models = []
aucs = []

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
0.9626429240005863
"""

'''
テストデータの予測
'''

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
sub.to_csv('./submit/liver_LightGBM_ver5.csv', header=None, index=False)

"""
スコア：
0.9394240
"""