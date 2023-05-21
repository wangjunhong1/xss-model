import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

dataframe = pd.read_csv('./dataset-1684681630.2595701.csv', header=None)
X = dataframe.values[:, :-2]
y = dataframe.values[:, -2]
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
bayes_modle = GaussianNB()
bayes_modle.fit(X_train, y_train)
y_pred = bayes_modle.predict(X_test)
np.column_stack((y_test, y_pred, y_test - y_pred))

metrics = {}
# 准确率
accuracy = accuracy_score(y_test, y_pred)
metrics['accuracy'] = accuracy
# 召回率
recall = recall_score(y_test, y_pred)
metrics['recall'] = recall
# 精确率
precision = precision_score(y_test, y_pred)
metrics['precision'] = precision
# F1度量
f1 = f1_score(y_test, y_pred)
metrics['f1'] = f1
# AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_ = auc(fpr, tpr)
metrics['auc'] = auc_
for key, value in metrics.items():
    print("%-10s : %.3f" % (key, value))
# 保存模型
with open('../models/naive_bayes.model', 'wb') as f:
    f.write(pickle.dumps(bayes_modle))
