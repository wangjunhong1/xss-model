from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

dataframe = pd.read_csv('./dataset-1684681630.2595701.csv', header=None)
X = dataframe.values[:, :-2]
y = dataframe.values[:, -2]
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
kmeans_model = KMeans(n_clusters=2, n_init='auto')
kmeans_model.fit(X_train, y_train)
y_pred = kmeans_model.predict(X_test)
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
with open('../models/k-means.model', 'wb') as f:
    f.write(pickle.dumps(kmeans_model))
