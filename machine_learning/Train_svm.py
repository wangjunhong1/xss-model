from sklearn import svm
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, \
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

data_frame = pd.read_csv("./dataset-1684681630.2595701.csv", header=None)
X = data_frame.values[:, :-2]
y = data_frame.values[:, -2]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
# 建立 svm 模型
clf = svm.SVC(kernel='sigmoid')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
np.column_stack((y_pred, y_test, y_pred - y_test))
confusion_matrix(y_test, y_pred)
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
with open('../models/svm.model', 'wb') as f:
    f.write(pickle.dumps(clf))
