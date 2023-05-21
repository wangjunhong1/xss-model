import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data_frame = pd.read_csv('./dataset-1684681630.2595701.csv')
dataset = data_frame.values[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1], dataset[:, -1], train_size=0.5)

logreg = LogisticRegression(solver='liblinear', max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
np.column_stack((y_pred, y_test))
from sklearn.metrics import confusion_matrix, \
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

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
with open('../models/logistic_regression.model', 'wb') as f:
    f.write(pickle.dumps(logreg))
