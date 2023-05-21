import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz
import pickle

data_frame = pd.read_csv('./dataset-1684681630.2595701.csv')
X = data_frame.values[:, :-2]
y = data_frame.values[:, -2]
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# 定义决策树分类器并训练模型
clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
np.column_stack((y_test, y_pred))
# 在测试集上进行预测并输出模型精度
accuracy = clf.score(X_test, y_test)
print("模型精度为: {:.2f}%".format(accuracy * 100))
# 可视化决策树
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=range(1, X.shape[1] + 1),
                           class_names=['否', '是'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('./决策树结果')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

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

from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 计算FNR和FPR
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)

print("FNR: ", fnr)
print("FPR: ", fpr)

# 保存决策树模型
with open('../models/decision_tree.model', 'wb') as f:
    f.write(pickle.dumps(clf))