from VarChiTfIdf import VarChiTfIdf
import pandas as pd
import pickle
import time
import numpy as np

tf_idf_model = VarChiTfIdf()
# 读取原始数据集
xss_dataframe = pd.read_csv('./xssed.csv')
normal_dataframe = pd.read_csv('./normal_examples.csv')
# 组合两个数据集
xssed_document = [s[0] for s in xss_dataframe.values]
normal_document = [s[0] for s in normal_dataframe.values]
documents = xssed_document + normal_document

tf_idf_model.fit(documents, [len(xssed_document)], dictionary_size=1000)
documents_vec = tf_idf_model.transform_imp_tf_idf(documents)

# 保存TF-IDF模型
with open('../models/VAR-CHI-TF-IDF.model', 'wb') as f:
    f.write(pickle.dumps(tf_idf_model))

# 合成数据集
xss_dataset = np.column_stack((documents_vec[:74063],
                               np.ones((74063, 1)),
                               np.zeros((74063, 1))))
normal_dataset = np.column_stack((documents_vec[74063:],
                                  np.zeros((31406, 1)),
                                  np.ones((31406, 1))))
dataset = np.row_stack((xss_dataset, normal_dataset))
dataset = dataset[np.random.permutation(dataset.shape[0]), :]

# 保存数据集
filename = './dataset-' + str(time.time()) + '.csv'
pd.DataFrame(dataset).to_csv(filename, header=None, index=None)