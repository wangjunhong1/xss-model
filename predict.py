import numpy as np
import sys
import re
from VarChiTfIdf import VarChiTfIdf
import pickle
from urllib.parse import unquote

args_map = {
    'csv_path': './test.csv',
    'model': 'decision_tree'
}

# 进行文本表示
with open("./models/VAR-CHI-TF-IDF.model", "rb") as f:
    tf_idf_model = pickle.loads(f.read())


# 读取参数
def parse_argument():
    args = sys.argv
    for i in range(1, len(args)):
        res = re.match("^-{1,2}(.*?)=(.*?)$", args[i]).groups()
        print(res, res[0])
        if res[0] not in args_map:
            raise RuntimeError("unknown argument : " + res[0])
        args_map[res[0]] = res[1]


# 设置csv文件路径
def set_csv_path(path):
    args_map['csv_path'] = path


# 设置使用的模型
def set_model(model_name):
    args_map['model'] = model_name


def predict(model_name, dataset):
    with open('./models/' + model_name + '.model', 'rb') as f:
        model = pickle.load(f)
    return model.predict(dataset)


def do_predict(model_name, dataset):
    # URL解析并取出URL协议后的部分
    for i in range(len(dataset)):
        dataset[i] = unquote(dataset[i]).lower()
        start = dataset[i].find('//')
        dataset[i] = dataset[i][start + 1:]
    dataset = tf_idf_model.transform_imp_tf_idf(dataset)
    # 使用模型预测
    if model_name == 'svm':
        y_pred = predict('svm', dataset)
    if model_name == 'decision_tree':
        y_pred = predict('decision_tree', dataset)
    if model_name == 'k-means':
        y_pred = predict('k-means', dataset)
    if model_name == 'random_forest':
        y_pred = predict('random_forest', dataset)
    if model_name == 'naive_bayes':
        y_pred = predict('naive_bayes', dataset)
    return np.array(y_pred)


def get_key_words():
    return tf_idf_model.top_word_list()


if __name__ == '__main__':
    tests = ['<script>alert(1)</script>',
        'query%3DSearch...%26Product%3D%27%22--%3E%3C/style%3E%3C/script%3E%3Cscript%3Ealert%28%27XSS%27%29%3C/script%3E%26Page%3D2']
    y_pred = do_predict('random_forest', tests)
    for _ in y_pred:
        print(_)
