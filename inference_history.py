import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score
import warnings
from data_cfg import cfg
warnings.simplefilter('ignore', category=Warning, lineno=0, append=False)

import pickle
from numpy import logical_and

# DATA_NAME = 'class_rehearsal'
# dimension = 1582
# activation_fn = tf.nn.sigmoid
activation_fn = tf.nn.tanh
np.random.seed(1)

def load_data():
    if 'flip' in DATA_NAME:
        data_dict = '/workspace/rll_origin/rll/raw_data/simu_data/{}/'.format(DATA_NAME)
    else:
        data_dict = '/workspace/rll_origin/rll/raw_data/{}/'.format(DATA_NAME)

    train_data = pd.read_csv(cfg[DATA_NAME]['DATA_NAME'] + cfg[DATA_NAME]['train_df_name'])
    test_data = pd.read_csv(cfg[DATA_NAME]['DATA_NAME'] + 'test.csv')

    # 如果考虑去掉中间票(2、3票)
    # train_data = train_data[(train_data.iloc[:, 1] > 3) | (train_data.iloc[:, 1] < 2)]

    train_y = np.array(train_data.iloc[:, 1])
    train_X = np.array(train_data.iloc[:, 3:])

    print('这里需要看一下每个人的信心')
    print(sum(train_y) / len(train_y))

    print(test_data.shape)
    test_y = np.array(test_data.iloc[:, 1])
    test_X = np.array(test_data.iloc[:, 3:])
    print('test:', sum(test_y) / len(test_y))

    return train_X, train_y, test_X, test_y


def reloadGraph(modelPath):
    tf.reset_default_graph()
    sess = tf.Session()

    metaFile = modelPath.split('/')[-1] + '.ckpt.meta'
    saver = tf.train.import_meta_graph(os.path.join(modelPath, metaFile))
    saver.restore(sess, tf.train.latest_checkpoint(modelPath))
    graph = tf.get_default_graph()
    return graph, sess


def inference(graph, loaded_sess, inputX):
    X = tf.placeholder(tf.float32, shape=[None, dimension], name='input')
    sess = loaded_sess
    # with loaded_sess as sess:
    w1 = graph.get_tensor_by_name('fc_l1/weights:0')
    b1 = graph.get_tensor_by_name('fc_l1/biases:0')
    w2 = graph.get_tensor_by_name('fc_l2/weights:0')
    b2 = graph.get_tensor_by_name('fc_l2/biases:0')
    embd = activation_fn(tf.matmul(activation_fn(tf.matmul(X, w1) + b1), w2) + b2)
    feed = {X: inputX}
    output = sess.run(embd, feed_dict=feed)
    return output


import pandas as pd

def inference_model():
    model_num = 0

    for model_path in list(os.listdir('../model_rll_history/rll/')):
        # 筛选出所需的model
        if (DATA_NAME not in model_path) | (('seq_8' not in model_path) & ('seq_9' not in model_path) & ('seq_10' not in model_path)):
            continue
        model_num += 1
        try:
            graph, session = reloadGraph('../model_rll_history/rll/' + model_path)
        except:
            continue
        print(model_path)
        embd_test = inference(graph, session, test_X)

        session.close()
        # break

        for C in [1]:
            model = LogisticRegression(penalty='l2', C=C, max_iter=300, fit_intercept=False, solver='liblinear')

            model.fit(embd, train_y)

            y_hat = model.predict(embd_test)
            y_proba = model.predict_proba(embd_test)

            # metrics
            acc = accuracy_score(test_y, y_hat)

            prec = precision_score(test_y, y_hat)
            recall = recall_score(test_y, y_hat)
            f1 = f1_score(test_y, y_hat)
            auc = roc_auc_score(test_y, y_proba[:, 1])

            print('Performance on learned embedding:{}'.format(model_path))
            print('LR C:', C)
            print('Accuracy {} \nprecision {} \nrecall {} \nAUC {}\n F1 {}'.format(acc, prec, recall, auc, f1))

    print(model_num)
    return 0


if __name__ == '__main__':

    inference_model()
