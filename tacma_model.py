import numpy as np
import os, time
import tensorflow as tf
import sys
import tensorflow.contrib.layers as layers
import utils
import pandas as pd
import glob
import pickle

from history_utils import get_average_confidence, calcucate_personal_confidence


groupSize = 4e4
warm_up = 5
# 是否用sampler靠下面这个参数决定
MODEL_NAME = 'RLL_l1_{}_l2_{}_lr_{}_penalty_{}_bs_{}_dropout_{}_sim_{}_anchor_{}_ON_{}_seq_{}_gamma_{}_history_0113'

LOG_NAME = "../train_logs/rll_history_2/{}"

# activation_fn = tf.nn.sigmoid
activation_fn = tf.nn.tanh

########################################################################################

config = tf.ConfigProto()
# config.gpu_options.visible_device_list = '0'
config.gpu_options.allow_growth = True
########################################################################################

# compute cosine distance between two vectors
def cos_sim(a, b):
    normalize_a = tf.nn.l2_normalize(a, 1)
    normalize_b = tf.nn.l2_normalize(b, 1)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)
    return cos_similarity


def euc_sim(a, b):
    euc_simimarity = 1 - tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1))
    return euc_simimarity

'''
Parameters
 
bs: batch size
lr_rate: learning rate
l1_n: number of neurons in the first layer
l2_n: number of neurons in the second layer
max_iter: max interation number for training
reg_scale: regularization penalty
dropout_rate: ratio to drop neurons in each layer
save_path: where you save the model
model_name: you can name your model however you like
'''


class RLL(object):
    def __init__(self, dimension, l1_n, l2_n, gamma, sim_type, anchor_type):

        self.dimension = dimension
        self.l1_n = l1_n
        self.l2_n = l2_n
        self.gamma = gamma
        self.sim_type = sim_type
        self.anchor_type = anchor_type

    # RLL的入口是batch，所以需要转换一下，入口变成输入整个原始数据集
    def loadData(self, raw_train_data, raw_validation_data):
        """
        Data loading.

        :param raw_data: Array of data
        :return:
        """
        # raw train data 的格式:
        # [filename, label, votes, features...]
        # 0           1      2       3:
        # 不要转成array. 后面准备输入的时候再转array

        self.raw_train_data = raw_train_data
        self.raw_validation_data = raw_validation_data


    # 注意，每个group的样本和weights都可以照常输入，无需改动。改动的是通过样本的representation来计算weights的方式
    def feedBatch(self, groups, weights, batchSize, lr_rate, dropout_rate, reg_scale, is_training):
        batchIndex = np.random.randint(low=0, high=groups[0].shape[0], size=batchSize)
        batchGroups = [groups[i][batchIndex] for i in range(len(groups))]
        batchWeights = [weights[i][batchIndex] for i in range(len(weights))]
        batchData = {
                            self.is_training: is_training,
                            self.lr_rate: lr_rate,
                            self.dropout_rate: dropout_rate,
                            self.reg_scale: reg_scale,

                            self.query: batchGroups[0],
                            self.posDoc: batchGroups[1],
                            self.negDoc0: batchGroups[2],
                            self.negDoc1: batchGroups[3],
                            self.negDoc2: batchGroups[4],

                            self.queryDocW: batchWeights[0].reshape(-1, ),
                            self.posDocW: batchWeights[1].reshape(-1, ),
                            self.negDoc0W: batchWeights[2].reshape(-1, ),
                            self.negDoc1W: batchWeights[3].reshape(-1, ),
                            self.negDoc2W: batchWeights[4].reshape(-1,)

                    }
        raw_batch_data = {
                            'is_training': is_training,
                            'lr_rate': lr_rate,
                            'dropout_rate': dropout_rate,
                            'reg_scale': reg_scale,

                            'query': batchGroups[0],
                            'posDoc': batchGroups[1],
                            'negDoc0': batchGroups[2],
                            'negDoc1': batchGroups[3],
                            'negDoc2': batchGroups[4],

                            'queryDocW': batchWeights[0].reshape(-1, ),
                            'posDocW': batchWeights[1].reshape(-1, ),
                            'negDoc0W': batchWeights[2].reshape(-1, ),
                            'negDoc1W': batchWeights[3].reshape(-1, ),
                            'negDoc2W': batchWeights[4].reshape(-1,)
                    }
        return batchData, raw_batch_data

    def single_net(self, input_layer, reuse=False):

        with tf.name_scope('fc_l1'):
            output = tf.contrib.layers.fully_connected(input_layer, self.l1_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),
                                                                 activation_fn = activation_fn, scope='fc_l1',
                                                                 biases_initializer=tf.random_normal_initializer(), reuse=reuse)

            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training)

        with tf.name_scope('fc_l2'):
            output = tf.contrib.layers.fully_connected(output, self.l2_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),
                                                             activation_fn = activation_fn, scope='fc_l2',
                                                            biases_initializer = tf.random_normal_initializer(), reuse=reuse)

            output = tf.layers.dropout(output, self.dropout_rate, training=self.is_training)

        return output

    def buildRLL(self):
        tf.reset_default_graph()

        self.is_training = tf.placeholder_with_default(False, shape=(), name='isTraining')

        self.query = tf.placeholder(tf.float32, shape=[None, self.dimension], name='queryInput')
        self.posDoc = tf.placeholder(tf.float32, shape=[None, self.dimension], name='posDocInput')
        self.negDoc0 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc0Input')
        self.negDoc1 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc1Input')
        self.negDoc2 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc2Input')

        self.queryDocW = tf.placeholder(tf.float32, shape=[None], name='queryDocWeight')
        self.posDocW = tf.placeholder(tf.float32, shape=[None], name='posDocWeight')
        self.negDoc0W = tf.placeholder(tf.float32, shape=[None], name='negDoc0Weight')
        self.negDoc1W = tf.placeholder(tf.float32, shape=[None], name='negDoc1Weight')
        self.negDoc2W = tf.placeholder(tf.float32, shape=[None], name='negDoc2Weight')

        self.lr_rate = tf.placeholder(tf.float32, shape=(), name='learningRate')
        self.reg_scale = tf.placeholder(tf.float32, shape=(), name='penaltyScale')
        self.dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropoutKeepRate')

        if self.sim_type == 'cos':
            sim = cos_sim
        elif self.sim_type == 'euc':
            sim = euc_sim

        # 全部共享
        outputQuery = self.single_net(input_layer=self.query)
        outputPosDoc = self.single_net(input_layer=self.posDoc, reuse=True)
        outputNegDoc0 = self.single_net(input_layer=self.negDoc0, reuse=True)
        outputNegDoc1 = self.single_net(input_layer=self.negDoc1, reuse=True)
        outputNegDoc2 = self.single_net(input_layer=self.negDoc2, reuse=True)

        #########

        with tf.name_scope('loss'):
            reg_ws_0 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1')
            reg_ws_1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2')
            reg_loss = tf.reduce_sum(reg_ws_0)+tf.reduce_sum(reg_ws_1)

            # average_anchor
            if self.anchor_type == 'average_anchor':
                outputQuery = tf.reduce_mean(outputQuery, axis=0, keepdims=True)

            # or mass anchor # weight有用

            elif self.anchor_type == 'mass_anchor':
                outputQuery = tf.reduce_mean(tf.multiply(self.queryDocW[:, tf.newaxis], outputQuery), axis=0, keep_dims=True)

            elif self.anchor_type == 'default':
                pass

            # 这里是一些相似度，大小是[batch size, ]
            self.query_pos_sim = sim(outputQuery, outputPosDoc)
            self.query_doc0_sim = sim(outputQuery, outputNegDoc0)
            self.query_doc1_sim = sim(outputQuery, outputNegDoc1)
            self.query_doc2_sim = sim(outputQuery, outputNegDoc2)

            nominator = tf.multiply(self.posDocW, tf.exp(tf.multiply(self.gamma, self.query_pos_sim)))
            doc0_similarity = tf.multiply(self.negDoc0W, tf.exp(tf.multiply(self.gamma, self.query_doc0_sim)))
            doc1_similarity = tf.multiply(self.negDoc1W, tf.exp(tf.multiply(self.gamma, self.query_doc1_sim)))
            doc2_similarity = tf.multiply(self.negDoc2W, tf.exp(tf.multiply(self.gamma, self.query_doc2_sim)))
            self.prob = prob = tf.add(nominator, tf.constant(1e-7))/tf.add(doc0_similarity+ doc1_similarity+doc2_similarity+nominator, tf.constant(1e-7))
            log_prob = tf.log(prob)
            self.loss = -tf.reduce_sum(log_prob) + reg_loss

            tf.summary.histogram('nominator', nominator)
            tf.summary.histogram('doc0_similarity', doc0_similarity)
            tf.summary.histogram('doc1_similarity', doc1_similarity)
            tf.summary.histogram('doc2_similarity', doc2_similarity)
            tf.summary.histogram('prob', prob)
            tf.summary.scalar('prob_loss', -tf.reduce_sum(log_prob))
            tf.summary.scalar('reg_loss', reg_loss)
            tf.summary.scalar('loss', self.loss)

        self.summaries = tf.summary.merge_all()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr_rate).minimize(self.loss)

    # 在RLLpp的RLL子网络中，训练变成单batch的了。
    def train(self, sess, batchData):
        batchData = {
            self.is_training: batchData['is_training'],
            self.lr_rate: batchData['lr_rate'],
            self.dropout_rate: batchData['dropout_rate'],
            self.reg_scale: batchData['reg_scale'],

            self.query: batchData['query'],
            self.posDoc: batchData['posDoc'],
            self.negDoc0: batchData['negDoc0'],
            self.negDoc1: batchData['negDoc1'],
            self.negDoc2: batchData['negDoc2'],

            self.queryDocW: batchData['queryDocW'],
            self.posDocW: batchData['posDocW'],
            self.negDoc0W: batchData['negDoc0W'],
            self.negDoc1W: batchData['negDoc1W'],
            self.negDoc2W: batchData['negDoc2W']

        }

        _, batch_loss = sess.run([self.optimizer, self.loss], feed_dict=batchData)
        return batch_loss

    def validate(self, sess, batchData):
        batchData = {
            self.is_training: batchData['is_training'],
            self.lr_rate: batchData['lr_rate'],
            self.dropout_rate: batchData['dropout_rate'],
            self.reg_scale: batchData['reg_scale'],

            self.query: batchData['query'],
            self.posDoc: batchData['posDoc'],
            self.negDoc0: batchData['negDoc0'],
            self.negDoc1: batchData['negDoc1'],
            self.negDoc2: batchData['negDoc2'],

            self.queryDocW: batchData['queryDocW'],
            self.posDocW: batchData['posDocW'],
            self.negDoc0W: batchData['negDoc0W'],
            self.negDoc1W: batchData['negDoc1W'],
            self.negDoc2W: batchData['negDoc2W']

        }

        batch_loss = sess.run(self.loss, feed_dict=batchData)
        return batch_loss

    def embedding_forward_loss(self, sess, batchData):
        batchData = {
            self.is_training: batchData['is_training'],
            self.lr_rate: batchData['lr_rate'],
            self.dropout_rate: batchData['dropout_rate'],
            self.reg_scale: batchData['reg_scale'],

            self.query: batchData['query'],
            self.posDoc: batchData['posDoc'],
            self.negDoc0: batchData['negDoc0'],
            self.negDoc1: batchData['negDoc1'],
            self.negDoc2: batchData['negDoc2'],

            self.queryDocW: batchData['queryDocW'],
            self.posDocW: batchData['posDocW'],
            self.negDoc0W: batchData['negDoc0W'],
            self.negDoc1W: batchData['negDoc1W'],
            self.negDoc2W: batchData['negDoc2W']

        }
        prob_as_difficulty = np.array(sess.run(self.prob, feed_dict=batchData))

        return prob_as_difficulty  # 输出的是prob，越大代表越简单

    def get_embedding(self, sess, X_input):

        graph = tf.get_default_graph()
        X = tf.placeholder(tf.float32, shape=[None, self.dimension], name='input')
        w1 = graph.get_tensor_by_name('fc_l1/weights:0')
        b1 = graph.get_tensor_by_name('fc_l1/biases:0')
        w2 = graph.get_tensor_by_name('fc_l2/weights:0')
        b2 = graph.get_tensor_by_name('fc_l2/biases:0')
        embd = activation_fn(tf.matmul(activation_fn(tf.matmul(X, w1) + b1), w2) + b2)
        feed = {X: X_input}
        output = sess.run(embd, feed_dict=feed)
        return np.array(output)


class RLLWORKER:
    # 实现整个流程
    def __init__(self, dimension, l1_n, l2_n, gamma, sim_type, anchor_type, grade):
        self.RLL = RLL(dimension, l1_n, l2_n, gamma, sim_type, anchor_type)
        self.rll_path = '../model_rll_history/rll'
        self.grade = grade

    def build(self, raw_train_data, raw_validation_data):
        self.RLL.buildRLL()
        self.RLL.loadData(raw_train_data=raw_train_data, raw_validation_data=raw_validation_data)

    # 整个训练过程
    def train(self, batchSize, lr_rate, reg_scale, dropout_rate, max_iter,
              filename=None, filename_valid=None, seq_length=7, json_file_list=None, positive_label_types=None, negative_label_types=None):
        # filename: 一个list，默认是输入数据的第一列

        # 连续n个epoch loss 不下降则停止训练
        earlyStopCount = 0
        saver = tf.train.Saver(max_to_keep=1)

        # 模型名称
        model_name = MODEL_NAME.format(self.RLL.l1_n, self.RLL.l2_n, lr_rate, reg_scale, batchSize, dropout_rate,
                                       self.RLL.sim_type, self.RLL.anchor_type, self.grade, seq_length, self.RLL.gamma)
        print('training model {}'.format(model_name))


        # 为模型提供存储位置
        currentModelPath = os.path.join(self.rll_path, model_name)
        if(not os.path.exists(currentModelPath)):
            os.makedirs(currentModelPath)

        """
        训练准备    
        """
        with tf.Session(config=config) as sess:

            tf.global_variables_initializer().run()
            batch_writer = tf.summary.FileWriter(LOG_NAME.format(model_name))
            """
            开始训练
            """
            '''
            这里是实验用的record
            '''
            # 训练epoch
            train_loss_record = []
            val_loss_list_record = []
            personal_confidence_record = []
            item_confidence_record = []
            '''
            实验用的记录信息结束
            '''
            best_val_loss = sys.maxsize
            start = time.time()


            # step 0: 初始化一个confidence df
            # 第一列是文件名，第二列是confidence
            confidence_df = pd.DataFrame(np.ones([self.RLL.raw_train_data.shape[0], 2]), columns=['filename', 'confidence'])
            confidence_df_valid = pd.DataFrame(np.ones([self.RLL.raw_validation_data.shape[0], 2]), columns=['filename', 'confidence'])
            print(confidence_df.shape, confidence_df_valid.shape)

            confidence_df.iloc[:, 0] = filename
            confidence_df_valid.iloc[:, 0] = filename_valid

            # step 0 结束，得到了初始的各个样本的置信度(=1)

            for epoch in range(max_iter):
                print((time.time() - start) // 60, '分钟')
                # 准备安全样本，首先是所有样本的特征和标签
                if epoch > warm_up:
                    # 不用votes做infer，而是用外部输入的confidence_
                    groupsTr, weightsTr = utils.prepareInput(self.RLL.raw_train_data, groupSize=int(groupSize), confidence_df=confidence_df,
                                                             use_history=True)
                    groupsVal, weightsVal = utils.prepareInput(self.RLL.raw_validation_data, groupSize=int(groupSize), confidence_df=confidence_df_valid,
                                                             use_history=True)

                else:
                    # 直接用votes做infer,和老方法一致
                    groupsTr, weightsTr = utils.prepareInput(self.RLL.raw_train_data, groupSize=int(groupSize), confidence_df=confidence_df,
                                                             use_history=False)
                    groupsVal, weightsVal = utils.prepareInput(self.RLL.raw_validation_data, groupSize=int(groupSize), confidence_df=confidence_df_valid,
                                                             use_history=False)

                if epoch == 0:
                    train_size = groupsTr[0].shape[0]
                    print('training group size is {}'.format(train_size))
                    val_size = groupsVal[0].shape[0]
                    print('validation group size is {}'.format(val_size))

                num_batch = train_size // batchSize

                # batch
                total_loss = 0
                for batch in range(num_batch):

                    _, batchData = self.RLL.feedBatch(groupsTr, weightsTr, batchSize, lr_rate,
                                                      dropout_rate, reg_scale, is_training=True)

                    batch_loss = self.RLL.train(sess, batchData)
                    total_loss += batch_loss

                # 这样就训练完了。
                print("Epoch {} train loss {}".format(epoch, total_loss / train_size))
                train_loss_record.append((epoch, total_loss / train_size))

                # 每个epoch结束
                if epoch % 3 == 0:
                    _, valData = self.RLL.feedBatch(groupsVal, weightsVal, groupsVal[0].shape[0], lr_rate, dropout_rate,
                                                 reg_scale, is_training=False)

                    valLoss = self.RLL.validate(sess, valData)
                    print('*' * 66)
                    print("Epoch {} validation loss {}".format(epoch, valLoss / val_size))
                    print('\n')
                    val_loss_list_record.append((epoch, valLoss / val_size))
                    if best_val_loss > valLoss:
                        best_val_loss = valLoss
                        print('best val loss is', best_val_loss)
                        earlyStopCount = 0
                        saver.save(sess, os.path.join(currentModelPath, model_name + '.ckpt'))
                    elif epoch > 10:
                        earlyStopCount += 1
                        # print(earlyStopCount)

                if (earlyStopCount >= 10):
                    print('Early stop at epoch {}!'.format(epoch))
                    break

                # 训练完毕, 获得每个样本在这个epoch结束时的表征
                # step 3 + 4
                if epoch > warm_up:
                    # 1. 不能一次输入太多，内存可能不足.
                    # 2. 记得train 和 valid 都要更新
                    # filename, label, votes, features...
                    repr_batch_cnt = 0
                    train_reprs = self.RLL.get_embedding(sess, self.RLL.raw_train_data.iloc[
                                                               repr_batch_cnt * 400: (repr_batch_cnt + 1) * 400, 3:])
                    for repr_batch_cnt in range(1, self.RLL.raw_train_data.shape[0] // 400 + 1):
                        train_reprs = np.concatenate([train_reprs, self.RLL.get_embedding(sess, self.RLL.raw_train_data.iloc[
                                                                   repr_batch_cnt * 400: (repr_batch_cnt + 1) * 400, 3:])])

                    repr_batch_cnt = 0
                    valid_reprs = self.RLL.get_embedding(sess, self.RLL.raw_validation_data.iloc[
                                                               repr_batch_cnt * 400: (repr_batch_cnt + 1) * 400, 3:])
                    for repr_batch_cnt in range(1, self.RLL.raw_validation_data.shape[0] // 400 + 1):
                        valid_reprs = np.concatenate([valid_reprs, self.RLL.get_embedding(sess, self.RLL.raw_validation_data.iloc[
                                                                   repr_batch_cnt * 400: (repr_batch_cnt + 1) * 400, 3:])])

                    # 得到了列表长度与原始输入长度相等的表征数据的数组。
                    repr_length = train_reprs.shape[1]
                    repr_col_names = [str(i) for i in range(repr_length)]
                    train_repr_df = pd.DataFrame(train_reprs, columns=repr_col_names)
                    train_repr_df['filename'] = filename
                    train_repr_df = train_repr_df[['filename'] + repr_col_names]

                    print('获得第{}个epoch的训练数据表征'.format(epoch))

                    # 验证集同理
                    # 得到了列表长度与原始输入长度相等的表征数据的数组。

                    valid_repr_df = pd.DataFrame(valid_reprs, columns=repr_col_names)
                    valid_repr_df['filename'] = filename_valid
                    valid_repr_df = valid_repr_df[['filename'] + repr_col_names]

                    # step 5 after train, valid, test 已经分好
                    # 开始更新confidence

                    ''' 这是之前存储的confidence的表格，它的 confidence_df.iloc[:, 1] 是要取出来的confidence
                    confidence_df.iloc[:, 0] = filename
                    confidence_df_valid.iloc[:, 0] = filename_valid
                    '''

                    person_confidence_collection = calcucate_personal_confidence(
                        json_file_list=json_file_list,
                        seq_length=seq_length,
                        feature_df=train_repr_df,
                        positive_label_types=positive_label_types,
                        negative_label_types=negative_label_types,
                    )
                    updated_confidence = get_average_confidence(person_confidence_collection)
                    confidence_df.iloc[:, 1] = updated_confidence
                    confidence_df_valid.iloc[:, 1] = updated_confidence

            print('$' * 66)
