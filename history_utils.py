import pandas as pd
import numpy as np
import traceback
import glob
from scipy.special import softmax
from scipy.spatial.distance import cdist

def KQ(K, Q):
    """
    :param K: n * d 最近n个样本的features
    :param Q: 1 * d 当前样本的features
    :return W: 1 * n 每个样本对K的相似度权重
    """

    W = np.dot(K, np.broadcast_to(Q, K.shape).T)
    softmax_W = softmax(W[:, 0])
    # print(softmax_W)

    return softmax_W


def WV(W, V):
    """

    :param W: n * 1  V 在时间序列上的权重, 例如  [0.3, 0.2, 0.5]T
    :param V: n * d labeler的标注 [0 0 1]  [0 1 0]  [0 0 1]
    :return: 当前样本基于注意力机制得到的每个类别的信心 W
    """

    return np.dot(V.T, W)


def label_confidence_inference(ordered_feature_list, label_list):
    """
    对给出的序列features和labels，求根据一致性猜测的label_list

    input params:
    ordered_feature_list: K, Q
    label_list: V

    """

    K = np.array(ordered_feature_list)
    Q = K[-1]
    V = np.array(label_list)
    # print(K.shape, V.shape)
    W = KQ(K, Q)
    result = WV(W, V)
    return result

def one_hot(x, positive_label_types=None, negative_label_types=None):

    if x in positive_label_types:
        return [0, 1]

    if x in negative_label_types:
        return [1, 0]

    return [-1, -1]


def one_hot_2_cls(x, positive_label_types=None, negative_label_types=None):
    if x in negative_label_types:
        return 0

    if x in positive_label_types:
        return 1

    return -1


def get_feature_list_by_order(current_file: str, seq_length: int, personal_label_list: list, feature_df: pd.DataFrame, positive_label_types: list, negative_label_types: list):
    """
    对给定的当前标注，获得最近n次标注的features和标注labels;
    如果包含自己在内，最近n次都是一样的，那么最终输出一定是确定的1. 这种情况后面集中看一下
    最开头的几次标注不计入。

    input params:
    current_file: 当前作为输入的文件名
    seq_length: 考虑最近n次标注
    personal_label_list: 这位标注人员的标注顺序序列(文件名) [{'file_1': 1}, {'file_2': 0}, ...]
    feature_df: 样本对应的特征的池子

    output params:
    ordered_feature_list: 按顺序给出这位标注人员的最近n次标注的features, 最后一位固定是current_file
    label_list: 对应的标签。每次标注的标签为one-hot形式

    """

    label_list = []
    # print('输入的seq len:', seq_length)
    # print(personal_label_list)
    ordered_feature_list = []
    # 从personal_label_dict里面找到current_file
    for idx, i in enumerate(personal_label_list):
        if list(i.keys())[0] == current_file:

            # 一共需要seq_length个数据
            seq_input_count = 0

            # 这一次处理相对于current_file的偏移量
            idx_count = 0

            # 从最近的开始往里面放, 逐渐追溯历史。
            while seq_input_count < seq_length:
                try:
                    label_ = one_hot(list(personal_label_list[idx - idx_count].values())[0],
                                     positive_label_types=positive_label_types,
                                     negative_label_types=negative_label_types)
                    # 无效音频不参与计算
                    if label_ == [-1, -1]:
                        if idx_count == 0:
                            return None, None

                        idx_count += 1
                        continue

                    if idx < idx_count:
                        return None, None

                    # 如果没越界，就继续找。如果越界了就返回无效
                    # 找前一个如果不在feature列表里，说明找了一个无效音频，那不管它，继续找前一个。
                    try:
                        feature_ = feature_df[
                                       feature_df['filename'] == list(personal_label_list[idx - idx_count].keys())[0]].iloc[
                                   0, 1:].to_numpy()
                        # print(feature_)
                        # print('aaa')

                        ordered_feature_list.append(feature_)
                        label_list.append(label_)

                        idx_count += 1
                        seq_input_count += 1
                    except:
                        idx_count += 1

                except Exception as e:
                    traceback.print_exc()
                    return None, None

    return ordered_feature_list, label_list


# 计算每个人置信度的主函数
def calcucate_personal_confidence(json_file_list=None, seq_length=5, feature_df=None,
                                  positive_label_types=None, negative_label_types=None):

    print('开始计算本轮所有人的信心列表')
    person_confidence_collection = []

    # 每个人有一个标注json
    for t, this_json in enumerate(json_file_list):
        print('开始计算第{}人'.format(t))
        with open(this_json, 'r') as f:
            l = f.read()
            json_content = eval(l)

        # 准备获取这个人标注的文件顺序和标注结果
        audio_personal, result_personal = [], []
        for i in json_content['true_message']['data']:
            audio_personal.append(i['audio_name'])
            result_personal.append(i['label'][0]) # 默认选第一位的标签作为输入. 后面是辅助的.

        # 整理成能处理的格式
        personal_label_seq = []
        for i in range(len(audio_personal)):
            personal_label_seq.append({audio_personal[i]: result_personal[i]})

        single_person_confidence = []

        # 转换格式，变成直接可查的
        output_f = {}
        for p in personal_label_seq:
            output_f[list(p.keys())[0]] = p[list(p.keys())[0]]

        # 按照filename的顺序进行置信度计算
        for fdx, file in enumerate(feature_df.filename.tolist()):

            if fdx % 1000 == 0:
                print(fdx)
            try:
                ordered_feature_list, label_list = get_feature_list_by_order(file, seq_length, personal_label_seq,
                                                                             feature_df, positive_label_types, negative_label_types)
                # print(np.concatenate(ordered_feature_list).shape, label_list)
                label_by_inference = label_confidence_inference(ordered_feature_list, label_list)

                # 从列表中取出这次所需的confidence是第一位还是第二位
                one_hot_res = one_hot_2_cls(output_f[file], positive_label_types=positive_label_types,
                                            negative_label_types=negative_label_types)
                if one_hot_res > -1:
                    single_person_confidence.append(label_by_inference[one_hot_res])
            except Exception as e:
                # traceback.print_exc()
                single_person_confidence.append(-1)
                # break
        person_confidence_collection.append(single_person_confidence)
    # print(person_confidence_collection)

    return person_confidence_collection


def get_average_confidence(person_confidence_collection):
    person_confidence_collection = np.array(person_confidence_collection).T
    person_confidence_collection = np.clip(person_confidence_collection, 0.0001, 1)
    return np.mean(person_confidence_collection, axis=1)


if __name__ == '__main__':

    # f_list = glob.glob('../raw_data/class_rehearsal/*-1.json')[0:2]
    #
    # # 实际用的时候传入的是filename+reprs的dataframe
    # feature_df = pd.read_csv('../raw_data/class_rehearsal/raw_data_3833.csv')
    # feature_df.drop(columns=['label', 'votes'], inplace=True)
    # cols = feature_df.columns.tolist()
    # print(feature_df)
    # cols[0] = 'filename'
    # feature_df.columns = cols
    #
    # seq_length = 7
    #
    # person_confidence_collection = calcucate_personal_confidence(
    #     json_file_list=f_list,
    #     seq_length=seq_length,
    #     feature_df=feature_df,
    #     positive_label_types=['情感丰富'],
    #     negative_label_types=['情感一般'],
    # )
    # print(person_confidence_collection)
    # print(get_average_confidence(person_confidence_collection))

    f_list = glob.glob('../raw_data/question_detection/*-1.json')
    # 实际用的时候传入的是filename+reprs的dataframe
    feature_df = pd.read_csv('../raw_data/question_detection/train_raw_3925.csv')
    positive_label_types = ['评估问句', '认知问句', '其它问句', '自问自答问句']
    negative_label_types = ['无人声', '非提问']

    feature_df.drop(columns=['label', 'votes'], inplace=True)
    cols = feature_df.columns.tolist()
    print(feature_df)
    cols[0] = 'filename'
    feature_df.columns = cols

    seq_length = 5

    person_confidence_collection = calcucate_personal_confidence(
        json_file_list=f_list,
        seq_length=seq_length,
        feature_df=feature_df,
        positive_label_types=positive_label_types,
        negative_label_types=negative_label_types,
    )
    print(person_confidence_collection)
    print(get_average_confidence(person_confidence_collection))

