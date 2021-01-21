import numpy as np
from sklearn.utils import shuffle


def prepareInput(data, groupSize=int(8e4), confidence_df=None, use_history=False):
    # confidence_ 是这次的confidence， 是从confidence_df里取出来的
    # data: 0 filename 1 label 2 votes 3: features

    # 去掉文件名
    data = data.iloc[:, 1:]
    if not use_history:
        # confidence_ 全等于1就是一般的RLL，confidence_等于自定义的值则是追踪history的做法。
        data = inferWeight(np.array(data))
        # 此时data的第二列固定是 1/5, 3/5这种数值。

    else:
        original_labels = data.iloc[:, 0].tolist()
        original_confidence = confidence_df.iloc[:, 1].tolist()
        for c_idx, c in confidence_df.iloc[:, 1]:
            if c < 0.5:
                original_labels[c_idx] = 1 - original_labels[c_idx]
                original_confidence[c_idx] = 1 - c

        # update estimated confidence and labels.
        data.iloc[:, 0] =original_labels
        data.iloc[:, 1] =original_confidence

    data = np.array(data)
    groups, weights = createGroupsRandom(data, groupSize)

    return groups, weights


def inferWeight(data, alpha=None, beta=None):    
    votes = data[:, 1]
    maxVote = max(votes)
    weights = []
    for i in range(votes.shape[0]):
        v = votes[i]

        if v >= (1+maxVote)/2:
            if alpha is None or beta is None:
                weights.append(float(v/maxVote))
            else:
                weights.append(float((v+alpha)/(maxVote+alpha+beta)))
        else:
            if alpha is None or beta is None:
                weights.append(1-float(v/maxVote))
            else:
                weights.append(float((maxVote-v+alpha)/(maxVote+alpha+beta)))
    data[:, 1] = weights
    return data


def splitFeatureWeight(x):
    return x[:,1:], x[:,0]


def createGroupsRandom(data, groupSize=int(8e4), rand_anchor_class=False):

    ########
    if rand_anchor_class:
        # 如果希望某个epoch中，负例也有机会作为anchor
        pos_label_sign = np.random.randint(0, 2, 1)[0]
        positive = data[np.where(data[:, 0] == pos_label_sign)]
        negative = data[np.where(data[:, 0] != pos_label_sign)]
        ########
    else:
        # 否则
        positive = data[np.where(data[:, 0] == 1)]
        negative = data[np.where(data[:, 0] == 0)]

    ########
    # 关于数据的第一列: 整数票数的时候，第一列是label，第二列是votes，第三列往后(2:)是features
    # 小数票数的时候，第一列是label，第二列是votes，第三列往后是features不变。


    posNum = positive.shape[0]
    negNum = negative.shape[0]

    idx = np.random.randint(low=0, high=posNum, size=groupSize)
    query = np.array([positive[i, 1:] for i in idx])
    posDoc = shuffle(query)

    idx = np.random.randint(low=0, high=negNum, size=groupSize)
    negDoc0 = np.array([negative[i, 1:] for i in idx])
    negDoc1 = shuffle(negDoc0)
    negDoc2 = shuffle(negDoc0)
    
    query, queryDocW = splitFeatureWeight(query)
    posDoc, posDocW = splitFeatureWeight(posDoc)
    negDoc0, negDoc0W = splitFeatureWeight(negDoc0)
    negDoc1, negDoc1W = splitFeatureWeight(negDoc1)
    negDoc2, negDoc2W = splitFeatureWeight(negDoc2)
    
    groups = (query, posDoc, negDoc0, negDoc1, negDoc2)
    weights = (queryDocW, posDocW, negDoc0W, negDoc1W, negDoc2W)
    return groups, weights