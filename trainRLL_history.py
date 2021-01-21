from RLL_history_learning import RLLWORKER
import numpy as np
import pandas as pd
import glob
import sys
from data_cfg import cfg
which_data = sys.argv[1]

dimension = cfg[which_data]['dimension']
DATA_NAME = cfg[which_data]['DATA_NAME']
train_df_name = cfg[which_data]['train_df_name']
valid_df_name = cfg[which_data]['valid_df_name']
json_file_pattern = cfg[which_data]['json_file_pattern']
positive_label_types = cfg[which_data]['positive_label_types']
negative_label_types = cfg[which_data]['negative_label_types']

# first column is ground truth label, second column is crowdsourced votes (set to 1 if you have no crowdsourced labels)
train = pd.read_csv(DATA_NAME + train_df_name)
validation = pd.read_csv(DATA_NAME + valid_df_name)
json_file_list = glob.glob(DATA_NAME + json_file_pattern)

#####
#####
# grid search for parameters
max_iter = 300
batchSize = 256

sorted(json_file_list)
lr_rate_lst = [1e-3]

l1_n_lst = [256, 128]

l2_n_lst = [64, 32]

reg_scale_lst = [2.0]
dropout_rate_lst = [0.2]
sim_type_lst = ['euc']
anchor_type_lst = ['default']
seq_length_list = [9, 10, 11]

gamma_list = [2.0]

for lr_rate in lr_rate_lst:
    for dropout_rate in dropout_rate_lst:
        for l1_n in l1_n_lst:
            for l2_n in l2_n_lst:
                for reg_scale in reg_scale_lst:
                        if l1_n < l2_n:
                            continue
                        for sim_type in sim_type_lst:
                            for anchor_type in anchor_type_lst:
                                for gamma in gamma_list:
                                    for seq_length in seq_length_list:
                                        model = RLLWORKER(dimension, l1_n, l2_n, gamma, sim_type, anchor_type, grade=which_data)
                                        model.build(train, validation)
                                        model.train(batchSize, lr_rate, reg_scale, dropout_rate, max_iter,
                                                    filename=train.iloc[:, 0].tolist(),
                                                    filename_valid=validation.iloc[:, 0].tolist(),
                                                    seq_length=seq_length,
                                                    json_file_list=json_file_list,
                                                    positive_label_types=positive_label_types,
                                                    negative_label_types=negative_label_types,)

