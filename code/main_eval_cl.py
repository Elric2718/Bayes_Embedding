""" main file for the evaluating the embedding on classification/regression tasks.
"""

import tensorflow as tf
import numpy as np
import input as _input
import pandas as pd
import evaluate as _eval
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', '-dt', type=int, default=3)
    parser.add_argument('--raw_or_new', '-ron', type=str, default="raw")
    parser.add_argument('--item_or_KG', '-ioK', type=str, default='1')
    parser.add_argument('--kg_type', '-kt', type=str, default="wiki_TransE")
    parser.add_argument('--behavior_type', '-bt', type=str, default="desc_doc2vec")
    parser.add_argument('--pair_or_single', '-pos', type=str, default="pair")
    parser.add_argument('--logdir', '-ld', type=str, default='')
    parser.add_argument('--result_file', '-rfile', type=str, default='')

    
    args = parser.parse_args()
    data_type = args.data_type
    raw_or_new = args.raw_or_new
    item_or_KG = args.item_or_KG
    
    kg_type = args.kg_type
    behavior_type = args.behavior_type
    pair_or_single = args.pair_or_single
    logdir = args.logdir
    result_file = args.result_file

    data_name = kg_type + "_" + behavior_type + "_" + pair_or_single #"wiki_net2"
    signal_name = "entity2type_20181229.txt" 
    data_path = "../data/data{data_type}/{data_name}_{raw_or_new}_dat{item_or_KG}.csv".format(data_type = data_type, data_name = data_name, raw_or_new = raw_or_new, item_or_KG = item_or_KG)
    signal_path = "../data/data{data_type}/{signal_name}".format(data_type = data_type, signal_name = signal_name)

    if not os.path.exists(data_path):
        _input.aggregate_data("../data/data"+ str(data_type) + "/" + data_name + "_" + raw_or_new + "_dat1.csv",\
                                  "../data/data"+ str(data_type) + "/" + data_name + "_" + raw_or_new + "_dat2.csv",\
                                  "../data/data"+ str(data_type) + "/" + data_name + "_" + raw_or_new + "_dat" + item_or_KG + ".csv",\
                                  item_or_KG[1:])
            
    
    nfold = 5
    task_id = 5
    supervision_type = "classification"
    dataset = _input.EvalDataSet(data_path, signal_path, nfold, supervision_type)


    print(dataset.n_signal)
    print(dataset.num_samples)
    mean_accuracy = 0.
    for fold_idx in range(nfold):
        print("*************** Start fold {fold_idx}. ***************".format(fold_idx = fold_idx))
        dataset.next_fold()
        n_batch = 500
        n_feat = dataset.n_feature
        n_signal = dataset.n_signal        
        network_dict = {}
        n_layers = 1
        layer_size = 64
        activation = tf.tanh
        output_activation = None
        nn_init_stddev = 0.2

        learning_rate = 0.001
        n_epoch = 50
        checkpoint_path = "checkpoints/data{data_type}/{supervision_type}_model{data_type}.ckpt".format(supervision_type = supervision_type, data_type = data_type)
        logdir = "../data/data{data_type}/log/task{task_id}_log".format(data_type = data_type, task_id = task_id)
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)

        if fold_idx == 0:
            seed = os.listdir(logdir)
            print(seed)
            seed = 0 if len(seed) == 0 else max([int(log_name.split("log")[1]) for log_name in seed]) + 1

        
        supervision_model = _eval.Supervisor(n_batch = n_batch,\
                                                    n_feat = n_feat,\
                                                    n_signal = n_signal,\
                                                    n_layers = n_layers,\
                                                    layer_size = layer_size,\
                                                    activation = activation,\
                                                    output_activation = output_activation,\
                                                    nn_init_stddev = nn_init_stddev,\
                                                    learning_rate = learning_rate,\
                                                    n_epoch = n_epoch,\
                                                    seed = seed,\
                                                    supervision_type = supervision_type,\
                                                    checkpoint_path = checkpoint_path,\
                                                    logdir = os.path.join(logdir, "task{task_id}_log{seed}".format(task_id = task_id, seed = seed))
                                                    )

        supervision_model.fit(dataset)

        predictions = supervision_model.predict(dataset.testdata()["features"])
        accuracy = _eval.measurement(predictions, dataset.testdata()["signals"], supervision_type)

        mean_accuracy += accuracy
        if result_file == '':
            result_file = os.path.join(logdir, "task{task_id}_log{seed}/result.txt".format(task_id = task_id, seed = seed)) 
        with open(result_file, "a+") as f:
            f.write(str(fold_idx) + "," + str(accuracy) + "\n")                   
        
        
        print("*************** Accuracy of Fold " + str(fold_idx) + " is: " + str(accuracy) + ". ***************")
    if result_file == '':
        result_file = os.path.join(logdir, "task{task_id}_log{seed}/result.txt".format(task_id = task_id, seed = seed)) 
    with open(result_file, "a+") as f:
        f.write("Mean: " + str(mean_accuracy/nfold) + "\n")                   
    

        
    
