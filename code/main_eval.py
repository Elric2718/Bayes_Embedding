""" main file for the evaluating the embedding on classification/regression tasks.
"""

import tensorflow as tf
import numpy as np
import input as _input
import pandas as pd
import evaluate as _eval
import os

if __name__ == "__main__":
    data_type = 3
    raw_or_new = "new"
    item_or_KG = 2
    data_name = "wiki_net2"
    signal_name = "entity2type_20181229.txt" 
    data_path = "../data/data{data_type}/{data_name}_{raw_or_new}_dat{item_or_KG}.csv".format(data_type = data_type, data_name = data_name, raw_or_new = raw_or_new, item_or_KG = item_or_KG)
    signal_path = "../data/data{data_type}/{signal_name}".format(data_type = data_type, signal_name = signal_name)

    
    nfold = 5
    task_id = 3
    supervision_type = "classification"
    dataset = _input.EvalDataSet(data_path, signal_path, nfold, supervision_type)


    print(dataset.n_signal)
    print(dataset.num_samples)
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
        checkpoint_path = "checkpoints/data{data_type}_{supervision_type}_model{data_type}.ckpt".format(supervision_type = supervision_type, data_type = data_type)
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

        with open(os.path.join(logdir, "task{task_id}_log{seed}/result.txt".format(task_id = task_id, seed = seed)), "a+") as f:
            f.write(str(fold_idx) + "," + str(accuracy) + "\n")
            
        
        
        
        print("*************** Accuracy of Fold " + str(fold_idx) + " is: " + str(accuracy) + ". ***************")

        
    
