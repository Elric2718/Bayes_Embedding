import tensorflow as tf
import numpy as np
import input as _input
import pandas as pd
import evaluate as _eval
import os

if __name__ == "__main__":
    data_type = 1
    item_or_KG = 1
    #data_path = "../data/data1/test_full_dat1.csv"
    data_path = "../data/data{data_type}/embedding_im_full_dat{item_or_KG}.csv".format(data_type = data_type, item_or_KG = item_or_KG)
    label_path = "../data/data{data_type}/embedding_cate2_im.txt".format(data_type = data_type)
    nfold = 5

    dataset = _input.EvalDataSet(data_path, label_path, nfold)

    print(dataset.n_label)
    print(dataset.num_samples)
    for fold_idx in range(nfold):
        print("*************** Start fold {fold_idx}. ***************".format(fold_idx = fold_idx))
        dataset.next_fold()
        n_batch = 1000
        n_feat = dataset.n_feature
        n_label = dataset.n_label
        
        network_dict = {}
        n_layers = 2
        layer_size = 128
        activation = tf.tanh
        output_activation = None
        nn_init_stddev = 0.2

        learning_rate = 0.001
        n_epoch = 20
        checkpoint_path = "checkpoints/classification_model{data_type}.ckpt".format(data_type = data_type)
        logdir = "../data/data{data_type}/log".format(data_type = data_type)
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)

        if fold_idx == 0:
            seed = os.listdir(logdir)
            seed = 0 if len(seed) == 0 else max([int(log_name.split("log")[1]) for log_name in seed]) + 1

        
        classification_model = _eval.Classifier(n_batch = n_batch,\
                                                    n_feat = n_feat,\
                                                    n_label = n_label,\
                                                    n_layers = n_layers,\
                                                    layer_size = layer_size,\
                                                    activation = activation,\
                                                    output_activation = output_activation,\
                                                    nn_init_stddev = nn_init_stddev,\
                                                    learning_rate = learning_rate,\
                                                    n_epoch = n_epoch,\
                                                    seed = seed,\
                                                    checkpoint_path = checkpoint_path,\
                                                    logdir = os.path.join(logdir, "task1_log" + str(seed))
                                                    )

        classification_model.fit(dataset)

        predictions = classification_model.predict(dataset.testdata()["features"])
        accuracy = _eval.measurement(predictions, dataset.testdata()["labels"])

        with open(os.path.join(logdir, "task1_log{seed}/result.txt".format(seed = seed)), "a+") as f:
            f.write(str(fold_idx) + "," + str(accuracy) + "\n")
            
        
        
        
        print("*************** Accuracy of Fold " + str(fold_idx) + " is: " + str(accuracy) + ". ***************")

        
    
