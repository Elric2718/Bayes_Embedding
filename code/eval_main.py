import tensorflow as tf
import numpy as np
import input as _input
import pandas as pd
import evaluate as _eval

if __name__ == "__main__":
    data_path = "../data/data1/embedding_im_full_new_dat1.csv"
    label_path = "../data/data1/embedding_cate_im.txt"
    nfold = 5

    dataset = _input.EvalDataSet(data_path, label_path, nfold)

    print(dataset.n_label)
    print(dataset.num_samples)
    for fold_idx in range(nfold):
        print("*************** Start fold " + str(fold_idx) + ". ***************")
        dataset.next_fold()
        n_batch = 1000
        n_feat = dataset.n_feature
        n_label = dataset.n_label
        
        network_dict = {}
        network_dict["n_layers"] = 2
        network_dict["layer_size"] = 128
        network_dict["activation"] = tf.tanh
        network_dict["output_activation"] = None
        network_dict["nn_init_stddev"] = 0.2

        learning_rate = 0.001
        n_epoch = 20
        seed = 0
        checkpoint_path = "checkpoints/classification_model.ckpt"

        classification_model = _eval.Classifier(n_batch = n_batch,\
                       n_feat = n_feat,\
                       n_label = n_label,\
                       network_dict = network_dict,\
                       learning_rate = learning_rate,\
                       n_epoch = n_epoch,\
                       seed = seed,\
                       checkpoint_path = checkpoint_path
                       )

        classification_model.fit(dataset)

        predictions = classification_model.predict(dataset.testdata()["features"])
        accuracy = _eval.measurement(predictions, dataset.testdata()["labels"])
        print("*************** Accuracy of Fold " + str(fold_idx) + " is: " + str(accuracy) + ". ***************")

        
    
