import tensorflow as tf
import numpy as np
import input
import bayes_embedding as be
import pandas as pd


if __name__ == "__main__":
    # parameters
    FLAGS = {}
    FLAGS["data_proc_sh"] = "./data_proc.sh"
    FLAGS["file_dir"] = "../data"
    # FLAGS["input_name"] = "test.txt"
    # FLAGS["output_name"] = "test"
    FLAGS["input_name"] = "embedding_im.txt"
    FLAGS["output_name"] = "embedding_im"

    train_or_pred = "pred"
    n_batch = 1000
    n_hidden = 500
    learning_rate = 0.001
    lambda1 = 1
    lambda2 = 1
    n_epoch = 20
    seed = 0    
    checkpoint_path = 'checkpoints/model.ckpt'

    # read in data
    if train_or_pred == "train":
        split_ratio = 0.8
        Decare_rounds = 10

        train, test = input.SplitData(FLAGS, None, None, split_ratio, Decare_rounds)
        n_prior = train.n_dat2
        n_obs = train.n_dat1
    else:
        output_name = FLAGS["file_dir"] + "/" + FLAGS["output_name"]
        dataset1 = pd.read_csv(output_name + "_dat1.csv", header = None).values
        dataset2 = pd.read_csv(output_name + "_dat2.csv", header = None).values

        n_prior = dataset2.shape[1]
        n_obs = dataset1.shape[1]    

    # build the model    
    model = be.BayesNetwork(n_batch = n_batch,\
                            n_prior = n_prior,\
                            n_hidden = n_hidden,\
                            n_obs = n_obs,\
                            learning_rate = learning_rate,\
                            lambda1 = lambda1,\
                            lambda2 = lambda2,\
                            n_epoch = n_epoch,\
                            seed = seed,\
                            checkpoint_path = checkpoint_path)


    # train the network or predict the results
    if train_or_pred == "train":                
        model.fit(train, test)
    else:
        model.graph = model._create_graph()
        mu = np.array(model.encode(dataset1)[0])
        new_z, new_h = model.decode(mu, dataset2)

        np.savetxt(output_name + "_new_dat1.csv", np.array(new_z), delimiter = ",")
        np.savetxt(output_name + "_new_dat2.csv", np.array(new_h), delimiter = ",")






