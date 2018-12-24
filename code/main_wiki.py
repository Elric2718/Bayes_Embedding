import tensorflow as tf
import numpy as np
import input as _input
import bayes_embedding as be
import pandas as pd


############################
###### infant-mother #######
############################

if __name__ == "__main__":

    train_or_pred = "pred"
    data_type = 1
    n_batch = 500
    n_hidden = 500
    learning_rate = 0.001
    lambda1 = 1
    lambda2 = 1
    n_epoch = 20
    seed = 0
    data_path = "../data/data2"
    output_path = data_path + "/wiki"
    checkpoint_path = 'checkpoints/data2/model.ckpt'

    # parameters
    if train_or_pred == "train":        
        input1_path = data_path + "/embedding_desc.txt"
        input2_path = data_path + "/item_embedding_kg.txt"

    else:
        id_path = data_path + "/wiki_id.txt"
        input1_path = data_path + "/wiki_raw_dat1.csv"
        input2_path = data_path + "/wiki_raw_dat2.csv"

    # read in data
    if train_or_pred == "train":
        split_ratio = 0.8
        Decare_rounds = 10
        
        train, test = _input.SplitData(input1_path, input2_path, output_path, split_ratio, Decare_rounds)
        n_prior = train.n_dat2
        n_obs = train.n_dat1
    else:
        data_id = np.array(pd.read_csv(id_path, sep = "#", header = None, names = ["item_id"])["item_id"].values)
        dataset1 = _input.decode_csv(pd.read_csv(input1_path, sep = "#", header = None, names = ["item_id", "data1"])["data1"].values)
        dataset2 = _input.decode_csv(pd.read_csv(input2_path, sep = "#", header = None, names = ["item_id", "data2"])["data2"].values)
            
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

        new_z_df = pd.DataFrame({"item_id": data_id, "new_data1": np.array([','.join(line.astype(str)) for line in new_z])})
        new_h_df = pd.DataFrame({"item_id": data_id, "new_data2": np.array([','.join(line.astype(str)) for line in new_h])})

        new_z_df.to_csv(output_path + "_new_dat1.csv", sep = "#", header = False)
        new_h_df.to_csv(output_path + "_new_dat2.csv", sep = "#", header = False)
        #np.savetxt(path_name + "_new_dat1.csv", np.array(new_z), delimiter = ",")
        #np.savetxt(path_name + "_new_dat2.csv", np.array(new_h), delimiter = ",")





        

    
    
