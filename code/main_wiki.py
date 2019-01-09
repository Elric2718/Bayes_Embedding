""" The main file to call Bayes Embedding on the wiki data.
"""
import tensorflow as tf
import numpy as np
import input as _input
import bayes_embedding as be
import pandas as pd
import argparse


############################
###### infant-mother #######
############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_or_pred', '-trpr', type=str, default='train')
    parser.add_argument('--data_type', '-dt', type=int, default=3)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--n_batch', '-nb', type=int, default=500)
    parser.add_argument('--n_hidden', '-nh', type=int, default=500)
    parser.add_argument('--lambda1', '-lbd1', type=float, default=1.0)
    parser.add_argument('--lambda2', '-lbd2', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epoch', '-nepoch', type=int, default=20)
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.8)
    parser.add_argument('--Decare_rounds', '-dr', type=int, default=10)
    parser.add_argument('--log_file', '-lf', type=str, default="test.txt")

    args = parser.parse_args()
    
    train_or_pred = args.train_or_pred
    data_type = args.data_type
    n_batch = args.n_batch
    n_hidden = args.n_hidden
    learning_rate = args.learning_rate
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    n_epoch = args.n_epoch
    seed = args.seed
    data_type = args.data_type
    log_file = args.log_file
    
    data_path = "../data/data{data_type}".format(data_type = data_type)
    output_prefix = "/wiki_net2" #"/wiki"
    output_path = data_path + output_prefix
    checkpoint_path = 'checkpoints/data{data_type}/model.ckpt'.format(data_type = data_type)

    # parameters
    if train_or_pred == "train":        
        input1_path = data_path + "/wiki_citation_spider.txt"#"/wiki_citation_net_emb_final.txt"#"/embedding_desc.txt"
        input2_path = data_path + "/item_embedding_kg.txt" 

        # extract the inhomogeneous data that only appears in kg
        entity2id_path = data_path + "/entity2id.txt"
        kg_shared_path = data_path + "/entity2shared_id2.txt"
    else:
        id_path = data_path + "/wiki_net2_id.txt" # "/wiki_id.txt"
        input1_path = data_path + output_prefix + "_raw_dat1.csv" #"/wiki_raw_dat1.csv"
        input2_path = data_path + output_prefix + "_raw_dat2.csv" #"/wiki_raw_dat2.csv"

    # read in data
    if train_or_pred == "train":
        split_ratio = args.split_ratio
        Decare_rounds = args.Decare_rounds

        train, test = _input.SplitData(input1_path, input2_path, output_path, split_ratio, Decare_rounds)        
        n_prior = train.n_dat2
        n_obs = train.n_dat1

        # extract the inhomogeneous data that only appears in kg
        _input.ExtractSharedData(entity2id_path, input1_path, kg_shared_path)
        
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
                            checkpoint_path = checkpoint_path,\
                            log_file = log_file)


    # train the network or predict the results
    if train_or_pred == "train":                
        model.fit(train, test)
    else:
        model.graph = model._create_graph()
        mu = np.array(model.encode(dataset1)[0])
        new_z, new_h = model.decode(mu, dataset2)

        new_z_df = pd.DataFrame({"item_id": data_id, "new_data1": np.array([','.join(line.astype(str)) for line in new_z])})
        new_h_df = pd.DataFrame({"item_id": data_id, "new_data2": np.array([','.join(line.astype(str)) for line in new_h])})

        new_z_df.to_csv(output_path + "_new_dat1.csv", sep = "#", header = False, index = False)
        new_h_df.to_csv(output_path + "_new_dat2.csv", sep = "#", header = False, index = False)

        #np.savetxt(path_name + "_new_dat1.csv", np.array(new_z), delimiter = ",")
        #np.savetxt(path_name + "_new_dat2.csv", np.array(new_h), delimiter = ",")





        

    
    
