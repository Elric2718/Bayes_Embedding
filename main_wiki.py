""" The main file to call Bayes Embedding on the wiki data.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse 
import os
import pickle

import bem.processing as _processing
import bem.bayes_embedding_nf as be

###################
###### wiki #######
###################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_or_pred', '-trpr', type=str, default='train')
    parser.add_argument('--data_folder', '-df', type=int, default=1)
    parser.add_argument('--kg_type', '-kt', type=str, default="wiki_TransE")
    parser.add_argument('--behavior_type', '-bt', type=str, default="pagelink_node2vec")
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--n_batch', '-nb', type=int, default=500)
    parser.add_argument('--n_hidden', '-nh', type=int, default=500)
    parser.add_argument('--lambda1', '-lbd1', type=float, default=1.0)
    parser.add_argument('--lambda2', '-lbd2', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epoch', '-nepoch', type=int, default=20)
    parser.add_argument('--nf_K', '-nfK', type=int, default=0)
    parser.add_argument('--pair_or_single', '-pos', type=str, default="pair")
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.8)
    parser.add_argument('--Decare_rounds', '-dr', type=int, default=10)
    parser.add_argument('--to_split', '-split', action = 'store_true')
    parser.add_argument('--to_normalize', '-norm', action = 'store_true')
    parser.add_argument('--log_file', '-lf', type=str, default="")
    parser.add_argument('--output_prefix', '-op', type=str, default="/wiki_net") 
    parser.add_argument('--pickle_prefix', '-pp', type=str, default="/wiki_net")
    parser.add_argument('--checkpoint', '-cp', type=str, default="model.ckpt")
    args = parser.parse_args()

    train_or_pred = args.train_or_pred
    data_folder = args.data_folder
    n_batch = args.n_batch
    n_hidden = args.n_hidden
    learning_rate = args.learning_rate
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    n_epoch = args.n_epoch
    nf_K = args.nf_K
    pair_or_single = args.pair_or_single
    seed = args.seed
    data_folder = args.data_folder
    kg_type = args.kg_type
    behavior_type = args.behavior_type
    checkpoint = args.checkpoint
    
    data_path = "./data/data{data_folder}".format(data_folder = data_folder)
    output_path = data_path + args.output_prefix 
    pickle_path = data_path + args.pickle_prefix 
    checkpoint_path = './checkpoints/data{data_folder}/{checkpoint}'.format(data_folder = data_folder, checkpoint = checkpoint)
    log_file = args.log_file

    if not(os.path.exists("./checkpoints")):
        os.makedirs("./checkpoints")
    
    # parameters
    if train_or_pred == "train":        
        input1_path = data_path + "/{behavior_type}".format(behavior_type = behavior_type)#e.g."/wiki_citation_net_emb_final.txt", "/embedding_desc.txt"
        input2_path = data_path + "/{kg_type}".format(kg_type = kg_type)# e.g. /item_embedding_kg.txt

        # extract the shared data that appears in both kg and behavior
        entity2id_path = data_path + "/entity2id.txt"
        kg_shared_path = data_path + "/entity2shared_id.txt"
    else:
        id_path = pickle_path + "_id.txt" # "/wiki_net_id.txt"
        input1_path = pickle_path + "_raw_dat1.csv" #"/wiki_net_raw_dat1.csv"
        input2_path = pickle_path + "_raw_dat2.csv" #"/wiki_net_raw_dat2.csv"

    # read in data
    if train_or_pred == "train":
        split_ratio = args.split_ratio
        Decare_rounds = args.Decare_rounds if pair_or_single == "pair" else 1
        to_split = args.to_split
        to_normalize = args.to_normalize

        # train-test split
        with tf.device("/cpu:0"):
            if os.path.exists(pickle_path + "_raw_dat.pkl"):
                with open(pickle_path + "_raw_dat.pkl", 'rb') as f:
                    train_test_dict = pickle.load(f)
                train = train_test_dict["train"]
                train.Decare_rounds = Decare_rounds
                test = train_test_dict["test"]
                test.Decare_rounds = Decare_rounds
                del train_test_dict
            else:
                train, test = _processing.SplitData(input1_path, input2_path, pickle_path, split_ratio, Decare_rounds, to_normalize, to_split)        
                # extract the inhomogeneous data that only appears in kg
                _processing.ExtractSharedData(entity2id_path, input1_path, kg_shared_path)
                with open(pickle_path + "_raw_dat.pkl", 'wb') as f:
                    pickle.dump({"train": train, "test": test}, f, protocol = pickle.HIGHEST_PROTOCOL)
                    
            n_prior = train.n_dat2
            n_obs = train.n_dat1
        print("Finish processing the data.")        
    else:
        print("Start prediction.")
        with tf.device("/cpu:0"):
            data_id = np.array(pd.read_csv(id_path, sep = "#", header = None, names = ["item_id"])["item_id"].values)
            dataset1 = _processing.decode_csv(input1_path)
            dataset2 = _processing.decode_csv(input2_path)
            
        print("Finish loading the prediction data.")            
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
                            nf_K = nf_K,\
                            pair_or_single = pair_or_single,\
                            seed = seed,\
                            checkpoint_path = checkpoint_path,\
                            log_file = log_file)


    # train the network or predict the results
    if train_or_pred == "train":                
        model.fit(train, test)
    else:
        model.graph = model._create_graph()
        mu = np.array(model.encode(dataset1, dataset2)[0])
        print("Finish estimating the parameters")
        new_z, new_h = model.decode(mu, dataset2)
        print("Finish predicting the data")

        new_z_df = pd.DataFrame({"item_id": data_id, "new_data1": np.array([','.join(line.astype(str)) for line in new_z])})
        new_h_df = pd.DataFrame({"item_id": data_id, "new_data2": np.array([','.join(line.astype(str)) for line in new_h])})

        new_z_df.to_csv(output_path + "_new_dat1.csv", sep = "#", header = False, index = False)
        new_h_df.to_csv(output_path + "_new_dat2.csv", sep = "#", header = False, index = False)
        print("Finish writing the new data.")




        

    
    
