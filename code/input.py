""" This script contains a number of functions and classes that are used to preprocess the data.
"""

import numpy as np
import subprocess
import pandas as pd
import json
import decimal
import os
import faiss
import tensorflow as tf
import itertools
import time
from sklearn.decomposition import PCA

# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 20


def set_decimal(num, digits):
    num_str = str(num)
    num_len = len(num_str)

    if num_len < digits:
        if '.' in num_str:
            num_str += '0' * (digits - num_len)
        else:
            num_str += ('.' + '0' * (digits - num_len))
    elif num_len > digits:
        if '.' in num_str[digits:] or '.' not in num_str:
            num_str = num_str[0] + '.' + num_str[1:(digits - 4 + 1)] + 'e' + str(int(np.log10(num)))
        else:
            num_str = num_str[:digits]

    return num_str
            
        
    
def cate2indices(category_labels):
    """
    """    
    label_dat = pd.DataFrame({'category': category_labels}, dtype = 'category')
    label_dat['cate_encoding'] =  label_dat['category'].cat.codes
    return label_dat

def decode_csv(input_path):
    new_dat = pd.read_csv(input_path, sep = ",", header = None)
    new_dat[0] = new_dat[0].str.split("#", expand = True)[1]
    return np.array(new_dat.values, dtype = "float")

def BashProcessData(FLAGS, process_type):
    data_proc_sh = FLAGS["data_proc_sh"]
    file_dir = FLAGS["file_dir"]
    input_name = file_dir + "/" + FLAGS["input_name"]        
    output_name = file_dir + "/" + FLAGS["output_name"]
    subprocess.call(["chmod", "777", data_proc_sh])
    subprocess.call([data_proc_sh, input_name, output_name, process_type])
    

def ExtractSharedData(dat1_path, dat2_path, output_path):
    """
    Extract the data that appears in both dat1 and dat2
    """

    dat1_df = pd.read_csv(dat1_path, sep = "\t", header = None, skiprows = 1, names = ["id", "data1"])
    dat1_df["id"] = dat1_df["id"].astype(str)    
    dat2_df = pd.read_csv(dat2_path, sep = "#", header = None, names = ["id", "data2"], usecols = ["id"])
    dat2_df["id"] = dat2_df["id"].astype(str)    
    dat_shared_df = dat1_df.join(dat2_df.set_index("id"), on = "id", how = 'inner')
    dat_shared_df = pd.concat([pd.DataFrame({"id": [dat_shared_df.shape[0]], "data1": [None]}), dat_shared_df],\
                               ignore_index = True,\
                               axis = 0)
    dat_shared_df.to_csv(output_path, sep = "\t", columns = ["id", "data1"], header = False, index = False)

def RetrieveItemByTrigger(embedding_id, trigger_df, embedding_data, n_batch, retrieve_upper_limit):
    """
    """    
    n_usrs = trigger_df.shape[0]
    item_bucket_size, n_dim = embedding_data.shape

    
    retrieved_item_per_usr = {usr_id: ['', (0, 0)] for usr_id in trigger_df["usr_id"].values}    
    embedding_id_dict = {embedding_id[i]: i for i in range(item_bucket_size)}

    if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0:
        # setup gpu
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(n_dim)
        faiss_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        faiss_index.add(np.ascontiguousarray(embedding_data))
    else:
        # use cpu
        faiss_index = faiss.IndexFlatL2(n_dim)
        faiss_index.add(np.ascontiguousarray(embedding_data))  
    
    start_time = time.time()
    for batch_id in range(int(np.ceil(n_usrs/n_batch))):

        count_list = []
        batch_emb_id = []
        for idx in range((n_batch * batch_id), min((batch_id + 1) * n_batch, n_usrs)):
            loc_cand_item_id = trigger_df["trigger_item"].iloc[idx].split(",")
            batch_item_id = [emb_id for emb_id in loc_cand_item_id if emb_id in embedding_id_dict]
            count_list.append(len(batch_item_id))
            batch_emb_id.append(batch_item_id)
            
        K = int(np.ceil(retrieve_upper_limit/max(min(count_list), 6))) # at least 6 items are recalled for each trigger (including itself)
        batch_emb_id = list(itertools.chain.from_iterable(batch_emb_id))
        batch_emb = np.array([embedding_data[embedding_id_dict[emb_id]] for emb_id in batch_emb_id])
        _, I = faiss_index.search(batch_emb, K)
        count_cumsum_list = np.cumsum([0] + count_list).tolist()
    
        for idx in range(len(count_list)):
            itm_idx_mat = I[count_cumsum_list[idx]:count_cumsum_list[idx + 1]]
            itm_idx_mat = np.transpose(itm_idx_mat[:,:min(itm_idx_mat.shape[1], int(np.ceil(retrieve_upper_limit/max(count_list[idx], 1))))])
            usr_emb_id_shape = itm_idx_mat.shape
            usr_emb_id_list = ",".join(embedding_id[itm_idx_mat.flatten()].tolist())
            retrieved_item_per_usr[trigger_df["usr_id"].values[n_batch * batch_id + idx]] = [usr_emb_id_list, usr_emb_id_shape]

        if batch_id % 10 == 0:
            print("Finish batch {batch_id} in the retrieving job. Elapsed time: {elapsed_time}s.".format(batch_id = batch_id, elapsed_time = time.time() - start_time))
            

    return retrieved_item_per_usr                                                                        
    
def normalize_e1(dataset):
    dataset = np.array(dataset)
    n_dim = dataset.shape[1]

    e1 = np.zeros(n_dim)
    e1[0] = 1
    
    return dataset - np.mean(dataset, axis = 0) + e1

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


def aggregate_data(item_file, kg_file, output_file, aggregation_method = "concat"):
    """
    """
    
    if aggregation_method == "concat":
        item_df = pd.read_csv(item_file, sep = "#", header = None, names = ["entity_id", "item_embedding"])
        kg_df = pd.read_csv(kg_file, sep = "#", header = None, names = ["entity_id", "kg_embedding"])
        full_df = item_df.join(kg_df.set_index("entity_id"), on = "entity_id", how = "inner")
        full_df["final_embedding"] = full_df.apply(lambda x: x.item_embedding + "," + x.kg_embedding, axis  = 1)
        final_df = full_df
        

    elif aggregation_method == "pca":
        ent_id = pd.read_csv(item_file, sep = "#", header = None, names = ["entity_id", "item_embedding"])["entity_id"].values
        item_dat = decode_csv(item_file)
        kg_dat = decode_csv(kg_file)

        pca = PCA(n_components = min(item_dat.shape[1], kg_dat.shape[1]))
        merge_dat = list(pca.fit_transform(item_dat) + pca.fit_transform(kg_dat))
        
        embeddings_str = [''] * len(merge_dat)
        for i in range(len(merge_dat)):
            embeddings_str[i] = ','.join(np.array(merge_dat[i]).astype(str))

        ent_emb_df = pd.DataFrame({"entity_id": ent_id, "final_embedding": embeddings_str})
        final_df = ent_emb_df
    final_df.to_csv(output_file, sep = "#", columns = ["entity_id", "final_embedding"], header = False, index = False)
        
        


def get_embedding_from_json(json_path, entity2id_path, embedding_type, output_file):
    """
    """

    # load json
    load_dict = json.load(open(json_path, 'r'))
    embeddings = load_dict[embedding_type]

    embeddings_str = [''] * len(embeddings)
    for i in range(len(embeddings)):
        embeddings_str[i] = ','.join(np.array(embeddings[i]).astype(str))

    # load entity2id
    ent_id = pd.read_csv(entity2id_path, sep = '\t', header = None, skiprows = 1, names = ["ent_id", "no_id"])["ent_id"].values
    
    ent_emb_df = pd.DataFrame({"entity_id": ent_id, "embeddings": embeddings_str})
    ent_emb_df.to_csv(output_file, sep = "#", columns = ["entity_id", "embeddings"], header = False, index = False)
    

def update_json_ent_embedding(ent_embedding_path, json_path, output_json_path,entity2id_path):
    """
    """

    # entity2id
    # file_raw = open(entity2id_path, 'r')

    # ent_dict={}
    # for line in file_raw:
    #     content=line.strip()
    #     content=content.split('\t')
    #     ent_dict[content[0].strip()]=int(content[1].strip())
    # file_raw.close()


    
    # entity2id
    ent_id_df = pd.read_csv(entity2id_path, sep = '\t', header = None, skiprows = 1, names = ["ent_id", "no_id"])

    n_ent = ent_id_df.shape[0]
    n_ent_100th = int(n_ent/100)
    
    # json
    load_dict = json.load(open(json_path, 'r'))
    embedding_list=load_dict['ent_embeddings']

    # ent_embedding_to_update
    ent_df = pd.read_csv(ent_embedding_path, sep = "#", header = None, names = ["ent_id", "embedding"])

    for _, row in ent_id_df.iterrows():
        ent_id, no_id = row["ent_id"], int(row["no_id"])
        if ent_id in ent_df["ent_id"].values:
            if no_id % n_ent_100th < 5:
                print (ent_id, no_id)
            embedding_list[no_id]=np.array(ent_df.loc[ent_df["ent_id"] == ent_id, "embedding"].values[0].split(",")).astype(float).tolist()

    load_dict['ent_embeddings'] = embedding_list

    with open(output_json_path, 'w') as outfile:        
        json.dump(load_dict, outfile)
    
def update_json_rel_embedding(json_path, train2id_path, relation2id_path):
    """
    """
    
    # relation2id
    # file_raw = open(relation2id_path, 'r')
    # _ = file_raw.readline()
    # rel_list=[]
    # for line in file_raw:
    #     content=line.strip()
    #     content=content.split('\t')
    #     rel_list.append(int(content[1].strip())) # 0: rel_id; 1: no_id
    # file_raw.close()
    
    # # train2id
    # train_df = pd.read_csv(train2id_path, sep = ' ', header = None, names = ["head", "tail", "relation"])    
    
    
    # json
    load_dict = json.load(open(json_path, 'r'))
    #ent_embedding_list=np.array(load_dict['ent_embeddings'])
    rel_embedding_list=load_dict['rel_embeddings']

    for i in range(len(rel_embedding_list)):
        tmp_embedding = np.array(rel_embedding_list[i])
        rel_embedding_list[i] = list(tmp_embedding * 1.0/np.linalg.norm(tmp_embedding))
        

    # estimate rel_embeddings
    # for no_rel_id in rel_list:
    #     head_set = np.array(train_df.loc[train_df["relation"] == no_rel_id, "head"].values).astype(int)
    #     tail_set = np.array(train_df.loc[train_df["relation"] == no_rel_id, "tail"].values).astype(int)

    #     if len(head_set) > 0:           
    #         rel_embedding_list[no_rel_id] = list(-np.mean(ent_embedding_list[head_set], axis = 0) + np.mean(ent_embedding_list[tail_set], axis = 0))

            
    load_dict['rel_embeddings'] = rel_embedding_list
    
    with open(json_path, 'w') as outfile:        
        json.dump(load_dict, outfile)
    
    
def SplitData(dataset1_path = None, dataset2_path = None, join_path = None, split_ratio = 1, Decare_rounds = 10, to_normalize=True, to_split=False):
    """
    """

    dat1_df = pd.read_csv(dataset1_path, delimiter = "#", header = None, names = ["item_id", "data1"])
    dat1_df["item_id"] = dat1_df["item_id"].astype(str)
    #dat1_df = dat1_df.drop_duplicates(subset = ["item_id"], keep = 'first')
    print("Finish loading dat1.")
    dat2_df = pd.read_csv(dataset2_path, delimiter = "#", header = None, names = ["item_id", "data2"])
    dat2_df["item_id"] = dat2_df["item_id"].astype(str)
    #dat2_df = dat2_df.drop_duplicates(subset = ["item_id"], keep = 'first')
    print("Finish loading dat2.")
    data_df = dat1_df.join(dat2_df.set_index("item_id"), on = "item_id", how = 'inner')    
    print("Finish joining dat1 and dat2.")
    
    data_df.to_csv(join_path + "_id.txt", sep = "#", columns = ["item_id"], header = False, index = False)
    data_df.to_csv(join_path + "_raw_dat1.csv", sep = "#", columns = ["item_id", "data1"], header = False, index = False)
    data_df.to_csv(join_path + "_raw_dat2.csv", sep = "#", columns = ["item_id", "data2"], header = False, index = False)
    print("Finish rewriting the data.")
   
    if to_split:         
        del dat1_df, dat2_df, data_df
        dataset1 = decode_csv(join_path + "_raw_dat1.csv")
        dataset2 = decode_csv(join_path + "_raw_dat2.csv")
        print("Finish converting data strings to data arrays.")
        
        n_dat1 = dataset1.shape[1]
        n_dat2 = dataset2.shape[1]
        if to_normalize:
            print("Start normalizing the data.")
            dataset1 = dataset1/np.linalg.norm(dataset1, axis = 1, keepdims = True)
            dataset2 = dataset2/np.linalg.norm(dataset2, axis = 1, keepdims = True)
    else:
        #dataset1 = np.array(data_df["data1"].str.split(",", expand = True).values, dtype = "float")
        #dataset2 = np.array(data_df["data2"].str.split(",", expand = True).values, dtype = "float")
        dataset1 = data_df["data1"].values
        dataset2 = data_df["data2"].values                
        
    n_sample = dataset1.shape[0]

    # permutation
    perm = np.arange(n_sample)
    np.random.shuffle(perm)
    dataset1 = dataset1[perm]
    dataset2 = dataset2[perm]
    
    # make training data and testing data        
    n_train = int(split_ratio * n_sample)
    n_test = n_sample - n_train

    if to_split:
        train1 = np.array(dataset1[:n_train]).reshape(n_train, n_dat1)
        test1 = np.array(dataset1[n_train:]).reshape(n_test, n_dat1)
        train2 = np.array(dataset2[:n_train]).reshape(n_train, n_dat2)
        test2 = np.array(dataset2[n_train:]).reshape(n_test, n_dat2)
    else:
        train1 = np.array(dataset1[:n_train])
        test1 = np.array(dataset1[n_train:])
        train2 = np.array(dataset2[:n_train])
        test2 = np.array(dataset2[n_train:])
    print("Finish creating the training and testing sets.")
    
    return DataSet(train1, train2, Decare_rounds, to_split), DataSet(test1, test2, Decare_rounds, to_split)



class DataSet(object):
    def __init__(self, dataset1, dataset2, Decare_rounds, is_splitted):
        # mean of dataset1 and dataset2
        #self.mean1 = np.mean(dataset1, 0)
        #self.mean2 = np.mean(dataset2, 0)
        
        
        # permutation        
        #self.dataset1 = normalize_e1(dataset1)
        #self.dataset2 = normalize_e1(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
        self.Decare_rounds = Decare_rounds
        self.is_splitted = is_splitted
        self.num_examples = dataset1.shape[0]
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.index_in_Decare = 0

        if self.is_splitted:
            self.n_dat1 = dataset1.shape[1]
            self.n_dat2 = dataset2.shape[1]
        else:
            self.n_dat1 = len(dataset1[0].split(','))
            self.n_dat2 = len(dataset2[0].split(','))            

                      
    def next_batch(self, batch_size):
        """Return the next `batch_size` samples from this data set."""

        if self.index_in_Decare < self.Decare_rounds:            
            self.index_in_Decare += 1
        else:
            self.index_in_Decare = 1
            self.index_in_epoch += batch_size
            if self.index_in_epoch + batch_size > self.num_examples:
                # Finished epoch
                self.epochs_completed += 1
                # Shuffle the data
                perm = np.arange(self.num_examples)
                np.random.shuffle(perm)
                self.dataset1 = self.dataset1[perm]
                self.dataset2 = self.dataset2[perm]
                # Start next epoch
                self.index_in_epoch = 0
                assert batch_size <= self.num_examples

        self.start = self.index_in_epoch
        self.end = self.index_in_epoch + batch_size

        perm1, perm2 = np.array([]), np.array([])
        while len(perm1) < batch_size:
            tmp_perm1, tmp_perm2 = np.random.choice(np.arange(self.start, self.end), batch_size, True), np.random.choice(np.arange(self.start, self.end), batch_size, True)
                
            idx = np.array([i for i in range(batch_size) if tmp_perm1[i] != tmp_perm2[i]])
        
            tmp_perm1, tmp_perm2 = tmp_perm1[idx], tmp_perm2[idx]
            perm1, perm2 = np.concatenate([perm1, tmp_perm1], 0), np.concatenate([perm2, tmp_perm2], 0)
            perm_pair = np.transpose(np.unique(np.transpose(np.concatenate([[perm1], [perm2]], 0)), axis = 0))
            perm1, perm2 = perm_pair[0], perm_pair[1]
        final_idx = np.random.choice(len(perm1), batch_size, False)
        perm1, perm2 = perm1[final_idx], perm2[final_idx]
        perm1, perm2 = perm1.astype(int), perm2.astype(int)

        if self.is_splitted:
            batch_dict = {"z1": self.dataset1[perm1], "z2": self.dataset1[perm2], "h1": self.dataset2[perm1], "h2": self.dataset2[perm2]}
        else:
            z1 = np.array(pd.Series(self.dataset1[perm1]).str.split(",", expand = True).values, dtype = "float")
            z2 = np.array(pd.Series(self.dataset1[perm2]).str.split(",", expand = True).values, dtype = "float")
            h1 = np.array(pd.Series(self.dataset2[perm1]).str.split(",", expand = True).values, dtype = "float")
            h2 = np.array(pd.Series(self.dataset2[perm2]).str.split(",", expand = True).values, dtype = "float")
            batch_dict = {"z1": z1, "z2": z2, "h1": h1, "h2": h2}

        return batch_dict
    
class EvalDataSet(object):
    def __init__(self, data_path, signal_path, n_fold, supervision_type):
        data_df = pd.read_csv(data_path, delimiter = "#", header = None, names = ["item_id", "data"])
        signal_df = pd.read_csv(signal_path, delimiter = "#", header = None, names = ["item_id", "signal"])
        data_df = data_df.join(signal_df.set_index("item_id"), on = "item_id", how = 'inner')
        
        self.data = np.array([row.split(",") for row in data_df["data"].values], dtype = "float")
        if supervision_type == "classification":
            self.signal_df = cate2indices(data_df["signal"].values)
            if -1 in self.signal_df["cate_encoding"].values:
                self.signal_df["cate_encoding"] = self.signal_df["cate_encoding"].values + 1
            self.signal = self.signal_df["cate_encoding"].values            
            self.signal_df = self.signal_df.drop_duplicates("category")
        elif supervision_type == "regression":
            self.signal_df = None
            self.signal = data_df["signal"].values

        self.num_samples = self.data.shape[0]
        self.n_feature = self.data.shape[1]
        self.n_signal = len(np.unique(self.signal)) if supervision_type == "classification" else 1
        
        # training and testing size
        self.n_fold = n_fold
        self.fold_size = int(np.ceil(self.num_samples/n_fold))
        self.test_fold = -1

        # batch 
        self.epochs_completed = 0
        self.index_in_epoch = 0        


    def next_fold(self):        
        self.test_fold = (self.test_fold + 1) % self.n_fold
                
        if self.test_fold == 0:
            # permutate
            perm = np.arange(self.num_samples)
            np.random.shuffle(perm)           
            self.data = self.data[perm]
            self.signal = self.signal[perm]
            
        self.test_id_set = np.arange(self.test_fold * self.fold_size, np.min([(self.test_fold + 1) * self.fold_size, self.num_samples]))
        self.train_id_set = np.array(list(set(np.arange(self.num_samples)) - set(self.test_id_set)))

        
        self.num_train = len(self.train_id_set)
        self.num_test = len(self.test_id_set)


    def next_batch(self, batch_size):
        assert batch_size <= self.num_train
        
        """Return the next `batch_size` samples from this data set."""
        if self.index_in_epoch + batch_size > self.num_train:
            # Finished epoch
            self.index_in_epoch = 0
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_train)
            np.random.shuffle(perm)
            self.train_id_set = self.train_id_set[perm]
            
        start = self.index_in_epoch
        end = start + batch_size
        self.index_in_epoch += batch_size

        batch_dict = {"features": self.data[self.train_id_set[start:end]], "signals": self.signal[self.train_id_set[start:end]]}
        
        return batch_dict

    def testdata(self):
        batch_dict = {"features": self.data[self.test_id_set], "signals": self.signal[self.test_id_set]}
        
        return batch_dict
        
  
        
        
      

        
        
        
        
