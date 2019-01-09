import numpy as np
import subprocess
import pandas as pd
import json
import decimal
import os

# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 20

def cate2indices(category_labels):
    """
    """
    
    label_dat = pd.DataFrame({'category': category_labels}, dtype = 'category')
    label_dat['cate_encoding'] =  label_dat['category'].cat.codes
    return label_dat

def decode_csv(dat):
    new_dat = [line.split(",") for line in dat]
    return np.array(new_dat).astype(float)

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

    dat1_df = pd.read_csv(dat1_path, sep = "\t", header = None, names = ["id", "data1"])
    dat2_df = pd.read_csv(dat2_path, sep = "#", header = None, names = ["id", "data2"])
    dat_shared_df = dat1_df.join(dat2_df.set_index("id"), on = "id", how = 'inner')
    dat_shared_df = pd.concat([pd.DataFrame({"id": [dat_shared_df.shape[0]], "data1": [None], "data2": [None]}), dat_shared_df],\
                               ignore_index = True,\
                               axis = 0)
    dat_shared_df.to_csv(output_path, sep = "\t", columns = ["id", "data1"], header = False, index = False)
    
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

def update_json_ent_embedding(ent_embedding_path, json_path, entity2id_path):
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
    
    # json
    load_dict = json.load(open(json_path, 'r'))
    nembedding_list=load_dict['ent_embeddings']

    # ent_embedding_to_update
    ent_df = pd.read_csv(ent_embedding_path, sep = "#", header = None, names = ["ent_id", "embedding"])



    for _, row in ent_id_df.iterrows():
        ent_id, no_id = row["ent_id"], int(row["no_id"])
        if ent_id in ent_df["ent_id"]:
            embedding_list[no_id]=np.array(ent_df.loc[ent_df["ent_id"] == ent_id, "embedding"].values[0].split(",")).astype(float).tolist()

    load_dict['ent_embeddings'] = embedding_list

    with open(json_path.replace('.json', '_new.json'), 'w') as outfile:        
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
    
    
def SplitData(dataset1_path = None, dataset2_path = None, join_path = None, split_ratio = 1, Decare_rounds = 10):
    """
    """
    dat1_df = pd.read_csv(dataset1_path, delimiter = "#", header = None, names = ["item_id", "data1"])
    dat2_df = pd.read_csv(dataset2_path, delimiter = "#", header = None, names = ["item_id", "data2"])    
    data_df = dat1_df.join(dat2_df.set_index("item_id"), on = "item_id", how = 'inner')        

    data_df.to_csv(join_path + "_id.txt", sep = "#", columns = ["item_id"], header = False, index = False)
    data_df.to_csv(join_path + "_raw_dat1.csv", sep = "#", columns = ["item_id", "data1"], header = False, index = False)
    data_df.to_csv(join_path + "_raw_dat2.csv", sep = "#", columns = ["item_id", "data2"], header = False, index = False)

    
    dataset1 = np.array([row.split(",") for row in data_df["data1"].values], dtype = "float")
    dataset2 = np.array([row.split(",") for row in data_df["data2"].values], dtype = "float")
       
    n_sample = dataset1.shape[0]
    
    # permutation
    perm = np.arange(n_sample)
    np.random.shuffle(perm)
    dataset1 = dataset1[perm]
    dataset2 = dataset2[perm]
    
    # make training data and testing data        
    n_train = int(split_ratio * n_sample)
    n_test = n_sample - n_train
    n_dat1 = dataset1.shape[1]
    n_dat2 = dataset2.shape[1]
        
    train1 = np.array(dataset1[:n_train]).reshape(n_train, n_dat1)
    test1 = np.array(dataset1[n_train:]).reshape(n_test, n_dat1)
    train2 = np.array(dataset2[:n_train]).reshape(n_train, n_dat2)
    test2 = np.array(dataset2[n_train:]).reshape(n_test, n_dat2)
        
    return DataSet(train1, train2, Decare_rounds), DataSet(test1, test2, Decare_rounds)



class DataSet(object):
    def __init__(self, dataset1, dataset2, Decare_rounds):
        # mean of dataset1 and dataset2
        #self.mean1 = np.mean(dataset1, 0)
        #self.mean2 = np.mean(dataset2, 0)
        
        
        # permutation        
        #self.dataset1 = normalize_e1(dataset1)
        #self.dataset2 = normalize_e1(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
        self.Decare_rounds = Decare_rounds
        self.num_examples = dataset1.shape[0]
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.index_in_Decare = 0
        self.n_dat1 = dataset1.shape[1]
        self.n_dat2 = dataset2.shape[1]

                      
    def next_batch(self, batch_size):
        """Return the next `batch_size` samples from this data set."""

        if self.index_in_Decare < self.Decare_rounds:            
            self.index_in_Decare += 1
        else:
            self.index_in_Decare = 0
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

        
        batch_dict = {"z1": self.dataset1[perm1], "z2": self.dataset1[perm2], "h1": self.dataset2[perm1], "h2": self.dataset2[perm2]}

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
        
  
        
        
      

        
        
        
        
