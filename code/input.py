import numpy as np
import subprocess
import pandas as pd

def BashProcessData(FLAGS, process_type):
    data_proc_sh = FLAGS["data_proc_sh"]
    file_dir = FLAGS["file_dir"]
    input_name = file_dir + "/" + FLAGS["input_name"]        
    output_name = file_dir + "/" + FLAGS["output_name"]
    subprocess.call(["chmod", "777", data_proc_sh])
    subprocess.call([data_proc_sh, input_name, output_name, process_type])    

def SplitData(dataset1_path = None, dataset2_path = None, split_ratio = 1, Decare_rounds = 10):
    """
    """
    
    dataset1 = np.array(pd.read_csv(dataset1_path, header = None).values)
    dataset2 = np.array(pd.read_csv(dataset2_path, header = None).values)
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


def normalize_e1(dataset):
    dataset = np.array(dataset)
    n_dim = dataset.shape[1]

    e1 = np.zeros(n_dim)
    e1[0] = 1
    
    return dataset - np.mean(dataset, axis = 0) + e1

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
    def __init__(dat_path, label_path, nfold):
        data_df = pd.read_csv(data_path, delimiter = "#", header = None, names = ["item_id", "data"])
        label_df = pd.read_csv(label_path, header = None, names = ["item_id", "label"])
        data_df.join(label_df.set_index("item_id"), on = "item_id", how = 'inner')

        
        self.data = np.array([row.split(",") for row in data_df["data"].values], dtype = "float")
        self.label = data_df["label"].values
        self.num_samples = self.data.shape[0]
        self.n_feature = self.data.shape[1]
        self.n_label = len(np.unique(self.label))
        
        # training and testing size
        self.nfold = nfold
        self.fold_size = int(np.ceil(self.num_samples/n_fold))
        self.test_fold = -1

        # batch 
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_fold(self):        
        self.test_fold = (self.test_fold + 1) % self.nfold
        if self.test_fold == 0:
            # permutate
            perm = np.random.shuffle(np.arange(self.num_samples))
            self.data = self.data[perm]
            self.label = self.label[perm]
            
        self.test_id_set = np.arange(self.test_fold * self_fold_size, np.min([(self.test_fold + 1) * self_fold_size, self.num_samples]))
        self.train_id_set = np.array(list(set(np.arange(self.num_samples)) - set(self.test_id_set)))

        self.num_train = len(self.train_id_set)
        self.num_test = len(self.test_id_set)


    def next_batch(self, batch_size):
        assert batch_size <= self.num_train
        
        """Return the next `batch_size` samples from this data set."""
        if self.index_in_epoch + batch_size > self.num_samples:
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

        batch_dict = {"features": self.data[self.train_id_set[start:end]], "labels": self.label[self.train_id_set[start:end]]}
        
        return batch_dict

    def testdata(self):
        batch_dict = {"features": self.data[self.test_id_set], "labels": self.label[self.test_id_set]}
        
        return batch_dict
        
  
        
        
      

        
        
        
        
