import numpy as np
import subprocess
import pandas as pd

def SplitData(FLAGS = None, dataset1 = None, dataset2 = None, split_ratio = 1, Decare_rounds = 10):
    assert not all([FLAGS is None, dataset1 is None, dataset2 is None])
    if FLAGS is not None:
        data_proc_sh = FLAGS["data_proc_sh"]
        file_dir = FLAGS["file_dir"]
        input_name = file_dir + "/" + FLAGS["input_name"]        
        output_name = file_dir + "/" + FLAGS["output_name"]
        subprocess.call(["chmod", "777", data_proc_sh])
        subprocess.call([data_proc_sh, input_name, output_name])

        dataset1 = pd.read_csv(output_name + "_dat1.csv", header = None).values
        dataset2 = pd.read_csv(output_name + "_dat2.csv", header = None).values

    dataset1 = np.array(dataset1)
    dataset2 = np.array(dataset2)        
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
    
