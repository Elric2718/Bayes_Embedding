import numpy as np
import pandas as pd
import argparse
import input as _input
import evaluate as _eval
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', '-dat', type=int, default=4)
    parser.add_argument('--raw_or_new', '-ron', type=str, default="raw")
    parser.add_argument('--item_or_KG', '-ioK', type=int, default=1)
    parser.add_argument('--data_name', '-dn', type=str, default="ali_selected")
    parser.add_argument('--act_type', '-at', type=str, default="buy")
    parser.add_argument('--test', '-test', action = 'store_true')
    parser.add_argument('--task', '-task', type=str, default="retrieve")
    parser.add_argument('--item_info', '-iif', type=str, default='item_id')
    parser.add_argument('--log_file', '-lf', type=str, default='/log/log_recomm.txt')
    #parser.add_argument('--to_split', '-split', action = 'store_true')
    
    args = parser.parse_args()
    data_type = args.data_type
    raw_or_new = args.raw_or_new
    item_or_KG = args.item_or_KG
    data_name = args.data_name
    task = args.task
    item_info = args.item_info
    if args.test:
        test_or_not = "_test"
    else:
        test_or_not = ''
           
    retrieve_upper_limit = 1000
    top_K = ['01', '05', '10', '30', '50']
    act_type = args.act_type
    n_retrieve_batch = 1000
    
    data_folder = "../data/data{data_type}".format(data_type = data_type)
    trigger_item_path = data_folder + "/trigger_list{test_or_not}.txt".format(test_or_not = test_or_not)
    test_item_path = data_folder + "/test_item_list{test_or_not}.txt".format(test_or_not = test_or_not)
    emb_path = data_folder + "/{data_name}_{raw_or_new}_dat{item_or_KG}.csv".format(data_name = data_name, raw_or_new = raw_or_new, item_or_KG = item_or_KG)
    item_info_path = data_folder + "/item_info.txt"
    retrieved_item_json_path = data_folder + "/retrieved_{act_type}_item_{raw_or_new}{test_or_not}.json".format(raw_or_new = raw_or_new, act_type = act_type, test_or_not = test_or_not)
    log_path = data_folder + args.log_file
    
    
    if task == "retrieve":
        trigger_df = pd.read_csv(trigger_item_path, delimiter = "#", header = None, names = ["usr_id", "trigger_item"], dtype = str)    
        print("Finish reading the trigger data.")    
        emb_data = np.array(_input.decode_csv(emb_path), dtype = np.float32)
        print("Finish reading the embedding data.")
        emb_id = pd.read_csv(emb_path, sep = "#", header = None, names = ["usr_id", "embeddings"], usecols = ["usr_id"], dtype = str).values.flatten()
        print("Finish reading the embedding id.")


        # retrieve items based on trigger items
        retrieved_item_per_user = _input.RetrieveItemByTrigger(emb_id, trigger_df, emb_data, n_batch = n_retrieve_batch, retrieve_upper_limit = retrieve_upper_limit)
        with open(retrieved_item_json_path, "w") as f:
            f.write(json.dumps(retrieved_item_per_user))
        print("Finish retrieving the items.")

    else:
        test_item_df = pd.read_csv(test_item_path, delimiter = "#", header = None, names = ["usr_id", "act_type", "test_item"], dtype = str)
        test_item_df = test_item_df.loc[test_item_df["act_type"] == act_type].reset_index(drop = True)
        print("Finish reading the testing-item data.")

        with open(retrieved_item_json_path, "r") as f:
            retrieved_item_per_user = json.loads(f.read())
        print("Finish reading the retrieved items.")


        info_df = pd.read_csv(item_info_path, delimiter = "#", header = None, names = ["item_id", "spu_id", "org_brand_id", "cate_id"], dtype = str)

        info_df = info_df[["item_id", item_info]]

        print("Finish loading the item info.")

        print("----- Parameters: {raw_or_new}; {item_or_KG}; {act_type}; {item_info} ----- \n".format(raw_or_new = raw_or_new, item_or_KG = item_or_KG, act_type = act_type, item_info = item_info))
        # check the results on the testing data
        results = _eval.measurement(retrieved_item_per_user, test_item_df, "hit-MR", top_K = top_K, info_df = info_df)
        print("Finish evaluation.")
    
        msg = "----- Parameters: {raw_or_new}; {item_or_KG}; {act_type}; {item_info} ----- \n".format(raw_or_new = raw_or_new, item_or_KG = item_or_KG, act_type = act_type, item_info = item_info) +\
        "measures |   mean  |   std   |   Q25   |   Q50   |   Q75   |   Q90   |   Q95   \n" +\
        ''.join(["  Hit@{K} | {mean} | {std} | {Q25} | {Q50} | {Q75} | {Q90} | {Q95} \n".format(K = top_K[i], mean = _input.set_decimal(results['mean'][i], 7), std = _input.set_decimal(results["std"][i], 7), Q25 = _input.set_decimal(results["Q25"][i], 7), Q50 = _input.set_decimal(results["Q50"][i], 7), Q75 = _input.set_decimal(results["Q75"][i], 7), Q90 = _input.set_decimal(results["Q90"][i], 7), Q95 = _input.set_decimal(results["Q95"][i], 7)) for i in range(len(top_K))]) +\
          "      MR | {mean} | {std} | {Q25} | {Q50} | {Q75} | {Q90} | {Q95} \n\n\n".format(K = top_K[-1], mean = _input.set_decimal(results['mean'][-1], 7), std = _input.set_decimal(results["std"][-1], 7), Q25 = _input.set_decimal(results["Q25"][-1], 7), Q50 = _input.set_decimal(results["Q50"][-1], 7), Q75 = _input.set_decimal(results["Q75"][-1], 7), Q90 = _input.set_decimal(results["Q90"][-1], 7), Q95 = _input.set_decimal(results["Q95"][-1], 7))

        #if os.path.exists(log_path):
        #    os.remove(log_path)
        with open(log_path, "a+") as f:
            f.write(msg)

