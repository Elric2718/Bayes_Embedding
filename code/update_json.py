""" This file updates the .json file by transE. It needs to manually change the intput files.
"""
import input as _input
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', '-dt', type=int, default=3)
    parser.add_argument('--kg_type', '-kt', type=str, default="wiki_TransE")
    parser.add_argument('--behavior_type', '-bt', type=str, default="desc_doc2vec")
    parser.add_argument('--pair_or_single', '-pos', type=str, default="pair")
    parser.add_argument('--task', '-task', type=str, default="1,2")
    args = parser.parse_args()

    data_type = args.data_type
    kg_type = args.kg_type
    behavior_type = args.behavior_type
    pair_or_single = args.pair_or_single

    if "1" in args.task:
        _input.update_json_ent_embedding("../data/data{data_type}/{kg_type}_{behavior_type}_{pair_or_single}_new_dat2.csv".format(data_type=data_type, kg_type = kg_type, behavior_type = behavior_type, pair_or_single = pair_or_single),\
                                            "../data/data{data_type}/{kg_type}_embedding.vec.json".format(data_type=data_type, kg_type = kg_type),\
                                            "../data/data{data_type}/{kg_type}_{behavior_type}_{pair_or_single}_embedding.vec.json".format(data_type=data_type, kg_type = kg_type, behavior_type = behavior_type, pair_or_single = pair_or_single),\
                                            "../data/data{data_type}/entity2id.txt".format(data_type = data_type))
    if "2" in args.task:
        _input.update_json_rel_embedding("../data/data{data_type}/{kg_type}_{behavior_type}_{pair_or_single}_embedding.vec.json".format(data_type=data_type,kg_type=kg_type, behavior_type = behavior_type, pair_or_single = pair_or_single), "../data/data{data_type}/train2id.txt".format(data_type=data_type), "../data/data{data_type}/relation2id.txt".format(data_type=data_type))
