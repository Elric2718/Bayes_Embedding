"""
Attach ids to embeddings output by a method w/ or w/o bem 
"""

import numpy as np
import pandas as pd
import json
import argparse


# outer_dir = "unsup-pagelink_data"
# innder_dir = "bem_n2v_small_0.000010"
# output_name = "val_mid_dat.csv"
parser = argparse.ArgumentParser()
parser.add_argument('--outer_dir', '-odir', type = str, default = '.')
parser.add_argument('--inner_dir', '-idir', type = str, default = '.')
parser.add_argument('--output_name', '-oname', type = str, default = 'val_mid_dat.csv')
parser.add_argument('--task_type', '-tt', type = str, default = 'concat_mid_emb')
args = parser.parse_args()

outer_dir = "./" + args.outer_dir + "/"
inner_dir = args.inner_dir + "/"
relative_dir = outer_dir + inner_dir
output_path = relative_dir + args.output_name
task_type = args.task_type

entity_id = pd.read_csv(outer_dir + "entity2type_20181229.txt", sep = "#", names = ["mid", "label"], usecols = ["mid"])
shared_id = pd.read_csv(outer_dir + "entity2shared_id.txt", sep = "\t", names = ["mid", "no_id"], usecols = ["mid", "no_id"], skiprows = 1)
ref_id = entity_id.join(shared_id.set_index('mid'), on = 'mid', how = 'left').dropna().loc[:, ['mid']]




if task_type == 'concat_mid_emb':
    # read in entitiey id
    ent_id = np.squeeze(pd.read_csv(relative_dir + "val.txt", sep = " ", header = None).values)

    # read in reference id
    # ref_id = pd.read_csv(relative_dir + "entity2type_20181229.txt", sep = "#", names = ["mid", "label"], usecols = ["mid"])
    ent_mid = ref_id["mid"].values[ent_id]

    # read in embeddings
    ent_emb = pd.DataFrame(np.transpose(np.load(relative_dir + "val.npy"))).astype(str).apply(','.join).values

    # save the data
    pd.DataFrame({"item_id": ent_mid, "data": ent_emb}).to_csv(output_path, sep = "#", columns = ["item_id", "data"], header = False, index = False)
elif task_type == "extract_kg":
    # get the ordering of the kg
    # read in reference id
    #ref_id = np.squeeze(pd.read_csv(relative_dir + "entity2type_20181229.txt", sep = "#", names = ["mid", "label"], usecols = ["mid"]).values)
    #ref_id = np.squeeze(ref_id.values)
    
    # read in kg embeddings
    kg_emb = pd.read_csv(relative_dir + "kg_wiki_TransE.txt", sep = "#", header = None, names = ["mid", "data"])
    short_kg_emb = ref_id.join(kg_emb.set_index('mid'), on = 'mid', how = 'left')
    
    # dim_emb = len(kg_emb["data"][0].split(','))
    # short_kg_emb = np.array([kg_emb["data"].values[np.where(kg_emb["mid"].values == mid)[0][0]] if mid in kg_emb["mid"].values else ','.join(list(np.random.normal(size = dim_emb).astype(str))) for mid in ref_id])
    # kg_extract_idx = np.squeeze(np.array([np.where(kg_emb["mid"].values == mid)[0][0] if mid in kg_emb["mid"].values else -1 for mid in ref_id]))
    # short_kg_emb = pd.DataFrame({"mid": ref_id, "data": short_kg_emb})
    short_kg_emb.to_csv(output_path, sep = "#", columns = ["mid", "data"], header = False, index = False)
   
    
    
    
    
