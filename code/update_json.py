import input as _input

if __name__ == "__main__":
    _input.update_json_ent_embedding("../data/data3/wiki_net_new_dat2.csv", "../data/data3/embedding.vec.json", "../data/data3/entity2id.txt")

    _input.update_json_rel_embedding("../data/data3/embedding.vec_new.json", "../data/data3/train2id.txt", "../data/data3/relation2id.txt")
