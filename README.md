This project targets at using Bayesian method agglomerate multiple embeddings from different sources. The paper can be found in https://arxiv.org/abs/1908.10611 

# Prerequisites:
1. Tensorflow 1.10 or above
2. Install faiss-cpu and faiss-gpu as per http://github.com/facebookresearch/faiss/blob/master/INSTALL.md


# Commands:
## Train BEM 
``CUDA_VISIBLE_DEVICES=1 python -u main_wiki.py  -trpr 'train' -df 1 -nb 500 -kt 'wiki_TransE.txt' -bt 'pagelink_node2vec.txt' -pos 'pair'``

## Prediction (Embedding Correction)
``CUDA_VISIBLE_DEVICES=1 python -u main_wiki.py  -trpr 'pred' -df 1 -nb 500 -kt 'wiki_TransE.txt' -bt 'pagelink_node2vec.txt' -pos 'pair'``

## Evaluation on the classification task 
* the raw embedding:

``CUDA_VISIBLE_DEVICES=1 python -u main_eval_cl.py -df 1 -ron 'raw' -ioK 1 -dn 'wiki_net'``

* the new/corrected embedding:

``CUDA_VISIBLE_DEVICES=1 python -u main_eval_cl.py -df 1 -ron 'new' -ioK 1 -dn 'wiki_net'``



