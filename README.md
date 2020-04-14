# Prerequisites:
1. Tensorflow 1.10 or above
2. Install faiss-cpu and faiss-gpu as per <http://github.com/facebookresearch/faiss/blob/master/INSTALL.md>.

---
# Example Commands:
### Train BEM 
> ``CUDA_VISIBLE_DEVICES=1 python -u main_wiki.py  -trpr 'train' -df 1 -nb 500 -kt 'wiki_TransE.txt' -bt 'pagelink_node2vec.txt' -pos 'pair'``

### Prediction (Embedding Correction)
> ``CUDA_VISIBLE_DEVICES=1 python -u main_wiki.py  -trpr 'pred' -df 1 -nb 500 -kt 'wiki_TransE.txt' -bt 'pagelink_node2vec.txt' -pos 'pair'``

### Evaluation on the classification task 
* the raw embedding:

> ``CUDA_VISIBLE_DEVICES=1 python -u main_eval_cl.py -df 1 -ron 'raw' -ioK 1 -dn 'wiki_net'``

* the new/corrected embedding:

> ``CUDA_VISIBLE_DEVICES=1 python -u main_eval_cl.py -df 1 -ron 'new' -ioK 1 -dn 'wiki_net'``

---
# Issues
If you encounter any bugs or have any specific feature requests, please [file an issue](https://github.com/Elric2718/Bayes_Embedding/issues).

---
# Citation

	@inproceedings{ye2019bayes,
	  title={Bayes EMbedding (BEM) Refining Representation by Integrating Knowledge Graphs and Behavior-specific Networks},
	  author={Ye, Yuting and Wang, Xuwu and Yao, Jiangchao and Jia, Kunyang and Zhou, Jingren and Xiao, Yanghua and Yang, Hongxia},
	  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
	  pages={679--688},
 	  year={2019}
	}


