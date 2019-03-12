#!/bin/bash

# A script that automatically runs Bayes Embedding, then updates the .json file from TransE, finally evaluates the embedding on link prediction and triplet classification tasks.


activation=$1
n_gpu=$2
log_file=$3
echo -e $activation

# create the log file
if [ ! -f $log_file ];then
    touch $log_file
fi


# Bayesian Embedding
if [[ $activation == *"1"* ]]; then
    echo -e "Bayesian Embedding:\n" >> $log_file
    CUDA_VISIBLE_DEVICES=$n_gpu python -u main_ali.py -trpr 'train' -nb 1800 -nfK 0 -nepoch 20 -dr 30 -split -lf $log_file 
    CUDA_VISIBLE_DEVICES='' python -u main_ali.py -trpr 'pred' -nb 1800 -nfK 0 -nepoch 20 -dr 30 -split 
fi

# retrieve items by trigger
if [[ $activation == *"2"* ]]; then
    echo -e "retrieve items by trigger:\n" >> $log_file
    CUDA_VISIBLE_DEVICES=$n_gpu python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'buy' -task 'retrieve' 
    CUDA_VISIBLE_DEVICES=$n_gpu python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'buy' -task 'retrieve' 
    CUDA_VISIBLE_DEVICES=$n_gpu python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'clk' -task 'retrieve' 
    CUDA_VISIBLE_DEVICES=$n_gpu python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'clk' -task 'retrieve' 
fi

# evaluation
if [[ $activation == *"3"* ]]; then
    echo -e "evaluation:\n" >> $log_file
    echo "\n-------------------\n-------------------\n" >> "../data/data4/log/log_recomm.txt"
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'buy' -task 'eval' -iif 'item_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'buy' -task 'eval' -iif 'item_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'clk' -task 'eval' -iif 'item_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'clk' -task 'eval' -iif 'item_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'buy' -task 'eval' -iif 'spu_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'buy' -task 'eval' -iif 'spu_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'clk' -task 'eval' -iif 'spu_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'clk' -task 'eval' -iif 'spu_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'buy' -task 'eval' -iif 'cate_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'buy' -task 'eval' -iif 'cate_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'clk' -task 'eval' -iif 'cate_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'clk' -task 'eval' -iif 'cate_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'buy' -task 'eval' -iif 'org_brand_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'buy' -task 'eval' -iif 'org_brand_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'raw' -dn 'ali_selected' -at 'clk' -task 'eval' -iif 'org_brand_id' 
    CUDA_VISIBLE_DEVICES='' python -u main_eval_recomm.py -ron 'new' -dn 'ali_selected' -at 'clk' -task 'eval' -iif 'org_brand_id'  
fi
