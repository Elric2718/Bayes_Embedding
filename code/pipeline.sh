#!/bin/bash

activation=$1
log_file=$2
echo -e $activation

# create the log file
if [ ! -f $log_file ];then
    touch $log_file
fi


for dr in 10
do
    # write the parameters into the index file
    para_index="Decare_rounds="$dr
    echo -e "---------------------------------------------------------------------------------------------------------------------------\n" >> $log_file
    echo -e "\n\nPipeline parameters: "$para_index"\n\n" >> $log_file

    # Bayesian Embedding
    if [[ $activation == *"1"* ]]; then
	echo -e "Bayesian Embedding:\n" >> $log_file
	CUDA_VISIBLE_DEVICES=0,1,3 python main_wiki.py -trpr 'train' -lf $log_file -dr $dr
	CUDA_VISIBLE_DEVICES=0,1,3 python main_wiki.py -trpr 'pred'
    fi


    # update the entity embedding in json
    if [[ $activation == *"2"* ]]; then
	echo -e "Update the entity embedding in json.\n" >> $log_file
	python update_json.py
	cp ../data/data3/embedding.vec_new.json ./OpenKE/datasets/wiki/
    fi


    # update the relation embedding in json
    if [[ $activation == *"3"* ]]; then
	echo -e "Update the relation embedding in json by fine-tuning:\n" >> $log_file
	cd OpenKE
	CUDA_VISIBLE_DEVICES=0,1,3 python train.py
	cd ..
    fi


    # test by link prediction and triplet classification
    if [[ $activation == *"4"* ]]; then
	echo -e "Test the KG embedding for link prediction and triplet classification:\n"  >> $log_file
	cd OpenKE
	CUDA_VISIBLE_DEVICES=0,1,3 python task.py
	cd ..
    fi
done
