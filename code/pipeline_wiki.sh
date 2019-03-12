#!/bin/bash

# A script that automatically runs Bayes Embedding, then updates the .json file from TransE, finally evaluates the embedding on link prediction and triplet classification tasks.

activation=$1
gpu_id=$2
log_file=$3
new_or_raw_type=$4
data_type=$5
kg_type=$6
behavior_type=$7
pair_or_single=$8

echo -e $activation

curr_path=$(dirname $(readlink -f "$0"))
result_file=$curr_path"/../data/data"$data_type"/"$kg_type"_"$behavior_type"_"$pair_or_single"_result.txt"
log_file=$(dirname $(readlink -f "$0"))"/../data/data"$data_type"/log/"$kg_type"_"$behavior_type"_"$pair_or_single"_log.txt"
echo -e 'Current Path: '$curr_path

# create the log file
if [ -f $log_file ];then
    touch $log_file
    touch $result_file
fi


echo -e "---------------------------------------------------------------------------------------------------------------------------\n" >> $log_file
echo -e "======================================\n\nTask:"$data_type"; "$new_or_raw_type"; "$kg_type"; "$behavior_type"; "$pair_or_single".\n\n======================================" >> $log_file

echo -e "---------------------------------------------------------------------------------------------------------------------------\n" >> $result_file
echo -e "======================================\n\nTask:"$data_type"; "$new_or_raw_type"; "$kg_type"; "$behavior_type"; "$pair_or_single".\n\n======================================" >> $result_file



# Bayesian Embedding
if  [[ $activation == *"1"* ]]; then
    echo -e "Bayesian Embedding:\n" >> $log_file
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_wiki.py -trpr 'train' -lf $log_file -dt $data_type -bt $behavior_type -kt $kg_type -pos $pair_or_single -pp "/"$kg_type$"_"$behavior_type"_"$pair_or_single -op "/"$kg_type$"_"$behavior_type"_"$pair_or_single -cp $kg_type$"_"$behavior_type"_"$pair_or_single".ckpt" -nb 500
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_wiki.py -trpr 'pred' -lf $log_file -dt $data_type -bt $behavior_type -kt $kg_type -pos $pair_or_single -pp "/"$kg_type$"_"$behavior_type"_"$pair_or_single -op "/"$kg_type$"_"$behavior_type"_"$pair_or_single -cp $kg_type$"_"$behavior_type"_"$pair_or_single".ckpt" -nb 500
fi


# update the entity embedding in json, then update the relation embedding in json
if [[ $activation == *"2"* ]]; then
    if [[ $new_or_raw_type == "new" ]]; then
	echo -e "Update the entity embedding in json.\n" >> $log_file
	python -u update_json.py -dt $data_type -kt $kg_type -bt $behavior_type -pos $pair_or_single -task "1,2"
    else
	cd "../data/data"$data_type
	cp $kg_type"_embedding.vec.json" $kg_type"_"$behavior_type"_"$pair_or_single"_embedding.vec.json"
	python -u update_json.py -dt $data_type -kt $kg_type -bt $behavior_type -pos $pair_or_single -task "2"
	cd ../../code
    fi
    cp "../data/data"$data_type"/"$kg_type"_"$behavior_type"_"$pair_or_single"_embedding.vec.json" ./OpenKE/datasets/wiki/
    cd "../data/data"$data_type
    cp "entity2shared_id.txt" "entity2shared_id"$data_type".txt"
    cd ../../code
    mv "../data/data"$data_type"/entity2shared_id"$data_type".txt" ./OpenKE/datasets/wiki/
    cd OpenKE
    if [[ $new_or_raw_type == "new" ]]; then
	CUDA_VISIBLE_DEVICES=$gpu_id python -u train_batch.py -kt $kg_type -bt $behavior_type -pos $pair_or_single -dt $data_type -lf $log_file -rfile $result_file
    else
	CUDA_VISIBLE_DEVICES=$gpu_id python -u train_batch.py -kt $kg_type -bt $behavior_type -pos $pair_or_single -dt $data_type -lf $log_file -rfile $curr_path"/../data/data"$data_type"/"$kg_type"_raw_result.txt"
    fi
    cd ..
    echo -e "Update the relation embedding in json by fine-tuning:\n" >> $log_file
fi

# test by link prediction and triplet classification
if [[ $activation == *"3"* ]]; then
    echo -e "Test the KG embedding for link prediction and triplet classification:\n" >> $log_file
    cd OpenKE
    if [[ $new_or_raw_type == "new" ]]; then
	CUDA_VISIBLE_DEVICES=$gpu_id python -u task_batch.py -kt $kg_type -bt $behavior_type -pos $pair_or_single -dt $data_type -lf $log_file -rfile $result_file
    else
	CUDA_VISIBLE_DEVICES=$gpu_id python -u task_batch.py -kt $kg_type -bt $behavior_type -pos $pair_or_single -dt $data_type -lf $log_file -rfile $curr_path"/../data/data"$data_type"/"$kg_type"_raw_result.txt"
    fi
    cd ..
fi

# test by category classification
if [[ $activation == *"4"* ]]; then
    echo -e "Test the corrected embeddings for category classification:\n"  >> $log_file    
    echo -e "========= raw; 1 ===========\n" >> $result_file
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_eval_cl.py -dt $data_type -ron 'raw' -ioK '1' -kt $kg_type -bt $behavior_type -pos $pair_or_single -rfile $result_file
    echo -e "========= new; 1 ===========\n" >> $result_file
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_eval_cl.py -dt $data_type -ron 'new' -ioK '1' -kt $kg_type -bt $behavior_type -pos $pair_or_single -rfile $result_file
    echo -e "========= raw; 2 ===========\n" >> $result_file
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_eval_cl.py -dt $data_type -ron 'raw' -ioK '2' -kt $kg_type -bt $behavior_type -pos $pair_or_single -rfile $result_file
    echo -e "========= new; 2 ===========\n" >> $result_file
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_eval_cl.py -dt $data_type -ron 'new' -ioK '2' -kt $kg_type -bt $behavior_type -pos $pair_or_single -rfile $result_file
    echo -e "========= raw; concat ===========\n" >> $result_file
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_eval_cl.py -dt $data_type -ron 'raw' -ioK '_concat' -kt $kg_type -bt $behavior_type -pos $pair_or_single -rfile $result_file
    echo -e "========= new; concat ===========\n" >> $result_file
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_eval_cl.py -dt $data_type -ron 'new' -ioK '_concat' -kt $kg_type -bt $behavior_type -pos $pair_or_single -rfile $result_file
    echo -e "========= raw; pca ===========\n" >> $result_file
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_eval_cl.py -dt $data_type -ron 'raw' -ioK '_pca' -kt $kg_type -bt $behavior_type -pos $pair_or_single -rfile $result_file
    echo -e "========= new; pca ===========\n" >> $result_file
    CUDA_VISIBLE_DEVICES=$gpu_id python -u main_eval_cl.py -dt $data_type -ron 'new' -ioK '_pca' -kt $kg_type -bt $behavior_type -pos $pair_or_single -rfile $result_file   
fi

# move all the generated files to a folder
if [[ $activation == *"5"* ]]; then
    cd "../data/data"$data_type
    if [ ! -e $kg_type'_'$behavior_type'_'$pair_or_single ]; then
	mkdir $kg_type'_'$behavior_type'_'$pair_or_single
    fi
    mv $kg_type'_'$behavior_type'_'$pair_or_single* $kg_type'_'$behavior_type'_'$pair_or_single"/"
    cd ../../code
fi
