#!/bin/bash
this_file=update.sh
for file in *
do
    if [[ $file != $this_file ]]&& [[ $file =~ .*\.[a-z]*$ ]]
    then
	   scp $file yyt193705@11.163.24.104:/home/yyt193705/disk/project/Bayes_Embedding/code
    fi
done    
	    