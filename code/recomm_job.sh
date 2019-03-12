#!/bin/bash

for ron in 'raw' 'new'
do
    for act in 'buy' 'clk'
    do
	for info in 'spu_id' 'org_brand_id' 'cate_id'
	do
	    nohup python -u main_eval_recomm.py -ron $ron -at $act -task 'eval' -iif $info  &
	done
    done    
done

