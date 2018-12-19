#!/bin/bash

input_file=$1
output_file=$2

awk -F '#' '{ print $1 }' $input_file > $output_file'_id.txt'
awk -F '#' '{ print $2 }' $input_file > $output_file'_dat1.csv'
awk -F '#' '{ print $3 }' $input_file > $output_file'_dat2.csv'
