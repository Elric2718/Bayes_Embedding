#!/bin/bash

input_file=$1
output_file=$2
process_type=$3

if [ "$process_type" -eq 0 ]; then
    awk -F '#' '{ print $1 }' $input_file'.txt' > $output_file'_id.txt'
    awk -F '#' '{ print $2 }' $input_file'.txt' > $output_file'_dat1.csv'
    awk -F '#' '{ print $3 }' $input_file'.txt' > $output_file'_dat2.csv'
elif [ "$process_type" -eq 1 ]; then    
    awk 'NR==FNR{a[FNR]=$1}NR!=FNR{print a[FNR]"#"$1}' $input_file'_id.txt' $input_file'_dat1.csv' > $output_file'_dat1.csv'
    awk 'NR==FNR{a[FNR]=$1}NR!=FNR{print a[FNR]"#"$1}' $input_file'_id.txt' $input_file'_new_dat1.csv' > $output_file'_new_dat1.csv'
    awk 'NR==FNR{a[FNR]=$1}NR!=FNR{print a[FNR]"#"$1}' $input_file'_id.txt' $input_file'_dat2.csv' > $output_file'_dat2.csv'
    awk 'NR==FNR{a[FNR]=$1}NR!=FNR{print a[FNR]"#"$1}' $input_file'_id.txt' $input_file'_new_dat2.csv' > $output_file'_new_dat2.csv'
else
    echo "Unknown process type."
fi
