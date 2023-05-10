#!/bin/bash

source activate.sh
python3 run_1.py --train_csv_file $1 --valid_csv_file $2 --test_csv_file $3 --output_csv_file taskA_StellEllaStars_run1_mediqaSum.csv

