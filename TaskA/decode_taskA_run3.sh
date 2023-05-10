#!/bin/bash

source activate.sh
python run_3.py --train_csv_file $1 --valid_csv_file $2 --test_csv_file $3 --output_csv_file taskA_StellEllaStars_run3_mediqaSum.csv

