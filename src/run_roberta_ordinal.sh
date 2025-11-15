#!/bin/bash

source ~/miniconda3/bin/activate

conda activate mlp

python finetune_roberta_ordinal.py

python evaluate_roberta_ordinal.py