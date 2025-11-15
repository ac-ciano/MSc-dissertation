#!/bin/bash

source ~/miniconda3/bin/activate

conda activate mlp

python finetune_roberta_softf1.py

python evaluate_roberta_softf1.py