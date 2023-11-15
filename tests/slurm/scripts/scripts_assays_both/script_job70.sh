#!/bin/bash
        srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-20-06-py3.sif $HOME/my_env/bin/python src/data_matching/main.py detect-similarity-all \
        -i1 ./tests/datasets/ChEMBL/Joinable/assays_both_50_1_ac1_ev/assays_both_50_1_ac1_ev_source.csv \
        -i2 ./tests/datasets/ChEMBL/Joinable/assays_both_50_1_ac1_ev/assays_both_50_1_ac1_ev_source.csv \
        -nd 64 \
        -ws basic \
        -ns 100000 \
        -wl 300 \
        -ta word2vec \
        -lm CBOW \
        -w 2
        