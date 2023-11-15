#!/bin/bash
        srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-20-06-py3.sif $HOME/my_env/bin/python src/data_matching/main.py detect-similarity-all \
        -i1 ./tests/datasets/TPC-DI/Joinable/prospect_both_50_1_ac1_ev/prospect_both_50_1_ac1_ev_source.csv \
        -i2 ./tests/datasets/TPC-DI/Joinable/prospect_both_50_1_ac1_ev/prospect_both_50_1_ac1_ev_source.csv \
        -nd 64 \
        -ws deepwalk \
        -ns 9000 \
        -wl 200 \
        -ta word2vec \
        -lm skipgram \
        -w 2
        