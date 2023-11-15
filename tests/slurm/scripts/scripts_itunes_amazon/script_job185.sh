#!/bin/bash
        srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-20-06-py3.sif $HOME/my_env/bin/python src/data_matching/main.py detect-similarity-all \
        -i1 ./tests/datasets/Magellan/itunes_amazon/itunes_amazon_source.csv\
        -i2 ./tests/datasets/Magellan/itunes_amazon/itunes_amazon_source.csv \
        -nd 64 \
        -ws node2vec \
        -ns 9000 \
        -wl 300 \
        -ta word2vec \
        -lm skipgram \
        -w 3
        