#!/usr/bin/env bash

loc=`dirname "%0"`

args="--n-steps=2"

CUDA_VISIBLE_DEVICES=2, python $loc/main.py --play --algorithm=action --identifier=debug --load-last-model --game=active --cuda-default=0 --n-actors=1 --actor-index=0 $args  &