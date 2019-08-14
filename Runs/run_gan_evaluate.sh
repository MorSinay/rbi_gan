#!/usr/bin/env bash

algorithm=$1
identifier=$2
game=$3
resume=$4
aux="${@:5}"

loc=`dirname "%0"`

args="--algorithm=$algorithm --env-iterations=938 --beta-lr=0.001 --value-lr=0.001 --replay-memory-size=20000 --replay-updates-interval=1000 --beta-init=label --tensorboard --load-last-model"

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --evaluate --identifier=$identifier --game=$game --resume=$resume $args $aux