#!/usr/bin/env bash

algorithm=$1
identifier=$2
game=$3
resume=$4
aux="${@:6}"

loc=`dirname "%0"`

echo $1 $2 $3 $4 $5

args="--algorithm=$algorithm --tensorboard --load-last-model --n-tot=1"

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --evaluate-random-policy --identifier=$identifier --game=$game --resume=$resume $args $aux