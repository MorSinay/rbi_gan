#!/usr/bin/env bash

algorithm=$1
identifier=$2
game=$3
resume=$4
aux="${@:5}"

loc=`dirname "%0"`

args="--algorithm=$algorithm --tensorboard --load-last-model"

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --evaluate --identifier=$identifier --game=$game --resume=$resume $args $aux