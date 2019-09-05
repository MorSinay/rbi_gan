#!/usr/bin/env bash

resume=$1
index=$2
aux="${@:3}"

echo $1 $2

args="--algorithm=bbo --identifier=debug --game=reinforce"

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --tensor --problem-index=$index --resume=$resume --load-last-model --budget=100 $args $aux &