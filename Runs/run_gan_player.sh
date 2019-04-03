#!/usr/bin/env bash

identifier=$1
game=$2
resume=$3
aux="${@:4}"

loc=`dirname "%0"`

tensor=""

if [ $game = "active" ]; then
    tensor="--no-tensorboard"
fi

if [ $game = "generate" ]; then
    tensor="--no-tensorboard"
fi


args="--algorithm=gan --n-steps=1 --batch=50 --cpu-workers=1 --update-target-interval=5 --n-tot=10 \
--checkpoint-interval=2 --update-memory-interval=1"

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --play --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --clean --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
