#!/usr/bin/env bash

algorithm=$1
identifier=$2
game=$3
resume=$4
aux="${@:5}"

loc=`dirname "%0"`

tensor=""

if [ $game = "active" ]; then
    tensor="--no-tensorboard"
fi

if [ $game = "generate" ]; then
    tensor="--no-tensorboard"
fi

args="--algorithm=$algorithm"


CUDA_VISIBLE_DEVICES=0, python $loc/main.py --clean --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=0, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=0, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &