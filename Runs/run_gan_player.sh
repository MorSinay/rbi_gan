#!/usr/bin/env bash

algorithm=$1
identifier=$2
game=$3
resume=$4
aux="${@:5}"

echo $1 $2 $3 $4

loc=`dirname "%0"`

tensor="--no-tensorboard"

args="--algorithm=$algorithm --beta-lr=0.001 --value-lr=0.001 --replay-memory-size=20000 --replay-updates-interval=1000"

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --clean --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux &

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=100  --save-beta&

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=2 &
CUDA_VISIBLE_DEVICES=0, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=3 &
CUDA_VISIBLE_DEVICES=0, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=4 &
CUDA_VISIBLE_DEVICES=0, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=5 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=6 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=7 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=8 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=9 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=10 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=11 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=12 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=13 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=14 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=15 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=16 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=17 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=18 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=19 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=20 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=21 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=22 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=23 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=24 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=25 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=26 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=27 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=28 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=29 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=30 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=31 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=32 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=33 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=34 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=35 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=36 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=37 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=38 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=39 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=40 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=41 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=42 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=43 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=44 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=45 &
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=46 &
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=47 &

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=51 & #--exploration-only&
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=52 & #--exploration-only&
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=53 & #--exploration-only&

CUDA_VISIBLE_DEVICES=1, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=54 & #--exploration-only&
CUDA_VISIBLE_DEVICES=2, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=55 & #--exploration-only&
CUDA_VISIBLE_DEVICES=3, python $loc/main.py --multiplay --identifier=$identifier --resume=$resume --load-last-model --game=$game $args $aux --actor-index=56 & #--exploration-only&