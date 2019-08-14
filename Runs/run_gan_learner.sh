#!/usr/bin/env bash

algorithm=$1
identifier=$2
game=$3
resume=$4
aux="${@:5}"

loc=`dirname "%0"`

if [ $resume != "new" ]; then
    resume="--resume=$resume --load-last-model"
    resume2="--resume=$resume --load-last-model"
    echo "Resume Experiment: $identifier $resume"
else
    resume=""
    resume2="--resume=-1 --load-last-model"
    echo "New Experiment"
fi

args="--algorithm=$algorithm --env-iterations=938 --beta-lr=0.001 --value-lr=0.001 --replay-memory-size=20000 --replay-updates-interval=1000 --beta-init=label"

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --learn --identifier=$identifier --game=$game $resume $args $aux