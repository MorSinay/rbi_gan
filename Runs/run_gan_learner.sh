#!/usr/bin/env bash

identifier=$1
game=$2
resume=$3
aux="${@:4}"

loc=`dirname "%0"`

if [ $resume != "new" ]; then
    resume="--resume=$3 --load-last-model"
    resume2="--resume=$3 --load-last-model"
    echo "Resume Experiment: $identifier $3"
else
    resume=""
    resume2="--resume=-1 --load-last-model"
    echo "New Experiment"
fi

args="--algorithm=gan"

CUDA_VISIBLE_DEVICES=0, python $loc/main.py --learn --identifier=$identifier --game=$game $resume $args $aux