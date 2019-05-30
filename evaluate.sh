#!/usr/bin/env bash

algorithm=$1
identifier=$2
game=$3
resume=$4
random=$5
all="${@:1}"

loc=`dirname "%0"`

echo $1 $2 $3 $4 $5

case "$random" in
    ("rand") bash $loc/Runs/run_gan_evaluate_policy.sh $all ;;
    (*) bash $loc/Runs/run_gan_evaluate.sh $all ;;
esac