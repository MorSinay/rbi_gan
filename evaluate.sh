#!/usr/bin/env bash

algorithm=$1
identifier=$2
game=$3
resume=$4
all="${@:1}"

loc=`dirname "%0"`

echo $1 $2 $3 $4

bash $loc/Runs/run_gan_evaluate.sh $all ;
