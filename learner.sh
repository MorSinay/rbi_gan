#!/usr/bin/env bash

algorithm=$1
all="${@:2}"

loc=`dirname "%0"`

case "$algorithm" in
    ("gan") bash $loc/Runs/run_gan_learner.sh $all ;;
    (*) echo "$algorithm: Not Implemented" ;;
esac