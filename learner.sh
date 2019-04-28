#!/usr/bin/env bash

algorithm=$1
all="${@:1}"

loc=`dirname "%0"`

case "$algorithm" in
    ("action") bash $loc/Runs/run_gan_learner.sh $all ;;
    ("policy") bash $loc/Runs/run_gan_learner.sh $all ;;
    (*) echo "$algorithm: Not Implemented" ;;
esac