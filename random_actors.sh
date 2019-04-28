#!/usr/bin/env bash

algorithm=$1
all="${@:1}"

loc=`dirname "%0"`

echo $1 $2 $3

case "$algorithm" in
    ("action") bash $loc/Runs/run_gan_random_player.sh $all ;;
    ("policy") bash $loc/Runs/run_gan_random_player.sh $all ;;
    (*) echo "$algorithm: Not Implemented" ;;
esac