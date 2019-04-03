#!/usr/bin/env bash

algorithm=$1
all="${@:2}"

loc=`dirname "%0"`

echo $1 $2 $3

case "$algorithm" in
    ("gan") bash $loc/Runs/run_gan_player.sh $all ;;
    (*) echo "$algorithm: Not Implemented" ;;
esac