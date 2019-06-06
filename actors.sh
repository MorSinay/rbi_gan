#!/usr/bin/env bash

algorithm=$1
all="${@:1}"

loc=`dirname "%0"`

case "$algorithm" in
    ("rbi") bash $loc/Runs/run_gan_player.sh $all ;;
    ("ddpg") bash $loc/Runs/run_gan_player.sh $all ;;
    (*) echo "$algorithm: Not Implemented" ;;
esac