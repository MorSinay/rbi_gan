#!/usr/bin/env bash

algorithm=$1
all="${@:1}"

loc=`dirname "%0"`

bash $loc/Runs/run_gan_learner.sh $all ;