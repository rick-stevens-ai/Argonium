#!/bin/bash
input=$1
model=$2
output=$input-MC.json

python make_v22.py $input --type mc --output $output --model $model --config model_servers.yaml --recursive --workers 10 --error-threshold 1800

python cleanup_mc.py $output
