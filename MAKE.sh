#!/bin/bash
input=$1
output=$input-MC.json

python make_v22.py $input --type mc --output $output --model gpt-4.1 --config argo_local.yaml --recursive --workers 10

python cleanup_mc.py $output
