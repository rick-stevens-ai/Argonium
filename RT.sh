#/bin/bash

python make_v22.py $1 --recursive --type rt --output $1-RT.json --model gpt-4.1 --config argo_local.yaml --workers 10
