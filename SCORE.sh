#/bin/bash

input=$1.json
output=$1-Score-llama70.json

python argonium_score_parallel_v9.py $input --model llama70 --config model_servers.yaml --parallel 10 --grader qwen3-235b-fp8 --output $output --save-incorrect --incorrect-output $1-Score-incorrect-llama70.json
