#/bin/bash

input=$1.json
output=$1-Score.json

python argonium_score_parallel_v9.py $input --model gpt-4.1 --config argo_local.yaml --parallel 10 --grader gpt-4.1 --output $output --save-incorrect --incorrect-output $1-Score-incorrect.json
