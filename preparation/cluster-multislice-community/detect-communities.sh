#!/bin/bash
#$ -l h_vmem=64.0G

source activate wikinet

run="now='${1}'"
run="$run; index=${2}"
run="$run; exec(open('detect-communities.py').read())"

echo "Running " $run
python -c "$run"


