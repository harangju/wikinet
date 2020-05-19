#!/bin/bash

source activate wikinet

#now=`date +"%Y%m%d_%H%M"`
#run="now='$now'"
run="now='${1}'"
run="$run; index=${2}"
run="$run; exec(open('simulate-models.py').read())"

echo "Running " $run
python -c "$run"

