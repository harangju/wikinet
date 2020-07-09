#!/bin/bash

now=`date +"%Y%m%d_%H%M"`
num_topics="38"
for (( i=0; i<$num_topics; i++ ))
#for i in 0 1 2
do
  qsub detect-communities.sh $now $i
  echo "Submitted job" $i "at" $now
done
