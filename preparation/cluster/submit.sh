#!/bin/bash

now=`date +"%Y%m%d_%H%M"`
num_topics="12" #37
i="11"
while [ $i -lt $num_topics ]
do
  qsub simulate-models.sh $now $i
  echo "Submitted job" $i "at" $now
  i=$[$i+1]
done

