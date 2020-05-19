#!/bin/bash

now=`date +"%Y%m%d_%H%M"`
num_topics=37
index=0
while [ $counter -le num_topics]
do
  ./simulate-models.sh $now $index
  ((counter++))
done

echo ""
