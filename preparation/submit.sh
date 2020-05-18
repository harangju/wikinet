#!/bin/bash

num_topics=37
index=0
while [ $counter -le num_topics]
do
  ./simulate-models.sh index
  ((counter++))
done

echo ""
