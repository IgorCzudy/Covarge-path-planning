#!/bin/bash

# episodes:
#   500 -> 0.99 0.98
#   1000 -> 0.99 0.995
#   1500 -> 0.995 0.997
#   2500 -> 0.997 0.999
#   4000 -> 0.999 0.998
#   6000 -> 0.999 0.998 0.9995
#   10000 -> 0.9995 0.9997 0.9993 0.999

# Define epsilon_decay values for each episodes count
declare -A epsilon_decay_map

epsilon_decay_map[500]="0.99 0.995"
epsilon_decay_map[1000]="0.99 0.995"
epsilon_decay_map[1500]="0.995 0.997"
epsilon_decay_map[2500]="0.997 0.999"
epsilon_decay_map[4000]="0.999 0.998"
epsilon_decay_map[6000]="0.999 0.998 0.9995"
epsilon_decay_map[10000]="0.9995 0.9997 0.9993 0.999"

for episodes in 500 1000 1500 2500 4000 6000 10000
do
  for epsilon_min in 0.01
  do
    for epsilon_decay in ${epsilon_decay_map[$episodes]}  # Loop over multiple values
    do
      echo "Running with epsilon_decay=$epsilon_decay, epsilon_min=$epsilon_min, episodes=$episodes"
      python TwoAgentsEnv.py --epsilon_decay "$epsilon_decay" --epsilon_min "$epsilon_min" --episodes "$episodes"
    done
  done
done

