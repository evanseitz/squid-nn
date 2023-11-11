#!/bin/sh

# model a consecutive series of inter-site distances
#   change 'step' and 'inter_dist' options below,
#   then run the following in command line:
#       bash example_global_pairwise_batch.sh

pyDir="${PWD}"
pyPath="$pyDir/example_global_pairwise.py"
step=1 # 'step'

for i in {0..35}; # 'inter_dist'
do
	foo=$(printf "%02d" $i)
	echo "${foo}"
	python $pyPath "${foo}" "${step}"
done
