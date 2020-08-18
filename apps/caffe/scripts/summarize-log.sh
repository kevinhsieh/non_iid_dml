#!/usr/bin/env sh

python scripts/summarize-loss.py $1/output.txt $2 $3 > $1/losses.txt
python scripts/summarize-accuracy.py $1/output.txt $2 0 $3 > $1/accuracy.txt
python scripts/summarize-accuracy.py $1/output.txt $2 1 $3 > $1/accuracy-training.txt
