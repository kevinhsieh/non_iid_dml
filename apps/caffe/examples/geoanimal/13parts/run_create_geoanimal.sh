#!/usr/bin/env sh

python scripts/duplicate.py examples/geoanimal/13parts/create_geoanimal.sh 13

#for n in 0 1 2 3 4 5 6 7 8 9 10 11 12 13
#do
#    ./examples/geoanimal/13parts/create_geoanimal.sh.${n}
#done
pdsh -R ssh -w ^examples/geoanimal/13parts/machinefile "cd $(pwd) && ./examples/geoanimal/13parts/create_geoanimal.sh.%n"
