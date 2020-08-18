#!/usr/bin/env sh

python scripts/duplicate.py examples/geoanimal/5parts/create_geoanimal.sh 5
pdsh -R ssh -w ^examples/geoanimal/5parts/machinefile "cd $(pwd) && ./examples/geoanimal/5parts/create_geoanimal.sh.%n"
