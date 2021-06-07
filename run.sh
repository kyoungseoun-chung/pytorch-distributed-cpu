#!/usr/bin/env bash

rm -rf lsf*

bsub -n 4 -W 01:00 -J 4cpus python distributed_test.py
bsub -n 8 -W 01:00 -J 8cpus python distributed_test.py
bsub -n 12 -W 01:00 -J 12cpus python distributed_test.py
# bsub -n 24 -W 01:00 -J 24cpus python distributed_test.py

