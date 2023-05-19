#!/bin/bash

python3 -m pytest ./test_relative_disp.py &
PID=$!

wait ${PID}

rm -rf __pycache__
rm -rf .pytest_cache

