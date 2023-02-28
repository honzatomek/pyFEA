#!/bin/bash

CWD="$(pwd)"
while [[ ! -d "pyFEA" ]]; do
  if [[ -d "__venv__" ]]; then
    VENV="$(pwd)/__venv__/bin/activate"
    break
  else
    cd ..
  fi
done
# cd "bin"

cd "${CWD}"

source "${VENV}"

# python -m pytest ../tests/unit
python -m pytest --verbosity=2 ../tests/unit

