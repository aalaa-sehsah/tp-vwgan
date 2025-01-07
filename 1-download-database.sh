#!/bin/bash

python3 pdb-database --mode=train
python3 pdb-database --mode=test

read -rsp 'Press <Enter> to exit...'