#!/bin/bash

python3 pdb-dataset --mode=train -128
python3 pdb-dataset --mode=test -128

read -rsp 'Press <Enter> to exit...'