#!/bin/bash
EPLTD_PROG_DIR=/mnt/c/Users/armand.chocr/Documents/armand_repo/epltd/epltd_all
MATLAB_CODE_DIR=/mnt/c/Users/armand.chocr/Documents/armand_repo/parsing
cd $MATLAB_CODE_DIR
command="$EPLTD_PROG_DIR -r $2"
eval $command