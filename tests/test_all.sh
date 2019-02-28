#!/bin/bash
export PYTHONPATH=$PWD
./tests/test_train.sh
pytest ./tests
