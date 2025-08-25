#!/bin/bash -x

find . -name __pycache__ -exec rm -r {} +
git clean -dfX
