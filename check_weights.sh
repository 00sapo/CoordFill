#!/bin/env sh

if [ ! -f pyproject.toml ]; then
	echo "Please run this script from the root of the repository!"
	exit 1
fi

if [ -f ./src/coordfill/encoder-epoch-last.pth ]; then
	echo "Weights found, including them in the wheel package!"
	exit 0
else
	echo "Weights not found, cannot include them in the wheel package!"
	exit 2
fi
