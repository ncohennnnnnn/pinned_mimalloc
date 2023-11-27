#!/usr/bin/env bash

# iterate over all files in the repo and reformat those that must be checked

CLANG_FORMAT_VERSION=clang-format-14

# make sure space in file names are handled
OIFS="$IFS"
IFS=$'\n'
for file in $(git ls-files | grep -E "\.(cpp|hpp|cu)(\.in)?$"); do
    # to allow for per-directory clang format files, we cd into the dir first
    DIR=$(dirname "$file")
    pushd ${DIR} >/dev/null
    ${CLANG_FORMAT_VERSION} -i $(basename -- "${file}")
    popd >/dev/null
done
