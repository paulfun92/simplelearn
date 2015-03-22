#!/bin/bash

# $1 - name of the directory - first argument
# $2 - regex - second argument

if [ $# -lt 2 ]; then
  echo Usage: ./myscript.sh DIR "REGEX"
  exit
fi

find "$1" -iname "$2" -type f -exec wc -l '{}' \; | awk '{ SUM += $1 } END { print SUM }'
