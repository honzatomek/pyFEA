#!/bin/bash -
#===============================================================================
#
#          FILE: download.sh
#
#         USAGE: ./download.sh
#
#   DESCRIPTION:
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (),
#  ORGANIZATION:
#       CREATED: 09/18/2022 10:57
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

IFS=$'\n' asc=($(cat curl_links.txt))
for ((i = 0; i < "${#asc[*]}"; i++)); do
  echo "${asc[$i]}:"
  curl 'https://www.ceas3.uc.edu/sdrluff/view.php' --data-raw 'fName='"${asc[i]}" > "${asc[i]}"
done

