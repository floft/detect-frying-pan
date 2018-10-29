#!/bin/bash
#
# Shrink images in my_images_large/ and convert to JPG
#
. /scripts/threading # https://floft.net/code/bash-threading/
thread_init

in="my_images_large/"
out="my_images/"

mkdir -p "$out"

for i in "$in/"*; do
    f="$(sed "
    s#\..*\$#\.jpg#g
    s#$in#$out#g
    " <<< "$i")"
    echo "$f"

    gm convert -quality 90 -resize 1280 "$i" "$f" &
    thread_wait
done

thread_finish
