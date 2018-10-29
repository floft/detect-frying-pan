#!/bin/bash
#
# Generate Sloth file for labeling from *_images/*
#
# From: http://sloth.readthedocs.io/en/latest/examples.html
out="sloth.json"
[[ ! -e "$out" ]] && \
    find *_images/ -type f -print0 | \
    sort -z | \
    xargs -0 sloth appendfiles "$out"
