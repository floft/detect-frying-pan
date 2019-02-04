#!/usr/bin/env python3
"""
Creating a new dataset from part of the old dataset, I only copied some of the
images, and I'd rather not re-label them all. This removes all the deleted
images from the Sloth JSON file. (Or, rather, just doesn't print them out.)

It can also remove a set of classes if desired.
"""
import os
import sys
import json
from sloth_common import getJson

def removeDeleted(data, ignoreClasses=[]):
    """
    If the image doesn't exist, then remove it from the JSON file
    """
    newData = []

    for d in data:
        if os.path.exists(d["filename"]):
            filename = d["filename"]
            className = d["class"]
            annotations = []

            for a in d["annotations"]:
                if a["class"] not in ignoreClasses:
                    annotations.append(a)

            newData.append({
                    "annotations": annotations,
                    "class": className,
                    "filename": filename
                })

    return newData

def outputJson(data):
    print(json.dumps(data, indent=4))

if __name__ == "__main__":
    f = os.path.join("sloth.json")
    data = getJson(f)
    data = removeDeleted(data)
    outputJson(data)
