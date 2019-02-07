#!/usr/bin/env python3
"""
I have way too many files. Delete all but every 10th one
"""
import os
import pathlib

def keep_nth_files(d, number=10):
    l = list(pathlib.Path(d).glob("*"))

    num = len(l)
    deleted = 0

    for i, n in enumerate(l):
        if i%number != 0:
            print("Deleting", n)
            os.remove(str(n))
            deleted += 1

    print("Removed", deleted, "/", num, "= %.01f%%"%(deleted/num*100))

if __name__ == "__main__":
    #keep_nth_files("rpi_images/00002")
    #keep_nth_files("rpi_images/00003")
    #keep_nth_files("rpi_images/00004")
    #keep_nth_files("rpi_images/00005")
    #keep_nth_files("rpi_images/00006")
    #keep_nth_files("rpi_images/00007")
    #keep_nth_files("rpi_images/00008")

