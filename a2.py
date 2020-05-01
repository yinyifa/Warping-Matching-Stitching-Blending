#!/usr/bin/python

# Authors : Xizi Wang/xiziwang, Yifan Ying/yinyifa, Chang Liu/liu472

import sys
import part1
import part2
import part3

if __name__ == '__main__':
    # python a2.py part1 2 bigben_6.jpg bigben_8.jpg eiffel_18.jpg eiffel_19.jpg part1_output.txt
    arguments = sys.argv
    which_part = arguments[1]
    if which_part == "part1":
        k = arguments[2]
        picture_list = arguments[3:-1]
        output_filename = arguments[-1]
        part1.k_cluster(picture_list, int(k), output_filename)
    elif which_part == "part2":
        n = arguments[2]
        picture_1, picture_2 = arguments[3:5]
        output_filename = arguments[5]
        coordinates = arguments[6:]
        # Todo: Change here
        part2.WHATEVER_FUNCTION(int(n), picture_1, picture_2, output_filename, coordinates)
    elif which_part == "part3":
        picture_1, picture_2, output_filename = arguments[2:]
        # Todo: Change here
        part3.WHATEVER_FUNCTION(picture_1, picture_2, output_filename)
