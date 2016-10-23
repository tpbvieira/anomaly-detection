# values.py
# This module will go through the file_list, each file in the list contains
# a bunch of key=value pair.
#
# This python script will create a dictionary from the key=value pair.

key_val_pair = {}

tag = 'key_val/'

file_list = ['attack.txt', 'protocols.txt', 'service.txt', 'flags.txt']

debug = False


def get_list():
    for each_file in file_list:
        with open(tag + each_file) as f:
            all_lines = f.readlines()

        for line in all_lines:
            if line == '' or line == '\n':
                continue

            val = line.split('=')

            key_val_pair[val[0]] = int(val[1])

    if debug:
        print key_val_pair

    return key_val_pair
