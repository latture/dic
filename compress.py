from __future__ import absolute_import, division, print_function
import argparse
from dic.dic_io import update_dic_data


def main():
    parser = argparse.ArgumentParser(description="A tool for compressing Matlab (.mat) files. " \
        "This program requires the -i (--input) and -o (--output) flags to be set via the command line. " \
        "All .mat file in the input directory will be compressed and saved into the output directory. " \
        "Ex: python compress.py -i \"/path/to/input_dir\" -o \"/path/to/output_dir\"")

    parser.add_argument("-i", "--input", type=str, dest="input", required=True,
            help="Path to input directory. Ex: \"/path/to/input_dir\"")
    parser.add_argument("-o", "--output", type=str, dest="output", required=True,
            help="Path to output directory. Ex: \"/path/to/ouput_dir\"")
    args = parser.parse_args()
    
    update_dic_data(args.input, args.output, compress=True)


if __name__ == '__main__':
    main()