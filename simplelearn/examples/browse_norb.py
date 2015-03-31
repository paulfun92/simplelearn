#! /usr/bin/env python

'''
Step through the images of NORB or Small NORB.
'''

import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description=("Steps through the images and labels of the NORB or "
                     "Small NORB dataset."))

    parser.add_argument('--which-norb',
                        choices=['big', 'small'],
                        required=True,
                        help="Which NORB dataset (big or small).")

    parser.add_argument('--which-set',
                        choices=['train', 'test'],
                        required=True,
                        help="Which subset (train or test)")

    return parser.parse_args()


def main():
    args = parse_args()

    norb = load_norb(args.which_norb, args.which_set)


if __name__ == '__main__':
    main()
