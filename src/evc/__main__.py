import argparse
import os
import hashlib
import numpy
import json

def update(args):
    print('Updating metadata...')
    os.system('make -f utils/meta.mk')

def meta(args):
    if args.target == 'data':
        from data import Dataset
        dataset = Dataset()
        X, y = dataset.load()

        checksum = hashlib.md5()
        checksum.update(X.view(numpy.uint8))
        checksum.update(y.view(numpy.uint8))
        checksum = checksum.hexdigest()

        jsondata = {
            'checksum': checksum,
            'n_samples': X.shape[0],
            'labels': y.tolist(),
        }

        with open('.meta/data.json', 'w') as f:
            json.dump(jsondata, f, indent=2)

    elif args.target == 'sampling':
        from data import Dataset
        dataset = Dataset()
        X, y = dataset.load()

        from sampling import Sampling
        sampling = Sampling()

        splits = sampling.split(X, y)

        jsondata = {
            'splits': splits,
            'rho': sampling.correction_factor(),
        }

        with open('.meta/sampling.json', 'w') as f:
            json.dump(jsondata, f, indent=2)

    else:
        print('Metadata generation for "{}" is not implemented'.format(args.target))
        raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # 'update' subcommand
    update_parser = subparsers.add_parser('update',
            help='Update metadata of the project')
    update_parser.set_defaults(func=update)

    # 'meta' subcommand
    meta_parser = subparsers.add_parser('meta',
            help='Generate specific metadata of the project')
    meta_parser.add_argument('--target', help='Name of the metadata',
            choices=['data', 'model', 'sampling', 'evaluation'],
            required=True)
    meta_parser.set_defaults(func=meta)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
