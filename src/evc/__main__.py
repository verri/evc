import argparse
import os
import hashlib
import numpy
import json
import pandas as pd
from git import Repo
import tempfile

metamk_contents = """
DATA_FILES = $(shell find src/data -type f)
SAMPLING_FILES = $(shell find src/sampling -type f)
EVALUATION_FILES = $(shell find src/evaluation -type f)
MODEL_FILES = $(shell find src/model -type f)

all: .meta/data.json .meta/sampling.json .meta/evaluation.json .meta/model.json

.meta/:
	@mkdir -p .meta

.meta/data.json: .meta/ $(DATA_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target data

.meta/sampling.json: .meta/data.json $(SAMPLING_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target sampling

.meta/evaluation.json: .meta/ $(EVALUATION_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target evaluation

.meta/model.json: .meta/ $(MODEL_FILES)
	@echo "Generating $@..."
	@python -m evc meta --target model
"""

def update(args):
    with tempfile.NamedTemporaryFile(suffix='.mk') as f:
        f.write(metamk_contents.encode())
        f.flush()
        print('Updating metadata...')
        os.system(f'make -f {f.name}')

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

    elif args.target == 'evaluation':
        from evaluation import Evaluation
        evaluation = Evaluation()

        # XXX: here we have a major limitation: how can we track if the
        # evaluation function is changed?  We could use the source code hash,
        # but I am unsure if this is enough.  Let's rely on the user to update
        # the name if the evaluation function is changed.
        # Another approach is to use a Git Hook that checks for changes inside
        # the evaluation/ directory.  If there is a change, the user must
        # explicitly update the version metadata.
        jsondata = {
            'name': Evaluation.name,
            'version': Evaluation.version,
        }

        with open('.meta/evaluation.json', 'w') as f:
            json.dump(jsondata, f, indent=2)

    elif args.target == 'model':
        from model import Model
        model = Model()

        # XXX: here we have a similar limitation as the evaluation metadata.
        jsondata = {
            'name': Model.name,
            'version': Model.version,
            'description': Model.__doc__,
        }

        with open('.meta/model.json', 'w') as f:
            json.dump(jsondata, f, indent=2)

    else:
        print('Metadata generation for "{}" is not implemented'.format(args.target))
        raise NotImplementedError

def evaluate(args):

    # TODO: we should check if the metadata is up-to-date before running the
    # evaluation.  Also, we must check whether all changes are committed to the
    # repository.

    from data import Dataset
    dataset = Dataset()
    X, y = dataset.load()

    from model import Model
    model = Model()

    from sampling import Sampling
    sampling = Sampling()

    from evaluation import Evaluation
    evaluation = Evaluation()

    splits = sampling.split(X, y)

    results = {
        'id': [],
        'scores': [],
    }

    for i, split in enumerate(splits):
        train_idx = split['train']
        test_idx = split['test']

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = evaluation.evaluate(y_test, y_pred)

        results['id'].append(i)
        results['scores'].append(score)

    os.makedirs('.cache', exist_ok=True)
    os.makedirs('.cache/evaluation', exist_ok=True)

    repo = Repo('.')
    assert not repo.bare

    head_commit_hash = repo.head.commit.hexsha
    pd.DataFrame(results).to_csv(f'.cache/evaluation/{head_commit_hash}.csv', index=False)

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

    # 'evaluate' subcommand
    evaluate_parser = subparsers.add_parser('evaluate',
            help='Evaluate the model')
    evaluate_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
