from argparse import ArgumentParser

from huggingface_datasets_converter import kaggle_to_hf

def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--kaggle_id', type=str, help='kaggle dataset name (ex. "evangower/airbnb-stock-price")')
    parser.add_argument('--repo_id', type=str, help='huggingface repo (ex: nateraw/airbnb-stock-price)')
    parser.add_argument('--use_zip', action='store_true', help='Pass this parameter if you just want to upload he whole kaggle dataset as zip')
    return parser.parse_args(args=args)


def main(args):
    kaggle_to_hf(args.kaggle_id, args.repo_id, unzip=not args.use_zip)

if __name__ == '__main__':
    main(parse_args())

    # from datasets import load_dataset
    # ds = load_dataset('nateraw/rice-image-dataset')