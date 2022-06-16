from argparse import ArgumentParser

from huggingface_datasets_converter import zenudo_to_hf

def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--zenudo_record', type=str, help='zenudo record name (ex. "6606485")')
    parser.add_argument('--repo_id', type=str, help='huggingface repo (ex: nateraw/espeni)')
    parser.add_argument('--workers', type=int, help='Number of download workers', default=1)
    return parser.parse_args(args=args)

def main(args):
    zenudo_to_hf(args.zenudo_record, args.repo_id, num_download_workers=args.workers)

if __name__ == '__main__':
    main(parse_args())

    # Then you can load like this
    # from datasets import load_dataset
    # ds = load_dataset('nateraw/espeni', data_files='espeni.csv')
