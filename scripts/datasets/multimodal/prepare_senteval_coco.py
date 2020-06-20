import argparse
import os
import pickle
import numpy as np
import pandas as pd
from gluonnlp.base import get_data_home_dir
from gluonnlp.utils.misc import download, load_checksum_stats
from gluonnlp.registry import DATA_MAIN_REGISTRY, DATA_PARSER_REGISTRY

_CITATIONS = """
@article{conneau2018senteval,
  title={Senteval: An evaluation toolkit for universal sentence representations},
  author={Conneau, Alexis and Kiela, Douwe},
  journal={arXiv preprint arXiv:1803.05449},
  year={2018}
}

@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}

@article{chen2015microsoft,
  title={Microsoft coco captions: Data collection and evaluation server},
  author={Chen, Xinlei and Fang, Hao and Lin, Tsung-Yi and Vedantam, Ramakrishna and Gupta, Saurabh and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  journal={arXiv preprint arXiv:1504.00325},
  year={2015}
}
"""


_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_URL_FILE_STATS = load_checksum_stats(os.path.join(_CURR_DIR,
                                                   '..', 'url_checksums', 'senteval_coco.txt'))

PATH = {
    'train': 'https://dl.fbaipublicfiles.com/senteval/coco_r101_feat/train.pkl',
    'valid': 'https://dl.fbaipublicfiles.com/senteval/coco_r101_feat/valid.pkl',
    'test': 'https://dl.fbaipublicfiles.com/senteval/coco_r101_feat/test.pkl'
}


@DATA_PARSER_REGISTRY.register('prepare_senteval_coco')
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default='.', type=str)
    parser.add_argument('--cache-path', type=str,
                        default=os.path.join(get_data_home_dir(), 'senteval_coco'),
                        help='The temporary path to download the dataset.')
    parser.add_argument('--overwrite', action='store_true')
    return parser


@DATA_MAIN_REGISTRY.register('prepare_senteval_coco')
def main(args):
    os.makedirs(args.cache_path, exist_ok=True)
    folder_name = 'senteval_coco'
    if os.path.exists(os.path.join(args.save_dir, 'senteval_coco')) and not args.overwrite:
        print('Found {}, Skip. You may add "--overwrite" to force overwrite.'
              .format(args.save_dir, folder_name))
    for split, url in PATH.items():
        print('Extract SentEval COCO Split={} to {}'.format(split,
                                                            os.path.join(args.save_dir,
                                                                         folder_name, split)))
        sha1_hash = _URL_FILE_STATS[url]
        target_path = download(url, path=args.cache_path, sha1_hash=sha1_hash)
        with open(target_path, 'rb') as f:
            dat = pickle.load(f, encoding='latin1')
            os.makedirs(os.path.join(args.save_dir, folder_name, split), exist_ok=True)
            np.save(os.path.join(args.save_dir, folder_name, split, 'image_features.npy'),
                    dat['features'])
            assert len(dat['image_to_caption_ids']) == len(dat['features'])
            np.save(os.path.join(args.save_dir, folder_name, split, 'image_to_caption_ids.npy'),
                    np.array(dat['image_to_caption_ids']))
            captions_df = pd.DataFrame(dat['captions'])
            captions_df.to_pickle(os.path.join(args.save_dir, folder_name, split, 'captions.pd.pkl'))


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
