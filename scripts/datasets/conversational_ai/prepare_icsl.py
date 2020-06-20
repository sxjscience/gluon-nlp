import argparse
import os
import zipfile
import tarfile
import shutil
from typing import List, Optional
from collections import Counter
from gluonnlp.base import get_data_home_dir
from gluonnlp.registry import DATA_MAIN_REGISTRY, DATA_PARSER_REGISTRY
from gluonnlp.utils.misc import download, load_checksum_stats
from gluonnlp.data.vocab import Vocab


_CITATIONS = """
@inproceedings{hemphill-etal-1990-atis,
    title = "The {ATIS} Spoken Language Systems Pilot Corpus",
    author = "Hemphill, Charles T.  and
      Godfrey, John J.  and
      Doddington, George R.",
    booktitle = "Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley,
     {P}ennsylvania, June 24-27,1990",
    year = "1990",
    url = "https://www.aclweb.org/anthology/H90-1021",
}

@article{coucke2018snips,
  title={Snips voice platform: an embedded spoken language understanding system for private-by-design voice interfaces},
  author={Coucke, Alice and Saade, Alaa and Ball, Adrien and Bluche, Th{\'e}odore and Caulier,
   Alexandre and Leroy, David and Doumouro, Cl{\'e}ment and Gisselbrecht, Thibault and Caltagirone, Francesco and Lavril, Thibaut and others},
  journal={arXiv preprint arXiv:1805.10190},
  year={2018}
}
"""

