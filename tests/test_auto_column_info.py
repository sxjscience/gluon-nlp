import os
import tempfile
import pandas as pd
from gluonnlp.base import get_repo_url
from gluonnlp.auto.column_info import TextColumnInfo, CategoricalColumnInfo,\
                                      NumericalColumnInfo, EntityColumnInfo
from gluonnlp.utils.misc import download


with tempfile.TemporaryDirectory() as root:
    test_snli_df_path = download(get_repo_url()
                                 + 'autonlp_test_datasets/snli_test_dataset_022d2d.pd.pkl',
                                 path=os.path.join(root, 'test.pkl'),
                                 sha1_hash='022d2de30ceae79c2f1dffa9eeb3ddd312f2de51')
    test_snli_df = pd.read_pickle(test_snli_df_path)
    test_snli_metadata = {'sentence1_entity_numeric': {'type': 'entity', 'parent': 'sentence1'},
                          'sentence1_entity_categorical': {'type': 'entity', 'parent': 'sentence1'},
                          'sentence2_entity': {'type': 'entity', 'parent': 'sentence2'}}


def test_text_column_info():
    text_column_info = TextColumnInfo(test_snli_df['sentence1'])
    assert text_column_info.num_sample == 1000
    assert text_column_info.lang == 'en'
    text_column_info2 = text_column_info.parse_test(test_snli_df['sentence2'])
    print(text_column_info2)



