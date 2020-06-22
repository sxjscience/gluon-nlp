import os
import tempfile
import pandas as pd
from gluonnlp.base import get_repo_url
from gluonnlp.auto.column_property import TextColumnProperty, CategoricalColumnProperty,\
                                      NumericalColumnProperty, EntityColumnProperty
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


def test_text_column_property():
    text_column_property = TextColumnProperty(test_snli_df['sentence1'])
    assert text_column_property.num_sample == 1000
    assert text_column_property.num_missing_sample == 0
    assert text_column_property.min_length == 16
    assert text_column_property.max_length == 229
    assert text_column_property.name == 'sentence2'
    assert text_column_property.lang == 'en'
    text_column_property2 = text_column_property.parse_other(test_snli_df['sentence2'])
    assert text_column_property2.min_length == 11
    assert text_column_property2.max_length == 116
    assert text_column_property2.name == 'sentence2'
    assert text_column_property2.lang == 'en'


def test_categorical_column_property():
    test_snli_df2 = test_snli_df.iloc[10:20]
    column_property = CategoricalColumnProperty(test_snli_df['label'])
    assert column_property.categories == ['contradiction', 'entailment', 'neutral']
    assert column_property.frequencies == [336, 331, 333]
    assert column_property.name == 'label'
    assert column_property.num_sample == 1000
    assert column_property.num_missing_sample == 0
    assert column_property.num_class == 3
    for idx, catalog in enumerate(column_property.categories):
        assert column_property.transform(catalog) == idx
        assert column_property.inv_transform(column_property.transform(catalog)) == catalog
    categorical_column_property2 = column_property.parse_other(test_snli_df2['label'])

