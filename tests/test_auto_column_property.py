import os
import tempfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from gluonnlp.base import get_repo_url
from gluonnlp.auto import constants as _C
from gluonnlp.auto.column_property import TextColumnProperty, CategoricalColumnProperty,\
                                      NumericalColumnProperty, EntityColumnProperty
from gluonnlp.utils.misc import download


with tempfile.TemporaryDirectory() as root:
    test_snli_df_path = download(get_repo_url()
                                 + 'autonlp_test_datasets/snli_test_dataset-0de8d633.pd.pkl',
                                 path=os.path.join(root, 'test.pkl'),
                                 sha1_hash='0de8d63354c33d66f34c1e4cc2b4289a9f8c8a3e')
    test_snli_df = pd.read_pickle(test_snli_df_path)
    test_snli_metadata = {'sentence1_entity_numeric': {'type': 'entity', 'parent': 'sentence1'},
                          'sentence1_entity_categorical': {'type': 'entity', 'parent': 'sentence1'},
                          'sentence2_entity': {'type': 'entity', 'parent': 'sentence2'}}


def test_text_column_property():
    text_column_property = TextColumnProperty(test_snli_df['sentence1'])
    print(text_column_property)  # Test printing
    assert text_column_property.type == _C.TEXT
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
    print(column_property)
    assert column_property.type == _C.CATEGORICAL
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
    assert categorical_column_property2.categories == ['contradiction', 'entailment', 'neutral']
    assert categorical_column_property2.frequencies == [4, 3, 3]


def test_numerical_column_property():
    test_series = pd.Series(np.arange(0, 100, dtype=np.float32))
    column_property = NumericalColumnProperty(test_series)
    print(column_property)
    assert column_property.type == _C.NUMERICAL
    assert column_property.num_sample == 100
    another_column_property = column_property.parse_other(pd.Series(np.arange(0, 10,
                                                                              dtype=np.float32)))
    assert another_column_property.num_sample == 10


def test_entity_column_property():
    # Test for Numeric Label
    sentence1_entity_numeric = test_snli_df['sentence1_entity_numeric']
    entity_with_numeric_label = EntityColumnProperty(
        sentence1_entity_numeric,
        parent=test_snli_metadata['sentence1_entity_numeric']['parent'])
    print(entity_with_numeric_label)
    assert entity_with_numeric_label.label_shape == (10,)
    assert entity_with_numeric_label.parent == 'sentence1'
    assert entity_with_numeric_label.label_type == _C.NUMERICAL
    assert entity_with_numeric_label.label_keys is None
    assert entity_with_numeric_label.label_freq is None
    assert entity_with_numeric_label.num_sample == 1000
    assert entity_with_numeric_label.num_missing_sample == 0
    assert entity_with_numeric_label.num_total_entity == 15917
    merged_char_offsets, merged_labels = entity_with_numeric_label.transform(
        sentence1_entity_numeric[0])
    assert merged_char_offsets.shape == (10, 2)
    assert merged_labels.shape == (10, 10)
    assert merged_char_offsets[0, 0] == sentence1_entity_numeric[0][0][0]
    assert merged_char_offsets[0, 1] == sentence1_entity_numeric[0][0][1]
    col_prop2 = entity_with_numeric_label.parse_other(sentence1_entity_numeric.iloc[1:2])
    assert col_prop2.label_shape == (10,)
    assert col_prop2.label_type == _C.NUMERICAL
    assert col_prop2.label_keys is None
    assert col_prop2.label_freq is None

    # Test for Categorical Label
    sentence1_entity_categorical = test_snli_df['sentence1_entity_categorical']
    column_prop = EntityColumnProperty(
        sentence1_entity_categorical,
        parent=test_snli_metadata['sentence1_entity_categorical']['parent'])
    print(column_prop)
    assert column_prop.parent == 'sentence1'
    assert column_prop.label_keys == ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'NOUN',
                                      'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'VERB']
    assert column_prop.label_freq == [1331, 2236, 194, 519, 384, 2861, 4508, 303,
                                      96, 176, 188, 1295, 200, 1626]
    with mp.Pool(2) as pool:
        transformed_data = pool.map(column_prop.transform, sentence1_entity_categorical.tolist())
    for (merged_char_offsets, merged_labels), original_labeled_entities in \
            zip(transformed_data, sentence1_entity_categorical.tolist()):
        for offset, label, entity in zip(merged_char_offsets, merged_labels,
                                         original_labeled_entities):
            assert offset[0] == entity['start']
            assert offset[1] == entity['end']
            assert column_prop.idx_to_label(label) == entity['label']

    # Test for None
    sentence1_entity_categorical_with_none = sentence1_entity_categorical.copy().iloc[1:5]
    sentence1_entity_categorical_with_none[0] = None
    column_prop2 = column_prop.parse_other(sentence1_entity_categorical_with_none)
    assert column_prop2.num_missing_sample == 1
    assert column_prop2.label_keys == column_prop.label_keys
    merged_char_offsets, merged_labels = column_prop2.transform(
        sentence1_entity_categorical_with_none[0])
    assert merged_char_offsets.shape == (0, 2)
    assert merged_labels.shape == (0,)

    # Test for Entity columns without label
    sentence2_entity = test_snli_df['sentence2_entity']
    column_prop = EntityColumnProperty(
        sentence2_entity,
        parent=test_snli_metadata['sentence2_entity']['parent'])
    print(column_prop)
    assert column_prop.num_sample == 1000
    assert column_prop.num_total_entity == 8344
    assert column_prop.name == 'sentence2_entity'
    assert column_prop.parent == 'sentence2'
    merged_char_offsets, merged_labels = column_prop.transform(sentence2_entity[0])
    assert merged_labels is None
    assert merged_char_offsets.shape == (15, 2)
