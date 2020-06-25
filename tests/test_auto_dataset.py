import os
import tempfile
import pytest
from gluonnlp.auto import constants as _C
from gluonnlp.auto.dataset import TabularDataset
from gluonnlp.utils.testing import autonlp_snli_testdata
from gluonnlp.cli.data.general_nlp_benchmark import prepare_glue


def test_tabular_nlp_snli_dataset():
    snli_sample_df, snli_sample_metadata = autonlp_snli_testdata()
    dataset = TabularDataset(snli_sample_df,
                             metadata=snli_sample_metadata)
    assert dataset.column_properties['sentence1'].type == _C.TEXT
    assert dataset.column_properties['sentence2'].type == _C.TEXT
    assert dataset.column_properties['sentence1_entity_numeric'].type == _C.ENTITY
    assert dataset.column_properties['sentence1_entity_categorical'].type == _C.ENTITY
    assert dataset.column_properties['sentence2_entity'].type == _C.ENTITY
    assert dataset.column_properties['label'].type == _C.CATEGORICAL


GLUE_TASKS_FOR_TEST = \
    [('cola', 'sentence', 'label'),
     ('sst', 'sentence', 'label'),
     ('mrpc', ['sentence1', 'sentence2'], 'label'),
     ('sts', ['sentence1', 'sentence2'], 'score'),
     ('qqp', ['sentence1', 'sentence2'], 'label'),
     ('mnli', ['sentence1', 'sentence2'], 'label'),
     ('qnli', ['question', 'sentence'], 'label'),
     ('rte', ['sentence1', 'sentence2'], 'label'),
     ('wnli', ['sentence1', 'sentence2'], 'label'),
     ('snli', ['sentence1', 'sentence2'], 'label')]


SUPERGLUE_TASKS_FOR_TEST = \
    [('boolq', ['passage', 'question'], 'label'),
     ('cb', ['premise', 'hypothesis'], 'label'),
     ('copa', ['premise', 'choice1', 'choice2', 'question'], 'label'),
     ('multirc', ['passage', 'question', 'answer'], 'label'),
     ('record', ['source', 'text', 'entities', 'query'], 'answers'),
     ('rte', ['premise', 'hypothesis'], 'label'),
     ('wic', ['sentence1', 'sentence2', 'entities1', 'entities2'], 'label'),
     ('wsc', ['text', 'noun', 'pronoun'], 'label')]


@pytest.mark.parametrize('task_name, feature_columns, label_columns', GLUE_TASKS_FOR_TEST)
def test_glue_datasets(task_name, feature_columns, label_columns):
    glue_parser = prepare_glue.get_parser()
    with tempfile.TemporaryDirectory() as root:
        args = glue_parser.parse_args(['--benchmark', 'glue',
                                       '--cache-path', root,
                                       '--data_dir', root,
                                       '-t', task_name])
        prepare_glue.main(args)
        if label_columns == 'label':
            expected_label_type = _C.CATEGORICAL
        elif label_columns == 'score':
            expected_label_type = _C.NUMERICAL
        else:
            expected_label_type = None
        train_path = os.path.join(root, task_name, 'train.pd.pkl')
        if task_name == 'mnli':
            dev_path = os.path.join(root, task_name, 'dev_mismatched.pd.pkl')
            test_path = os.path.join(root, task_name, 'test_mismatched.pd.pkl')
            columns = ['sentence1', 'sentence2', 'label']
        else:
            dev_path = os.path.join(root, task_name, 'dev.pd.pkl')
            test_path = os.path.join(root, task_name, 'test.pd.pkl')
            columns = None
        # We test for the parsing
        train_data = TabularDataset(train_path, columns=columns)
        dev_data = TabularDataset(dev_path,
                                  columns=columns,
                                  column_properties=train_data.column_properties)
        test_data = TabularDataset(test_path,
                                   columns=columns,
                                   column_properties=train_data.column_properties)
        if expected_label_type is not None:
            assert train_data.column_properties[label_columns].type == expected_label_type
            assert dev_data.column_properties[label_columns].type == expected_label_type


@pytest.mark.parametrize('task_name, feature_columns, label_columns', SUPERGLUE_TASKS_FOR_TEST)
def test_superglue_datasets(task_name, feature_columns, label_columns):
    glue_parser = prepare_glue.get_parser()
    with tempfile.TemporaryDirectory() as root:
        args = glue_parser.parse_args(['--benchmark', 'superglue',
                                       '--cache-path', root,
                                       '--data_dir', root,
                                       '-t', task_name])
        prepare_glue.main(args)
        if label_columns == 'label':
            expected_label_type = _C.CATEGORICAL
        elif label_columns == 'score':
            expected_label_type = _C.NUMERICAL
        elif label_columns == 'answers':
            expected_label_type = _C.ENTITY
        if os.path.exists(os.path.join(root, task_name, 'metadata.json')):
            metadata_path = os.path.join(root, task_name, 'metadata.json')
        else:
            metadata_path = None
        train_path = os.path.join(root, task_name, 'train.pd.pkl')
        dev_path = os.path.join(root, task_name, 'dev.pd.pkl')
        test_path = os.path.join(root, task_name, 'test.pd.pkl')
        # We test for the parsing
        train_data = TabularDataset(train_path,
                                    metadata=metadata_path)
        dev_data = TabularDataset(dev_path,
                                  column_properties=train_data.column_properties,
                                  metadata=metadata_path)
        test_data = TabularDataset(test_path,
                                   column_properties=train_data.column_properties,
                                   metadata=metadata_path)
        assert train_data.column_properties[label_columns].type == expected_label_type
        assert dev_data.column_properties[label_columns].type == expected_label_type
