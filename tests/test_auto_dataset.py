import os
import tempfile
from gluonnlp.auto import constants as _C
from gluonnlp.auto.dataset import TabularDataset
from gluonnlp.utils.testing import autonlp_snli_testdata
from gluonnlp.cli.data.general_nlp_benchmark import prepare_glue


def test_tabular_nlp_snli_dataset():
    snli_sample_df, snli_sample_metadata = autonlp_snli_testdata()
    dataset = TabularDataset(snli_sample_df,
                             label='label',
                             metadata=snli_sample_metadata)
    assert dataset.column_properties['sentence1'].type == _C.TEXT
    assert dataset.column_properties['sentence2'].type == _C.TEXT


def test_glue_datasets(tmp_path):
    glue_parser = prepare_glue.get_parser()
    with tempfile.TemporaryDirectory() as root:
        args = glue_parser.parse_args(['--benchmark', 'glue',
                                       '--cache-path', tmp_path,
                                       '--data_dir', root])
        prepare_glue.main(args)
        tasks = [('cola', 'sentence', 'label'),
                 ('sst2', 'sentence', 'label'),
                 ('mrpc', ['sentence1', 'sentence2'], 'label'),
                 ('sts', ['sentence1', 'sentence2'], 'score'),
                 ('qqp', ['sentence1', 'sentence2'], 'label'),
                 ('mnli', ['sentence1', 'sentence2'], 'label'),
                 ('qnli', ['question', 'sentence'], 'label'),
                 ('rte', ['sentence1', 'sentence2'], 'label'),
                 ('wnli', ['sentence1', 'sentence2'], 'label'),
                 ('snli', ['sentence1', 'sentence2'], 'label'),
                 ('rte_diagnostic', None, None)]
        for dir_name, feature_columns, label_columns in tasks:
            if label_columns == 'label':
                expected_label_type = _C.CATEGORICAL
            elif label_columns == 'score':
                expected_label_type = _C.NUMERICAL
            else:
                expected_label_type = None
            train_path = os.path.join(root, dir_name, 'train.pd.pkl')
            dev_path = os.path.join(root, dir_name, 'dev.pd.pkl')
            test_path = os.path.join(root, dir_name, 'test.pd.pkl')
            # We test for the parsing
            train_data = TabularDataset(train_path,
                                        feature=feature_columns,
                                        label=label_columns)
            dev_data = TabularDataset(dev_path,
                                      feature=train_data.feature_columns,
                                      label=train_data.label_columns,
                                      column_properties=train_data.column_properties)
            test_data = TabularDataset(test_path,
                                       feature=train_data.feature_columns,
                                       label=train_data.label_columns,
                                       column_properties=train_data.column_properties)
            if expected_label_type is not None:
                assert train_data.column_properties[label_columns] == expected_label_type
                assert dev_data.column_properties[label_columns] == expected_label_type
                assert test_data.column_properties[label_columns] == expected_label_type


def test_superglue_datasets(tmp_path):
    glue_parser = prepare_glue.get_parser()
    with tempfile.TemporaryDirectory() as root:
        args = glue_parser.parse_args(['--benchmark', 'superglue',
                                       '--cache-path', tmp_path,
                                       '--data_dir', os.path.join(root, 'superglue')])
        prepare_glue.main(args)
        tasks = [('boolq', ['passage', 'question'], 'label'),
                 ('cb', ['premise', 'hypothesis'], 'label'),
                 ('copa', ['premise', 'choice1', 'choice2', 'question'], 'label'),
                 ('multirc', ['passage', 'question', 'answer'], 'label'),
                 ('record', ['source', 'text', 'entities', 'query'], 'answers'),
                 ('rte', ['premise', 'hypothesis'], 'label'),
                 ('wic', ['sentence1', 'sentence2', 'entities1', 'entities2'], 'label'),
                 ('wsc', ['text', 'entities'], 'label'),
                 ('AX-b', None, 'label'),
                 ('AX-g', None, 'label')]
        for dir_name, feature_columns, label_columns in tasks:
            if label_columns == 'label':
                expected_label_type = _C.CATEGORICAL
            elif label_columns == 'score':
                expected_label_type = _C.NUMERICAL
            elif label_columns == 'answers':
                expected_label_type = _C.TEXT
            if os.path.exists(os.path.join(root, dir_name, 'metadata.json')):
                metadata_path = os.path.join(root, dir_name, 'metadata.json')
            else:
                metadata_path = None
            train_path = os.path.join(root, dir_name, 'train.pd.pkl')
            dev_path = os.path.join(root, dir_name, 'dev.pd.pkl')
            test_path = os.path.join(root, dir_name, 'test.pd.pkl')
            # We test for the parsing
            train_data = TabularDataset(train_path,
                                        feature=feature_columns,
                                        label=label_columns,
                                        metadata=metadata_path)
            dev_data = TabularDataset(dev_path,
                                      feature=train_data.feature_columns,
                                      label=train_data.label_columns,
                                      column_properties=train_data.column_properties,
                                      metadata=metadata_path)
            test_data = TabularDataset(test_path,
                                       feature=train_data.feature_columns,
                                       label=train_data.label_columns,
                                       column_properties=train_data.column_properties,
                                       metadata=metadata_path)
            assert train_data.column_properties[label_columns] == expected_label_type
            assert dev_data.column_properties[label_columns] == expected_label_type
            assert test_data.column_properties[label_columns] == expected_label_type
