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
                                       '--data_dir', os.path.join(root, 'glue')])
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
                 ('snli', ['sentence1', 'sentence2'], 'label')]


def test_superglue_datasets(tmp_path):
    glue_parser = prepare_glue.get_parser()
    with tempfile.TemporaryDirectory() as root:
        args = glue_parser.parse_args(['--benchmark', 'superglue',
                                       '--cache-path', tmp_path,
                                       '--data_dir', os.path.join(root, 'superglue')])
        prepare_glue.main(args)
