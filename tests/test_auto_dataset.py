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


def test_glue_datasets():
    glue_parser = prepare_glue.get_parser()


def test_superglue_datasets():
    glue_parser = prepare_glue.get_parser()

