from gluonnlp.auto.dataset import TabularNLPDataset
from gluonnlp.utils.testing import autonlp_snli_testdata


def test_tabular_nlp_dataset():
    test_snli_df, test_snli_metadata = autonlp_snli_testdata()
    dataset = TabularNLPDataset(test_snli_df,
                                feature_columns=None,
                                label_columns='label',
                                metadata=test_snli_metadata)
