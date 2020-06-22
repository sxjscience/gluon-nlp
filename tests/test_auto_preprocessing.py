from gluonnlp.models.bert import get_pretrained_bert
from gluonnlp.auto.preprocessing import TabularBERTPreprocessor
from gluonnlp.utils.testing import autonlp_snli_testdata

test_snli_df, test_snli_metadata = autonlp_snli_testdata()


def test_tabular_bert_preprocessor():
    cfg, tokenizer, _, _ = get_pretrained_bert()
    preprocessor = TabularBERTPreprocessor(tokenizer=tokenizer,
                                           column_property_dict=)

