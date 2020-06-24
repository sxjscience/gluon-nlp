from gluonnlp.models.bert import get_pretrained_bert
from gluonnlp.auto.dataset import TabularDataset
from gluonnlp.auto.preprocessing import TabularBERTPreprocessor
from gluonnlp.utils.testing import autonlp_snli_testdata

test_snli_df, test_snli_metadata = autonlp_snli_testdata()


def test_tabular_bert_preprocessor():
    cfg, tokenizer, _, _ = get_pretrained_bert()
    dataset = TabularDataset(test_snli_df, label='label', metadata=test_snli_metadata)
    preprocessor = TabularBERTPreprocessor(tokenizer=tokenizer,
                                           column_properties=dataset.column_properties,
                                           max_length=512,
                                           merge_text=True)
    np_table = dataset.table.to_numpy()
    for row in np_table:
        out = preprocessor(row)
        print(out)
        ch = input()
