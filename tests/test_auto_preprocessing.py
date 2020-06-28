import multiprocessing as mp
import pytest
import mxnet as mx
from mxnet.gluon.data import ArrayDataset, DataLoader
from gluonnlp.models.bert import get_pretrained_bert
from gluonnlp.auto.dataset import TabularDataset
from gluonnlp.auto.preprocessing import TabularClassificationBERTPreprocessor
from gluonnlp.utils.testing import autonlp_snli_testdata
from gluonnlp.utils.misc import num_mp_workers
from gluonnlp.auto import constants as _C

test_snli_df, test_snli_metadata = autonlp_snli_testdata()


@pytest.mark.parametrize('merge_text', [False, True])
def test_tabular_bert_preprocessor(merge_text):
    _, tokenizer, _, _ = get_pretrained_bert()
    dataset = TabularDataset(test_snli_df, metadata=test_snli_metadata)
    max_length = 60
    preprocessor = TabularClassificationBERTPreprocessor(tokenizer=tokenizer,
                                                         column_properties=dataset.column_properties,
                                                         max_length=max_length,
                                                         label_columns='label',
                                                         merge_text=True)
    with mp.Pool(num_mp_workers()) as pool:
        preprocessed_samples = pool.map(preprocessor, np_table)
    mx_dataset = ArrayDataset(preprocessed_samples)
    dataloader = DataLoader(mx_dataset, batch_size=4, shuffle=False,
                            batchify_fn=preprocessor.batchify())
    for batch in dataloader:
        for i, (field_type_code, filed_attrs) in enumerate(field_infos):
            if field_type_code == _C.TEXT:
                batch_token_ids, batch_segment_ids, batch_valid_length = batch[i]
                assert batch_token_ids.shape[1] <= max_length
                assert batch_segment_ids.shape == batch_token_ids.shape
                assert batch_valid_length.max() <= max_length
            elif field_type_code == _C.ENTITY:
                batch_spans, batch_labels, batch_num_entity = batch[i]
            elif field_type_code == _C.CATEGORICAL:
                batch_data = batch[i]
            elif field_type_code == _C.NUMERICAL:
                batch_data = batch[i]
        mx.npx.waitall()
