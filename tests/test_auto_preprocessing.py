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
def test_tabular_bert_preprocessor_case1(merge_text):
    _, tokenizer, _, _ = get_pretrained_bert()
    dataset = TabularDataset(test_snli_df, metadata=test_snli_metadata)
    max_length = 60
    preprocessor = TabularClassificationBERTPreprocessor(tokenizer=tokenizer,
                                                         column_properties=dataset.column_properties,
                                                         max_length=max_length,
                                                         label_columns='label',
                                                         merge_text=True)
    train_preprocessed = preprocessor.process_train(dataset.table)
    test_preprocessed = preprocessor.process_test(dataset.table)
    train_mx_dataset = ArrayDataset(train_preprocessed)
    test_mx_dataset = ArrayDataset(test_preprocessed)
    train_dataloader = DataLoader(train_mx_dataset, batch_size=4, shuffle=False,
                                  batchify_fn=preprocessor.batchify(is_test=False))
    test_dataloader = DataLoader(test_mx_dataset, batch_size=4, shuffle=False,
                                 batchify_fn=preprocessor.batchify(is_test=True))
    num_shift = 0
    train_missed_entity = {key: [0, 0] for key in preprocessor.entity_columns}
    for feature_batch, label_batch in train_dataloader:
        for i, (field_type_code, field_attrs) in enumerate(preprocessor.feature_field_info()):
            if field_type_code == _C.TEXT:
                batch_token_ids, batch_segment_ids, batch_valid_length = feature_batch[i]
                assert batch_token_ids.shape[1] <= max_length
                assert batch_segment_ids.shape == batch_token_ids.shape
                assert batch_valid_length.max() <= max_length
                assert (batch_token_ids[:, 0].asnumpy() == tokenizer.vocab.cls_id).all()
            elif field_type_code == _C.ENTITY:
                batch_spans, batch_labels, batch_num_entity = feature_batch[i]
                parent_idx = field_attrs['parent_idx']
                original_name = field_attrs['prop'].name
                parent_token_ids = feature_batch[parent_idx][0].asnumpy()
                for idx, spans in enumerate(batch_spans):
                    original_idx = idx + num_shift
                    char_spans =
                    for span_idx in range(batch_num_entity[idx].asnumpy().item()):
                        span = spans[span_idx].asnumpy()
                        span_token_ids = parent_token_ids[idx][span[0]:(span[1] + 1)].tolist()
                        print(tokenizer.decode(span_token_ids))
                        ch = input()
            elif field_type_code == _C.CATEGORICAL:
                batch_data = feature_batch[i]
            elif field_type_code == _C.NUMERICAL:
                batch_data = feature_batch[i]
        assert len(label_batch) == 1
        assert label_batch[0]
        num_shift += len(feature_batch)
        mx.npx.waitall()
