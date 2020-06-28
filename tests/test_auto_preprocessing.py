import multiprocessing as mp
import pytest
import mxnet as mx
from mxnet.gluon.data import ArrayDataset, DataLoader
from gluonnlp.models.bert import get_pretrained_bert
from gluonnlp.auto.dataset import TabularDataset
from gluonnlp.auto.preprocessing import TabularClassificationBERTPreprocessor
from gluonnlp.utils.testing import autonlp_snli_testdata
from gluonnlp.utils.preprocessing import convert_token_level_span_to_char
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
                                                         merge_text=merge_text)
    train_preprocessed = preprocessor.process_train(dataset.table)
    test_preprocessed = preprocessor.process_test(dataset.table)
    train_dataloader = DataLoader(train_preprocessed, batch_size=2, shuffle=False,
                                  batchify_fn=preprocessor.batchify(is_test=False))
    test_dataloader = DataLoader(test_preprocessed, batch_size=2, shuffle=False,
                                 batchify_fn=preprocessor.batchify(is_test=True))
    for typ, dataloader in [('train', train_dataloader), ('test', test_dataloader)]:
        num_shift = 0
        missed_total_entity = {key: [0, 0] for key in preprocessor.entity_columns}
        missed_entity_cases = {key: [] for key in preprocessor.entity_columns}
        for batch in dataloader:
            if typ == 'train':
                feature_batch, label_batch = batch
            else:
                feature_batch = batch
                label_batch = None
            for i, (field_type_code, field_attrs) in enumerate(preprocessor.feature_field_info()):
                if field_type_code == _C.TEXT:
                    batch_token_ids, batch_valid_length, batch_segment_ids, batch_token_offsets = feature_batch[i]
                    assert batch_token_ids.shape[1] <= max_length
                    assert batch_segment_ids.shape == batch_token_ids.shape
                    assert batch_valid_length.max() <= max_length
                    assert (batch_token_ids[:, 0].asnumpy() == tokenizer.vocab.cls_id).all()
                    for idx, val_length in enumerate(batch_valid_length.asnumpy()):
                        assert batch_token_ids[idx, val_length - 1].asnumpy() == tokenizer.vocab.sep_id
                elif field_type_code == _C.ENTITY:
                    batch_spans, batch_labels, batch_num_entity = feature_batch[i]
                    parent_idx = field_attrs['parent_idx']
                    parent_name = field_attrs['prop'].parent
                    parent_token_ids = feature_batch[parent_idx][0].asnumpy()
                    parent_token_offsets = feature_batch[parent_idx][-1].asnumpy()
                    for idx, spans in enumerate(batch_spans):
                        num_span = batch_num_entity[idx].asnumpy().item()
                        token_spans = spans[:num_span].asnumpy()
                        original_idx = idx + num_shift
                        print('original_idx:', original_idx, idx, num_shift)
                        original_parent_text = dataset.table[parent_name][original_idx]
                        char_spans = convert_token_level_span_to_char(parent_token_offsets[idx], token_spans)
                        for token_span, char_span in zip(token_spans, char_spans):
                            span_token_ids = parent_token_ids[idx][token_span[0]:(token_span[1] + 1)].tolist()
                            text_by_decode = tokenizer.decode(span_token_ids)
                            text_by_slice = original_parent_text[char_span[0]:char_span[1]]
                            missed_total_entity[field_attrs['prop'].name][1] += 1
                            if text_by_decode.strip() == text_by_slice.strip():
                                missed_total_entity[field_attrs['prop'].name][0] += 1
                            else:
                                missed_entity_cases[field_attrs['prop'].name]\
                                    .append((text_by_decode, text_by_slice))
                elif field_type_code == _C.CATEGORICAL:
                    batch_data = feature_batch[i]
                elif field_type_code == _C.NUMERICAL:
                    batch_data = feature_batch[i]
            if label_batch is not None:
                assert len(label_batch) == 1
                assert ((label_batch[0] >= 0) * label_batch[0] <= 2).asnumpy().all()
            num_shift += len(feature_batch[0])
            print('typ:', num_shift)
            mx.npx.waitall()
        assert num_shift == len(dataset.table)
        for key, (num_match, num_total) in missed_total_entity.items():
            assert num_match / num_total >= 0.95
