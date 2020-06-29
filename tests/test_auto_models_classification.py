import mxnet as mx
import os
import tempfile
from mxnet.gluon.data import DataLoader
from gluonnlp.models.bert import get_pretrained_bert, BertModel
from gluonnlp.auto.dataset import TabularDataset
from gluonnlp.auto.preprocessing import TabularClassificationBERTPreprocessor
from gluonnlp.cli.data.general_nlp_benchmark import prepare_glue
from gluonnlp.auto.preprocessing import get_problem_type
from gluonnlp.auto.models.classification import BERTForTabularClassificationV1
mx.npx.set_np()

GLUE_TASKS_FOR_TEST = \
    [('cola', 'sentence', 'label'),
     ('sst', 'sentence', 'label'),
     ('mrpc', ['sentence1', 'sentence2'], 'label'),
     ('sts', ['sentence1', 'sentence2'], 'score'),
     ('qqp', ['sentence1', 'sentence2'], 'label')]


def test_bert_for_tabular_classification_v1():
    glue_parser = prepare_glue.get_parser()
    dataset_name = 'sst'
    with tempfile.TemporaryDirectory() as root:
        args = glue_parser.parse_args(['--benchmark', 'glue',
                                       '--cache-path', root,
                                       '--data_dir', root,
                                       '-t', dataset_name])
        prepare_glue.main(args)
        train_dataset = TabularDataset(os.path.join(root, dataset_name, 'train.pd.pkl'),
                                       columns=['sentence', 'label'])
        dev_dataset = TabularDataset(os.path.join(root, dataset_name, 'dev.pd.pkl'),
                                     columns=['sentence', 'label'],
                                     column_properties=train_dataset.column_properties)
        test_dataset = TabularDataset(os.path.join(root, dataset_name, 'test.pd.pkl'),
                                      columns=['sentence'],
                                      column_properties=train_dataset.column_properties)
    cfg, tokenizer, param_path, _ = get_pretrained_bert()
    backbone = BertModel.from_cfg(cfg)
    backbone.load_parameters(param_path)
    column_properties = train_dataset.column_properties
    preprocessor = TabularClassificationBERTPreprocessor(tokenizer=tokenizer,
                                                         column_properties=column_properties,
                                                         max_length=backbone.max_length,
                                                         label_columns='label',
                                                         merge_text=True)
    problem_type, label_shape = get_problem_type(column_properties['label'])
    train_preprocessed = preprocessor.process_train(train_dataset.table)
    dev_preprocessed = preprocessor.process_train(dev_dataset.table)
    test_preprocessed = preprocessor.process_test(test_dataset.table)
    model = BERTForTabularClassificationV1(text_backbone=backbone,
                                           feature_field_info=preprocessor.feature_field_info(),
                                           label_shape=label_shape)
    model.hybridize()
    model.initialize()
    train_dataloader = DataLoader(train_preprocessed, batch_size=2, shuffle=False,
                                  batchify_fn=preprocessor.batchify(is_test=False))
    dev_dataloader = DataLoader(dev_preprocessed, batch_size=2, shuffle=False,
                                batchify_fn=preprocessor.batchify(is_test=True))
    feature_batch, label_batch = next(iter(train_dataloader))
    out = model(feature_batch)
