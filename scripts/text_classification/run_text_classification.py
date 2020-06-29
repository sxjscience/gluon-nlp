import mxnet as mx
from mxnet.lr_scheduler import PolyScheduler
from collections import OrderedDict
from gluonnlp.auto.dataset import TabularDataset
from gluonnlp.auto.preprocessing import TabularClassificationBERTPreprocessor, get_problem_type
from gluonnlp.auto.models.classification import BERTForTabularClassificationV1
from gluonnlp.models import get_backbone
from gluonnlp.data.filtering import LanguageIdentifier
from gluonnlp.utils.config import CfgNode
from gluonnlp.utils.misc import parse_ctx
import pandas as pd
from functools import partial
import multiprocessing as mp
import argparse
import numpy as np
mx.npx.set_np()


TASKS = \
    {'cola': ('sentence', 'label'),
     'sst': ('sentence', 'label'),
     'mrpc': (['sentence1', 'sentence2'], 'label'),
     'sts': (['sentence1', 'sentence2'], 'score'),
     'qqp': (['sentence1', 'sentence2'], 'label'),
     'mnli': (['sentence1', 'sentence2'], 'label'),
     'qnli': (['question', 'sentence'], 'label'),
     'rte': (['sentence1', 'sentence2'], 'label'),
     'wnli': (['sentence1', 'sentence2'], 'label'),
     'snli': (['sentence1', 'sentence2'], 'label')}


class OptimizationV1Config:
    @staticmethod
    def get_cfg():
        cfg = CfgNode()
        cfg.lr_scheduler = 'poly_scheduler'
        cfg.begin_lr = 0.0
        cfg.batch_size = 32
        cfg.lr = 1E-4
        cfg.final_lr = 0.0
        cfg.num_train_epochs = 3.0
        cfg.warmup_portion = 0.1
        cfg.layerwise_lr_decay = 0.9  # The layer_wise decay
        cfg.wd = 0.01  # Weight Decay
        cfg.max_grad_norm = 1.0 # Maximum Gradient Norm
        cfg.version = 'v1'
        return cfg

class TabularModelV1Config:
    @staticmethod
    def get_cfg():
        cfg = CfgNode()
        cfg.BACKBONE = CfgNode()
        cfg.BACKBONE.max_length = 128
        cfg.BACKBONE.name = 'google_electra_base'
        cfg.TABULAR_CLASSIFICATION = BERTForTabularClassificationV1.get_cfg()
        return cfg

class Config:
    @staticmethod
    def get_cfg():
        cfg = CfgNode()
        cfg.version = 1
        cfg.OPTIMIZATION = OptimizationV1Config.get_cfg()
        cfg.MODEL = TabularModelV1Config.get_cfg()
        return cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Text classification example.')
    parser.add_argument('--train_file', type=str,
                        help='The training pandas dataframe.',
                        default=None)
    parser.add_argument('--metadata', type=str, help='The metadata of the problem.',
                        default=None)
    parser.add_argument('--dev_file', type=str,
                        help='The validation pandas dataframe',
                        default=None)
    parser.add_argument('--test_file', type=str,
                        help='The test pandas dataframe',
                        default=None)
    parser.add_argument('--task', type=str,
                        help='The default tasks',
                        default=None)
    parser.add_argument('--eval_metric', type=str,
                        help='The metrics for evaluating the models.',
                        default=None)
    parser.add_argument('--stop_metric', type=str,
                        help='The metrics for early stopping',
                        default=None)
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to evaluate the model')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The output directory where the model params will be written.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args()
    return args


def train(args):
    ctx_l = parse_ctx(args.gpus)
    all_cfg = Config.get_cfg()
    if args.config_file is not None:
        all_cfg = all_cfg.clone_merge(args.config_file)
    optimization_cfg = all_cfg.OPTIMIZATION
    model_cfg = all_cfg.MODEL
    backbone_model_cls, backbone_cfg, tokenizer, local_params_path\
        = get_backbone(model_cfg.BACKBONE.name)
    text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
    # Load train and dev dataset
    train_df = pd.read_pickle(args.train_file)
    dev_df = pd.read_pickle(args.dev_file)
    test_df = pd.read_pickle(args.test_file)
    feature_columns, label_columns = TASKS[args.task]
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    elif isinstance(label_columns, str):
        label_columns = [label_columns]
    train_dataset = TabularDataset(train_df, columns=feature_columns + label_columns)
    dev_dataset = TabularDataset(dev_df, columns=feature_columns + label_columns,
                                 column_properties=train_dataset.column_properties)
    test_dataset = TabularDataset(test_df, columns=feature_columns,
                                  column_properties=train_dataset.column_properties)
    column_properties = train_dataset.column_properties
    preprocessor = TabularClassificationBERTPreprocessor(tokenizer=tokenizer,
                                                         column_properties=column_properties,
                                                         label_columns=label_columns,
                                                         max_length=model_cfg.BACKBONE.max_length)
    problem_type, label_shape =
    net = BERTForTabularClassificationV1(text_backbone=text_backbone,
                                         feature_field_info=preprocessor.feature_field_info(),


    column_infos = parse_columns(train_df, args.feature_columns + args.label_columns)
    assert len(args.label_columns) == 1, 'Currently, only a single label is supported'

    backbone = backbone_model_cls.from_cfg(cfg, prefix='backbone_')
    if args.problem_type == 'regression':
        net = BertForTextClassification(backbone=backbone, num_class=1)
    elif args.problem_type == 'classification':
        unique_labels = np.unique(train_df[args.label_columns[0]])
        label_name_id_map = {i: ele for i, ele in enumerate(unique_labels)}
        net = BertForTextClassification(backbone=backbone, num_class=len(unique_labels))
    else:
        raise NotImplementedError
    # Preprocess the data
    col_data = dict()
    col_statistics = dict()
    for col_name in args.feature_columns:
        if column_infos[col_name].type == 'text':
            with mp.Pool(4) as pool:
                lang_id = LanguageIdentifier()
                lang_id_out = pool.map(lang_id, train_df[col_name])
                unique_langs, counts = np.unique(
                    np.array([ele[0] for ele in lang_id_out]),
                    return_counts=True)
                lang = unique_langs[counts.argmax()]
                encoded_col = pool.map(partial(tokenizer.encode, output_type=int),
                                       train_df[col_name])
                # Calculate some basic statistics
                encoded_col = pool.map(np.array, encoded_col)
                lengths = pool.map(len, encoded_col)
                unk_counts = pool.map(partial(get_num_unk,
                                              unk_id=tokenizer.vocab.unk_id),
                                      encoded_col)
                col_data[col_name] = pd.Series(encoded_col)
                col_statistics[col_name] = {'type': 'text',
                                            'lang': lang,
                                            'length_stats': (np.min(lengths),
                                                             np.max(lengths),
                                                             np.mean(lengths)),
                                            'unk_stats': (np.min(unk_counts),
                                                          np.max(unk_counts),
                                                          np.mean(unk_counts))}
        else:
            raise NotImplementedError
    preprocessed_train_df = pd.DataFrame(col_data)
    print(preprocessed_train_df)
    print(col_statistics)
    ch = input()



def predict(args):
    assert args.test_file is not None
    test_df = pd.read_pickle(args.test_file)


if __name__ == '__main__':
    args = parse_args()
    if args.do_train:
        train(args)
    if args.do_eval:
        predict(args)
