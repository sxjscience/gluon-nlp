import mxnet as mx
import os
import json
import argparse
import numpy as np
from gluonnlp.auto.dataset import load_pandas_df
from gluonnlp.auto.estimators.tabular_basic import BertForTabularPredictionBasic
mx.npx.set_np()


TASKS = \
    {'cola': ('sentence', 'label', 'mcc', ['mcc']),
     'sst': ('sentence', 'label', 'acc', ['acc']),
     'mrpc': (['sentence1', 'sentence2'], 'label', 'f1', ['acc', 'f1']),
     'sts': (['sentence1', 'sentence2'], 'score', 'pearsonr', ['pearsonr', 'spearmanr']),
     'qqp': (['sentence1', 'sentence2'], 'label', 'f1', ['acc', 'f1']),
     'mnli': (['sentence1', 'sentence2'], 'acc', ['acc']),
     'qnli': (['question', 'sentence'], 'acc', ['acc']),
     'rte': (['sentence1', 'sentence2'], 'acc', ['acc']),
     'wnli': (['sentence1', 'sentence2'], 'acc', ['acc']),
     'snli': (['sentence1', 'sentence2'], 'acc', ['acc'])}


def parse_args():
    parser = argparse.ArgumentParser(description='AutoML for Tabular Prediction Basic Example.')
    parser.add_argument('--train_file', type=str,
                        help='The training pandas dataframe.',
                        default=None)
    parser.add_argument('--dev_file', type=str,
                        help='The validation pandas dataframe',
                        default=None)
    parser.add_argument('--test_file', type=str,
                        help='The test pandas dataframe',
                        default=None)
    parser.add_argument('--metadata', type=str, help='The metadata of the problem.',
                        default=None)
    parser.add_argument('--batch_size', type=int, help='Total batch_size', default=None)
    parser.add_argument('--num_accumulated', type=int, help='Num of gradient accumulation',
                        default=None)
    parser.add_argument('--backbone_name', type=str, help='Name of the backbone model',
                        default=None)
    parser.add_argument('--seed', type=int, help='The seed',
                        default=None)
    parser.add_argument('--task', type=str,
                        help='The default tasks',
                        default=None)
    parser.add_argument('--save_dir', type=str,
                        help='The directory to save the experiment',
                        default=None)
    parser.add_argument('--eval_metrics', type=str,
                        help='The metrics for evaluating the models.',
                        default=None)
    parser.add_argument('--stop_metric', type=str,
                        help='The metrics for early stopping',
                        default=None)
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to evaluate the model')
    parser.add_argument('--exp_dir', type=str, default='output',
                        help='The experiment directory where the model params will be written.')
    parser.add_argument('--ctx', type=str, default='gpu0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args()
    return args


def train(args):
    cfg = BertForTabularPredictionBasic.get_cfg()
    if args.task is not None:
        feature_columns, label_columns, stop_metric, eval_metrics = TASKS[args.task]
    else:
        raise NotImplementedError
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    all_columns = feature_columns + [label_columns]
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.LEARNING.stop_metric = stop_metric
    cfg.LEARNING.log_metrics = ','.join(eval_metrics)
    if args.batch_size is not None:
        cfg.OPTIMIZATION.batch_size = args.batch_size
    if args.exp_dir is not None:
        cfg.MISC.exp_dir = args.exp_dir
    if args.ctx is not None:
        cfg.MISC.context = args.ctx
    cfg.freeze()
    train_data = load_pandas_df(args.train_file)
    dev_data = load_pandas_df(args.dev_file)
    test_data = load_pandas_df(args.test_file)
    train_data = train_data[all_columns]
    dev_data = dev_data[all_columns]
    test_data = test_data[feature_columns]
    model = BertForTabularPredictionBasic(cfg)
    model.fit(train_data=train_data, label=label_columns)
    dev_metrics_scores = model.evaluate(dev_data, metrics=eval_metrics)
    with open(os.path.join(cfg.MISC.exp_dir, 'final_model_dev_score.json'), 'w') as of:
        json.dump(of, dev_metrics_scores)
    test_prediction = model.predict(test_data)
    np.savetxt(os.path.join(cfg.MISC.exp_dir, 'test_predictions.txt'),
               test_prediction)


def predict(args):
    raise NotImplementedError


if __name__ == '__main__':
    args = parse_args()
    if args.do_train:
        train(args)
    if args.do_eval:
        predict(args)
