import mxnet as mx
import os
import json
import argparse
from gluonnlp.auto.tasks import AutoNLP
mx.npx.set_np()


TASKS = \
    {'cola': ('sentence', 'label', 'mcc', ['mcc']),
     'sst': ('sentence', 'label', 'acc', ['acc']),
     'mrpc': (['sentence1', 'sentence2'], 'label', 'f1', ['acc', 'f1']),
     'sts': (['sentence1', 'sentence2'], 'score', 'spearmanr', ['pearsonr', 'spearmanr']),
     'qqp': (['sentence1', 'sentence2'], 'label', 'f1', ['acc', 'f1']),
     'mnli': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'qnli': (['question', 'sentence'], 'label', 'acc', ['acc']),
     'rte': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'wnli': (['sentence1', 'sentence2'], 'label', 'acc', ['acc']),
     'snli': (['sentence1', 'sentence2'], 'label', 'acc', ['acc'])}


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
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='The experiment directory where the model params will be written.')
    parser.add_argument('--ctx', type=str, default='gpu0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    parser.add_argument('--config_file', type=str, default=None)
    args = parser.parse_args()
    return args


def train(args):
    if args.task is not None:
        feature_columns, label_columns, stop_metric, eval_metrics = TASKS[args.task]
    else:
        raise NotImplementedError
    if args.backbone_name is None:
        args.backbone_name = 'google_electra_base'
    if args.exp_dir is None:
        args.exp_dir = '{}_{}'.format(args.task, args.backbone_name)
    model = AutoNLP.fit(train_data=args.train_file,
                        feature_columns=feature_columns,
                        label=label_columns,
                        exp_dir=args.exp_dir,
                        stop_metric=stop_metric,
                        eval_metrics=eval_metrics)
    dev_metrics_scores = model.evaluate(args.dev_file, metrics=eval_metrics)
    with open(os.path.join(args.exp_dir, 'final_model_dev_score.json'), 'w') as of:
        json.dump(dev_metrics_scores, of)
    dev_prediction = model.predict(args.dev_file)
    test_prediction = model.predict(args.test_file)
    with open(os.path.join(args.exp_dir, 'dev_predictions.txt'), 'w') as of:
        for ele in dev_prediction:
            of.write(str(ele) + '\n')
    with open(os.path.join(args.exp_dir, 'test_predictions.txt'), 'w') as of:
        for ele in test_prediction:
            of.write(str(ele) + '\n')
    model.save(os.path.join(args.exp_dir, 'saved_model'))


def predict(args):
    raise NotImplementedError


if __name__ == '__main__':
    args = parse_args()
    if args.do_train:
        train(args)
    if args.do_eval:
        predict(args)
