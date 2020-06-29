import pickle
import time
import mxnet as mx
import os
import math
import logging
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from scipy.stats import pearsonr
from mxnet.lr_scheduler import PolyScheduler, CosineScheduler
from gluonnlp.lr_scheduler import InverseSquareRootScheduler
from mxnet.gluon.data import DataLoader
from gluonnlp.auto import constants as _C
from gluonnlp.auto.dataset import TabularDataset
from gluonnlp.auto.preprocessing import TabularClassificationBERTPreprocessor, infer_problem_type
from gluonnlp.auto.models.classification import BERTForTabularClassificationV1
from gluonnlp.models import get_backbone
from gluonnlp.utils.config import CfgNode
from gluonnlp.utils.misc import parse_ctx, set_seed, grouper, repeat, logging_config
from gluonnlp.utils.parameter import move_to_ctx, clip_grad_global_norm
import pandas as pd
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
        cfg.optimizer = 'adamw'
        cfg.optimizer_params = [('beta1', 0.9), ('beta2', 0.999), ('epsilon', 1e-6),
                                ('correct_bias', False)]
        cfg.begin_lr = 0.0
        cfg.batch_size = 32
        cfg.num_accumulated = 1
        cfg.val_batch_size_mult = 2  # By default, we double the batch size for validation
        cfg.lr = 1E-4
        cfg.final_lr = 0.0
        cfg.num_train_epochs = 3.0
        cfg.warmup_portion = 0.1
        cfg.layerwise_lr_decay = 0.8  # The layer_wise decay
        cfg.wd = 0.01  # Weight Decay
        cfg.max_grad_norm = 1.0  # Maximum Gradient Norm
        # The validation frequency = validation frequency * num_updates_in_an_epoch
        cfg.valid_frequency = 0.2
        # Logging frequency = log frequency * num_updates_in_an_epoch
        cfg.log_frequency = 0.05
        cfg.stop_metrics = []
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


class TaskConfig:
    @staticmethod
    def get_cfg():
        cfg = CfgNode()
        cfg.train_file = ''
        cfg.dev_file = ''
        cfg.test_file = ''
        cfg.metadata = ''
        cfg.eval_metric = ''


class Config:
    @staticmethod
    def get_cfg():
        cfg = CfgNode()
        cfg.VERSION = 1
        cfg.SEED = 123
        cfg.OPTIMIZATION = OptimizationV1Config.get_cfg()
        cfg.MODEL = TabularModelV1Config.get_cfg()
        return cfg


def get_optimizer(cfg, updates_per_epoch):
    max_update = int(updates_per_epoch * cfg.num_train_epochs)
    warmup_steps = int(updates_per_epoch * cfg.num_train_epochs
                       * cfg.warmup_portion)
    if cfg.lr_scheduler == 'poly_scheduler':
        assert warmup_steps < max_update
        lr_scheduler = PolyScheduler(max_update=max_update,
                                     base_lr=cfg.lr,
                                     warmup_begin_lr=cfg.begin_lr,
                                     pwr=1,
                                     final_lr=cfg.final_lr,
                                     warmup_steps=warmup_steps,
                                     warmup_mode='linear')
    elif cfg.lr_scheduler == 'inv_sqrt':
        warmup_steps = int(updates_per_epoch * cfg.num_train_epochs
                           * cfg.warmup_portion)
        lr_scheduler = InverseSquareRootScheduler(warmup_steps=warmup_steps,
                                                  base_lr=cfg.lr,
                                                  warmup_init_lr=cfg.begin_lr)
    elif cfg.lr_scheduler == 'constant':
        lr_scheduler = None
    elif cfg.lr_scheduler == 'cosine':
        max_update = int(updates_per_epoch * cfg.num_train_epochs)
        warmup_steps = int(updates_per_epoch * cfg.num_train_epochs
                           * cfg.warmup_portion)
        assert warmup_steps < max_update
        lr_scheduler = CosineScheduler(max_update=max_update,
                                       base_lr=cfg.lr,
                                       final_lr=cfg.final_lr,
                                       warmup_steps=warmup_steps,
                                       warmup_begin_lr=cfg.begin_lr)
    else:
        raise ValueError('Unsupported lr_scheduler="{}"'
                         .format(cfg.lr_scheduler))
    optimizer_params = {'learning_rate': cfg.lr,
                        'wd': cfg.wd,
                        'lr_scheduler': lr_scheduler}
    optimizer = cfg.optimizer
    additional_params = {key: value for key, value in cfg.optimizer_params}
    optimizer_params.update(additional_params)
    return optimizer, optimizer_params, max_update


def parse_args():
    parser = argparse.ArgumentParser(description='Text classification example.')
    parser.add_argument('--train_file', type=str,
                        help='The training pandas dataframe.',
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
    parser.add_argument('--dev_file', type=str,
                        help='The validation pandas dataframe',
                        default=None)
    parser.add_argument('--test_file', type=str,
                        help='The test pandas dataframe',
                        default=None)
    parser.add_argument('--task', type=str,
                        help='The default tasks',
                        default=None)
    parser.add_argument('--save_dir', type=str,
                        help='The directory to save the experiment',
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


# TODO(sxjscience) Move to the backbone models.
def apply_layerwise_decay(model, layerwise_decay, not_included=None):
    """Apply the layer-wise gradient decay
    .. math::
        lr = lr * layerwise_decay^(max_depth - layer_depth)

    Parameters:
    ----------
    model
        qa_net
    layerwise_decay: int
        layer-wise decay power
    not_included: list of str
        A list or parameter names that not included in the layer-wise decay
    """
    if not_included is None:
        not_included = []
    # consider the task specific fine-tuning layer as the last layer, following with pooler
    # In addition, the embedding parameters have the smaller learning rate based on this setting.
    all_layers = model.encoder.all_encoder_layers
    max_depth = len(all_layers)
    if 'pool' in model.collect_params().keys():
        max_depth += 1
    for key, value in model.collect_params().items():
        if 'scores' in key:
            value.lr_mult = layerwise_decay**(0)
        if 'pool' in key:
            value.lr_mult = layerwise_decay**(1)
        if 'embed' in key:
            value.lr_mult = layerwise_decay**(max_depth + 1)

    for (layer_depth, layer) in enumerate(all_layers):
        layer_params = layer.collect_params()
        for key, value in layer_params.items():
            for pn in not_included:
                if pn in key:
                    continue
            value.lr_mult = layerwise_decay**(max_depth - layer_depth)


def validate(net, dataloader, ctx_l, problem_type, eval_metrics=None, pos_label=1):
    """

    Parameters
    ----------
    net
    dataloader
    ctx_l
    problem_type
    eval_metrics
        The evaluation metrics
    pos_label
        Will only be used if it's F1 score

    Returns
    -------
    predictions
    gt_label
    metric_scores
    """
    predictions = []
    gt_labels = []
    metric_scores = dict()
    for sample_l in grouper(dataloader, len(ctx_l)):
        iter_pred_l = []
        iter_label_l = []
        for sample, ctx in zip(sample_l, ctx_l):
            if sample is None:
                continue
            batch_feature, batch_label = sample
            iter_label_l.append(batch_label[0])
            batch_feature = move_to_ctx(batch_feature, ctx)
            pred = net(batch_feature)
            if problem_type == _C.CLASSIFICATION:
                pred = mx.npx.softmax(pred, axis=-1)
            iter_pred_l.append(pred)
        for pred in iter_pred_l:
            predictions.append(pred.asnumpy())
        for label in iter_label_l:
            gt_labels.append(label.asnumpy())
    predictions = np.concatenate(predictions, axis=0)
    gt_labels = np.concatenate(gt_labels, axis=0)
    for metric_name in eval_metrics:
        if metric_name == 'acc':
            metric_scores[metric_name] = accuracy_score(gt_labels, predictions.argmax(axis=-1))
        elif metric_name == 'f1':
            metric_scores[metric_name] = f1_score(gt_labels, predictions.argmax(axis=-1),
                                                  pos_label=pos_label)
        elif metric_name == 'mcc':
            metric_scores[metric_name] = matthews_corrcoef(gt_labels, predictions.argmax(axis=-1))
        elif metric_name == 'auc':
            metric_scores[metric_name] = roc_auc_score(gt_labels, predictions[:, pos_label])
        elif metric_name == 'nll':
            metric_scores[metric_name] = - np.log(predictions[np.arange(gt_labels.shape[0]),
                                                              gt_labels]).mean()
        elif metric_name == 'pearsonr':
            metric_scores[metric_name] = pearsonr(gt_labels, predictions)
        elif metric_name == 'mse':
            metric_scores[metric_name] = np.square(predictions - gt_labels).mean()
        else:
            raise ValueError('Unknown metric = {}'.format(metric_name))
    return predictions, gt_labels, metric_scores


def train(args):
    ctx_l = parse_ctx(args.gpus)
    all_cfg = Config.get_cfg()
    if args.config_file is not None:
        all_cfg = all_cfg.clone_merge(args.config_file)
    all_cfg.defrost()
    if args.batch_size is not None:
        all_cfg.OPTIMIZATION.batch_size = args.batch_size
    if args.num_accumulated is not None:
        all_cfg.OPTIMIZATION.num_accumulated = args.num_accumulated
    if args.backbone_name is not None:
        all_cfg.MODEL.BACKBONE.name = args.backbone_name
    if args.seed is not None:
        all_cfg.SEED = args.seed
    all_cfg.freeze()
    if args.save_dir is None:
        args.save_dir = '{}_{}'.format(args.task, all_cfg.MODEL.BACKBONE.name)

    logging_config(args.save_dir, name='text_classification')
    set_seed(all_cfg.SEED)
    optimization_cfg = all_cfg.OPTIMIZATION
    model_cfg = all_cfg.MODEL
    backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path, _\
        = get_backbone(model_cfg.BACKBONE.name)
    with open(os.path.join(args.save_dir, 'cfg.yml'), 'w') as f:
        f.write(all_cfg.dump())
    with open(os.path.join(args.save_dir, 'backbone_cfg.yml'), 'w') as f:
        f.write(backbone_cfg.dump())
    text_backbone = backbone_model_cls.from_cfg(backbone_cfg)
    # Load train and dev dataset
    train_df = pd.read_pickle(args.train_file)
    dev_df = pd.read_pickle(args.dev_file)
    test_df = pd.read_pickle(args.test_file)
    feature_columns, label_columns = TASKS[args.task]
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    if isinstance(label_columns, str):
        label_columns = [label_columns]
    assert len(label_columns) == 1, 'Multi-label classification is currently not supported!'
    train_dataset = TabularDataset(train_df, columns=feature_columns + label_columns)
    dev_dataset = TabularDataset(dev_df, columns=feature_columns + label_columns,
                                 column_properties=train_dataset.column_properties)
    test_dataset = TabularDataset(test_df, columns=feature_columns,
                                  column_properties=train_dataset.column_properties)

    logging.info('Train Dataset:')
    logging.info(train_dataset)
    logging.info('Dev Dataset:')
    logging.info(dev_dataset)
    logging.info('Test Dataset:')
    logging.info(test_dataset)
    column_properties = train_dataset.column_properties
    label_column_property = column_properties[label_columns[0]]
    # Build Preprocessor + Preprocess the training dataset + Inference problem type
    preprocessor = TabularClassificationBERTPreprocessor(tokenizer=tokenizer,
                                                         column_properties=column_properties,
                                                         label_columns=label_columns,
                                                         max_length=model_cfg.BACKBONE.max_length)
    processed_train = preprocessor.process_train(train_dataset.table)
    processed_dev = preprocessor.process_train(dev_dataset.table)
    processed_test = preprocessor.process_test(test_dataset.table)
    problem_type, label_shape = infer_problem_type(label_column_property)

    batch_size = optimization_cfg.batch_size // len(ctx_l) // optimization_cfg.num_accumulated
    inference_batch_size = batch_size * optimization_cfg.val_batch_size_mult
    assert batch_size * optimization_cfg.num_accumulated * len(ctx_l) == optimization_cfg.batch_size
    train_dataloader = DataLoader(processed_train, batch_size=batch_size,
                                  shuffle=True, batchify_fn=preprocessor.batchify(is_test=False))
    dev_dataloader = DataLoader(processed_dev, batch_size=inference_batch_size,
                                shuffle=False, batchify_fn=preprocessor.batchify(is_test=False))
    test_dataloader = DataLoader(processed_test, batch_size=inference_batch_size,
                                 shuffle=False, batchify_fn=preprocessor.batchify(is_test=True))
    # Get the evaluation metrics
    if args.eval_metrics is None:
        if problem_type == _C.CLASSIFICATION:
            eval_metrics = ['acc', 'f1', 'mcc', 'auc', 'nll']
        elif problem_type == _C.REGRESSION:
            eval_metrics = ['mse']
        else:
            raise NotImplementedError
    else:
        eval_metrics = args.eval_metrics

    # Build the network

    net = BERTForTabularClassificationV1(text_backbone=text_backbone,
                                         feature_field_info=preprocessor.feature_field_info(),
                                         label_shape=label_shape,
                                         cfg=model_cfg.TABULAR_CLASSIFICATION)
    net.text_backbone.load_parameters(backbone_params_path, ctx=ctx_l)
    net.initialize(ctx=ctx_l)
    net.hybridize()

    # Initialize the optimizer
    updates_per_epoch = int(len(train_dataloader) / (optimization_cfg.num_accumulated * len(ctx_l)))
    optimizer, optimizer_params, max_update = get_optimizer(optimization_cfg,
                                                            updates_per_epoch=updates_per_epoch)
    valid_interval = math.ceil(optimization_cfg.valid_frequency * updates_per_epoch)
    train_log_interval = math.ceil(optimization_cfg.log_frequency * updates_per_epoch)
    trainer = mx.gluon.Trainer(net.collect_params(),
                               optimizer, optimizer_params,
                               update_on_kvstore=False)
    if optimization_cfg.layerwise_lr_decay > 0:
        apply_layerwise_decay(net.text_backbone, optimization_cfg.layerwise_lr_decay)
    # Do not apply weight decay to all the LayerNorm and bias
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [p for p in net.collect_params().values() if p.grad_req != 'null']
    # Set grad_req if gradient accumulation is required
    if optimization_cfg.num_accumulated > 1:
        logging.info('Using gradient accumulation. Global batch size = {}'
                     .format(optimization_cfg.batch_size))
        for p in params:
            p.grad_req = 'add'
        net.collect_params().zero_grad()
    train_loop_dataloader = grouper(repeat(train_dataloader), len(ctx_l))
    log_loss_l = [mx.np.array(0.0, dtype=np.float32, ctx=ctx) for ctx in ctx_l]
    log_num_samples_l = [0 for _ in ctx_l]
    logging_start_tick = time.time()
    for update_idx in range(max_update):
        num_samples_per_update_l = [0 for _ in ctx_l]
        for accum_idx in range(optimization_cfg.num_accumulated):
            sample_l = next(train_loop_dataloader)
            loss_l = []
            for i, (sample, ctx) in enumerate(zip(sample_l, ctx_l)):
                feature_batch, label_batch = sample
                feature_batch = move_to_ctx(feature_batch, ctx)
                label_batch = move_to_ctx(label_batch, ctx)
                with mx.autograd.record():
                    pred = net(feature_batch)
                    if problem_type == _C.CLASSIFICATION:
                        logits = mx.npx.log_softmax(pred, axis=-1)
                        loss = - mx.npx.pick(logits, label_batch[0])
                    elif problem_type == _C.REGRESSION:
                        loss = mx.np.square(pred - label_batch[0])
                    loss_l.append(loss.sum() / batch_size)
                    num_samples_per_update_l[i] += np.prod(loss.shape)
            for loss in loss_l:
                loss.backward()
            for i in range(len(ctx_l)):
                log_loss_l[i] += loss_l[i]
                log_num_samples_l[i] += num_samples_per_update_l[i]
        # Begin to update
        trainer.allreduce_grads()
        # Here, the accumulated gradients are
        # \sum_{n=1}^N g_n / batch_size
        # Thus, in order to clip the average gradient
        #   \frac{1}{N} \sum_{n=1}^N      -->  clip to args.max_grad_norm
        # We need to change the ratio to be
        #  \sum_{n=1}^N g_n / batch_size  -->  clip to args.max_grad_norm  * N / batch_size
        num_samples_per_update = sum(num_samples_per_update_l)
        total_norm, ratio, is_finite =\
            clip_grad_global_norm(params,
                                  optimization_cfg.max_grad_norm * num_samples_per_update / batch_size)
        total_norm = total_norm / (num_samples_per_update / batch_size)
        trainer.update(num_samples_per_update / batch_size)

        # Clear after update
        if optimization_cfg.num_accumulated > 1:
            net.collect_params().zero_grad()
        if (update_idx + 1) % train_log_interval == 0:
            log_loss = sum([ele.as_in_ctx(ctx_l[0]) for ele in log_loss_l]).asnumpy()
            log_num_samples = sum(log_num_samples_l)
            avg_log_loss = log_loss / log_num_samples * batch_size
            logging.info('[Iter {}/{}, Epoch {}] train loss={}, gnorm={}, lr={}, #samples processed={},'
                         ' #sample per second={}'
                         .format(update_idx, max_update, int(update_idx / updates_per_epoch),
                                 avg_log_loss, total_norm, trainer.learning_rate,
                                 log_num_samples,
                                 log_num_samples / (time.time() - logging_start_tick)))
            logging_start_tick = time.time()
            log_loss_l = [mx.np.array(0.0, dtype=np.float32, ctx=ctx) for ctx in ctx_l]
            log_num_samples_l = [0 for _ in ctx_l]
        if (update_idx + 1) % valid_interval == 0:
            valid_start_tick = time.time()
            predictions, gt_labels, metric_scores = validate(net, dataloader=dev_dataloader,
                                                             ctx_l=ctx_l,
                                                             problem_type=problem_type)
            valid_time_spent = time.time() - valid_start_tick
            np.savez_compressed('iter{}_prediction.npz'.format(update_idx),
                                predictions=predictions, labels=gt_labels)
            loss_string = ''
            for i, key in enumerate(sorted(metric_scores.keys())):
                if i < len(metric_scores) - 1:
                    loss_string += '{}={}, '.format(key, metric_scores[key])
                else:
                    loss_string += '{}={}'.format(key, metric_scores[key])
            logging.info('[Iter {}/{}, Epoch {}] valid {}, time spent={}'.format(
                update_idx, max_update, int(update_idx / updates_per_epoch),
                loss_string, valid_time_spent))


def predict(args):
    raise NotImplementedError


if __name__ == '__main__':
    args = parse_args()
    if args.do_train:
        train(args)
    if args.do_eval:
        predict(args)
