import pickle
import mxnet as mx
import itertools
from mxnet.lr_scheduler import PolyScheduler, CosineScheduler
from gluonnlp.lr_scheduler import InverseSquareRootScheduler
from mxnet.gluon.data import DataLoader
from gluonnlp.auto.dataset import TabularDataset
from gluonnlp.auto.preprocessing import TabularClassificationBERTPreprocessor, infer_problem_type
from gluonnlp.auto.models.classification import BERTForTabularClassificationV1
from gluonnlp.models import get_backbone
from gluonnlp.utils.config import CfgNode
from gluonnlp.utils.misc import parse_ctx, set_seed, grouper, infinite_loop
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
        cfg.layerwise_lr_decay = 0.9  # The layer_wise decay
        cfg.wd = 0.01  # Weight Decay
        cfg.max_grad_norm = 1.0 # Maximum Gradient Norm
        # The validation frequency = validation frequency * num_updates_in_an_epoch
        cfg.valid_frequency = 0.5
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
        cfg.VERSION = 1
        cfg.SEED = 123
        cfg.OPTIMIZATION = OptimizationV1Config.get_cfg()
        cfg.MODEL = TabularModelV1Config.get_cfg()
        return cfg


def get_optimizer(optimization_cfg, updates_per_epoch):
    max_update = int(updates_per_epoch * optimization_cfg.num_train_epochs)
    warmup_steps = int(updates_per_epoch * optimization_cfg.num_train_epochs
                       * optimization_cfg.warmup_portion)
    if optimization_cfg.lr_scheduler == 'poly_scheduler':
        assert warmup_steps < max_update
        lr_scheduler = PolyScheduler(max_update=max_update,
                                     base_lr=optimization_cfg.lr,
                                     warmup_begin_lr=optimization_cfg.begin_lr,
                                     pwr=1,
                                     final_lr=optimization_cfg.final_lr,
                                     warmup_steps=warmup_steps,
                                     warmup_mode='linear')
    elif optimization_cfg.lr_scheduler == 'inv_sqrt':
        warmup_steps = int(updates_per_epoch * optimization_cfg.num_train_epochs
                           * optimization_cfg.warmup_portion)
        lr_scheduler = InverseSquareRootScheduler(warmup_steps=warmup_steps,
                                                  base_lr=optimization_cfg.lr,
                                                  warmup_init_lr=optimization_cfg.begin_lr)
    elif optimization_cfg.lr_scheduler == 'constant':
        lr_scheduler = None
    elif optimization_cfg.lr_scheduler == 'cosine':
        max_update = int(updates_per_epoch * optimization_cfg.num_train_epochs)
        warmup_steps = int(updates_per_epoch * optimization_cfg.num_train_epochs
                           * optimization_cfg.warmup_portion)
        assert warmup_steps < max_update
        lr_scheduler = CosineScheduler(max_update=max_update,
                                       base_lr=optimization_cfg.lr,
                                       final_lr=optimization_cfg.final_lr,
                                       warmup_steps=warmup_steps,
                                       warmup_begin_lr=optimization_cfg.begin_lr)
    else:
        raise ValueError('Unsupported lr_scheduler="{}"'
                         .format(optimization_cfg.lr_scheduler))
    optimizer_params = {'learning_rate': args.lr,
                        'wd': args.wd,
                        'lr_scheduler': lr_scheduler}
    optimizer = optimization_cfg.optimizer
    additional_params = {key: value for key, value in optimization_cfg.optimizer_params}
    optimizer_params.update(additional_params)
    return optimizer, optimizer_params, max_update


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


def train(args):
    ctx_l = parse_ctx(args.gpus)
    all_cfg = Config.get_cfg()
    if args.config_file is not None:
        all_cfg = all_cfg.clone_merge(args.config_file)
    set_seed(all_cfg.SEED)
    optimization_cfg = all_cfg.OPTIMIZATION
    model_cfg = all_cfg.MODEL
    backbone_model_cls, backbone_cfg, tokenizer, backbone_params_path\
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
    assert len(label_columns) == 1, 'Multi-label classification is currently not supported!'
    train_dataset = TabularDataset(train_df, columns=feature_columns + label_columns)
    dev_dataset = TabularDataset(dev_df, columns=feature_columns + label_columns,
                                 column_properties=train_dataset.column_properties)
    test_dataset = TabularDataset(test_df, columns=feature_columns,
                                  column_properties=train_dataset.column_properties)
    column_properties = train_dataset.column_properties

    # Build Preprocessor + Preprocess the training dataset + Inference problem type
    preprocessor = TabularClassificationBERTPreprocessor(tokenizer=tokenizer,
                                                         column_properties=column_properties,
                                                         label_columns=label_columns,
                                                         max_length=model_cfg.BACKBONE.max_length)
    processed_train = preprocessor.process_train(train_dataset.table)
    processed_dev = preprocessor.process_train(dev_dataset.table)
    processed_test = preprocessor.process_test(test_dataset.table)
    problem_type, label_shape = infer_problem_type(label_columns[0])

    batch_size = optimization_cfg.batch_size // len(ctx_l) // optimization_cfg.num_accumulated
    inference_batch_size = batch_size * optimization_cfg.val_batch_size_mult
    assert batch_size * optimization_cfg.num_accumulated * len(ctx_l) == optimization_cfg.batch_size
    train_dataloader = DataLoader(processed_train, batch_size=batch_size,
                                  shuffle=True, batchify_fn=preprocessor.batchify(is_test=False))
    dev_dataloader = DataLoader(processed_dev, batch_size=inference_batch_size,
                                shuffle=False, batchify_fn=preprocessor.batchify(is_test=False))
    test_dataloader = DataLoader(processed_test, batch_size=inference_batch_size,
                                 shuffle=False, batchify_fn=preprocessor.batchify(is_test=True))
    # Build the network

    net = BERTForTabularClassificationV1(text_backbone=text_backbone,
                                         feature_field_info=preprocessor.feature_field_info(),
                                         label_shape=label_shape,
                                         cfg=model_cfg.TABULAR_CLASSIFICATION)
    net.text_backbone.load_parameters(backbone_params_path, ctx=ctx_l)
    net.initialize(ctx=ctx_l)
    net.hybridize()

    # Initialize the optimizer
    updates_per_epoch = len(train_dataloader) * optimization_cfg.num_accumulated
    optimizer, optimizer_params, max_update = get_optimizer(optimization_cfg,
                                                            updates_per_epoch=updates_per_epoch)
    valid_every_k_update = int(optimization_cfg.valid_frequency * updates_per_epoch)
    trainer = mx.gluon.Trainer(net.collect_params(),
                               optimizer, optimizer_params,
                               update_on_kvstore=False)
    if optimization_cfg.layerwise_lr_decay > 0:
        apply_layerwise_decay(net.text_backbone, optimization_cfg.layerwise_lr_decay)
    # Do not apply weight decay to all the LayerNorm and bias
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    train_infinite_iter = grouper(itertools.cycle(train_dataloader), len(ctx_l))
    is_last_batch = False
    update_count = 0
    while True:
        try:
            sample_data_l = next(train_multi_data_loader)
        except StopIteration:
            train_multi_data_loader = grouper(train_dataloader, len(ctx_l))
            sample_data_l = next(train_multi_data_loader)
        for sample_data, ctx in zip(sample_data_l, )


def predict(args):
    assert args.test_file is not None
    test_df = pd.read_pickle(args.test_file)


if __name__ == '__main__':
    args = parse_args()
    if args.do_train:
        train(args)
    if args.do_eval:
        predict(args)
