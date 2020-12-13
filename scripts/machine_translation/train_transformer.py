"""
Transformer
=================================

This example shows how to implement the Transformer model with GluonNLP Toolkit.

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones,
          Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6000--6010},
  year={2017}
}
"""

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import time
import random
import os
import logging
import itertools
import math
import numpy as np
import mxnet as mx
from mxnet import gluon
from gluonnlp.models.transformer import TransformerModel
from gluonnlp.utils.misc import logging_config, AverageSGDTracker, count_parameters,\
    md5sum, grouper, init_comm, repeat
from gluonnlp.data.sampler import (
    ConstWidthBucket,
    LinearWidthBucket,
    ExpWidthBucket,
    FixedBucketSampler,
    BoundedBudgetSampler,
    ShardedIterator
)
from tensorboardX import SummaryWriter
import gluonnlp.data.batchify as bf
from gluonnlp.data import Vocab
from gluonnlp.data import tokenizers
from gluonnlp.data.tokenizers import BaseTokenizerWithVocab, huggingface
from gluonnlp.lr_scheduler import InverseSquareRootScheduler
from gluonnlp.loss import LabelSmoothCrossEntropyLoss
from gluonnlp.utils.parameter import grad_global_norm, clip_grad_global_norm


try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

mx.npx.set_np()


CACHE_PATH = os.path.realpath(os.path.join(os.path.realpath(__file__), '..', 'cached'))
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH, exist_ok=True)


def get_parser():
    parser = argparse.ArgumentParser(description='Transformer for Neural Machine Translation.')
    parser.add_argument('--train_src_corpus', type=str,
                        help='The source training corpus.')
    parser.add_argument('--train_tgt_corpus', type=str,
                        help='The target training corpus.')
    parser.add_argument('--dev_src_corpus', type=str,
                        help='The source dev corpus.')
    parser.add_argument('--dev_tgt_corpus', type=str,
                        help='The target dev corpus.')
    parser.add_argument('--src_tokenizer', choices=['spm',
                                                    'subword_nmt',
                                                    'yttm',
                                                    'hf_bytebpe',
                                                    'hf_wordpiece',
                                                    'hf_bpe',
                                                    'whitespace'],
                        default='whitespace', type=str,
                        help='The source tokenizer. '
                             'Whitespace tokenizer supports processing pre-encoded corpus, '
                             'and the tokenizers besides whitespace supports online encoding.')
    parser.add_argument('--tgt_tokenizer', choices=['spm',
                                                    'subword_nmt',
                                                    'yttm',
                                                    'hf_bytebpe',
                                                    'hf_wordpiece',
                                                    'hf_bpe',
                                                    'whitespace'],
                        default='whitespace', type=str,
                        help='The target tokenizer.')
    parser.add_argument('--src_subword_model_path', type=str,
                        help='Path to the source subword model.')
    parser.add_argument('--src_vocab_path', type=str,
                        help='Path to the source vocab.')
    parser.add_argument('--tgt_subword_model_path', type=str,
                        help='Path to the target subword model.')
    parser.add_argument('--tgt_vocab_path', type=str,
                        help='Path to the target vocab.')
    parser.add_argument('--seed', type=int, default=100, help='The random seed.')
    parser.add_argument('--epochs', type=int, default=30, help='Upper epoch limit, '
                        'the model will keep training when epochs < 0 and max_update < 0.')
    parser.add_argument('--max_update', type=int, default=-1,
                        help='Max update steps, when max_update > 0, epochs will be set to -1, '
                             'each update step contains gpu_num * num_accumulated batches.')
    parser.add_argument('--save_interval_update', type=int, default=500,
                         help='Update interval of saving checkpoints while using max_update.')
    parser.add_argument('--cfg', type=str, default='transformer_base',
                        help='Configuration of the transformer model. '
                             'You may select a yml file or use the prebuild configurations.')
    parser.add_argument('--label_smooth_alpha', type=float, default=0.1,
                        help='Weight of label smoothing')
    parser.add_argument('--sampler', type=str, choices=['BoundedBudgetSampler',
                                                        'FixedBucketSampler'],
                        default='FixedBucketSampler', help='Type of sampler')
    parser.add_argument('--batch_size', type=int, default=2700,
                        help='Batch size. Number of tokens per gpu in a minibatch.')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='Batch size for evaluation.')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Max gradient norm. When not specified, the gradient '
                             'clipping will not be used.')
    parser.add_argument('--num_buckets', type=int, default=20, help='Bucket number.')
    parser.add_argument('--bucket_scheme', type=str, default='exp',
                        help='Strategy for generating bucket keys. It supports: '
                             '"constant": all the buckets have the same width; '
                             '"linear": the width of bucket increases linearly; '
                             '"exp": the width of bucket increases exponentially')
    parser.add_argument('--bucket_ratio', type=float, default=0.0,
                        help='Ratio for increasing the throughput of the bucketing')
    parser.add_argument('--max_num_tokens', type=int, default=-1,
                        help='max tokens num of each batch, applicable while using BoundedBudgetSampler')
    parser.add_argument('--max_num_sentences', type=int, default=-1,
                        help='max sentences num of each batch, applicable while using BoundedBudgetSampler')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='The learning rate at the end of the warmup stage. '
                             'If it is not given, we will use the formula suggested in the '
                             'original Transformer paper:'
                             ' 1.0 / sqrt(d_model) / sqrt(warmup_steps). '
                             'Otherwise, we will use the given lr as the final learning rate in '
                             'the warmup phase.')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='number of warmup steps used in NOAM\'s stepsize schedule')
    parser.add_argument('--warmup_init_lr', type=float, default=0.0,
                        help='Initial learning rate at the beginning of the warm-up stage')
    parser.add_argument('--num_accumulated', type=int, default=32,
                        help='Number of steps to accumulate the gradients. '
                             'This is useful to mimic large batch training with limited gpu memory')
    parser.add_argument('--magnitude', type=float, default=3.0,
                        help='Magnitude of Xavier initialization')
    parser.add_argument('--num_averages', type=int, default=-1,
                        help='Perform final testing based on the '
                             'average of last num_averages checkpoints. '
                             'Use num_average will cause extra gpu memory usage.')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--save_dir', type=str, default='transformer_out',
                        help='directory path to save the final model and training log')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use dtype float16')
    parser.add_argument('--comm_backend', type=str, default='device',
                        choices=['horovod', 'dist_sync_device', 'device'],
                        help='Communication backend.')
    parser.add_argument('--gpus', type=str,
                        help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
    return parser


def validation(model, data_loader, ctx_l):
    """Validate the model on the dataset

    Parameters
    ----------
    model : TransformerModel
        The transformer model
    data_loader : DataLoader
        DataLoader
    ctx_l : list
        List of mx.ctx.Context


    Returns
    -------
    avg_nll_loss : float
        The average negative log-likelihood loss
    """
    avg_nll_loss = mx.np.array(0, dtype=np.float32, ctx=mx.cpu())
    ntokens = 0
    for sample_data_l in grouper(data_loader, len(ctx_l)):
        loss_l = []
        ntokens += sum([ele[3].sum().asnumpy() - ele[0].shape[0] for ele in sample_data_l
                        if ele is not None])
        for sample_data, ctx in zip(sample_data_l, ctx_l):
            if sample_data is None:
                continue
            src_token_ids, tgt_token_ids, src_valid_length, tgt_valid_length, sample_ids = sample_data
            src_token_ids = src_token_ids.as_in_ctx(ctx)
            tgt_token_ids = tgt_token_ids.as_in_ctx(ctx)
            src_valid_length = src_valid_length.as_in_ctx(ctx)
            tgt_valid_length = tgt_valid_length.as_in_ctx(ctx)
            tgt_pred = model(src_token_ids, src_valid_length, tgt_token_ids[:, :-1],
                             tgt_valid_length - 1)
            tgt_labels = tgt_token_ids[:, 1:]
            tgt_pred_logits = mx.npx.log_softmax(tgt_pred, axis=-1)
            nll_loss = - mx.npx.pick(tgt_pred_logits, tgt_labels, axis=-1)
            loss = mx.npx.sequence_mask(nll_loss,
                                        sequence_length=tgt_valid_length - 1,
                                        use_sequence_length=True,
                                        axis=1)
            loss_l.append(loss.sum())
        avg_nll_loss += sum([loss.as_in_ctx(mx.cpu()) for loss in loss_l])
        mx.npx.waitall()
    avg_loss = avg_nll_loss.asnumpy() / ntokens
    return avg_loss


def load_dataset_with_cache(src_corpus_path: str,
                            tgt_corpus_path: str,
                            src_tokenizer: BaseTokenizerWithVocab,
                            tgt_tokenizer: BaseTokenizerWithVocab,
                            overwrite_cache: bool,
                            local_rank: int):
    src_md5sum = md5sum(src_corpus_path)
    tgt_md5sum = md5sum(tgt_corpus_path)
    cache_filepath = os.path.join(CACHE_PATH,
                                  '{}_{}.cache.npz'.format(src_md5sum[:6], tgt_md5sum[:6]))
    if os.path.exists(cache_filepath) and not overwrite_cache:
        if local_rank == 0:
            logging.info('Load cache from {}'.format(cache_filepath))
        npz_data = np.load(cache_filepath, allow_pickle=True)
        src_data, tgt_data = npz_data['src_data'][:], npz_data['tgt_data'][:]
    else:
        assert src_tokenizer.vocab.eos_id is not None,\
            'You will need to add the EOS token to the vocabulary used in the tokenizer of ' \
            'the source language.'
        assert tgt_tokenizer.vocab.bos_id is not None and tgt_tokenizer.vocab.eos_id is not None, \
            'You will need to add both the BOS token and the EOS tokens to the vocabulary used ' \
            'in the tokenizer of the target language.'
        src_data = []
        tgt_data = []
        # TODO(sxjscience) Optimize the speed of converting to cache
        with open(src_corpus_path) as f:
            for line in f:
                sample = np.array(src_tokenizer.encode(line.strip(), output_type=int) +
                                  [src_tokenizer.vocab.eos_id], dtype=np.int32)
                src_data.append(sample)
        with open(tgt_corpus_path) as f:
            for line in f:
                sample = np.array([tgt_tokenizer.vocab.bos_id] +
                                  tgt_tokenizer.encode(line.strip(), output_type=int) +
                                  [tgt_tokenizer.vocab.eos_id], dtype=np.int32)
                tgt_data.append(sample)
        src_data = np.array(src_data)
        tgt_data = np.array(tgt_data)
        np.savez(cache_filepath, src_data=src_data, tgt_data=tgt_data)
    return src_data, tgt_data


def create_tokenizer(tokenizer_type, model_path, vocab_path):
    if tokenizer_type == 'whitespace':
        return tokenizers.create(tokenizer_type, vocab=Vocab.load(vocab_path))
    elif tokenizer_type == 'spm':
        return tokenizers.create(tokenizer_type, model_path=model_path, vocab=vocab_path)
    elif tokenizer_type == 'subword_nmt':
        return tokenizers.create(tokenizer_type, model_path=model_path, vocab=vocab_path)
    elif tokenizer_type == 'yttm':
        return tokenizers.create(tokenizer_type, model_path=model_path)
    elif tokenizer_type in ['hf_bytebpe', 'hf_wordpiece', 'hf_bpe']:
        if huggingface.is_new_version_model_file(model_path):
            return tokenizers.create('hf_tokenizer', model_path=model_path, vocab=vocab_path)
        elif tokenizer_type == 'hf_bytebpe':
            return tokenizers.create(tokenizer_type, merges_file=model_path, vocab_file=vocab_path)
        elif tokenizer_type == 'hf_wordpiece':
            return tokenizers.create(tokenizer_type, vocab_file=vocab_path)
        elif tokenizer_type == 'hf_bpe':
            return tokenizers.create(tokenizer_type, merges_file=model_path, vocab_file=vocab_path)
    else:
        raise NotImplementedError


def train(args):
    _, num_parts, rank, local_rank, _, ctx_l = init_comm(
        args.comm_backend, args.gpus)
    use_amp = args.fp16
    if use_amp:
        from mxnet import amp
    src_tokenizer = create_tokenizer(args.src_tokenizer,
                                     args.src_subword_model_path,
                                     args.src_vocab_path)
    tgt_tokenizer = create_tokenizer(args.tgt_tokenizer,
                                     args.tgt_subword_model_path,
                                     args.tgt_vocab_path)
    src_vocab = src_tokenizer.vocab
    tgt_vocab = tgt_tokenizer.vocab
    train_src_data, train_tgt_data = load_dataset_with_cache(args.train_src_corpus,
                                                             args.train_tgt_corpus,
                                                             src_tokenizer,
                                                             tgt_tokenizer,
                                                             args.overwrite_cache,
                                                             local_rank)
    dev_src_data, dev_tgt_data = load_dataset_with_cache(args.dev_src_corpus,
                                                         args.dev_tgt_corpus,
                                                         src_tokenizer,
                                                         tgt_tokenizer,
                                                         args.overwrite_cache,
                                                         local_rank)
    data_train = gluon.data.SimpleDataset(
        [(src_tokens, tgt_tokens, len(src_tokens), len(tgt_tokens), i)
         for i, (src_tokens, tgt_tokens) in enumerate(zip(train_src_data, train_tgt_data))])
    data_val = gluon.data.SimpleDataset(
        [(src_tokens, tgt_tokens, len(src_tokens), len(tgt_tokens), i)
         for i, (src_tokens, tgt_tokens) in enumerate(zip(dev_src_data, dev_tgt_data))])
    # Construct the model + loss function
    if args.cfg.endswith('.yml'):
        cfg = TransformerModel.get_cfg().clone_merge(args.cfg)
    else:
        cfg = TransformerModel.get_cfg(args.cfg)
    cfg.defrost()
    cfg.MODEL.src_vocab_size = len(src_vocab)
    cfg.MODEL.tgt_vocab_size = len(tgt_vocab)
    cfg.freeze()
    model = TransformerModel.from_cfg(cfg)
    model.initialize(mx.init.Xavier(magnitude=args.magnitude),
                     ctx=ctx_l)
    model.hybridize()
    if local_rank == 0:
        logging.info(model)
    with open(os.path.join(args.save_dir, 'config.yml'), 'w') as cfg_f:
        cfg_f.write(cfg.dump())
    label_smooth_loss = LabelSmoothCrossEntropyLoss(num_labels=len(tgt_vocab),
                                                    alpha=args.label_smooth_alpha,
                                                    from_logits=False)
    label_smooth_loss.hybridize()

    if args.comm_backend == 'horovod':
        hvd.broadcast_parameters(model.collect_params(), root_rank=0)
    
    # Construct the trainer
    if args.lr is None:
        base_lr = 2.0 / math.sqrt(args.num_units) / math.sqrt(args.warmup_steps)
    else:
        base_lr = args.lr
    lr_scheduler = InverseSquareRootScheduler(warmup_steps=args.warmup_steps, base_lr=base_lr,
                                              warmup_init_lr=args.warmup_init_lr)
    optimizer_params = {'learning_rate': args.lr, 'beta1': 0.9,
                        'beta2': 0.98, 'epsilon': 1e-9,
                        'lr_scheduler': lr_scheduler}
    if args.fp16:
        optimizer_params.update({'multi_precision': True})
    if args.comm_backend == 'horovod':
        trainer = hvd.DistributedTrainer(model.collect_params(), 'adam', optimizer_params)
    else:
        trainer = gluon.Trainer(model.collect_params(), 'adam', optimizer_params,
                                update_on_kvstore=False)
    # Load Data
    if args.sampler == 'BoundedBudgetSampler':
        train_batch_sampler = BoundedBudgetSampler(lengths=[(ele[2], ele[3]) for ele in data_train],
                                                   max_num_tokens=args.max_num_tokens,
                                                   max_num_sentences=args.max_num_sentences,
                                                   shuffle=True,
                                                   seed=args.seed)
    elif args.sampler == 'FixedBucketSampler':
        if args.comm_backend == 'horovod':
            raise NotImplementedError('FixedBucketSampler does not support horovod at present')

        if args.bucket_scheme == 'constant':
            bucket_scheme = ConstWidthBucket()
        elif args.bucket_scheme == 'linear':
            bucket_scheme = LinearWidthBucket()
        elif args.bucket_scheme == 'exp':
            bucket_scheme = ExpWidthBucket(bucket_len_step=1.2)
        else:
            raise NotImplementedError
        # TODO(sxjscience) Support auto-bucket-size tuning
        train_batch_sampler = FixedBucketSampler(lengths=[(ele[2], ele[3]) for ele in data_train],
                                                 batch_size=args.batch_size,
                                                 num_buckets=args.num_buckets,
                                                 ratio=args.bucket_ratio,
                                                 shuffle=True,
                                                 use_average_length=True,
                                                 bucket_scheme=bucket_scheme,
                                                 seed=args.seed)
    else:
        raise NotImplementedError

    num_updates_per_epoch = int(math.ceil(len(train_batch_sampler)
                                          / (num_parts * len(ctx_l) * args.num_accumulated)))
    # Convert the batch sampler to multiple shards
    if num_parts > 1:
        train_batch_sampler = ShardedIterator(train_batch_sampler,
                                              num_parts=num_parts,
                                              part_index=rank,
                                              even_size=True,
                                              seed=args.seed + 1000 * rank)

    logging.info(train_batch_sampler)

    batchify_fn = bf.Tuple(bf.Pad(), bf.Pad(), bf.Stack(), bf.Stack(), bf.Stack())
    train_data_loader = gluon.data.DataLoader(data_train,
                                              batch_sampler=train_batch_sampler,
                                              batchify_fn=batchify_fn,
                                              num_workers=0)

    val_data_loader = gluon.data.DataLoader(data_val,
                                            batch_size=args.val_batch_size,
                                            batchify_fn=batchify_fn,
                                            num_workers=0,
                                            shuffle=False)
    for v in model.collect_params().values():
        if v.grad_req != 'null':
            v.grad_req = 'add'
    params = [p for p in model.collect_params().values() if p.grad_req != 'null']
    model_averager = AverageSGDTracker(model.collect_params())
    log_start_time = time.time()
    num_params, num_fixed_params = None, None

    # TODO(sxjscience) Add a log metric class
    log_avg_loss_l = [mx.np.array(0.0, ctx=ctx) for ctx in ctx_l]
    # Maintain the denominator of the loss.
    log_avg_loss_denom_l = [mx.np.array(0.0, ctx=ctx) for ctx in ctx_l]
    log_wc_l = [mx.np.array(0, dtype=np.int64, ctx=ctx) for ctx in ctx_l]
    log_tgt_wc_l = [mx.np.array(0, dtype=np.int64, ctx=ctx) for ctx in ctx_l]
    log_avg_grad_norm = 0
    log_iter_num = 0

    if local_rank == 0:
        writer = SummaryWriter(logdir=os.path.join(args.save_dir, 'tensorboard'))
    if use_amp:
        amp.init_trainer(trainer)
    train_multi_data_loader = grouper(repeat(train_data_loader), len(ctx_l))
    # when args.epochs < 0, the model will keep training
    if args.epochs < 0:
        if args.max_update > 0:
            total_train_iters = args.max_update
            if args.num_averages > 0:
                assert args.num_averages <= total_train_iters // args.save_iterval_update
                avg_start_iter = (total_train_iters // args.save_iterval_update
                                  - args.num_averages) * args.save_iterval_update
            else:
                avg_start_iter = -1
        else:
            total_train_iters = np.inf
            avg_start_iter = -1
    else:
        total_train_iters = args.epochs * num_updates_per_epoch
        if args.num_averages > 0:
            assert args.num_averages <= args.epochs
            avg_start_iter = (args.epochs - args.num_average) * num_updates_per_epoch
        else:
            avg_start_iter = -1

    # Here, we are manually setting up the scale to 1.0 because
    # in horovod, the scale can be the number of workers:
    # See the code here: https://github.com/horovod/horovod/blob/125115583b7029196e2ec530decd4209459d5479/horovod/mxnet/__init__.py#L141
    # Since we will need to use the dynamic scaling in amp, we will manually call amp.unscale().
    # A scale that is larger than 1.0 can be problematic in this case.
    trainer._scale = 1.0
    if args.max_num_tokens > 0:
        const_scale = args.max_num_tokens
    else:
        const_scale = 100


    for train_iter in range(total_train_iters):
        model.zero_grad()
        loss_denom_l = [mx.np.array(0.0, ctx=ctx) for ctx in ctx_l]
        for i in range(args.num_accumulated):
            loss_l = []
            sample_data_l = next(train_multi_data_loader)
            for j, (sample_data, ctx) in enumerate(zip(sample_data_l, ctx_l)):
                src_token_ids, tgt_token_ids, src_valid_length,\
                tgt_valid_length, sample_ids = sample_data
                src_token_ids = src_token_ids.as_in_ctx(ctx)
                tgt_token_ids = tgt_token_ids.as_in_ctx(ctx)
                src_valid_length = src_valid_length.as_in_ctx(ctx)
                tgt_valid_length = tgt_valid_length.as_in_ctx(ctx)
                src_wc, tgt_wc, bs = src_valid_length.sum(), \
                                     tgt_valid_length.sum(), src_token_ids.shape[0]
                log_wc_l[j] += src_wc + tgt_wc
                log_tgt_wc_l[j] += tgt_wc
                token_count = (tgt_valid_length - 1).sum()
                loss_denom_l[j] += token_count / const_scale
                log_avg_loss_denom_l[j] += token_count / const_scale
                with mx.autograd.record():
                    tgt_pred = model(src_token_ids, src_valid_length, tgt_token_ids[:, :-1],
                                     tgt_valid_length - 1)
                    tgt_labels = tgt_token_ids[:, 1:]
                    loss = label_smooth_loss(tgt_pred, tgt_labels)
                    loss = mx.npx.sequence_mask(loss,
                                                sequence_length=tgt_valid_length - 1,
                                                use_sequence_length=True,
                                                axis=1)
                    loss = loss.sum() / const_scale
                    loss_l.append(loss)
                log_avg_loss_l[j] += loss
            if use_amp:
                with mx.autograd.record():
                    with amp.scale_loss(loss_l, trainer) as amp_loss_l:
                        for loss in amp_loss_l:
                            loss.backward()
            else:
                with mx.autograd.record():
                    for loss in loss_l:
                        loss.backward()

        # Print the total number of parameters
        if local_rank == 0 and num_params is None:
            num_params, num_fixed_params = count_parameters(model.collect_params())
            logging.info('Total Number of Parameters (not-fixed/fixed): {}/{}'
                         .format(num_params, num_fixed_params))
        # All-Reduce the gradient
        trainer.allreduce_grads()
        if args.comm_backend == 'horovod':
            # All-Reduce the loss denominator
            assert len(loss_denom_l) == 1
            loss_denom = hvd.allreduce(loss_denom_l[0], average=False).asnumpy()
        else:
            loss_denom = sum([ele.asnumpy() for ele in loss_denom_l])
        if use_amp:
            # We need to first unscale the gradient and then perform allreduce.
            grad_scale = trainer.amp_loss_scale * loss_denom
        else:
            grad_scale = loss_denom
        if args.max_grad_norm is not None:
            total_norm, ratio, is_finite\
                = clip_grad_global_norm(params, args.max_grad_norm * grad_scale)
            total_norm = total_norm / grad_scale
        else:
            total_norm = grad_global_norm(params)
            total_norm = total_norm / grad_scale
        log_avg_grad_norm += total_norm
        log_iter_num += 1

        trainer.update(loss_denom, ignore_stale_grad=True)

        if avg_start_iter > 0 and train_iter >= avg_start_iter:
            model_averager.step()

        if ((train_iter + 1) % args.log_interval == 0 or train_iter + 1 == total_train_iters):
            if args.comm_backend == 'horovod':
                # Use allreduce to get the total number of tokens and loss
                log_wc = hvd.allreduce(log_wc_l[0], average=False).asnumpy()
                log_tgt_wc = hvd.allreduce(log_tgt_wc_l[0], average=False).asnumpy()
                log_avg_loss = hvd.allreduce(log_avg_loss_l[0] / log_avg_loss_denom_l[0],
                                             average=True)
                log_avg_loss = log_avg_loss.asnumpy()
            else:
                log_wc = sum([ele.asnumpy() for ele in log_wc_l])
                log_tgt_wc = sum([ele.asnumpy() for ele in log_tgt_wc_l])
                log_avg_loss =\
                    sum([log_avg_loss_l[i].asnumpy() / log_avg_loss_denom_l[i].asnumpy()
                         for i in range(len(log_avg_loss_l))]) / len(log_avg_loss_l)
            log_avg_grad_norm = log_avg_grad_norm / log_iter_num
            if local_rank == 0:
                log_end_time = time.time()
                wps = log_wc / (log_end_time - log_start_time)
                epoch_id = train_iter // num_updates_per_epoch
                logging.info('[Epoch {} Iter {}/{}] loss={:.4f}, ppl={:.4f}, '
                             'throughput={:.2f}K wps, total wc={:.2f}K, wpb={:.2f}K,'
                             ' LR={}, gnorm={:.4f}'
                             .format(epoch_id, train_iter % num_updates_per_epoch + 1,
                                     num_updates_per_epoch,
                                     log_avg_loss, np.exp(log_avg_loss),
                                     wps / 1000, log_wc / 1000,
                                     log_tgt_wc / 1000 / log_iter_num,
                                     trainer.learning_rate,
                                     log_avg_grad_norm))
                writer.add_scalar('throughput_wps', wps, train_iter)
                writer.add_scalar('train_loss', log_avg_loss, train_iter)
                writer.add_scalar('lr', trainer.learning_rate, train_iter)
                writer.add_scalar('grad_norm', log_avg_grad_norm, train_iter)
            # Reinitialize the log variables
            log_start_time = time.time()
            log_avg_loss_l = [mx.np.array(0.0, ctx=ctx) for ctx in ctx_l]
            log_avg_loss_denom_l = [mx.np.array(0.0, ctx=ctx) for ctx in ctx_l]
            log_avg_grad_norm = 0
            log_iter_num = 0
            log_wc_l = [mx.np.array(0, dtype=np.int64, ctx=ctx) for ctx in ctx_l]
            log_tgt_wc_l = [mx.np.array(0, dtype=np.int64, ctx=ctx) for ctx in ctx_l]

        if local_rank == 0 and\
                ((args.max_update > 0 and (train_iter + 1) % args.save_interval_update == 0)
                 or ((train_iter + 1) % num_updates_per_epoch == 0)
                 or train_iter + 1 == total_train_iters):
            epoch_id = (train_iter + 1) // num_updates_per_epoch
            if args.max_update <= 0:
                model.save_parameters(os.path.join(args.save_dir,
                                                   'epoch{}.params'.format(epoch_id)),
                                      deduplicate=True)
            else:
                model.save_parameters(os.path.join(args.save_dir,
                                                   'iter{}.params'.format(train_iter + 1)),
                                      deduplicate=True)

            # TODO(?) One way to improve the current validation is to support multi-node validation.
            avg_valid_loss = validation(model, val_data_loader, ctx_l)
            logging.info('[Epoch {}][Iter {}/{}] validation loss/ppl={:.4f}/{:.4f}'
                         .format(epoch_id, train_iter, total_train_iters,
                                 avg_valid_loss, np.exp(avg_valid_loss)))
            writer.add_scalar('valid_loss', avg_valid_loss, train_iter)

    if args.num_averages > 0:
        model_averager.copy_back(model.collect_params())  # TODO(sxjscience) Rewrite using update
        model.save_parameters(os.path.join(args.save_dir, 'average.params'),
                              deduplicate=True)


if __name__ == '__main__':
    os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
    parser = get_parser()
    args = parser.parse_args()
    if args.max_update > 0:
        args.epochs = -1
    logging_config(args.save_dir, console=True)
    logging.info(args)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    random.seed(args.seed)
    if args.fp16:
        # Initialize amp if it's fp16 training
        from mxnet import amp
        amp.init()
    train(args)
