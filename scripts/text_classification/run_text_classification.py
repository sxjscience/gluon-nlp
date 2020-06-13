from collections import OrderedDict
from models import BertForTextClassification
from gluonnlp.data.batchify import Pad, Tuple
from gluonnlp.models import get_backbone
from gluonnlp.auto.dataset import get_column_info
from gluonnlp.auto.constants import TEXT, CATEGORICAL, NUMERICAL, ENTITY, CLASSIFICATION
from gluonnlp.base import INT_TYPES, FLOAT_TYPES
from gluonnlp.data.filtering import LanguageIdentifier
from gluonnlp.utils.preprocessing import get_trimmed_lengths
from mxnet.gluon.data import batchify as bf
import operator
import pandas as pd
from functools import partial
import multiprocessing as mp
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Text classification example.')
    parser.add_argument('--train_file', type=str,
                        help='The training pandas dataframe.',
                        default=None)
    parser.add_argument('--dev_file', type=str,
                        help='The validation pandas dataframe',
                        default=None)
    parser.add_argument('--test_file', type=str,
                        help='The test pandas dataframe',
                        default=None)
    parser.add_argument('--feature_columns', type=str, nargs='+',
                        help='Feature columns', required=True)
    parser.add_argument('--label_columns', type=str, nargs='+',
                        required=True)
    parser.add_argument('--problem_type', type=str,
                        choices=['classification', 'regression'], default='classification')
    parser.add_argument('--eval_metric', type=str,
                        help='The metrics for evaluating the models.',
                        default=None)
    parser.add_argument('--stop_metric', type=str,
                        help='The metrics for early stopping')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to train the model')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to evaluate the model')
    parser.add_argument('--batch_size', type=int,
                        help='The batch-size')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The output directory where the model params will be written.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. -1 means using cpu.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm.')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        help='optimization algorithm. default is adamw')
    parser.add_argument('--backbone_name', type=str,
                        default='google_albert_base_v2',
                        help='Name of the pretrained model.')
    args = parser.parse_args()
    return args


class TextClassificationPretrainTransform:
    def __init__(self, tokenizer,
                 column_info: OrderedDict,
                 max_length: int,
                 merge_text: bool = True):
        """Preprocess the inputs to work with a pretrained model.

        Parameters
        ----------
        tokenizer
            The tokenizer
        column_info
            The column information of the sample
        max_length
            The maximum length of the encoded token sequence.
        merge_text
            Whether to merge the token_ids when there are multiple text fields.
            For example, we will merge the text fields as
            [CLS] token_ids1 [SEP] token_ids2 [SEP] token_ids3 [SEP] token_ids4 [SEP] ...
        """
        self._tokenizer = tokenizer
        #TODO(sxjscience) We may need a better data structure to store the column info
        self._column_info = column_info
        self._max_length = max_length
        self._merge_text = merge_text
        self._text_columns = []
        self._entity_columns = []
        self._categorical_columns = []
        self._numerical_columns = []
        for col_id, (col_name, col_info) in enumerate(self._column_info.items()):
            if col_info.type == TEXT:
                self._text_columns.append((col_name, col_id))
            elif col_info.type == ENTITY:
                self._entity_columns.append((col_name, col_id))
            elif col_info.type == CATEGORICAL:
                self._categorical_columns.append((col_name, col_id))
            elif col_info.type == NUMERICAL:
                self._numerical_columns.append((col_name, col_id))
            else:
                raise NotImplementedError
        self._text_column_require_offsets = {col_name: False for col_name, _ in self.text_columns}
        for col_name, _ in self.categorical_columns:
            self._text_column_require_offsets[self.column_info[col_name].parent] = True

    @property
    def max_length(self):
        return self._max_length

    @property
    def column_info(self):
        return self._column_info

    @property
    def merge_text(self):
        return self._merge_text

    @property
    def text_columns(self):
        return self._text_columns

    @property
    def text_column_require_offsets(self):
        return self._text_column_require_offsets

    @property
    def entity_columns(self):
        return self._entity_columns

    @property
    def categorical_columns(self):
        return self._categorical_columns

    @property
    def numerical_columns(self):
        return self._numerical_columns

    def out_feature_types(self):
        """Get the types of the output features after this transformation

        TEXT --> (token_ids, valid_length)
        ENTITY --> [#ENTITY, 3], each will be (start, end, label)


        Returns
        -------
        out_types
            A list of output feature types
        """
        out_types = []
        text_col_idx = dict()
        if len(self.text_columns) > 0:
            if self.merge_text:
                out_types.append((TEXT, dict()))
            else:
                for i, (col_name, _) in enumerate(self.text_columns):
                    text_col_idx[col_name] = i
                    out_types.append((TEXT,
                                      {'info': self.column_info[col_name]}))
        if len(self.entity_columns) > 0:
            for col_name, _ in self.entity_columns:
                parent = self.column_info[col_name].parent
                if self.merge_text:
                    parent_idx = 0
                else:
                    parent_idx = text_col_idx[parent]
                out_types.extend((ENTITY,
                                  {'parent_idx': parent_idx,
                                   'info': self.column_info[col_name]}))
        if len(self.categorical_columns) > 0:
            out_types.extend([(CATEGORICAL, {'info': self.column_info[col_name]})
                              for col_name, _ in self.entity_columns])
        if len(self.numerical_columns) > 0:
            out_types.extend([(NUMERICAL, {'info': self.column_info[col_name]})
                              for col_name, _ in self.entity_columns])
        return out_types

    def __call__(self, sample):
        """Transform a sample into a list of features.

        We organize and represent the features in the following format:

        - Text features
            We transform text into a sequence of token_ids.
            If there are multiple text fields, we have the following options
            1) merge_text = True
                We will concatenate these text fields and inserting CLS, SEP ids, i.e.
                [CLS] text_ids1 [SEP] text_ids2 [SEP] text_ids3, ...
            2) merge_text = False
                We will transform each text field separately:
                [CLS] text_ids1 [SEP], [CLS] text_ids2 [SEP], ...
            For empty text / missing text data, we will just convert it to [CLS] [SEP]
        - Entity features
            The raw entities are stored as character-level start and end offsets.
            After the preprocessing step, we will store them as the token-level
            start + end. Different from the raw character-level start + end offsets, the
            token-level end will be used.
            - [(token_level_start, token_level_end, entity_label_id)]
            or
            - [(token_level_start, token_level_end)]
        - Categorical features
            We transform the categorical features to the its ids.
            We indicate the missing value with a special flag.
        - Numerical features
            We keep the numerical features and indicate the missing value
            with a special flag.

        Parameters
        ----------
        sample
            A single data sample.

        Returns
        -------
        features
            Generated features.
            - TEXT
                The encoded value will be the (token_ids, valid_length) tuple
                    data: Shape (seq_length,)
                    valid_length: integer scalar

            - ENTITY
                The encoded feature will be the (data, num_entity) tuple
                - has_label = True
                    data: Shape (num_entity, 3)
                        Each item will be (start, end, label_id)
                - has_label = False
                    data: Shape (num_entity, 2)
                        Each item will be (start, end)

            - CATEGORICAL
                The categorical feature. Will be an integer

            - NUMERICAL
                The numerical feature. Will be a scalar
        labels
            The labels
        """
        features = []
        labels = []
        # Step 1: Get the features of all text columns
        sentence_start_in_merged = None
        sentence_slice_info = dict()
        text_token_ids = OrderedDict()
        text_token_offsets = OrderedDict()
        if len(self.text_columns) > 0:
            for col_name, col_id in self.text_columns:
                if isinstance(sample[col_id], str):
                    if self.text_column_require_offsets[col_name]:
                        token_ids, token_offsets =\
                            self._tokenizer.encode_with_offsets(sample[col_id], int)
                        token_ids = np.array(token_ids)
                        token_offsets = np.array(token_offsets)
                    else:
                        token_ids = self._tokenizer.encode(sample[col_id], int)
                        token_ids = np.array(token_ids)
                elif isinstance(sample[col_id], np.ndarray):
                    if self.text_column_require_offsets[col_name]:
                        raise ValueError('Must get the offsets of all text tokens!')
                    token_ids = sample[col_id]
                elif isinstance(sample[col_id], tuple):
                    token_ids, token_offsets = sample[col_id]
                else:
                    raise NotImplementedError('The input format of the text column '
                                              'cannot be understood!')
                text_token_ids[col_name] = token_ids
                if self.text_column_require_offsets[col_name]:
                    text_token_offsets[col_name] = token_offsets
            lengths = [len(text_token_ids[col_name]) for col_name, _ in self.text_columns]
            if self.merge_text:
                # We will merge the text tokens by
                # [CLS] token_ids1 [SEP] token_ids2 [SEP]
                trimmed_lengths = get_trimmed_lengths(lengths,
                                                      max_length=self.max_length - len(lengths) - 1,
                                                      do_merge=True)
                encoded_token_ids = [np.array([self._tokenizer.cls_id])]
                sentence_start_in_merged = dict()
                for trim_length, (col_name, _) in zip(trimmed_lengths, self.text_columns):
                    sentence_start_in_merged[col_name] = len(encoded_token_ids)
                    sentence_slice_info[col_name] = (0, trim_length)
                    encoded_token_ids.append(text_token_ids[col_name][:trim_length])
                    encoded_token_ids.append(np.array([self._tokenizer.sep_id]))
                encoded_token_ids = np.concatenate(encoded_token_ids)
                valid_length = len(encoded_token_ids)
                features.append((encoded_token_ids, valid_length))
            else:
                trimmed_lengths = get_trimmed_lengths(lengths,
                                                      max_length=self.max_length - 2,
                                                      do_merge=False)
                for trim_length, (col_name, _) in zip(trimmed_lengths, self.text_columns):
                    sentence_slice_info[col_name] = (0, trim_length)
                    encoded_token_ids = np.concatenate(np.array([self._tokenizer.cls_id]),
                                                       text_token_ids[col_name][:trim_length],
                                                       np.array([self._tokenizer.sep_id]))
                    features.append((encoded_token_ids, len(encoded_token_ids)))
        # Step 2: Transform all entity columns
        # Slice from the first column
        for col_name, col_id in self.entity_columns:
            entities = sample[col_id]
            if isinstance(entities, tuple):
                entities = [entities]
            entities = np.array(entities)
            token_offsets = text_token_offsets[col_name]
            if self.merge_text:


        for col_info, ele in zip(self._column_info, sample):
            if col_info.type == TEXT:
                token_ids = ele[]


class TextClassificationSample:
    def __init__(self):
        pass

    def to_json(self):
        pass


def generate_samples(tokenizer, features, labels=None,
                     stride=20, max_length=None):
    """Generate the examples for training/inference the network

    Parameters
    ----------
    tokenizer
        The tokenizer
    features
        Text features
    labels
        Labels
    stride
        The stride for generating the chunks.
    max_length
        The maximum length of the merged text sentence

    Returns
    -------
    samples
        A list of samples
    """
    samples = []
    for i, (feature, label) in enumerate(zip(features, labels)):
        token_ids = tokenizer.encode(feature[0], int)
        pass

    return samples


def get_num_unk(data, unk_id):
    return (data == unk_id).sum()


def train(args):
    assert args.train_file is not None
    assert args.dev_file is not None
    train_df = pd.read_pickle(args.train_file)
    dev_df = pd.read_pickle(args.dev_file)
    column_infos = get_column_info(train_df, args.feature_columns + args.label_columns)
    assert len(args.label_columns) == 1, 'Currently, only a single label is supported'
    backbone_model_cls, cfg, tokenizer, local_params_path = get_backbone(args.backbone_name)
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
        assert args.train_file is not None
        assert args.dev_file is not None
        train(args)
    if args.do_eval:
        predict(args)
