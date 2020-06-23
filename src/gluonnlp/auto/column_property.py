import abc
import pprint
import pandas as pd
import numpy as np
import multiprocessing as mp
import collections
from typing import List, Optional, Union, Tuple, Hashable
from . import constants as _C
from ..data.vocab import Vocab
from ..data.filtering import LanguageIdentifier
from ..utils.misc import num_mp_workers
__all__ = ['CategoricalColumnProperty', 'TextColumnProperty', 'NumericalColumnProperty', 'EntityColumnProperty']


class ColumnProperty(abc.ABC):
    type = None

    def __init__(self, column_data: pd.Series):
        assert isinstance(column_data, pd.Series), 'Currently, the input column data must be a Pandas Series.'
        self._num_sample = len(column_data)
        self._num_missing_samples = column_data.isnull().sum().sum().item()
        self._name = column_data.name

    @property
    def name(self):
        return self._name

    @property
    def num_sample(self):
        return self._num_sample

    @property
    def num_missing_sample(self):
        return self._num_missing_samples

    @property
    def num_valid_sample(self):
        return self.num_sample - self.num_missing_sample

    def transform(self, ele):
        return ele

    @abc.abstractmethod
    def parse_other(self, column_data: pd.Series):
        """Parse another column data with the given property"""
        pass

    def info(self, additional_attributes=None):
        basename = self.__class__.__name__
        if basename.endswith('ColumnProperty'):
            basename = basename[:(-len('ColumnProperty'))]
        padding = 3
        ret = '{}(\n'.format(basename)
        ret += ' ' * padding + 'name="{}"\n'.format(self.name)
        ret += ' ' * padding + '#total/missing={}/{}\n'.format(self.num_sample,
                                                             self.num_missing_sample)
        if additional_attributes is not None:
            for key, info in additional_attributes:
                ret += ' ' * padding + '{}={}\n'.format(key, str(info))
        ret += ')\n'
        return ret

    def __repr__(self):
        return self.info()


class CategoricalColumnProperty(ColumnProperty):
    type = _C.CATEGORICAL

    def __init__(self, column_data: pd.Series,
                 categories: Optional[List[Hashable]] = None):
        """

        Parameters
        ----------
        column_data
            The value counts
        name
            Name of the column
        categories
            The possible categories
        """
        super().__init__(column_data=column_data)
        value_counts = column_data.value_counts()
        if categories is None:
            categories = sorted(list(value_counts.keys()))
        self._vocab = Vocab(tokens=categories, unk_token=None)
        self._freq = [value_counts[ele] for ele in categories]

    def transform(self, data: Hashable) -> int:
        """Transform the input data

        Parameters
        ----------
        data
            Element in the input data

        Returns
        -------
        idx
            The transformed idx
        """
        return self.to_idx(data)

    def inv_transform(self, idx: int) -> Hashable:
        """Transform the idx back to the category

        Parameters
        ----------
        idx

        Returns
        -------
        category
        """
        return self.to_category(idx)

    @property
    def num_class(self):
        return len(self._vocab)

    def to_idx(self, item):
        return self._vocab[item]

    def to_category(self, idx):
        return self._vocab.all_tokens[idx]

    @property
    def categories(self):
        return self._vocab.all_tokens

    @property
    def frequencies(self):
        return self._freq

    def parse_other(self, column_data: pd.Series):
        return CategoricalColumnProperty(column_data=column_data,
                                     categories=self.categories)

    def info(self):
        return super().info(
            [('num_class', self.num_class),
             ('categories', self.categories),
             ('freq', self.frequencies)])


class NumericalColumnProperty(ColumnProperty):
    type = _C.NUMERICAL

    def __init__(self, column_data: pd.Series,
                 shape: Optional[Tuple] = None):
        """

        Parameters
        ----------
        column_data
            Column data
        shape
            The shape of the numerical values
        """
        super().__init__(column_data=column_data)
        idx = column_data.first_valid_index()
        val = column_data[idx]
        inferred_shape = np.array(val).shape
        if shape is not None:
            assert shape == inferred_shape, 'Shape mismatch!. Expected shape={},' \
                                     ' shape in the dataset is {}'.format(shape, inferred_shape)
        self._shape = inferred_shape

    def transform(self, ele):
        return np.array(ele, dtype=np.float32)

    @property
    def shape(self):
        return self._shape

    def parse_other(self, column_data: pd.Series):
        return NumericalColumnProperty(column_data=column_data, shape=self.shape)

    def info(self):
        return super().info([('shape', self.shape)])


class TextColumnProperty(ColumnProperty):
    type = _C.TEXT

    def __init__(self, column_data: pd.Series, lang=None):
        """

        Parameters
        ----------
        column_data
            Column data
        lang
            The language of the text column
        """
        super().__init__(column_data=column_data)
        lengths = column_data.apply(len)
        self._min_length = lengths.min()
        self._avg_length = lengths.mean()
        self._max_length = lengths.max()
        self._num_samples = len(column_data)
        if lang is not None:
            self._lang = lang
        else:
            # Determine the language
            lang_id = LanguageIdentifier(algo='langid')
            merged_string = ' '.join(column_data[:100].tolist())
            lang = lang_id(merged_string)
            self._lang = lang

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def lang(self):
        return self._lang

    @property
    def min_length(self):
        return self._min_length

    @property
    def max_length(self):
        return self._max_length

    @property
    def avg_length(self):
        return self._avg_length

    def parse_other(self, column_data: pd.Series):
        return TextColumnProperty(column_data=column_data, lang=self.lang)

    def info(self):
        return super().info([('length, min/avg/max',
                              '{:d}/{:.2f}/{:d}'.format(self.min_length,
                                                        self.avg_length, self.max_length)),
                             ('language', self.lang)])


def _get_entity_label_type(label) -> str:
    """

    Parameters
    ----------
    label
        The label of an entity

    Returns
    -------
    type_str
        The type of the label. Will either be null, categorical or numerical
    """
    if label is None:
        return _C.NULL
    if isinstance(label, (int, str)):
        return _C.CATEGORICAL
    else:
        return _C.NUMERICAL


class EntityColumnProperty(ColumnProperty):
    """The Entities Column.

    The elements inside the column can be
    - a single dictionary -> 1 entity
    - a single tuple -> 1 entity
    - a list of dictionary -> K entities
    - a list of tuples -> K entities
    - an empty list -> 0 entity
    - None -> 0 entity

    For each entity, it will be
    1) a dictionary that contains these keys
    - start
        The character-level start of the entity
    - end
        The character-level end of the entity
    - label
        The label information of this entity.
        We support
        - categorical labels
            Each label can be either a unicode string or a int value.
        - numpy array/vector labels/numerical labels
            Each label should be a fixed-dimensional array/numerical value
    2) a tuple with (start, end) or (start, end, label)

    """
    type = _C.ENTITY

    def __init__(self, column_data, parent,
                 label_type=None,
                 label_shape=None,
                 label_vocab=None):
        """

        Parameters
        ----------
        column_data
            Column data
        parent
            The column name of its parent
        label_type
            The type of the labels.
            Can be the following:
            - null
            - categorical
            - numerical
        label_shape
            The shape of the label. Only be available when the entity contains numerical label
        label_vocab
            The vocabulary of the categorical label.
            It is only available when the entity contains categorical label.
        """
        super().__init__(column_data=column_data)
        self._parent = parent
        self._label_type = label_type
        self._label_shape = label_shape
        self._label_vocab = label_vocab
        self._label_freq = None

        # Store statistics
        all_span_lengths = []
        categorical_label_counter = collections.Counter()
        for idx, entities in column_data.items():
            if entities is None:
                continue
            if isinstance(entities, dict) or isinstance(entities, tuple):
                entities = [entities]
            assert isinstance(entities, list),\
                'The entity type is "{}" and is not supported by ' \
                'GluonNLP. Received entities={}'.format(type(entities), entities)
            for entity in entities:
                if isinstance(entity, dict):
                    start = entity['start']
                    end = entity['end']
                    label = entity.get('label', None)
                else:
                    assert isinstance(entity, tuple)
                    if len(entity) == 2:
                        start, end = entity
                        label = None
                    else:
                        start, end, label = entity
                all_span_lengths.append(end - start)
                label_type = _get_entity_label_type(label)
                if label_type == _C.CATEGORICAL:
                    categorical_label_counter[label] += 1
                elif label_type == _C.NUMERICAL and self._label_shape is None:
                    self._label_shape = np.array(label).shape
                if self._label_type is not None:
                    assert self._label_type == label_type, \
                        'Unmatched label types. ' \
                        'The type of labels of all entities should be consistent. ' \
                        'Received label type="{}".' \
                        ' Stored label_type="{}"'.format(label_type, self._label_type)
                else:
                    self._label_type = label_type
        self._num_total_entity = len(all_span_lengths)
        self._avg_entity_per_sample = len(all_span_lengths) / self.num_valid_sample
        self._avg_span_length = np.mean(all_span_lengths).item()
        if self._label_type == _C.CATEGORICAL:
            if self._label_vocab is None:
                keys = sorted(categorical_label_counter.keys())
                self._label_vocab = Vocab(tokens=keys,
                                          unk_token=None)
                self._label_freq = [categorical_label_counter[ele] for ele in keys]
            else:
                for key in categorical_label_counter.keys():
                    if key not in self._label_vocab:
                        raise ValueError('The entity label="{}" is not found in the provided '
                                         'vocabulary. The provided labels="{}"'
                                         .format(key,
                                                 self._label_vocab.all_tokens))
                self._label_freq = [categorical_label_counter[ele]
                                    for ele in self._label_vocab.all_tokens]

    def transform(self,
                  data: Optional[Union[dict, List[dict],
                                       Tuple, List[Tuple]]]) -> Tuple[np.ndarray,
                                                                      Optional[np.ndarray]]:
        """Transform the element to a formalized format

        Returns
        -------
        char_offsets
            Numpy array. Shape is (#entities, 2)
        labels
            Either None, or the transformed label
            - None
                None
            - Categorical:
                (#entities,)
            - Numerical:
                (#entities,) + label_shape
        """
        if data is None:
            if self.label_type == _C.CATEGORICAL:
                return np.zeros((0, 2), dtype=np.int32),\
                       np.zeros((0,), dtype=np.int32)
            elif self.label_type == _C.NUMERICAL:
                return np.zeros((0, 2), dtype=np.int32), \
                       np.zeros((0,) + self.label_shape, dtype=np.float32)
            elif self.label_type == _C.NULL:
                return np.zeros((0, 2), dtype=np.int32), None
            else:
                raise NotImplementedError
        labels = None if self.label_type == _C.NULL else []
        char_offsets = []
        if isinstance(data, dict) or isinstance(data, tuple):
            data = [data]
        for ele in data:
            if isinstance(ele, dict):
                start = ele['start']
                end = ele['end']
                if self.label_type == _C.CATEGORICAL:
                    labels.append(self.label_to_idx(ele['label']))
                elif self.label_type == _C.NUMERICAL:
                    labels.append(ele['label'])
            else:
                start = ele[0]
                end = ele[1]
                if self.label_type == _C.CATEGORICAL:
                    labels.append(self.label_to_idx(ele[2]))
                elif self.label_type == _C.NUMERICAL:
                    labels.append(ele[2])
            char_offsets.append((start, end))
        char_offsets = np.stack(char_offsets)
        if self.label_type != _C.NULL:
            labels = np.stack(labels)
        return char_offsets, labels

    @property
    def label_shape(self) -> Optional[Tuple[int]]:
        """The shape of each individual label of the entity.

        Will only be enabled when label_type == numerical

        Returns
        -------
        ret
        """
        return self._label_shape

    @property
    def label_type(self) -> str:
        """Type of the label.

        If there is no label attached to the entities, it will return None.

        Returns
        -------
        ret
            The type of the label. Should be either
            - 'null'
            - 'categorical'
            - 'numerical'
        """
        return self._label_type

    def label_to_idx(self, label):
        assert self.label_type == _C.CATEGORICAL
        return self._label_vocab[label]

    def idx_to_label(self, idx):
        assert self.label_type == _C.CATEGORICAL
        return self._label_vocab.all_tokens[idx]

    @property
    def label_keys(self):
        if self._label_vocab is None:
            return None
        else:
            return self._label_vocab.all_tokens

    @property
    def label_freq(self):
        return self._label_freq

    @property
    def has_label(self):
        return self._label_type != _C.NULL

    @property
    def parent(self):
        return self._parent

    @property
    def avg_entity_per_sample(self):
        return self._avg_entity_per_sample

    @property
    def avg_span_length(self):
        return self._avg_span_length

    @property
    def num_total_entity(self):
        return self._num_total_entity

    def parse_other(self, column_data: pd.Series):
        return EntityColumnProperty(column_data=column_data,
                                    parent=self.parent,
                                    label_type=self.label_type,
                                    label_shape=self.label_shape,
                                    label_vocab=self._label_vocab)

    def info(self):
        additional_attributes = [('parent', '"{}"'.format(self._parent)),
                                 ('#total entity', self.num_total_entity),
                                 ('num entity per sample',
                                  '{:.2f}'.format(self.avg_entity_per_sample)),
                                 ('avg span length', '{:.2f}'.format(self._avg_span_length))]
        if self.label_type == _C.CATEGORICAL:
            additional_attributes.append(('num categories', len(self.label_keys)))
            additional_attributes.append(('max/min freq',
                                          '{}/{}'.format(max(self.label_freq),
                                                         min(self.label_freq))))
        elif self.label_type == _C.NUMERICAL:
            additional_attributes.append(('label_shape', self.label_shape))
        return super().info(additional_attributes)
