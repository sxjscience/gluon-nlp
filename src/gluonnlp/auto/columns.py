import abc
import pandas as pd
import numpy as np
import multiprocessing as mp
import collections
from typing import List, Optional, Union, Tuple, Hashable
from . import constants as _C
from ..data.vocab import Vocab
from ..data.filtering import LanguageIdentifier
from ..utils.misc import num_mp_workers


class ColumnProperty(abc.ABC):
    type = None

    def __init__(self, column_data: pd.Series, name: str):
        self._num_sample = len(column_data)
        self._num_missing_samples = column_data.isnull().sum().sum()
        self._name = name

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
    def parse_test(self, column_data: pd.Series):
        """Parse the test column data"""
        pass


class CategoricalColumn(ColumnProperty):
    type = _C.CATEGORICAL

    def __init__(self, column_data: pd.Series, name: str,
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
        super().__init__(column_data=column_data, name=name)
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

    def parse_test(self, column_data: pd.Series):
        return CategoricalColumn(column_data=column_data,
                                 name=self.name,
                                 categories=self.categories)

    def __repr__(self):
        ret = 'Categorical(\n' \
              '   name={},\n' \
              '   total/missing={}/{},\n' \
              '   num_class={},\n' \
              '   categories={},\n' \
              '   freq={}\n' \
              ')'.format(self.name,
                         self.num_sample, self.num_missing_sample,
                         self.num_class,
                         self.categories,
                         self.frequencies)
        return ret


class NumericalColumn(ColumnProperty):
    type = _C.NUMERICAL

    def __init__(self, column_data: pd.Series, name: str,
                 shape: Optional[Tuple] = None):
        """

        Parameters
        ----------
        column_data
            Column data
        name
            Name of the column
        shape
            The shape of the numerical values
        """
        super().__init__(column_data=column_data, name=name)
        if shape is None:
            idx = column_data.first_valid_index()
            val = column_data[idx]
            self._shape = np.array(val).shape
        else:
            self._shape = shape

    @property
    def shape(self):
        return self._shape

    def parse_test(self, column_data: pd.Series):
        return NumericalColumn(column_data=column_data, name=self.name, shape=self.shape)

    def __repr__(self):
        ret = 'Numerical(\n' \
              '   name={},\n' \
              '   shape={}\n' \
              ')'.format(self.name, self.shape)
        return ret


class TextColumn(ColumnProperty):
    type = _C.TEXT

    def __init__(self, column_data: pd.Series, name: str, lang=None):
        """

        Parameters
        ----------
        column_data
            Column data
        name
            Name of the column
        lang
            The language of the text column
        """
        super().__init__(column_data=column_data, name=name)
        lengths = column_data.apply(len)
        self._min_length = lengths.min()
        self._avg_length = lengths.mean()
        self._max_length = lengths.max()
        self._num_samples = len(column_data)
        if lang is not None:
            self._lang = lang
        else:
            # Determine the language
            lang_id = LanguageIdentifier()
            with mp.Pool(num_mp_workers()) as pool:
                langs = pool.map(lang_id, column_data)
            unique_langs, counts = np.unique(
                np.array([ele[0] for ele in langs]),
                return_counts=True)
            self._lang = unique_langs[counts.argmax()]

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

    def parse_test(self, column_data: pd.Series):
        return TextColumn(column_data=column_data, name=self.name, lang=self.lang)

    def __repr__(self):
        ret = 'Text(\n' \
              '   name={},\n' \
              '   total/missing={}/{},\n' \
              '   min/avg/max length={:d}/{:.2f}/{:d}\n' \
              ')'.format(self.name,
                         self.num_sample, self.num_missing_sample,
                         self.min_length, self.avg_length, self.max_length)
        return ret


def _get_entity_label_type(label) -> str:
    """

    Parameters
    ----------
    label
        The label of an entity

    Returns
    -------
    type_str
        The type of the label. Will either be categorical or numerical
    """
    if isinstance(label, (int, str)):
        return _C.CATEGORICAL
    else:
        return _C.NUMERICAL


class EntitiesColumn(ColumnProperty):
    """The Entities Column.

    The elements inside the column can be
    - a single dictionary -> 1 entity
    - a list of dictionary -> K entities
    - an empty list -> 0 entity
    - None -> 0 entity

    For each entity, it will be a dictionary that contains these keys
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

    """
    type = _C.ENTITIES

    def __init__(self, column_data, name, parent,
                 label_type=None,
                 label_shape=None,
                 label_vocab=None):
        """

        Parameters
        ----------
        column_data
            Column data
        name
            Name of the column
        parent
            The column name of its parent
        label_type
            The type of the labels.
            Can be the following:
            - null
            - categorical
            - numerical
        """
        super().__init__(column_data=column_data, name=name)
        self._parent = parent
        self._label_type = label_type
        self._label_shape = label_shape
        self._label_vocab = label_vocab
        self._label_freq = None

        # Store statistics
        all_span_lengths = []
        categorical_label_counter = collections.Counter()
        for entities in column_data:
            if entities is None:
                continue
            if isinstance(entities, dict):
                entities = [entities]
            assert isinstance(entities, list),\
                'The entity type is "{}" and is not supported by ' \
                'GluonNLP. Received entities={}'.format(type(entities), entities)
            for entity in entities:
                start = entity['start']
                end = entity['end']
                all_span_lengths.append(end - start)
                if 'label' in entity:
                    label = entity['label']
                    label_type = _get_entity_label_type(label)
                    if label_type == _C.CATEGORICAL:
                        categorical_label_counter[label] += 1
                    elif label_type == _C.NUMERICAL and self._label_shape is None:
                        self._label_shape = np.array(label).shape
                else:
                    label_type = _C.NULL
                if self._label_type is not None:
                    assert self._label_type == label_type, \
                        'Unmatched label types. ' \
                        'The type of labels of all entities should be consistent. ' \
                        'Received label type="{}".' \
                        ' Stored label_type="{}"'.format(label_type, self._label_type)
                else:
                    self._label_type = label_type
        self._num_total_entities = len(all_span_lengths)
        self._avg_entity_per_sample = len(all_span_lengths) / self.num_valid_sample
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
                  data: Optional[Union[dict, List[dict]]]) -> Tuple[np.ndarray,
                                                                    Optional[np.ndarray]]:
        """Transform the element to a formalized format

        Returns
        -------
        entities
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
        entities = []
        if isinstance(data, dict):
            data = [data]
        for ele in data:
            start = ele['start']
            end = ele['end']
            if self.label_type == _C.CATEGORICAL:
                labels.append(self.label_to_idx(ele['label']))
            elif self.label_type == _C.NUMERICAL:
                labels.append(ele['label'])
            else:
                raise NotImplementedError
            entities.append((start, end))
        entities = np.stack(entities)
        if self.label_type is not None:
            labels = np.stack(labels)
        return entities, labels

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
    def label_vocab(self):
        return self._label_vocab

    @property
    def has_label(self):
        return self._label_type is not None

    @property
    def parent(self):
        return self._parent

    @property
    def avg_entity_per_sample(self):
        return self._avg_entity_per_sample

    def __repr__(self):
        if self.label_type is None:
            ret = 'Entities(' \
                  '   name={},' \
                  '   #entity_per_sample={},' \
                  '   '
        ret = 'Entity(' \
              '   name={},' \
              '   min/avg/max length={:d}/{:.2f}/{:d}' \
              ')'.format(self.name, self.min_length, self.avg_length, self.max_length)
        return ret
