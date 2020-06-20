import abc
import pandas as pd
import numpy as np
import multiprocessing as mp
import warnings
import collections
from typing import List, Optional, Dict, Union
from . import constants as _C
from ..data.filtering import LanguageIdentifier
from ..base import INT_TYPES, FLOAT_TYPES
from ..utils.misc import num_mp_workers


class BaseColumn(abc.ABC):
    type = None


class CategoricalColumn(BaseColumn):
    type = _C.CATEGORICAL

    def __init__(self, column_data: pd.Series):
        """

        Parameters
        ----------
        column_data
            The value counts
        """
        super().__init__()
        self._value_counts = column_data.value_counts()
        self._idx_to_items = list(self._value_counts.keys())
        self._item_to_idx = {ele: i for i, ele in enumerate(self._idx_to_items)}

    @property
    def num_class(self):
        return len(self._value_counts)

    def to_idx(self, item):
        return self._item_to_idx[item]

    def to_category(self, idx):
        return self._idx_to_items[idx]

    @property
    def value_counts(self):
        return self._value_counts


class NumericalColumn(BaseColumn):
    type = _C.NUMERICAL

    def __init__(self, column_data: pd.Series):
        super().__init__()
        self._min_value = column_data.min()
        self._max_value = column_data.max()
        self._avg_value = column_data.mean()

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def avg_value(self):
        return self._avg_value


class TextColumn(BaseColumn):
    type = _C.TEXT

    def __init__(self, column_data: pd.Series):
        super().__init__()
        lengths = column_data.apply(len)
        self._min_length = lengths.min()
        self._avg_length = lengths.mean()
        self._max_length = lengths.max()
        lang_id = LanguageIdentifier()
        with mp.Pool(num_mp_workers()) as pool:
            langs = pool.map(lang_id, column_data)
        unique_langs, counts = np.unique(
            np.array([ele[0] for ele in langs]),
            return_counts=True)
        self._lang = unique_langs[counts.argmax()]

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


class EntityColumn(BaseColumn):
    """The Entity Column.

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
    type = _C.ENTITY

    def __init__(self, column_data, parent):
        super().__init__()
        self._parent = parent
        self._has_label = False
        self._label_type = None
        self._avg_entity_per_sample = 0
        # Store count the labels
        categorical_label_counter = collections.Counter()
        all_span_lengths = []
        all_entity_labels = []
        for entities in column_data:
            if entities is None:
                continue
            if isinstance(entities, dict):
                entities = [entities]
            assert isinstance(entities, list),\
                'The entity type is "{}" and is not supported by ' \
                'GluonNLP. Received entities={}'.format(type(entities), entities)
            self._avg_entity_per_sample += len(entities)
            for entity in entities:
                start = entity['start']
                end = entity['end']
                if 'label' in entity:
                    label = entity['label']
                    if self._has_label is False:
                        self._has_label = True
                    label_type = _get_entity_label_type(label)
                    if self._label_type is not None:
                        assert self._label_type == label_type,\
                            'Unmatched label types. ' \
                            'The type of labels of all entities should be consistent. ' \
                            'Received label type="{}".' \
                            ' Stored label_type="{}"'.format(label_type, self._label_type)
                    else:
                        self._label_type = label_type
                    if label_type == _C.CATEGORICAL:
                        categorical_label_counter[label] += 1

                has_label = len(entities[0]) == 3
                all_span_lengths.extend([ele[1] - ele[0] for ele in entities])
                all_entity_labels.extend([ele[2] for ele in entities])
        if self._has_label:
            unique_entity_labels, entity_label_counts = np.unique(all_entity_labels,
                                                                  return_counts=True)
            self._unique_entity_labels = unique_entity_labels
            self._entity_label_counts = entity_label_counts
            self._label_to_idx = {ele: i for i, ele in enumerate(self._unique_entity_labels)}
        else:
            self._unique_entity_labels = None
            self._entity_label_counts = None

    @property
    def label_type(self) -> Optional[str]:
        """

        Returns
        -------
        ret
            The type of the label. Should be either
            - 'categorical'
            - 'numerical'
        """
        return self._label_type

    def to_idx(self, label):
        assert self._has_label
        return self._label_to_idx[label]

    def to_label(self, idx):
        assert self._has_label
        return self._unique_entity_labels[idx]

    @property
    def entity_labels(self):
        return self._unique_entity_labels

    @property
    def entity_label_counts(self):
        return self._entity_label_counts

    @property
    def has_label(self):
        return self._has_label

    @property
    def parent(self):
        return self._parent


def is_categorical_column(data: pd.Series,
                          threshold: int = 1000,
                          ratio: float = 0.1) -> bool:
    """Check whether the column is a categorical column.

    If the number of unique elements in the column is smaller than

        min(#Total Sample * ratio, threshold),

    it will be treated as a categorical column

    Parameters
    ----------
    data
        The column data
    threshold
        The threshold for detecting categorical column
    ratio
        The ratio for detecting categorical column

    Returns
    -------
    is_categorical
        Whether the column is a categorical column
    """
    threshold = min(int(len(data) * ratio), threshold)
    sample_set = set()
    for sample in data:
        sample_set.add(sample)
        if len(sample_set) > threshold:
            return False
    return True


def parse_columns(df, column_names: Optional[List[str]] = None,
                  metadata: Optional[Dict] = None) -> collections.OrderedDict:
    """Inference the column types of the data frame

    Parameters
    ----------
    df
        Pandas Dataframe
    column_names
        The chosen column names of the table
    metadata
        The additional metadata object to help specify the column types

    Returns
    -------
    column_info
        Information of the columns
    """
    column_info = collections.OrderedDict()
    # Process all feature columns
    if column_names is None:
        column_names = df.columns
    for col_name in column_names:
        if metadata is not None and col_name in metadata:
            col_type = metadata[col_name]['type']
            if col_type == CATEGORICAL:
                column_info[col_name] = CategoricalColumn(df[col_name])
            elif col_type == TEXT:
                column_info[col_name] = TextColumn(df[col_name])
            elif col_type == NUMERICAL:
                column_info[col_name] = NumericalColumn(df[col_name])
            elif col_type == ENTITY:
                parent = metadata[col_name]['parent']
                column_info[col_name] = EntityColumn(column_data=df[col_name], parent=parent)
            else:
                raise KeyError('Column type is not supported.'
                               ' Type="{}"'.format(col_type))
        idx = df[col_name].first_valid_index()
        if idx is None:
            # No valid index, it's safe to ignore the column
            continue
        ele = df[col_name][idx]
        if isinstance(ele, str):
            # Try to tell if the column is a text column / categorical column
            is_categorical = is_categorical_column(df[col_name])
            if is_categorical:
                column_info[col_name] = CategoricalColumn(df[col_name])
            else:
                column_info[col_name] = TextColumn(df[col_name])
        elif isinstance(ele, INT_TYPES + FLOAT_TYPES):
            is_categorical = is_categorical_column(df[col_name])
            if is_categorical:
                column_info[col_name] = CategoricalColumn(df[col_name])
            else:
                column_info[col_name] = NumericalColumn(df[col_name])
        else:
            raise KeyError('The type of the column is "{}" and is not yet supported.'
                           ' Please consider to update your input dataframe.'.format(type(ele)))
    return column_info


class AutoNLPDataset:
    def __init__(self, path_or_df: Union[str, pd.DataFrame],
                 feature_columns: List[str],
                 label_columns: List[str] = None,
                 metadata: Dict[str, str] = None,
                 problem_type: Dict[str, str] = None):
        """

        Parameters
        ----------
        path_or_df
            The path or dataframe of the tabular dataset for NLP.
        feature_columns
            The feature columns
        label_columns
            The label columns
        metadata
            The metadata object that describes the property of the columns in the dataset
        problem_type
            The type of the problem that we will need to solve for each label column.
            If it is not given, we will infer the problem type from the input.
        """
        if not isinstance(path_or_df, pd.DataFrame):
            # Assume pickle.
            df = pd.read_pickle(path_or_df)
        else:
            df = path_or_df
        self._feature_columns = feature_columns
        self._label_columns = label_columns
        all_columns = self._feature_columns
        if self._label_columns is not None:
            all_columns.extend(self._label_columns)
        self._table = df[all_columns]
        if metadata is None:
            metadata = dict()
        if problem_type is not None:
            for col_name in self._label_columns:
                label_problem_type = problem_type[col_name]
                if label_problem_type == CLASSIFICATION:
                    if col_name not in metadata:
                        metadata[col_name]['type'] = CATEGORICAL
                    else:
                        assert metadata[col_name]['type'] == CATEGORICAL
                elif label_problem_type == REGRESSION:
                    if col_name not in metadata:
                        metadata[col_name]['type'] = NUMERICAL
                    else:
                        assert metadata[col_name]['type'] == NUMERICAL
                else:
                    raise NotImplementedError
        self._column_info = parse_columns(self._table, metadata=metadata)
        if label_columns is not None:
            if problem_type is None:
                problem_type = dict()
                # Try to infer the problem type of each label
                for col_name in label_columns:
                    if self._column_info[col_name].type == CATEGORICAL:
                        problem_type[col_name] = CLASSIFICATION
                    elif self._column_info[col_name].type == NUMERICAL:
                        problem_type[col_name] = REGRESSION
                    elif self._column_info[col_name].type == ENTITY:
                        problem_type[col_name] = CLASSIFICATION
                    else:
                        raise NotImplementedError
            self._problem_type = problem_type
        else:
            self._problem_type = None

    @property
    def problem_type(self):
        return self._problem_type

    @property
    def table(self):
        return self._table

    @property
    def column_info(self):
        return self._column_info

    @property
    def feature_columns(self):
        return self._feature_columns

    @property
    def label_columns(self):
        return self._label_columns
