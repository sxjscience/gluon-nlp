import collections
import pandas as pd
from . import constants as _C
from .column_info import CategoricalColumn, EntityColumn, TextColumn, NumericalColumn
from typing import List, Optional, Union, Dict


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
            if col_type == _C.CATEGORICAL:
                column_info[col_name] = CategoricalColumn(df[col_name])
            elif col_type == _C.TEXT:
                column_info[col_name] = TextColumn(df[col_name])
            elif col_type == _C.NUMERICAL:
                column_info[col_name] = NumericalColumn(df[col_name])
            elif col_type == _C.ENTITIES:
                parent = metadata[col_name]['parent']
                column_info[col_name] = EntityColumn(column_data=df[col_name], parent=parent)
            else:
                raise KeyError('Column type is not supported.'
                               ' Type="{}"'.format(col_type))
        idx = df[col_name].first_valid_index()
        if idx is None:
            # No valid index, it's safe to ignore the column
            warnings.warn('Column Name="{}" has no valid data and is ignored.'.format(col_name))
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
                           ' Please consider to update your input pandas dataframe.'
                           .format(type(ele)))
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
                if label_problem_type == _C.CLASSIFICATION:
                    if col_name not in metadata:
                        metadata[col_name]['type'] = _C.CATEGORICAL
                    else:
                        assert metadata[col_name]['type'] == _C.CATEGORICAL
                elif label_problem_type == _C.REGRESSION:
                    if col_name not in metadata:
                        metadata[col_name]['type'] = _C.NUMERICAL
                    else:
                        assert metadata[col_name]['type'] == _C.NUMERICAL
                else:
                    raise NotImplementedError
        self._column_info = parse_columns(self._table, metadata=metadata)
        if label_columns is not None:
            if problem_type is None:
                problem_type = dict()
                # Try to infer the problem type of each label
                for col_name in label_columns:
                    if self._column_info[col_name].type == _C.CATEGORICAL:
                        problem_type[col_name] = _C.CLASSIFICATION
                    elif self._column_info[col_name].type == _C.NUMERICAL:
                        problem_type[col_name] = _C.REGRESSION
                    elif self._column_info[col_name].type == _C.ENTITY:
                        problem_type[col_name] = _C.CLASSIFICATION
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
