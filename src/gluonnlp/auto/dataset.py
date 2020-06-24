import collections
import warnings
import pandas as pd
import json
from . import constants as _C
from .column_property import CategoricalColumnProperty, EntityColumnProperty,\
                             TextColumnProperty, NumericalColumnProperty
from ..base import INT_TYPES, FLOAT_TYPES
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
    for idx, sample in data.items():
        sample_set.add(sample)
        if len(sample_set) > threshold:
            return False
    return True


def get_column_properties(df, column_names: Optional[List[str]] = None,
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
    column_properties
        Dictionary of column properties

    """
    column_properties = collections.OrderedDict()
    # Process all feature columns
    if column_names is None:
        column_names = df.columns
    for col_name in column_names:
        if metadata is not None and col_name in metadata:
            col_type = metadata[col_name]['type']
            if col_type == _C.CATEGORICAL:
                column_properties[col_name] = CategoricalColumnProperty(df[col_name])
                continue
            elif col_type == _C.TEXT:
                column_properties[col_name] = TextColumnProperty(df[col_name])
                continue
            elif col_type == _C.NUMERICAL:
                column_properties[col_name] = NumericalColumnProperty(df[col_name])
                continue
            elif col_type == _C.ENTITY:
                parent = metadata[col_name]['parent']
                column_properties[col_name] = EntityColumnProperty(column_data=df[col_name],
                                                                      parent=parent)
                continue
            else:
                raise KeyError('Column type is not supported.'
                               ' Type="{}"'.format(col_type))
        idx = df[col_name].first_valid_index()
        if idx is None:
            # No valid index, it's safe to ignore the column
            warnings.warn('Column Name="{}" has no valid data and is ignored.'.format(col_name))
            continue
        ele = df[col_name][idx]
        # Try to inference the categorical column
        if isinstance(ele, collections.Hashable) and not isinstance(ele, FLOAT_TYPES):
            # Try to tell if the column is a categorical column
            is_categorical = is_categorical_column(df[col_name])
            if is_categorical:
                column_properties[col_name] = CategoricalColumnProperty(df[col_name])
                continue
        if isinstance(ele, str):
            column_properties[col_name] = TextColumnProperty(df[col_name])
            continue
        # Raise error if we find an entity column
        if isinstance(ele, list):
            if isinstance(ele[0], (tuple, dict)):
                raise ValueError('An Entity column "{}" is found but no metadata is given.'
                                 .format(col_name))
        elif isinstance(ele, dict):
            raise ValueError('An Entity column "{}" is found but no metadata is given.'
                             .format(col_name))
        column_properties[col_name] = NumericalColumnProperty(df[col_name])
    return column_properties


def convert_text_to_numeric_df(df):
    """Try to convert the text columns in the input data-frame to numerical columns

    Returns
    -------
    new_df
    """
    conversion_cols = dict()
    for col_name in df.columns:
        try:
            dat = pd.to_numeric(df[col_name])
            conversion_cols[col_name] = dat
        except Exception:
            pass
        finally:
            pass
    if len(conversion_cols) == 0:
        return df
    else:
        series_l = dict()
        for col_name in df.columns:
            if col_name in conversion_cols:
                series_l[col_name] = conversion_cols[col_name]
            else:
                series_l[col_name] = df[col_name]
        return pd.DataFrame(series_l)


class TabularDataset:
    def __init__(self, path_or_df: Union[str, pd.DataFrame],
                 feature: Optional[Union[str, List[str]]] = None,
                 label: Union[str, List[str]] = None,
                 metadata: Optional[Union[str, Dict]] = None,
                 column_properties: Optional[collections.OrderedDict] = None):
        """

        Parameters
        ----------
        path_or_df
            The path or dataframe of the tabular dataset for NLP.
        feature
            Name of the feature columns
        label
            Name of the label columns
        metadata
            The metadata object that describes the property of the columns in the dataset
        column_properties
            The given column properties
        """
        if not isinstance(path_or_df, pd.DataFrame):
            # Assume pickle.
            df = pd.read_pickle(path_or_df)
        else:
            df = path_or_df
        df = convert_text_to_numeric_df(df)
        # Parse the feature + label columns
        feature_columns = feature
        label_columns = label
        if feature_columns is None and label_columns is None:
            feature_columns = df.columns
            label_columns = None
        elif feature_columns is None and label_columns is not None:
            if isinstance(label, str):
                label_columns = [label]
            # Inference the feature columns based on label_columns
            feature_columns = [ele for ele in df.columns if ele not in label_columns]
        else:
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]
            if isinstance(label_columns, str):
                label_columns = [label_columns]
        all_columns = feature_columns.copy()
        if label_columns is not None:
            all_columns.extend(label_columns)
        table = df[all_columns]
        if metadata is None:
            metadata = dict()
        elif isinstance(metadata, str):
            with open(metadata, 'r') as f:
                metadata = json.load(f)
        # Inference the column properties
        if column_properties is None:
            column_properties = get_column_properties(table,
                                                      metadata=metadata)
        else:
            column_properties = collections.OrderedDict(
                [(col_name, column_properties[col_name].parse_other(df[col_name]))
                 for col_name in all_columns])
        # Ignore some unused columns
        feature_columns = [col_name for col_name in feature_columns
                           if col_name in column_properties]
        if label_columns is not None:
            label_columns = [col_name for col_name in label_columns
                             if col_name in column_properties]
        table = df[list(column_properties.keys())]
        self._table = table
        self._column_properties = column_properties
        self._feature_columns = feature_columns
        self._label_columns = label_columns

    @property
    def table(self):
        return self._table

    @property
    def column_properties(self):
        return self._column_properties

    @property
    def feature_columns(self):
        return self._feature_columns

    @property
    def label_columns(self):
        return self._label_columns

    def infer_problem_type(self):
        problem_type = dict()
        for col_name in self.label_columns:
            if self.column_properties[col_name].type == _C.CATEGORICAL:
                problem_type[col_name] = _C.CLASSIFICATION
            elif self.column_properties[col_name].type == _C.NUMERICAL:
                problem_type[col_name] = _C.REGRESSION
            else:
                raise NotImplementedError('Cannot infer the problem type')
        return problem_type

    def __str__(self):
        ret = 'Feature Columns:\n\n'
        for col_name in self.feature_columns:
            ret += '- ' + str(self.column_properties[col_name])
        ret += '\n'
        if self.label_columns is not None:
            ret += 'Label Columns:\n\n'
            for col_name in self.label_columns:
                ret += '- ' + str(self.column_properties[col_name])
        return ret
