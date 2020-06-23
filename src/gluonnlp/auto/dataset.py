import collections
import warnings
import pandas as pd
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
    for sample in data:
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
    column_property_dict
        Dictionary of column properties
    """
    column_property_dict = collections.OrderedDict()
    # Process all feature columns
    if column_names is None:
        column_names = df.columns
    for col_name in column_names:
        if metadata is not None and col_name in metadata:
            col_type = metadata[col_name]['type']
            if col_type == _C.CATEGORICAL:
                column_property_dict[col_name] = CategoricalColumnProperty(df[col_name])
                continue
            elif col_type == _C.TEXT:
                column_property_dict[col_name] = TextColumnProperty(df[col_name])
                continue
            elif col_type == _C.NUMERICAL:
                column_property_dict[col_name] = NumericalColumnProperty(df[col_name])
                continue
            elif col_type == _C.ENTITY:
                parent = metadata[col_name]['parent']
                column_property_dict[col_name] = EntityColumnProperty(column_data=df[col_name],
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
        if isinstance(ele, (str,) + INT_TYPES + FLOAT_TYPES):
            # Try to tell if the column is a text column / categorical column
            is_categorical = is_categorical_column(df[col_name])
            if is_categorical:
                column_property_dict[col_name] = CategoricalColumnProperty(df[col_name])
                continue
        if isinstance(ele, str):
            column_property_dict[col_name] = TextColumnProperty(df[col_name])
            continue
        # Raise error if we find an entity column
        if isinstance(ele, list):
            if isinstance(ele[0], (tuple, dict)):
                raise ValueError('An Entity column "{}" is found but no metadata is given.'
                                 .format(col_name))
        elif isinstance(ele, dict):
            raise ValueError('An Entity column "{}" is found but no metadata is given.'
                             .format(col_name))
        column_property_dict[col_name] = NumericalColumnProperty(df[col_name])
    return column_property_dict


class TabularNLPDataset:
    def __init__(self, path_or_df: Union[str, pd.DataFrame],
                 feature_columns: Optional[Union[str, List[str]]] = None,
                 label_columns: Union[str, List[str]] = None,
                 metadata: Dict[str, str] = None,
                 column_property_dict: Optional[collections.OrderedDict] = None,
                 problem_type: Dict[str, str] = None,
                 is_test: bool = False):
        """

        Parameters
        ----------
        path_or_df
            The path or dataframe of the tabular dataset for NLP.
        feature_columns
            Name of the feature columns
        label_columns
            Name of the label columns
        metadata
            The metadata object that describes the property of the columns in the dataset
        column_property_dict
            The given column properties
        problem_type
            The type of the problem that we will solve for the label column.
            If it is not given, we will infer the problem type from the input.
        is_test
            Whether it is a test dataset
        """
        self._is_test = is_test
        if not isinstance(path_or_df, pd.DataFrame):
            # Assume pickle.
            df = pd.read_pickle(path_or_df)
        else:
            df = path_or_df
        # Parse the feature + label columns
        if feature_columns is None and label_columns is None:
            raise ValueError('Must specify either the "feature_columns" or "label_columns".')
        if feature_columns is None and label_columns is not None:
            if isinstance(label_columns, str):
                label_columns = [label_columns]
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
        if problem_type is not None:
            for col_name in label_columns:
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
                    raise NotImplementedError('Unsupported problem_type="{}"'.format(problem_type))
        # Inference the column properties
        if column_property_dict is None:
            column_property_dict = get_column_properties(table, metadata=metadata)
        # Ignore some unused columns
        feature_columns = [col_name for col_name in feature_columns
                           if col_name in column_property_dict]
        label_columns = [col_name for col_name in label_columns
                         if col_name in column_property_dict]
        self._column_property_dict = column_property_dict
        self._feature_columns = feature_columns
        self._label_columns = label_columns
        if label_columns is not None:
            if isinstance(label_columns, str):
                label_columns = [label_columns]
            if problem_type is None:
                problem_type = dict()
                # Try to infer the problem type of each label
                for col_name in label_columns:
                    if self._column_property_dict[col_name].type == _C.CATEGORICAL:
                        problem_type[col_name] = _C.CLASSIFICATION
                    elif self._column_property_dict[col_name].type == _C.NUMERICAL:
                        problem_type[col_name] = _C.REGRESSION
                    elif self._column_property_dict[col_name].type == _C.ENTITY:
                        raise NotImplementedError('Cannot does not support the label column '
                                                  'to be entity column')
                    else:
                        raise NotImplementedError
            self._problem_type = problem_type
        else:
            self._problem_type = None

    @property
    def problem_type(self):
        return self._problem_type

    @property
    def is_test(self):
        return self._is_test

    @property
    def table(self):
        return self._table

    @property
    def column_property_dict(self):
        return self._column_property_dict

    @property
    def feature_columns(self):
        return self._feature_columns

    @property
    def label_columns(self):
        return self._label_columns

    def __repr__(self):
        ret = 'Problem Type: {}, Is Test Dataset: {}\n'.format(self.problem_type, self.is_test)
        ret += 'Feature Columns:\n\n'
        for col_name in self.feature_columns:
            ret += '- ' + str(self.column_property_dict[col_name])
        ret += '\n'
        ret += 'Label Columns:\n\n'
        for col_name in self.label_columns:
            ret += '- ' + str(self.column_property_dict[col_name])
        return ret
