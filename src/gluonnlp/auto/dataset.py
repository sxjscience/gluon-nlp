import collections
import warnings
import pandas as pd
import json
from . import constants as _C
from .column_property import CategoricalColumnProperty, EntityColumnProperty,\
                             TextColumnProperty, NumericalColumnProperty
from ..base import INT_TYPES, FLOAT_TYPES
from typing import List, Optional, Union, Dict, Tuple


def load_pandas_df(data: Union[str, pd.DataFrame]):
    if isinstance(data, pd.DataFrame):
        return data
    loading_error = dict()
    df = None
    for loader, fmt in [(pd.read_pickle, 'pickle'),
                        (pd.read_parquet, 'parquet'),
                        (pd.read_csv, 'csv')]:
        try:
            df = loader(data)
            break
        except Exception as err:
            loading_error[fmt] = err
    if df is not None:
        return df
    else:
        raise Exception(loading_error)


def get_column_properties_from_metadata(metadata):
    """Generate the column properties from metadata

    Parameters
    ----------
    metadata
        The path to the metadata json file. Or the loaded meta data

    Returns
    -------
    column_properties
        The column properties
    """
    column_properties = collections.OrderedDict()
    if metadata is None:
        return column_properties
    if isinstance(metadata, str):
        with open(metadata, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        assert isinstance(metadata, dict)
    for col_name in metadata:
        col_type = metadata[col_name]['type']
        col_attrs = metadata[col_name]['attrs']
        if col_type == _C.TEXT:
            column_properties[col_name] = TextColumnProperty(**col_attrs)
        elif col_type == _C.ENTITY:
            column_properties[col_name] = EntityColumnProperty(**col_attrs)
        elif col_type == _C.NUMERICAL:
            column_properties[col_name] = NumericalColumnProperty(**col_attrs)
        elif col_type == _C.CATEGORICAL:
            column_properties[col_name] = CategoricalColumnProperty(**col_attrs)
        else:
            raise KeyError('Column type is not supported.'
                           ' Type="{}"'.format(col_type))
    return column_properties


def is_categorical_column(data: pd.Series,
                          threshold: int = 100,
                          ratio: float = 0.1,
                          is_label_columns: bool = False,
                          default_allow_missing: bool = True) -> Tuple[bool, bool]:
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
    is_label_columns
        Whether the column is a label column
    ratio
        The ratio for detecting categorical column

    Returns
    -------
    is_categorical
        Whether the column is a categorical column
    parsed_allow_missing
    """
    threshold = min(int(len(data) * ratio), threshold)
    sample_set = set()
    dtype = type(data[data.first_valid_index])
    if isinstance(dtype, str):
        for idx, sample in data.items():
            sample_set.add(sample)
            if len(sample_set) > threshold:
                return False, False
        if is_label_columns:
            return True, False
        else:
            return True, default_allow_missing
    elif isinstance(dtype, int):
        value_counts = data.value_counts()
        if data.min() == 0 and data.max() == len(value_counts) - 1:
            return True, False
        else:
            return False, False
    elif isinstance(dtype, bool):
        return True, False
    else:
        return False, False



def get_column_properties(
        df: 'DataFrame',
        label_columns: Union[str, List[str]],
        metadata: Optional[Dict] = None,
        provided_column_properties: Optional[Dict] = None,
        categorical_default_handle_missing_value: bool = True) -> collections.OrderedDict:
    """Inference the column types of the data frame

    Parameters
    ----------
    df
        Pandas Dataframe
    label_columns
        The chosen column names of the table
    metadata
        The additional metadata object to help specify the column types
        {'col_name': {'type':
    provided_column_properties
        The column properties provided.
        For example, these can be the column properties of the training set and you provide this
        to help inference the column properties of the dev/test set.
    categorical_default_handle_missing_value
        Whether to handle missing values for categorical columns by default

    Returns
    -------
    column_properties
        Dictionary of column properties

    """
    if label_columns is None:
        label_columns_set = set()
    elif isinstance(label_columns, str):
        label_columns_set = set([label_columns])
    else:
        label_columns_set = set(label_columns)
    column_properties = collections.OrderedDict()
    # Process all feature columns
    column_properties_from_metadata = get_column_properties_from_metadata(metadata)
    for col_name in df.columns:
        if provided_column_properties is not None and col_name in provided_column_properties:
            column_properties[col_name] = provided_column_properties[col_name].clone()
            column_properties[col_name].parse(df[col_name])
            continue
        if col_name in column_properties_from_metadata:
            column_properties[col_name] = column_properties_from_metadata[col_name].clone()
            column_properties[col_name].parse(df[col_name])
            continue
        idx = df[col_name].first_valid_index()
        if idx is None:
            # No valid index, it should have been handled previously
            raise ValueError('Column Name="{}" has no valid data and is ignored.'.format(col_name))
        ele = df[col_name][idx]
        # Try to inference the categorical column
        if isinstance(ele, collections.abc.Hashable) and not isinstance(ele, FLOAT_TYPES):
            # Try to tell if the column is a categorical column
            is_categorical, allow_missing = is_categorical_column(
                df[col_name],
                is_label_columns=col_name in label_columns_set)
            if is_categorical:
                column_properties[col_name] = CategoricalColumnProperty(allow_missing=allow_missing)
                column_properties[col_name].parse(df[col_name])
                continue
        if isinstance(ele, str):
            column_properties[col_name] = TextColumnProperty()
            column_properties[col_name].parse(df[col_name])
            continue
        # Raise error if we find an entity column
        if isinstance(ele, list):
            if isinstance(ele[0], (tuple, dict)):
                raise ValueError('An Entity column "{}" is found but no metadata is given.'
                                 .format(col_name))
        elif isinstance(ele, dict):
            raise ValueError('An Entity column "{}" is found but no metadata is given.'
                             .format(col_name))
        column_properties[col_name] = NumericalColumnProperty()
        column_properties[col_name].parse(df[col_name])
    return column_properties


def normalize_df(df, convert_text_to_numerical=False, remove_none=True):
    """Try to convert the text columns in the input data-frame to numerical columns

    Parameters
    ----------
    df
        The DataFrame
    convert_text_to_numerical
        Whether to convert text columns to numerical columns
    remove_none
        Whether to try to remove None values in the sample.

    Returns
    -------
    new_df
        The normalized dataframe
    """
    conversion_cols = dict()
    for col_name in df.columns:
        col = df[col_name]
        idx = col.first_valid_index()
        if idx is not None:
            val = col[idx]
            if isinstance(val, str):
                num_missing = col.isnull().sum().sum().item()
                if num_missing > 0 and remove_none:
                    col = col.fillna('')
                    conversion_cols[col_name] = col
                if convert_text_to_numerical:
                    try:
                        col = pd.to_numeric(col)
                        conversion_cols[col_name] = col
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
                 *,
                 columns=None,
                 label_columns=None,
                 column_metadata: Optional[Union[str, Dict]] = None,
                 column_properties: Optional[collections.OrderedDict] = None,
                 categorical_default_handle_missing_value=True):
        """

        Parameters
        ----------
        path_or_df
            The path or dataframe of the tabular dataset for NLP.
        columns
            The chosen columns to load the data
        label_columns
            The name of the label columns. This helps to infer the column properties.
        column_metadata
            The metadata object that describes the property of the columns in the dataset
        column_properties
            The given column properties
        categorical_default_handle_missing_value
            Whether to handle missing value in categorical columns by default
        """
        df = load_pandas_df(path_or_df)
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            df = df[columns]
        df = normalize_df(df)
        if column_metadata is None:
            column_metadata = dict()
        elif isinstance(column_metadata, str):
            with open(column_metadata, 'r') as f:
                column_metadata = json.load(f)
        # Inference the column properties
        column_properties = get_column_properties(
            df,
            metadata=column_metadata,
            label_columns=label_columns,
            provided_column_properties=column_properties,
            categorical_default_handle_missing_value=categorical_default_handle_missing_value)
        self._table = df
        self._column_properties = column_properties

    @property
    def columns(self):
        return list(self._table.columns)

    @property
    def table(self):
        return self._table

    @property
    def column_properties(self):
        return self._column_properties

    def column_metadata(self):
        metadata = dict()
        for col_name, col_prop in self.column_properties.items():
            metadata[col_name] = {'type': col_prop.type,
                                  'attrs': col_prop.get_attributes()}
        return metadata

    def save_column_metadata(self, save_path):
        with open(save_path, 'w', encoding='utf-8') as of:
            json.dump(self.column_metadata(), of, ensure_ascii=False)

    def infer_problem_type(self, label_col_name):
        if self.column_properties[label_col_name].type == _C.CATEGORICAL:
            return _C.CLASSIFICATION, self.column_properties[label_col_name].num_class
        elif self.column_properties[label_col_name].type == _C.NUMERICAL:
            return _C.REGRESSION, self.column_properties[label_col_name].shape
        else:
            raise NotImplementedError('Cannot infer the problem type')

    def __str__(self):
        ret = 'Columns:\n\n'
        for col_name in self.column_properties.keys():
            ret += '- ' + str(self.column_properties[col_name])
        ret += '\n'
        return ret
