from collections import OrderedDict
from typing import Dict
import numpy as np
import mxnet.gluon.data.batchify as bf
from ..utils.preprocessing import get_trimmed_lengths, match_tokens_with_char_spans
from . import constants as _C
from .column_property import ColumnProperty


class TextTokenIdsField:
    def __init__(self, token_ids, segment_ids=None):
        """

        Parameters
        ----------
        token_ids
            The token_ids, shape (seq_length,)
        segment_ids
            The segment_ids, shape (seq_length,)
        """
        self.token_ids = token_ids
        self.length = len(token_ids)
        if segment_ids is None:
            self.segment_ids = np.zeros_like(token_ids)
        else:
            self.segment_ids = segment_ids

    def batchify(self, round_to=None):
        """Get the batchify function. The batchify function takes a list of samples.

        Parameters
        ----------
        round_to
            The round to option

        Returns
        -------
        batchify_fn
            The batchify function
        """
        pad_batchify = bf.Pad(round_to=round_to)
        stack_batchify = bf.Stack()

        def batchify_fn(data):
            batch_token_ids = pad_batchify([ele.token_ids for ele in data])
            batch_segment_ids = pad_batchify([ele.segment_ids for ele in data])
            batch_valid_length = stack_batchify([ele.length for ele in data])
            return batch_token_ids, batch_segment_ids, batch_valid_length
        return batchify_fn

    def __repr__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'token_ids={}\n'.format(self.token_ids)
        ret += 'length={}\n'.format(self.token_ids)
        ret += 'segment_ids={}\n'.format(self.token_ids)
        ret += ')\n'
        return ret


class EntityField:
    def __init__(self, data, label=None):
        """

        Parameters
        ----------
        data
            (#Num Entities, 2)
        label
            (#Num Entities,)
        """
        self.data = data
        self.label = label

    def batchify(self):
        raise NotImplementedError

    def __repr__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'data={}\n'.format(self.data)
        ret += 'label={}\n'.format(None if self.label is None else self.label)
        ret += ')\n'
        return ret


class ArrayField:
    def __init__(self, data):
        self.data = data

    def batchify(self):
        stack_batchify = bf.Stack()

        def batchify_fn(samples):
            return stack_batchify([ele.data for ele in samples])
        return batchify_fn

    def __repr__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'data={}\n'.format(self.data)
        ret += ')\n'
        return ret


class TabularBERTPreprocessor:
    def __init__(self, tokenizer,
                 column_properties: Dict[str, ColumnProperty],
                 max_length: int,
                 merge_text: bool = True):
        """Preprocess the inputs to work with a pretrained model.

        Parameters
        ----------
        tokenizer
            The tokenizer
        column_properties
            A dictionary that contains the column properties
        max_length
            The maximum length of the encoded token sequence.
        merge_text
            Whether to merge the token_ids when there are multiple text fields.
            For example, we will merge the text fields as
            [CLS] token_ids1 [SEP] token_ids2 [SEP] token_ids3 [SEP] token_ids4 [SEP] ...
        """
        self._tokenizer = tokenizer
        self._column_properties = column_properties
        self._max_length = max_length
        self._merge_text = merge_text
        self._text_columns = []
        self._entity_columns = []
        self._categorical_columns = []
        self._numerical_columns = []
        self._col_idx_map = {col_name: col_id
                             for col_id, col_name in enumerate(self._column_properties.keys())}
        # TODO(sxjscience) Refactor the implementation
        for col_name, col_info in self._column_properties.items():
            if col_info.type == _C.TEXT:
                self._text_columns.append(col_name)
            elif col_info.type == _C.ENTITY:
                self._entity_columns.append(col_name)
            elif col_info.type == _C.CATEGORICAL:
                self._categorical_columns.append(col_name)
            elif col_info.type == _C.NUMERICAL:
                self._numerical_columns.append(col_name)
            else:
                raise NotImplementedError
        self._text_column_require_offsets = {col_name: False for col_name in self.text_columns}
        for col_name in self._entity_columns:
            self._text_column_require_offsets[self.column_properties[col_name].parent] = True

    @property
    def max_length(self):
        return self._max_length

    @property
    def column_properties(self):
        return self._column_properties

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
                out_types.append((_C.TEXT, dict()))
            else:
                for i, col_name in enumerate(self.text_columns):
                    text_col_idx[col_name] = i
                    out_types.append((_C.TEXT, dict()))
        if len(self.entity_columns) > 0:
            for col_name in self.entity_columns:
                parent = self.column_properties[col_name].parent
                if self.merge_text:
                    parent_idx = 0
                else:
                    parent_idx = text_col_idx[parent]
                out_types.extend((_C.ENTITY,
                                  {'parent_idx': parent_idx,
                                   'col_prop': self.column_properties[col_name]}))
        if len(self.categorical_columns) > 0:
            out_types.extend([(_C.CATEGORICAL,
                               {'col_prop': self.column_properties[col_name]})
                              for col_name in self.entity_columns])
        if len(self.numerical_columns) > 0:
            out_types.extend([(_C.NUMERICAL,
                               {'col_prop': self.column_properties[col_name]})
                              for col_name in self.numerical_columns])
        return out_types

    def __call__(self, sample):
        """Transform a sample into a list of fields.

        We organize and represent the features in the following format:

        - Text fields
            We transform text into a sequence of token_ids.
            If there are multiple text fields, we have the following options
            1) merge_text = True
                We will concatenate these text fields and inserting CLS, SEP ids, i.e.
                [CLS] text_ids1 [SEP] text_ids2 [SEP]
            2) merge_text = False
                We will transform each text field separately:
                [CLS] text_ids1 [SEP], [CLS] text_ids2 [SEP], ...
            For empty text / missing text data, we will just convert it to [CLS] [SEP]
        - Entity fields
            The raw entities are stored as character-level start and end offsets.
            After the preprocessing step, we will store them as the token-level
            start + end. Different from the raw character-level start + end offsets, the
            token-level start + end offsets will be used.
            - [(token_level_start, token_level_end, entity_label_id)]
            or
            - [(token_level_start, token_level_end)]
        - Categorical fields
            We transform the categorical features to its ids.
            We indicate the missing value with a special flag.
        - Numerical fields
            We keep the numerical features and indicate the missing value
            with a special flag.

        Parameters
        ----------
        sample
            A single data sample.

        Returns
        -------
        ret
            Sample after the transformation. Will contain the following
            - TEXT
                The encoded value will be a TextTokenIdsField

            - ENTITY
                The encoded feature will be:
                data: Shape (num_entity, 2)
                    Each item will be (start, end)
                if has_label:
                    label:
                        - Categorical: Shape (num_entity,)
                        - Numerical: (num_entity,) + label_shape

            - CATEGORICAL
                The categorical feature. Will be an integer

            - NUMERICAL
                The numerical feature. Will be a numpy array
        """
        fields = []
        # Step 1: Get the features of all text columns
        sentence_start_in_merged = None  # The start of each sentence in the merged text
        text_token_ids = OrderedDict()
        text_token_offsets = OrderedDict()
        if len(self.text_columns) > 0:
            for col_name in self.text_columns:
                col_idx = self._col_idx_map[col_name]
                if isinstance(sample[col_idx], str):
                    if self.text_column_require_offsets[col_name]:
                        token_ids, token_offsets =\
                            self._tokenizer.encode_with_offsets(sample[col_idx], int)
                        token_ids = np.array(token_ids)
                        token_offsets = np.array(token_offsets)
                    else:
                        token_ids = self._tokenizer.encode(sample[col_idx], int)
                        token_ids = np.array(token_ids)
                elif isinstance(sample[col_idx], np.ndarray):
                    if self.text_column_require_offsets[col_name]:
                        raise ValueError('Must get the offsets of all text tokens!')
                    token_ids = sample[col_idx]
                elif isinstance(sample[col_idx], tuple):
                    token_ids, token_offsets = sample[col_idx]
                else:
                    raise NotImplementedError('The input format of the text column '
                                              'cannot be understood!')
                text_token_ids[col_name] = token_ids
                if self.text_column_require_offsets[col_name]:
                    text_token_offsets[col_name] = token_offsets
            lengths = [len(text_token_ids[col_name]) for col_name in self.text_columns]
            if self.merge_text:
                # We will merge the text tokens by
                # [CLS] token_ids1 [SEP] token_ids2 [SEP]
                # 0       0          0       1        1
                trimmed_lengths = get_trimmed_lengths(lengths,
                                                      max_length=self.max_length - len(lengths) - 1,
                                                      do_merge=True)
                encoded_token_ids = [np.array([self._tokenizer.vocab.cls_id])]
                segment_ids = [np.array([0])]
                sentence_start_in_merged = dict()
                for idx, (trim_length, col_name) in enumerate(zip(trimmed_lengths,
                                                                  self.text_columns)):
                    sentence_start_in_merged[col_name] = len(encoded_token_ids)
                    encoded_token_ids.append(text_token_ids[col_name][:trim_length])
                    segment_ids.append(np.full_like(encoded_token_ids[-1], idx))
                    encoded_token_ids.append(np.array([self._tokenizer.vocab.sep_id]))
                    segment_ids.append(np.array([idx]))
                encoded_token_ids = np.concatenate(encoded_token_ids).astype(np.int32)
                segment_ids = np.concatenate(segment_ids).astype(np.int32)
                fields.append(TextTokenIdsField(encoded_token_ids, segment_ids))
            else:
                # We encode each sentence independently
                # [CLS] token_ids1 [SEP], [CLS] token_ids2 [SEP]
                #  0     0           0  ,  0     0           0
                trimmed_lengths = get_trimmed_lengths(lengths,
                                                      max_length=self.max_length - 2,
                                                      do_merge=False)
                for trim_length, col_name in zip(trimmed_lengths, self.text_columns):
                    encoded_token_ids = np.concatenate(np.array([self._tokenizer.vocab.cls_id]),
                                                       text_token_ids[col_name][:trim_length],
                                                       np.array([self._tokenizer.vocab.sep_id]))
                    fields.append(TextTokenIdsField(encoded_token_ids.astype(np.int32),
                                                    np.zeros_like(encoded_token_ids,
                                                                  dtype=np.int32)))
        # Step 2: Transform all entity columns
        for col_name in self.entity_columns:
            col_idx = self._col_idx_map[col_name]
            entities = sample[col_idx]
            col_prop = self.column_properties[col_name]
            parent_name = col_prop.parent
            char_offsets, transformed_labels = col_prop.transform(entities)
            # Get the offsets output by the tokenizer
            token_offsets = text_token_offsets[parent_name]
            entity_token_offsets = match_tokens_with_char_spans(token_offsets=token_offsets,
                                                                spans=char_offsets)
            if self.merge_text:
                entity_token_offsets += sentence_start_in_merged[parent_name]
            else:
                entity_token_offsets += 1  # Add the offset w.r.t the cls token.
            fields.append(EntityField(entity_token_offsets, transformed_labels))

        # Step 3: Transform all categorical columns
        for col_name in self.categorical_columns:
            col_idx = self._col_idx_map[col_name]
            col_prop = self.column_properties[col_name]
            transformed_labels = col_prop.transform(sample[col_idx])
            fields.append(ArrayField(transformed_labels))

        # Step 4: Transform all numerical columns
        for col_name in self.numerical_columns:
            col_idx = self._col_idx_map[col_name]
            col_prop = self.column_properties[col_name]
            fields.append(ArrayField(col_prop.transform(sample[col_idx])))
        return fields
