from collections import namedtuple
from . import constants as _C
import numpy as np
from mxnet.gluon.data import batchify as bf


class TextTokenIdsField:
    type = _C.TEXT

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
        if segment_ids is None:
            self.segment_ids = np.zeros_like(token_ids)
        else:
            self.segment_ids = segment_ids

    @classmethod
    def batchify(cls, round_to=None):
        """Get the batchify function. The batchify function takes a list of samples.

        Parameters
        ----------
        round_to
            The round_to option. Usually, the

        Returns
        -------
        batchify_fn
            The returned batchify function
        """
        pad_batchify = bf.Pad(round_to=round_to)
        stack_batchify = bf.Stack()

        def batchify_fn(data):
            """

            Parameters
            ----------
            data

            Returns
            -------
            batch_token_ids
                (batch_size, sequence_length)
            batch_segment_ids
                (batch_size, sequence_length)
            batch_valid_length
                (batch_size,)
            """
            batch_token_ids = pad_batchify([ele.token_ids for ele in data])
            batch_segment_ids = pad_batchify([ele.segment_ids for ele in data])
            batch_valid_length = stack_batchify([len(ele.token_ids) for ele in data])
            return batch_token_ids, batch_segment_ids, batch_valid_length
        return batchify_fn

    def __str__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'token_ids={}\n'.format(self.token_ids)
        ret += 'length={}\n'.format(self.token_ids)
        ret += 'segment_ids={}\n'.format(self.token_ids)
        ret += ')\n'
        return ret


class EntityField:
    type = _C.ENTITY

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

    @classmethod
    def batchify(cls):
        pad_batchify = bf.Pad()
        stack_batchify = bf.Stack()

        def batchify_fn(data):
            """The internal batchify function

            Parameters
            ----------
            data
                The input data.

            Returns
            -------
            batch_span
                Shape (batch_size, #num_entities, 2)
            batch_label
                Shape (batch_size, #num_entities) + label_shape
            batch_num_entity
                Shape (batch_size,)
            """
            batch_span = pad_batchify([ele.data for ele in data])
            no_label = data[0].label is None
            if no_label:
                batch_label = None
            else:
                batch_label = pad_batchify([ele.label for ele in data])
            batch_num_entity = stack_batchify([len(ele.data) for ele in data])
            return batch_span, batch_label, batch_num_entity
        return batchify_fn

    def __str__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'data={}\n'.format(self.data)
        ret += 'label={}\n'.format(None if self.label is None else self.label)
        ret += ')\n'
        return ret


class NumericalField:
    type = _C.NUMERICAL

    def __init__(self, data):
        self.data = data

    @classmethod
    def batchify(cls):
        stack_batchify = bf.Stack()

        def batchify_fn(samples):
            """

            Parameters
            ----------
            samples

            Returns
            -------
            dat
                Shape (batch_size,) + sample_shape
            """
            return stack_batchify([ele.data for ele in samples])
        return batchify_fn

    def __str__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'data={}\n'.format(self.data)
        ret += ')\n'
        return ret


class CategoricalField(NumericalField):
    type = _C.CATEGORICAL
    pass