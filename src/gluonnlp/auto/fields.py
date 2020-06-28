from . import constants as _C
from mxnet.gluon.data import batchify as bf


class FeatureTuple:
    def __init__(self, *args):
        self.data = args

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def as_in_ctx(self, ctx):
        new_data = [self.data.as_in_ctx(ctx) if self.data is not None else None]
        return FeatureTuple(*new_data)


class TextTokenIdsField:
    type = _C.TEXT

    def __init__(self, token_ids, segment_ids=None, token_offsets=None):
        """

        Parameters
        ----------
        token_ids
            The token_ids, shape (seq_length,)
        segment_ids
            The segment_ids, shape (seq_length,)
        token_offsets
            The character-level offsets of the token, shape (seq_length, 2)
        """
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.token_offsets = token_offsets

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
            batch_valid_length
                (batch_size,)
            batch_segment_ids
                (batch_size, sequence_length)
            batch_token_offsets
                (batch_size, seq_length, 2)
            """
            batch_token_ids = pad_batchify([ele.token_ids for ele in data])
            batch_valid_length = stack_batchify([len(ele.token_ids) for ele in data])
            if data[0].segment_ids is None:
                batch_segment_ids = None
            else:
                batch_segment_ids = pad_batchify([ele.segment_ids for ele in data])
            if data[0].token_offsets is None:
                batch_token_offsets = None
            else:
                batch_token_offsets = pad_batchify([ele.token_offsets for ele in data])
            return FeatureTuple(batch_token_ids, batch_valid_length,
                                batch_segment_ids, batch_token_offsets)
        return batchify_fn

    def __str__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'token_ids={}\n'.format(self.token_ids)
        ret += 'segment_ids={}\n'.format(self.segment_ids)
        ret += 'token_offsets={}\n'.format(self.token_offsets)
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
            return FeatureTuple(batch_span, batch_label, batch_num_entity)
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
