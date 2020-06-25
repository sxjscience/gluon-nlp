import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.util import use_np
from . import constants as _C


@use_np
class BERTForTabularClassification(HybridBlock):
    """The basic model for tabular data classification + regression with
    BERT (and its variants like ALBERT + ELECTRA) as the backbone for handling text data.

    Here, we use the backbone network to extract the contextual embeddings and use
    another dense layer to map the contextual embeddings to the class scores

    """
    def __init__(self, text_backbone,
                 field_infos,
                 label_name,
                 weight_initializer=None,
                 bias_initializer=None,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        text_backbone
            Backbone network for handling the text data
        field_infos
            The field information. Each will be a tuple:
            - (field_type, attributes)
        weight_initializer
        bias_initializer
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.text_backbone = text_backbone
            self.field_infos = field_infos
            self.out_layer = nn.Dense(units=num_class,
                                      flatten=False,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      prefix='out_layer_')

    def forward(self, features):
        """

        Parameters
        ----------
        features

        Returns
        -------
        logits_or_scores
            Shape (batch_size, V)
        """
