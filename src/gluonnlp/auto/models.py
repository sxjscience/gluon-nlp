import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.util import use_np
from gluonnlp.layers import get_activation
from gluonnlp.op import select_vectors_by_position
from gluonnlp.attention_cell import masked_logsoftmax, masked_softmax


@use_np
class BERTForTextClassification(HybridBlock):
    """The basic model for NLP classification.

    Here, we use the backbone network to extract the contextual embeddings and use
    another dense layer to map the contextual embeddings to the class logits / score

    """
    def __init__(self, backbone, num_class,
                 weight_initializer=None, bias_initializer=None,
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.backbone = backbone
            self.out_layer = nn.Dense(units=num_class, flatten=False,
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
