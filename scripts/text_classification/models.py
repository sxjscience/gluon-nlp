import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.util import use_np
from gluonnlp.layers import get_activation
from gluonnlp.op import select_vectors_by_position
from gluonnlp.attention_cell import masked_logsoftmax, masked_softmax


@use_np
class BertForTextClassification(HybridBlock):
    """The basic model for text classification.

    Here, we directly use the backbone network to extract the contextual embeddings and use
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

    def hybrid_forward(self, F, tokens, token_types, valid_length, mask):
        """

        Parameters
        ----------
        F
        tokens
            Shape (batch_size, seq_length)
        token_types
            Shape (batch_size, seq_length)
        valid_length
            Shape (batch_size,)
        mask
            Whether the tokens are masked or not
            Shape (batch_size, seq_length)

        Returns
        -------
        logits_or_scores
            Shape (batch_size, V)
        """
        contextual_embedding, pooling_out = self.backbone(tokens, token_types, valid_length)
        logits_or_scores = self.out_layer(pooling_out)
        return logits_or_scores
