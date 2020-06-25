import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.util import use_np
from . import constants as _C
from ..utils.config import CfgNode

class VanillaCategoricalFeatureNet(HybridBlock):
    def __init__(self, num_class, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim=num_class)

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.units =


class VanillaNumericalFeatureNet(HybridBlock):
    def __init__(self, input_shape, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.input_shape = input_shape

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.normalization = 'batch_norm'
        else:
            raise NotImplementedError
        return cfg


class FeatureAggregator(HybridBlock):
    def __init__(self, num_fields, out_shape, in_units,
                 cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = self.get_cfg()
        self.cfg = cfg
        self.num_fields = num_fields
        self.out_shape = out_shape
        self.in_units = in_units
        with self.name_scope():
            self.proj_layers = nn.HybridSequential()
            self.activations = nn.HybridSequential()
            self.norm_layers = nn.HybridSequential()
            for i in range(cfg.num_layers):
                if i == cfg.num_layers - 1:
                    # Process the output layer


    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.pool_type = 'mean'
            cfg.num_layers = 1
            cfg.units = 128
            cfg.dropout = 0.2
            cfg.activation = 'leaky'
            cfg.normalization = 'batch_norm'
        else:
            raise NotImplementedError
        return cfg

    def forward(self, field_proj_features):
        """

        Parameters
        ----------
        field_proj_features
            List of projection features. All elements must have the same shape.

        Returns
        -------
        scores
            Shape (batch_size,) + out_shape
        """
        if self.cfg.pool_type == 'mean':
            if len(field_proj_features) == 0:
                agg_features = field_proj_features[0]
            else:
                agg_features = mx.np.stack(field_proj_features)
                agg_features = mx.np.mean(agg_features, axis=0)
        else:
            raise NotImplementedError
        for layer in self.layers:



class BackboneTextFeatureNet(HybridBlock):
    def __init__(self, backbone, ):


@use_np
class BERTForTabularClassificationV1(HybridBlock):
    """The basic model for tabular classification + regression with
    BERT (and its variants like ALBERT, MobileBERT, ELECTRA, etc.)
     as the backbone for handling text data.

    Here, we use the backbone network to extract the contextual embeddings and use
    another dense layer to map the contextual embeddings to the class scores.

    Input:

    TextField ---------> TextNet ---------> TextFeature
        ...
    CategoricalField --> CategoricalNet --> CategoricalFeature  ==> AggregateNet --> Logits/Scores
        ...
    NumericalField ----> NumericalNet ----> NumericalFeature
    """
    def __init__(self, text_backbone,
                 field_infos,
                 label_name,
                 cfg=None,
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
        label_name
            The name of the label column
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = self.get_cfg()
        with self.name_scope():
            self.text_backbone = text_backbone
            self.field_infos = field_infos
            self.label_name = label_name
            label_field_type_code, label_field_attrs = self.field_infos[label_name]
            if label_field_type_code == _C.CATEGORICAL:
                num_class = label_field_attrs['col_prop'].num_class
                self.out_layer = nn.Dense(units=num_class,
                                          in_units=text_backbone.units,
                                          weight_initializer=weight_initializer,
                                          flatten=False)
            self.out_layer = nn.Dense(units=num_class,
                                      in_units=text_backbone.units,
                                      flatten=False,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      prefix='out_layer_')

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]
            cfg.INITIALIZER.bias = ['zeros']
            cfg.TEXT_NET = CfgNode()
            cfg.TEXT_NET.MODEL = CfgNode()
            cfg.TEXT_NET.INITIALIZER = CfgNode()
            cfg.TEXT_NET.MODEL.intermediate_units = []
            cfg.CATEGORICAL_NET = CfgNode()
            cfg.NUMERICAL_NET = CfgNode()
            return cfg
        else:
            raise NotImplementedError

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


