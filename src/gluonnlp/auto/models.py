import numpy as np
import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.util import use_np
from . import constants as _C
from ..utils.config import CfgNode
from ..layers import get_activation, get_norm_layer


class CategoricalFeatureNet(HybridBlock):
    def __init__(self, num_class, units, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = self.get_cfg()
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim=num_class,
                                          output_dim=cfg.emb_units,
                                          weight_initializer=embed_initializer)
            self.

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.emb_units = 64
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.embed = ['xavier', 'gaussian', 'in', 1.0]
            return cfg
        else:
            raise NotImplementedError


class NumericalFeatureNet(HybridBlock):
    def __init__(self, input_shape, units, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = self.get_cfg()
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
    def __init__(self, num_fields, out_shape, units, in_units,
                 cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = self.get_cfg()
        self.cfg = cfg
        self.num_fields = num_fields
        self.out_shape = out_shape
        self.in_units = in_units
        weight_initializer = mx.init.create(*self.cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*self.cfg.INITIALIZER.bias)
        num_out_units = int(np.prod(out_shape))
        with self.name_scope():
            self.proj = nn.HybridSequential()
            in_units = in_units
            for i in range(cfg.num_layers):
                if i == cfg.num_layers - 1:
                    # Generate the output layer
                    self.proj.add(nn.Dense(units=num_out_units,
                                           in_units=in_units,
                                           weight_initializer=weight_initializer,
                                           bias_initializer=bias_initializer,
                                           flatten=False))
                    in_units = num_out_units
                else:
                    # We apply the Dense + BN + Activation
                    self.proj.add(nn.Dense(units=self.cfg.units,
                                           in_units=in_units,
                                           flatten=False,
                                           weight_initializer=weight_initializer,
                                           bias_initializer=bias_initializer,
                                           use_bias=False))
                    self.proj.add(get_norm_layer(self.cfg.normalization,
                                                 axis=-1,
                                                 epsilon=self.cfg.norm_eps,
                                                 in_channels=in_units))
                    self.proj.add(get_activation(self.cfg.activation))
                    in_units = self.cfg.units

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.pool_type = 'mean'
            cfg.num_layers = 1
            cfg.dropout = 0.2
            cfg.activation = 'leaky'
            cfg.normalization = 'batch_norm'
            cfg.norm_eps = 1e-5
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]
            cfg.INITIALIZER.bias = ['zeros']
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
            # TODO(sxjscience) May try to implement more advanced pooling methods for
            #  multimodal data.
            raise NotImplementedError
        scores = self.proj(agg_features)
        if len(self.out_shape) != 1:
            scores = mx.np.reshape((-1,) + self.out_shape)
        return scores


@use_np
class BERTForTabularClassificationV1(HybridBlock):
    """The basic model for tabular classification + regression with
    BERT (and its variants like ALBERT, MobileBERT, ELECTRA, etc.)
     as the backbone for handling text data.

    Here, we use the backbone network to extract the contextual embeddings and use
    another dense layer to map the contextual embeddings to the class scores.

    Input:

    TextField + EntityField ---------> TextNet ---------> TextFeature
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
        units = cfg.units
        with self.name_scope():
            self.text_backbone = text_backbone
            self.field_infos = field_infos
            self.label_name = label_name
            label_field_type_code, label_field_attrs = self.field_infos[label_name]
            if label_field_type_code == _C.CATEGORICAL:
                num_class = label_field_attrs['col_prop'].num_class
                self.out_layer = FeatureAggregator(num_fields=len(field_infos) - 1,
                                                   in_units=units)
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
            cfg.units = 768
            cfg.AGG_NET = FeatureAggregator.get_cfg()
            cfg.CATEGORICAL_NET = CategoricalFeatureNet.get_cfg()
            cfg.NUMERICAL_NET = NumericalFeatureNet.get_cfg()
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


