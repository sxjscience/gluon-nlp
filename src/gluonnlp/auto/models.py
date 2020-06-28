import numpy as np
import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.util import use_np
from . import constants as _C
from ..utils.config import CfgNode
from ..layers import get_activation, get_norm_layer


class BasicMLP(HybridBlock):
    def __init__(self, in_shape, mid_units, out_units,
                 num_layers=1, normalization='batch_norm',
                 norm_eps=1E-5, dropout=0.1,
                 activation='leaky',
                 weight_initializer=None,
                 bias_initializer=None,
                 prefix=None, params=None):
        """
        data -> [Dense -> BN -> ACT] * N -> dropout -> Dense -> out

        Parameters
        ----------
        in_shape
        mid_units
        out_units
        num_layers
        normalization
        norm_eps
        dropout
        activation
        """
        super().__init__(prefix=prefix, params=params)
        self.in_units = int(np.prod(in_shape))
        in_units = self.in_units
        with self.name_scope():
            self.proj = nn.HybridSequential()
            with self.proj.name_scope():
                for i in range(num_layers - 1):
                    self.proj.add(nn.Dense(units=mid_units,
                                           in_units=in_units,
                                           flatten=False,
                                           weight_initializer=weight_initializer,
                                           bias_initializer=bias_initializer,
                                           use_bias=False))
                    self.proj.add(get_norm_layer(normalization,
                                                 axis=-1,
                                                 epsilon=norm_eps,
                                                 in_channels=in_units))
                    self.proj.add(get_activation(activation))
                    in_units = mid_units
                self.proj.add(nn.Dropout(dropout))
                self.proj.add(nn.Dense(units=out_units,
                                       in_units=in_units,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       flatten=False))

    def hybrid_forward(self, F, x):
        return self.proj(F.np.reshape(x, (-1, self.in_units)))


class CategoricalFeatureNet(HybridBlock):
    def __init__(self, num_class, out_units, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = self.get_cfg()
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim=num_class,
                                          output_dim=cfg.emb_units,
                                          weight_initializer=embed_initializer)
            with self.name_scope():
                self.proj = BasicMLP(in_shape=(cfg.emb_units,),
                                     mid_units=cfg.mid_units,
                                     out_units=out_units,
                                     num_layers=cfg.num_layers,
                                     normalization=cfg.normalization,
                                     norm_eps=cfg.norm_eps,
                                     dropout=cfg.dropout,
                                     activation=cfg.activation,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer)

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.emb_units = 64
            cfg.mid_units = 128
            cfg.num_layers = 1
            cfg.dropout = 0.1
            cfg.activation = 'leaky'
            cfg.normalization = 'batch_norm'
            cfg.norm_eps = 1e-5
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.embed = ['xavier', 'gaussian', 'in', 1.0]
            cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 3.0]
            cfg.INITIALIZER.bias = ['zeros']
            return cfg
        else:
            raise NotImplementedError

    def hybrid_forward(self, F, feature):
        embed = self.embedding(feature)
        return self.proj(embed)


class NumericalFeatureNet(HybridBlock):
    def __init__(self, input_shape, out_units, cfg=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = self.get_cfg()
        self.input_shape = input_shape
        self.cfg = cfg
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        with self.name_scope():
            self.proj = BasicMLP(in_shape=input_shape,
                                 mid_units=cfg.mid_units,
                                 out_units=out_units,
                                 num_layers=cfg.num_layers,
                                 normalization=cfg.normalization,
                                 norm_eps=cfg.norm_eps,
                                 dropout=cfg.dropout,
                                 activation=cfg.activation,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer)

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.mid_units = 128
            cfg.num_layers = 1
            cfg.dropout = 0.1
            cfg.activation = 'leaky'
            cfg.normalization = 'batch_norm'
            cfg.norm_eps = 1e-5
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 3.0]
            cfg.INITIALIZER.bias = ['zeros']
        else:
            raise NotImplementedError
        return cfg

    def hybrid_forward(self, F, feature):
        return self.proj(feature)


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
        weight_initializer = mx.init.create(*self.cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*self.cfg.INITIALIZER.bias)
        out_units = int(np.prod(out_shape))
        with self.name_scope():
            self.proj = nn.HybridSequential()
            if self.cfg.agg_type == 'mean':
                in_units = in_units
            elif self.cfg.agg_type == 'concat':
                in_units = in_units * num_fields
            else:
                raise NotImplementedError
            self.proj = BasicMLP(in_shape=(in_units,),
                                 mid_units=cfg.mid_units,
                                 out_units=out_units,
                                 num_layers=cfg.num_layers,
                                 normalization=cfg.normalization,
                                 norm_eps=cfg.norm_eps,
                                 dropout=cfg.dropout,
                                 activation=cfg.activation,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer)

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.agg_type = 'mean'
            cfg.mid_units = 128
            cfg.num_layers = 1
            cfg.dropout = 0.1
            cfg.activation = 'leaky'
            cfg.normalization = 'batch_norm'
            cfg.norm_eps = 1e-5
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 3.0]
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
        if len(field_proj_features) == 0:
            agg_features = field_proj_features[0]
        else:
            if self.cfg.agg_type == 'mean':
                agg_features = mx.np.stack(field_proj_features)
                agg_features = mx.np.mean(agg_features, axis=0)
            elif self.cfg.agg_type == 'concat':
                agg_features = mx.np.concatenate(field_proj_features, axis=-1)
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

    TextField + EntityField --> TextNet -------> TextFeature
        ...
    CategoricalField --> CategoricalNet --> CategoricalFeature  ==> AggregateNet --> logits/scores
        ...
    NumericalField ----> NumericalNet ----> NumericalFeature
    """
    def __init__(self, text_backbone,
                 feature_field_info,
                 label_field_info,
                 cfg=None,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        text_backbone
            Backbone network for handling the text data
        feature_field_info
            The field information of the training data. Each will be a tuple:
            - (field_type, attributes)
        label_field_info
            The idx of the label column
        cfg
            Configuration
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        if cfg is None:
            cfg = self.get_cfg()
        self.cfg = cfg
        feature_units = self.cfg.feature_units
        if feature_units == -1:
            feature_units = text_backbone.units
        with self.name_scope():
            self.text_backbone = text_backbone
            self.field_infos = field_infos
            self.label_idx = label_idx
            self.categorical_fields = []
            self.numerical_fields = []
            self.agg_layer = None
            self.categorical_networks = nn.HybridSequential()
            self.numerical_networks = nn.HybridSequential()
            for i, (field_type_code, field_attrs) in enumerate(self.field_infos):
                if i == label_idx:
                    if field_type_code == _C.CATEGORICAL:
                        num_class = field_attrs['prop'].num_class
                        out_shape = (num_class,)
                    elif field_type_code == _C.NUMERICAL:
                        out_shape = field_attrs['prop'].shape
                    else:
                        raise NotImplementedError
                    self.agg_layer = FeatureAggregator(num_fields=len(field_infos) - 1,
                                                       out_shape=out_shape,
                                                       in_units=feature_units,
                                                       cfg=cfg.AGG_NET)
                else:
                    if field_type_code == _C.CATEGORICAL:
                        categorical_mapping_net = nn.HybridSequential()
                        categorical_mapping_net.add(
                            CategoricalFeatureNet(num_class=field_attrs['prop'].num_class,
                                                  out_units=feature_units,
                                                  cfg=cfg.CATEGORICAL_NET))
                        self.categorical_networks.add(categorical_mapping_net)
                    elif field_type_code == _C.NUMERICAL:
                        numerical_mapping_net = nn.HybridSequential()
                        numerical_mapping_net.add(
                            NumericalFeatureNet(input_shape=field_attrs['prop'].shape,
                                                out_units=feature_units))

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            cfg = CfgNode()
            cfg.feature_units = -1  # -1 means not given and we will use the units of BERT
            cfg.TEXT_NET = CfgNode()
            cfg.TEXT_NET.backbone_type = 'bert'
            cfg.TEXT_NET.agg_type = 'cls'
            cfg.AGG_NET = FeatureAggregator.get_cfg()
            cfg.CATEGORICAL_NET = CategoricalFeatureNet.get_cfg()
            cfg.NUMERICAL_NET = NumericalFeatureNet.get_cfg()
            cfg.INITIALIZER = CfgNode()
            cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]
            cfg.INITIALIZER.bias = ['zeros']
            return cfg
        else:
            raise NotImplementedError

    def forward(self, features):
        """

        Parameters
        ----------
        features
            A list of field data

        Returns
        -------
        logits_or_scores
            Shape (batch_size,) + out_shape
        """
        features_l = []
        for i, (field_type_code, field_attrs) in enumerate(self.field_infos):
            if i != self.
