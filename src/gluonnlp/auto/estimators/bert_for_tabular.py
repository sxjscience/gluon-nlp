from ..modules.classification import BERTForTabularClassificationV1
from ...utils.config import CfgNode
from ...utils.misc import set_seed
from . import BaseTabularEstimator
from ..dataset import TabularDataset


def base_optimization_config():
    cfg = CfgNode()
    cfg.lr_scheduler = 'poly_scheduler'
    cfg.optimizer = 'adamw'
    cfg.early_stopping = False  # Whether to use early stopping
    cfg.model_average = 10      # When this value is larger than 1, we will use
    cfg.optimizer_params = [('beta1', 0.9),
                            ('beta2', 0.999),
                            ('epsilon', 1e-6),
                            ('correct_bias', False)]
    cfg.begin_lr = 0.0
    cfg.batch_size = 32
    cfg.num_accumulated = 1
    cfg.val_batch_size_mult = 2  # By default, we double the batch size for validation
    cfg.lr = 1E-4
    cfg.final_lr = 0.0
    cfg.num_train_epochs = 3.0
    cfg.warmup_portion = 0.1
    cfg.layerwise_lr_decay = 0.8  # The layer_wise decay
    cfg.wd = 0.01  # Weight Decay
    cfg.max_grad_norm = 1.0  # Maximum Gradient Norm
    # The validation frequency = validation frequency * num_updates_in_an_epoch
    cfg.valid_frequency = 0.2
    # Logging frequency = log frequency * num_updates_in_an_epoch
    cfg.log_frequency = 0.05
    cfg.stop_metrics = 'auto'
    return cfg


def base_tabular_model_config():
    cfg = CfgNode()
    cfg.PREPROCESS = CfgNode()
    cfg.PREPROCESS.merge_text = True
    cfg.PREPROCESS.max_length = 128
    cfg.BACKBONE = CfgNode()
    cfg.BACKBONE.name = 'google_electra_base'
    cfg.TABULAR_CLASSIFICATION = BERTForTabularClassificationV1.get_cfg()
    return cfg


def base_learning_config():
    cfg = CfgNode()
    cfg.valid_ratio = 0.1       # The ratio of to split the validation data
    return cfg


def base_misc_config():
    cfg = CfgNode()
    cfg.seed = 123
    cfg.context = 'gpus[0,1,2]'
    cfg.exp_dir = './bert_for_tabular'
    return cfg


def base_cfg():
    cfg = CfgNode()
    cfg.VERSION = 1
    cfg.OPTIMIZATION = base_optimization_config()
    cfg.MODEL = base_tabular_model_config()
    cfg.MISC = base_misc_config()
    return cfg


class BertForTabularClassification(BaseTabularEstimator):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self._inferred_problem = None
        self.net = None

    @property
    def cfg(self):
        return self._cfg

    @staticmethod
    def get_cfg(key=None):
        """

        Parameters
        ----------
        key
            Prebuilt configurations

        Returns
        -------
        cfg
        """
        if key is None:
            return base_cfg()
        else:
            raise NotImplementedError

    def fit(self, train_data, label, valid_data=None):
        """

        Parameters
        ----------
        train_data
            The training data.
            Should be a format that can be converted to a tabular dataset
        valid_data
            The validation data.
        label
            The label column
        """
        assert label is not None
        set_seed(self.cfg.MISC.seed)
        if not isinstance(train_data, TabularDataset):
            train_data = TabularDataset(train_data, label_columns=label)
        column_properties = train_data.column_properties
        if valid_data is None:
            train_table =
            valid_table =


    def predict_proba(self, test_data):


    def predict(self, test_data):
        pass

    def save(self, file_path):
        pass

    @classmethod
    def load(cls, file_path):
        pass



class BERTForTabularRegression(BaseEstimator):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self._inferred_problem = None
        self.net = None

