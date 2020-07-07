import abc


class BaseTabularEstimator(abc.ABC):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = BaseTabularEstimator.get_cfg()
            self._cfg = cfg
        else:
            base_cfg = BaseTabularEstimator.get_cfg()
            self._cfg = base_cfg.clone_merge(cfg)

    @property
    def cfg(self):
        return self._cfg

    @staticmethod
    @abc.abstractmethod
    def get_cfg(key=None):
        pass

    @abc.abstractmethod
    def fit(self, train_data, label, valid_data=None):
        pass

    @abc.abstractmethod
    def predict(self, test_data):
        pass

    @abc.abstractmethod
    def save(self, dir_path):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, dir_path):
        pass
