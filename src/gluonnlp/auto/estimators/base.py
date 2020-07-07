import abc


class BaseTabularEstimator(abc.ABC):
    def __init__(self, cfg):
        if cfg is None:
            cfg = self.get_cfg()
        self._cfg = self.get_cfg().clone_merge(cfg)

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
