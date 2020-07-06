import abc


class BaseEstimator(abc.ABC):
    def __init__(self, cfg):
        if cfg is None:
            cfg = self.get_cfg()
        self._cfg = cfg

    @property
    def cfg(self):
        return self._cfg

    @staticmethod
    @abc.abstractmethod
    def get_cfg(key=None):
        pass

    @abc.abstractmethod
    def fit(self, train_data, valid_data=None, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, test_data, **kwargs):
        pass

    @abc.abstractmethod
    def save(self, dir_path):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, dir_path):
        pass
