import abc


class BaseTabularEstimator(abc.ABC):
    @property
    @abc.abstractmethod
    def cfg(self):
        pass

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
