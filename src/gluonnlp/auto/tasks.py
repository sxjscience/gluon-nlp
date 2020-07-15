from .dataset import load_pandas_df, random_split_train_val, TabularDataset
from .estimators.basic import BertForTabularPredictionBasic


class AutoNLP:
    @staticmethod
    def fit(train_data,
            valid_data=None, feature_columns=None,
            label=None, valid_ratio=0.15,
            exp_dir='./autonlp',
            stop_metric=None,
            eval_metrics=None,
            log_metrics=None,
            time_limits=3,
            hyperparameters=None,
            network_configs='google_electra_base'):
        """

        Parameters
        ----------
        train_data
        valid_data
        feature_columns
        label
        valid_ratio
            Valid ratio
        exp_dir
            The experiment directory
        stop_metric
            Stop metric for model selection
        eval_metrics
            How you may potentially evaluate the model
        log_metrics
            The logging metrics
        timelimits

        network_configs

        Returns
        -------
        estimator
            An estimator with fit called
        """
        train_data = load_pandas_df(train_data)
        if label is None:
            # Perform basic label inference
            if 'label' in train_data.columns:
                label = 'label'
            elif 'score' in train_data.columns:
                label = 'score'
            else:
                label = train_data.columns[-1]
        if feature_columns is None:
            used_columns = train_data.columns
            feature_columns = [ele for ele in used_columns if ele is not label]
        else:
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]
            used_columns = feature_columns + [label]
        train_data = TabularDataset(train_data,
                                    columns=used_columns,
                                    label_columns=label)
        column_properties = train_data.column_properties
        if valid_data is None:
            train_data, valid_data = random_split_train_val(train_data.table,
                                                            valid_ratio=valid_ratio)
            train_data = TabularDataset(train_data,
                                        columns=used_columns,
                                        column_properties=column_properties)
        else:
            valid_data = load_pandas_df(valid_data)
        valid_data = TabularDataset(valid_data,
                                    columns=used_columns,
                                    column_properties=column_properties)
        cfg = BertForTabularPredictionBasic.get_cfg()
        cfg.defrost()
        if exp_dir is not None:
            cfg.MISC.exp_dir = exp_dir
        if log_metrics is not None:
            cfg.LEARNING.log_metrics = log_metrics
        if stop_metric is not None:
            cfg.LEARNING.stop_metric = stop_metric
        if backbone_name is not None:
            cfg.MODEL.BACKBONE.name = backbone_name
        cfg.freeze()
        estimator = BertForTabularPredictionBasic(cfg)
        estimator.fit(train_data=train_data, valid_data=valid_data,
                      feature_columns=feature_columns,
                      label=label)
        return estimator

    @staticmethod
    def load(dir_path):
        """

        Parameters
        ----------
        dir_path

        Returns
        -------

        """
        BertForTabularPredictionBasic.load(dir_path)
