from .dataset import load_pandas_df, random_split_train_val, TabularDataset
from .estimators.basic import BertForTabularPredictionBasic


class AutoNLP:
    @staticmethod
    def fit(train_data, valid_data=None, feature_columns=None,
            label=None, valid_ratio=0.15, exp_dir='./autonlp', eval_metrics=None):
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
        eval_metrics
            How you may potentially evaluate the model

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
        cfg.freeze()
        estimator = BertForTabularPredictionBasic(cfg)
        estimator.fit(train_data=train_data, valid_data=valid_data,
                      feature_columns=feature_columns,
                      label=label)
        return estimator
