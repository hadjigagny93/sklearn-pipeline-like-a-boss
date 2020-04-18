# import modules
import os
import pandas as pd
import numpy as np
import src.settings.base as base
import joblib
from sklearn.impute import SimpleImputer
from sklearn_pandas import CategoricalImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

class  ModelDiagnostic:
    """class model that get the whole info about the pipelines
    steps which have already been performed and the others to come
    """

    def __init__(self):
        pass

class RemoveSaveParams(BaseEstimator, TransformerMixin):
    """A class that abstracts two taks: remove some uninteresting features
    from dataframe and load the others in pkl file (see "../data/joblib_load" repo)
    when job variable is set to train, loading some features name is performed and in the case that
    job given to be equal to test, then one load the previous variables saved in training process
    and return then by default for testing tasks.

    BaseEstimator and TransformerMixin will serve classes which will inherit from this one
    attributes
    ----------
    job: train or test
    save: save the features for testing processes -- if not then no test will be runned
    hash: to link train and test processes
    method_name: missing_values or low_std_remove class

    method
    ------
    the same paradigm as sklearn class that will be used in Pipeline building task"""

    def __init__(self, job="train", save=True, hashed=None, method_name=None):
        self.job = job
        self.save = save
        self.hash = hashed
        self.method_name = method_name
        self.MAP = {"missing_values":"MV", "remove_low_std": "SD"}

    def fit(self, X):
        """return self nothing else"""
        return self

    def transform(self, X):
        func = self.method_name
        path_to_features = os.path.join(
            base.DATA_DIR,
            "joblib_load/{}-{}.pkl".format(self.MAP[self.method_name], self.hash))
        if self.job == "train":
            features_of_interest = eval("self.{}(X)".format(func))
            if self.save:
                try:
                    joblib.dump(features_of_interest, path_to_features)
                except:
                    raise ValueError("hash not given")
            return X[features_of_interest]
        try:
            features_of_interest = joblib.load(path_to_features)
        except:
            raise ValueError("No training task performed before")
        return  X[features_of_interest]

class HandlingMissingValue(RemoveSaveParams):
    """detect features with hign missing values level, remove them and save/load the others
    depending on the fact that one perform train or test pipeline

    attributes
    ----------
    thrs: threshold defining features to removed"""

    def __init__(self, thrs=60, job="train", save=True, hashed=None, method_name="missing_values"):
        super().__init__(job=job, save=save, hashed=hashed, method_name=method_name)
        self.thrs = thrs

    def missing_values(self, X):
        total = X.isnull().sum().sort_values(ascending=False)
        percent = X.isnull().sum()/X.isnull().count().sort_values(ascending=False)*100
        missing_data = pd.concat([total,percent], axis=1, keys=['Total', 'Pourcentage'])
        features_of_interest = list(missing_data[(percent<=self.thrs)].index)
        return features_of_interest

class RemoveLowStdFeature(RemoveSaveParams):
    "removed low std features"

    def __init__(self, std_level,job="train",save=True,hashed=None,method_name="remove_low_std"):
        super().__init__(job=job, save=save, hashed=hashed, method_name=method_name)
        self.std_level = std_level

    def remove_low_std(self, X):
        std_frame = X.describe().loc["std"].to_frame()
        std_frame_thrs = std_frame[std_frame["std"] > self.std_level]
        high_std_features = list(std_frame_thrs.index)
        return high_std_features

class RemoveOutliers(BaseEstimator, TransformerMixin):
    """ for numeric variable"""

    def __init__(self, job="train"):
        self.job = job

    def fit(self, X):
        return self

    def transform(self, X):
        "Do not touch anything if job equals test"
        if self.job == "test":
            return X
        return X[np.abs(X - X.mean()) <= (3 * X.std())]

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, features, typ="numerical"):
        self.features = features
        self.typ = typ

    def fit(self, X):
        return self

    def transform(self, X):
        col = list(set(X.columns.tolist()).intersection(self.features))
        if self.typ == "numerical":
            return X[col]
        return X[col].values

class CombinedAttrAdder(BaseEstimator, TransformerMixin):

    def __init__(self, name_columns):
        pass

    def fit(self):
        return self

    def transform(self):
        pass

class ProcessPipeline:

    def __init__(self, job="train", thrs=60, pca_enable=False, save=True, pipeline_hash="hash", std=2):
        self.job = self._get_job(job=job)
        self.thrs = thrs
        self.pca_enable = pca_enable
        self.save = save
        self.std = std
        self.pipeline_hash = pipeline_hash

    def _load_data(self):
        path = os.path.join(base.DATA_DIR, self.job + ".csv")
        data = pd.read_csv(path, sep=",").drop([base.DATE, base.DURATION_CONTACT_COL], axis=1)
        return data.drop("SUBSCRIPTION", axis=1), data.SUBSCRIPTION.values

    def _get_job(self, job):
        if job not in {"train", "test"}:
            print("job must be equal to 'train' or 'test'")
            raise
        return job

    def build_pipelines(self):
        process_pipeline = Pipeline(steps=[
            ("handling_missing_values", HandlingMissingValue(
                thrs=self.thrs,
                job=self.job,
                save=self.save,
                hashed=self.pipeline_hash)
            )])
        numeric_pipeline = Pipeline(steps=[
           ("selector", DataFrameSelector(base.NUM_FEATURES, typ="numerical")),
           ("remove_low_std_features", RemoveLowStdFeature(
               std_level=self.std,
               job=self.job,
               save=self.save,
               hashed=self.pipeline_hash)
           ),
           ("remove_outliers", RemoveOutliers(
               job=self.job)
           ),
           ("imputer", SimpleImputer(
             strategy="median"
            ))
           ])
        categorical_pipeline = Pipeline(steps=[
            ("selector", DataFrameSelector(base.CAT_FEATURES, typ="categorical")),
            ("categorical_imputer", CategoricalImputer()),
            ("onehotencoder", OneHotEncoder())])

        num_and_cat_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", numeric_pipeline),
            ("cat_pipeline", categorical_pipeline)])


        # general common process between train and test
        full_common_pipeline = Pipeline(steps=[
           ("process_pipeline", process_pipeline),
           ("num_and_cat_pipeline", num_and_cat_pipeline)])
        return full_common_pipeline

class TestPipeline:
    def __init__(self, process_pipeline):

        if isinstance(process_pipeline, ProcessPipeline):
            self.process_pipeline = process_pipeline

    def run_pipeline(self):
        pipeline, pipeline_hash = self.process_pipeline, self.process_pipeline.pipeline_hash
        features, target = pipeline._load_data()
        features_transformed = pipeline.build_pipelines().fit_transform(features)
        model = self.load_model(hashed=pipeline_hash)
        predicted = model.predict(features_transformed)
        return predicted, accuracy_score(target, predicted)

    @staticmethod
    def load_model(hashed):
        path_to_model = os.path.join(
            base.DATA_DIR,
            "joblib_load/{}-{}.pkl".format("ML", hashed))
        return joblib.load(path_to_model)


class TrainPipeline:
    def __init__(self, process_pipeline):

        if isinstance(process_pipeline, ProcessPipeline):
            self.process_pipeline = process_pipeline

    def build_model_pipeline(self):
        from imblearn.pipeline import Pipeline
        model = Pipeline([
            ('sampling', RandomOverSampler(random_state=0)),
            ('classification', RandomForestClassifier())])
        return model

    def run_pipeline(self):
        pipeline, pipeline_hash = self.process_pipeline, self.process_pipeline.pipeline_hash
        features, target = pipeline._load_data()
        features_transformed = pipeline.build_pipelines().fit_transform(features)
        model = self.build_model_pipeline()
        model.fit(features_transformed, target)
        self.save_model(model_hyperparams=model, hashed=pipeline_hash)

    @staticmethod
    def save_model(model_hyperparams, hashed):
        path_to_model = os.path.join(
            base.DATA_DIR,
            "joblib_load/{}-{}.pkl".format("ML", hashed))
        joblib.dump(model_hyperparams, path_to_model)



def main():

    HASH = "XX-ML"
    train_process_pipeline = ProcessPipeline(pipeline_hash=HASH, job="train")
    test_process_pipeline = ProcessPipeline(pipeline_hash=HASH, job="test")
    train_pipeline = TrainPipeline(train_process_pipeline)
    train_pipeline.run_pipeline()
    test_pipeline = TestPipeline(test_process_pipeline)
    print("accuracy -- {}".format(test_pipeline.run_pipeline()[1]))


if __name__ == "__main__":
    main()