import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.datasets.dataset import Dataset
from src.evaluators.quality_metric import QualityMetric
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints


class MachineLearningAccuracyQualityMetric(QualityMetric):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, private_data: Dataset, synthetic_data: Dataset, repaired_data: Dataset,
                 mcs: MarginalsConstraints) -> dict[str, float]:
        return {
            "synthetic_data": self.evaluate(synthetic_data, private_data),
            "repaired_data": self.evaluate(repaired_data, private_data)
        }

    def evaluate(self, train: Dataset, test: Dataset) -> float:
        if train.target is None or test.target is None:
            return -1

        if len(train) == 0:
            return 1 / len(test.data[test.target].unique())  # If no training data, return baseline accuracy

        x_train, x_test = self._get_features(train, test)
        y_train, y_test = self._get_targets(train, test)
        model = self._build_model()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        return accuracy_score(y_test, predictions)

    def _get_features(self, train: Dataset, test: Dataset):
        train_df = train.data.drop(columns=[train.target])
        test_df = test.data.drop(columns=[test.target])
        combined = pd.concat([train_df, test_df], axis=0)
        combined_dummies = pd.get_dummies(combined, drop_first=True)
        x_train_dummies = combined_dummies.iloc[:len(train_df)]
        x_test_dummies = combined_dummies.iloc[len(train_df):]
        scaler = StandardScaler()
        return scaler.fit_transform(x_train_dummies), scaler.transform(x_test_dummies)

    def _get_targets(self, train: Dataset, test: Dataset):
        y_train = train.data[train.target]
        y_test = test.data[test.target]
        if y_train.dtype == 'object' or y_test.dtype == 'object':
            y_train = pd.factorize(y_train)[0]
            y_test = pd.factorize(y_test)[0]
        return y_train, y_test

    def _build_model(self):
        match self.model_name:
            case "logistic_regression":
                return LogisticRegression(solver='liblinear',
                                          max_iter=1000,
                                          random_state=42)
            case "random_forest":
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=0,
                    n_jobs=-1,
                )
            case "mlp":
                return MLPClassifier(hidden_layer_sizes=(64, 32),
                                     activation='relu',
                                     solver='adam',
                                     max_iter=500,
                                     random_state=42)
            case _:
                raise ValueError(f"Unknown model name: {self.model_name}")
