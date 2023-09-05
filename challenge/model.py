import pandas as pd
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression  # Import necessary for Logistic Regression

from utils import get_period_day, get_min_diff, is_high_season

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        # Load the data from the CSV file directly in __init__
        self._data = pd.read_csv("data\data.csv")

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        # Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        #Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Handle missing values (drop rows with missing data)
        data.dropna(inplace=True)

        features = data.copy()

        # Feature Generation
        features['period_day'] = features['Fecha-I'].apply(get_period_day)
        features['high_season'] = features['Fecha-I'].apply(is_high_season)
        features['min_diff'] = features.apply(get_min_diff, axis=1)

        # Calculate 'delay' feature
        threshold_in_minutes = 15
        features['delay'] = np.where(features['min_diff'] > threshold_in_minutes, 1, 0)

        # Check if the target column is specified
        if target_column is not None:
            # Split the data into features (X) and the target (y)
            X = features.drop(columns=[target_column])
            y = features[target_column]

            # Return features and target as a tuple
            return X, y
        else:
            # If target_column is not specified, return only features
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        return
