import pandas as pd
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import xgboost as xgb

from utils import get_period_day, get_min_diff, is_high_season

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model is saved in this attribute.
        # Load the data from the CSV file
        self._data = pd.read_csv("data\data.csv")


    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
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
        # data.dropna(inplace=True)

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

            # training_data = shuffle(features[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)

            # features = pd.concat([
            #     features,
            #     pd.get_dummies(training_data['OPERA'], prefix = 'OPERA'),
            #     pd.get_dummies(training_data['TIPOVUELO'], prefix = 'TIPOVUELO'),
            #     pd.get_dummies(training_data['MES'], prefix = 'MES'),
            #     # pd.get_dummies(training_data[target_column], prefix = target_column),
            #     ],
            #     axis = 1
            # )

            X = features.drop(columns=[target_column])
            y = features[target_column]

            # Apply one-hot encoding to categorical columns
            X = pd.get_dummies(X, columns=['Fecha-I', 'Vlo-I', 'Ori-I', 'Des-I', 'Emp-I', 'Fecha-O', 'Vlo-O', 'Ori-O', 'Des-O', 'Emp-O', 'DIANOM', 'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES', 'period_day'])

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
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
        # print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")
        # print(y_train.value_counts('%')*100)
        # print(y_test.value_counts('%')*100)

        model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)

        # Train the model
        model.fit(x_train, y_train)

        # Make predictions on the test data
        y_preds = model.predict(x_test)

        # Convert predicted probabilities to binary predictions using a threshold
        y_preds = [1 if y_pred > 0.5 else 0 for y_pred in y_preds]

        # Calculate performance metrics (e.g., classification report, confusion matrix)
        confusion = confusion_matrix(y_test, y_preds)
        report = classification_report(y_test, y_preds)

        # Print or return these metrics as needed
        print("Confusion Matrix:")
        print(confusion)
        print("\nClassification Report:")
        print(report)

        # save the trained model to the class attribute
        self._model = model


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
        # Use the trained model to make predictions
        # features = self.preprocess(features)
        if self._model is not None:
            predictions = self._model.predict(features)
            return predictions
        else:
            raise ValueError("Model has not been trained yet. Call the 'fit' method first.")
