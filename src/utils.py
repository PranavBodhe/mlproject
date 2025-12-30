import os
import sys
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves any Python object to a file using dill
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models):
    """
    Trains multiple models and evaluates them using R2 score
    """
    try:
        report = {}

        for model_name, model in models.items():
            # Train model
            model.fit(x_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(x_test)

            # Calculate R2 score
            test_model_score = r2_score(y_test, y_test_pred)

            # Store result
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
