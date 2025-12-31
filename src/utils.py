import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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


def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    """
    Performs GridSearchCV for multiple models and
    returns their R2 scores
    """
    try:
        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})

            gs = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=3,
                scoring="r2",
                n_jobs=-1
            )

            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            # update model with best version
            models[model_name] = best_model

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)