from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def custom_permutation_importance(model_wrapper, X_scalar, y, n_repeats=10):
    baseline_scores = []
    permuted_scores = []
    for i in range(n_repeats):
        # Shuffle the scalar data
        X_scalar_shuffled = shuffle(X_scalar)
        # Calculate the baseline score
        baseline_scores.append(model_wrapper.score(X_scalar, y))
        # Calculate the permuted score
        permuted_scores.append(model_wrapper.score(X_scalar_shuffled, y))
    # Calculate the importances
    importances = np.array(baseline_scores) - np.array(permuted_scores)
    # Calculate the standard deviation
    std = np.std(importances)
    return importances, std


class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, keras_model, full_image_data, full_scalar_data):
        self.keras_model = keras_model
        self.full_image_data = full_image_data
        self.full_scalar_data = full_scalar_data

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.keras_model.predict([self.full_image_data, X])

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            "keras_model": self.keras_model,
            "full_image_data": self.full_image_data,
            "full_scalar_data": self.full_scalar_data,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


# Define a function to select the corresponding image data
def get_corresponding_image_data(scalar_data, full_image_data, full_scalar_data):
    indices = [
        np.where(np.all(full_scalar_data == row, axis=1))[0][0] for row in scalar_data
    ]
    return full_image_data[indices]


# test the above functions and classes
if __name__ == "__main__":
    # Create some dummy data
    full_image_data = np.random.rand(100, 224, 224, 3)
    full_scalar_data = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    # Create a dummy Keras model
    keras_model = KerasClassifierWrapper(
        keras_model=None,
        full_image_data=full_image_data,
        full_scalar_data=full_scalar_data,
    )

    # Test the custom_permutation_importance function
    importances, std = custom_permutation_importance(
        model_wrapper=keras_model, X_scalar=full_scalar_data, y=y, n_repeats=10
    )
    print(importances)
    print(std)

    # Test the get_corresponding_image_data function
    scalar_data = np.random.rand(10, 10)
    image_data = get_corresponding_image_data(
        scalar_data=scalar_data,
        full_image_data=full_image_data,
        full_scalar_data=full_scalar_data,
    )
    print(image_data.shape)
