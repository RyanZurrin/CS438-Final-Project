from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


def plot_confusion_matrix(true_labels, predictions, title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    Plots a confusion matrix.

    :param true_labels: array-like, true class labels
    :param predictions: array-like, predicted class labels
    :param title: str, title of the plot
    :param cmap: matplotlib colormap, default is plt.cm.Blues
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()


def plot_roc_curve(true_labels, probability_scores, title='ROC Curve'):
    """
    Plots an ROC curve.

    :param true_labels: array-like, true class labels
    :param probability_scores: array-like, probability estimates of the positive class
    :param title: str, title of the plot
    """
    fpr, tpr, _ = roc_curve(true_labels, probability_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()


def plot_precision_recall_curve(true_labels, probability_scores,
                                title='Precision-Recall Curve'):
    """
    Plots a precision-recall curve.

    :param true_labels: array-like, true class labels
    :param probability_scores: array-like, probability estimates of the positive class
    :param title: str, title of the plot
    """
    average_precision = average_precision_score(true_labels, probability_scores)
    precision, recall, _ = precision_recall_curve(true_labels, probability_scores)
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', label=f'AP={average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()


def plot_permutation_importance(importance_means, feature_names,
                                title='Permutation Importance'):
    """
    Plots permutation importance.

    :param importance_means: array-like, mean importances
    :param feature_names: array-like, names of features corresponding to the importances
    :param title: str, title of the plot
    """
    sorted_idx = np.argsort(importance_means)
    plt.figure(figsize=(10, 12))
    plt.barh(range(len(feature_names)), importance_means[sorted_idx])
    plt.yticks(range(len(feature_names)), feature_names[sorted_idx])
    plt.xlabel('Importance')
    plt.title(title)
    plt.show()


def plot_logistic_regression_feature_importance(coef, feature_names,
                                                title='Logistic Regression Feature Importance'):
    """
    Plots feature importance for a logistic regression model.

    :param coef: array-like, coefficients from a logistic regression model
    :param feature_names: array-like, names of features corresponding to the coefficients
    :param title: str, title of the plot
    """
    feature_importance = abs(coef)
    sorted_idx = np.argsort(feature_importance)

    plt.figure(figsize=(10, 12))
    plt.barh(range(len(feature_names)), feature_importance[sorted_idx],
             align='center')
    plt.yticks(range(len(feature_names)), np.array(feature_names)[sorted_idx])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(title)
    plt.show()


def plot_learning_curve(estimator, X, y, train_sizes, cv, scoring, n_jobs=-1,
                        shuffle=False):
    """
    Plots a learning curve for a given estimator.

    :param estimator: the model (estimator) to use
    :param X: feature data
    :param y: target data
    :param train_sizes: array-like, list of training set sizes
    :param cv: int, cross-validation generator or an iterable
    :param scoring: str or callable, scorer to use
    :param n_jobs: int, number of jobs to run in parallel (-1 uses all available cores)
    :param shuffle: bool, whether to shuffle the data before splitting into batches
    """
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring,
        n_jobs=n_jobs, shuffle=shuffle)

    # Calculate mean and standard deviation for training set scores
    train_mean = -np.mean(train_scores, axis=1) if 'neg' in scoring else np.mean(
        train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for validation set scores
    validation_mean = -np.mean(validation_scores,
                               axis=1) if 'neg' in scoring else np.mean(
        validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    # Plot the learning curve
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, validation_mean - validation_std,
                     validation_mean + validation_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_mean, 'o-', color="r", label=f"Training {scoring}")
    plt.plot(train_sizes, validation_mean, 'o-', color="g",
             label=f"Cross-validation {scoring}")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel(scoring.capitalize())
    plt.legend(loc="best")
    plt.show()


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
        np.where(np.all(full_scalar_data == row, axis=1))[0][0] for row in
        scalar_data
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
