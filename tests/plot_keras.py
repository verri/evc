from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, KFold
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from evc import BayesianEvaluation

iris = load_iris()


baseline = Pipeline([
    ('dt', DecisionTreeClassifier(max_depth=2, random_state=0))
])


def create_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(4,)),  # Adjust based on actual input features
        Dense(3, activation='softmax')  # 3 output neurons for 3 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


competitors = [
    ('my_model', Pipeline([
        ('classifier', KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0))
    ])),
    ('my_model', Pipeline([
        ('classifier', KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0))
    ])),
]

n_splits = 10
n_repeats = 2

def cross_validate_model(model, X, y, n_splits, n_repeats):
    scores = np.array([])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)
    label_binarizer = LabelBinarizer()
    y_encoded = label_binarizer.fit_transform(y)

    for i in range(n_repeats):
        print(f"Repeat: {i+1}")
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            print(f" - Fold {fold+1}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_encoded[train_index], y[test_index]

            model.fit(X_train, y_train)

            # Evaluate the model
            predictions = model.predict(X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(y_test, predicted_classes)
            scores = np.append(scores, accuracy)
            print(f"   - Accuracy: {accuracy}")
    return scores

baseline_score = cross_validate_model(baseline, iris.data, iris.target, n_splits, n_repeats)
competitors_score = [
    (name, cross_validate_model(competitor, iris.data, iris.target, n_splits, n_repeats)) for name, competitor in competitors]

be = BayesianEvaluation(baseline_score, rope=0.01)

be.plot(competitors_score)