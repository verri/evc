from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, KFold
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
import numpy as np


baseline = Pipeline([
    ('dt', DecisionTreeClassifier(max_depth=2, random_state=0))
])

# Define the Keras model
def create_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(4,)),  # Adjust based on actual input features
        Dense(3, activation='softmax')  # 3 output neurons for 3 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# One-hot encode labels
label_binarizer = LabelBinarizer()
y_encoded = label_binarizer.fit_transform(y)

pipeline = Pipeline(
    [('classifier', KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0))]
)

pipeline.fit(X, y_encoded)  # Fit the model

new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = pipeline.predict(new_sample)
print("Prediction:", prediction)

'''
# Setup cross-validator
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)

# Perform cross-validation
scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')  # Use one-hot encoded y
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", np.mean(scores))
'''

# Perform cross-validation

n_splits = 10
n_repeats = 2

def cross_validate_model(model, X, y, n_splits, n_repeats):
    scores = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)
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
            scores.append(accuracy)
            print(f"   - Accuracy: {accuracy}")
    return scores
    
