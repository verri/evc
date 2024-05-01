from evc import BayesianEvaluation

# # Create random vector of scores.
# baseline = np.random.normal(0, 1, 100)
# score = np.random.normal(0.1, 1, 100)
#
# # Create an instance of the BayesianEvaluation class.
# be = BayesianEvaluation(baseline)
#
# # Plot the distribution of the score.
# be.plot(score)

from sklearn.datasets import load_iris
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

# Load the iris dataset.
iris = load_iris()

# Create baseline pipeline

baseline = Pipeline([
    ('dt', DecisionTreeClassifier(max_depth=2, random_state=0))
])

competitors = [
    ('svc', Pipeline([
        ('svc', SVC(kernel='rbf', C=1, random_state=0))
    ])),
    ('knn', Pipeline([
        ('knn', KNeighborsClassifier(n_neighbors=3))
    ])),
    ('nb', Pipeline([
        ('nb', GaussianNB())
    ])),
    ('rf', Pipeline([
        ('rf', RandomForestClassifier(n_estimators=10, random_state=0))
    ]))
]

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)

baseline_score = cross_val_score(baseline, iris.data, iris.target, cv=cv, scoring='accuracy')
competitors_score = [
        (name, cross_val_score(competitor, iris.data, iris.target, cv=cv,
            scoring='accuracy')) for name, competitor in competitors]

be = BayesianEvaluation(baseline_score, rope=0.01)

be.plot(competitors_score)

# Now plot a boxplot of the difference of the scores.

scores = [score - baseline_score for name, score in competitors_score]
names = [name for name, score in competitors_score]

from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
ax.boxplot(scores)
ax.set_xticklabels(names)
ax.set_ylabel('Score difference')
ax.set_title('Score difference with respect to baseline')

plt.show()
