from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("GaussianNB Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


texts = [
    "Free entry in a contest win money now",
    "Hey how are you doing today",
    "Win cash prize click now",
    "Let's meet for lunch tomorrow",
    "Congratulations you won a lottery",
    "Are we still meeting today"
]

labels = [1, 0, 1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

y_pred = mnb.predict(X_test)

print("MultinomialNB Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))