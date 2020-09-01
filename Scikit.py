from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import linear_model

dataset = load_boston()

print dataset.DESCR

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42, test_size=0.33)

model = linear_model.LinearRegression()

model.fit(x_train, y_train)
print model.score(x_test, y_test)

print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

predicted= model.predict(x_test)

print predicted





from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = load_iris()

print dataset.DESCR

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42, test_size=0.33)

model = LogisticRegression()
model.fit(x_train, y_train)
print model.score(x_test, y_test)
predicted= model.predict(x_test)

print predicted


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42, test_size=0.33)

model = GaussianNB() 
# Train the model using the training sets and check score
model.fit(x_train, y_train)
print model.score(x_test, y_test)
#Predict Output
predicted= model.predict(x_test)

print predicted
from sklearn.datasets import load_iris
from sklearn.svm import SVC
dataset = load_iris()
data = dataset.data
target = dataset.target

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42, test_size=0.3)

model = SVC()
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, model.predict(X_test))
model.support_vectors_.shape
model.support_vectors_


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm

dataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42, test_size=0.33)

model = svm.SVC() 
model.fit(x_train, y_train)
print model.score(x_test, y_test)
predicted= model.predict(x_test)
print predicted


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)

model.fit(X_train, y_train)
model.score(X_test, y_test)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42, test_size=0.33)

model = RandomForestClassifier()
model.fit(x_train, y_train)
print model.score(x_test, y_test)
predicted= model.predict(x_test)

print predicted

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42, test_size=0.33)

model = KNeighborsClassifier(n_neighbors=6)
model.fit(x_train, y_train)
print model.score(x_test, y_test)
predicted= model.predict(x_test)

print predicted

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

digits = load_digits()
dataset = digits.data
ss = StandardScaler()
dataset = ss.fit_transform(dataset)

model = KMeans(n_clusters= 10, init="k-means++", n_init=10)
model.fit(dataset

model.labels_
model.inertia_
model.cluster_centers_
