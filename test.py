# import streamlit as st
# st.title("Streamlit Totorial")
# st.title("#Explore Different classifiers")
# st.write("""
# #Explore Different classifiers
# Which one is best?
# Hello Shubham How are you?
# """)
# dataset_name=st.selectbox("Select DATASET",["iris","wine","breast_cancer"])
# classifier_name=st.selectbox("Select Classifier",["Logistic Regression","Desicion Tree","Random Forest"])
# # print(dataset_name)
# # print(classifier_name)
# def get_dataset(dataset_name):
#     if dataset_name=="iris":
#         data=dataset.load_iris()
#     elif dataset_name=="iris":
#         data=datasets.load_iris()    
from re import A
import streamlit as st
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
st.title("Streamlit Tutorial")

st.write("""
# Explore different classifers
Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ["iris", "wine", "breast_cancer"])

classifier_name = st.sidebar.selectbox("Select Classifier", ["Logistic Regression", "Decision Tree", "Random Forest"])

def get_dataset(data_name):
  if data_name == "iris":
    data = datasets.load_iris()
  elif data_name == "wine":
    data = datasets.load_wine()
  elif data_name == "breast_cancer":
    data = datasets.load_breast_cancer()

  X = data.data
  y = data.target

  return X, y

X, y = get_dataset(dataset_name)
# print("X ",X.shape)
# print("y ",y.shape)

st.write("Shape of the dataset: ", X.shape)
st.write("Shape of the target: ", y.shape)
# A = data.data
#   b = data.target

#   return X, y

# X, y = get_dataset(dataset_name)
# # print("X ",X.shape)
# # print("y ",y.shape)

# st.write("Shape of the dataset: ", X.shape)
# st.write("Shape of the target: ", y.shape)

def add_perimeter_ui(clf_name):
  params = dict()
  if clf_name == "Logistic Regression":
    params["C"] = st.sidebar.slider("C", 0.01, 10.0, 0.1)
  elif clf_name == "Decision Tree":
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 10, 3)
  elif clf_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("# of Estimators", 1, 100, 10)
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 10, 3)
  return params

params = add_perimeter_ui(classifier_name)

def get_classifier(clf_name, params):
  if clf_name == "Logistic Regression":
    clf = LogisticRegression(**params)
  elif clf_name == "Decision Tree":
    clf = DecisionTreeClassifier(**params)
  elif clf_name == "Random Forest":
    clf = RandomForestClassifier(**params)
  return clf

# Classification

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write("Classifier ", classifier_name)
st.write("Accuracy: ", acc)

# # PLOT
pca = PCA(2)
x_pojected = pca.fit_transform(X)

x1 = x_pojected[:, 0]
x2 = x_pojected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Pricipal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar()
st.pyplot(plt)


