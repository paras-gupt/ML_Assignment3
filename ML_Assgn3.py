import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        return datasets.load_iris()
    elif dataset_name == "Wine":
        return datasets.load_wine()
    elif dataset_name == "Breast Cancer":
        return datasets.load_breast_cancer()
    elif dataset_name == "Digits":
        return datasets.load_digits()

st.set_page_config(page_title="SVM Kernels Visualization", page_icon=":chart_with_upwards_trend:")
st.title("SVM Kernels Visualization")

dataset_name = st.sidebar.selectbox("Select a dataset:", ("Iris", "Wine", "Breast Cancer", "Digits"))
dataset = get_dataset(dataset_name)
X = dataset.data
y = dataset.target

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

kernel_name = st.sidebar.selectbox("Select a kernel function:", ("Linear", "Polynomial", "RBF"))
if kernel_name == "Polynomial":
    p = st.sidebar.slider("Select the degree of the polynomial kernel (p):", 1, 10, 3)
    q = st.sidebar.slider("Select the coefficient of the polynomial kernel (q):", 1, 10, 2)
    model = SVC(kernel='poly', degree = p, coef0 = q)
    model.fit(X_pca, y)
elif kernel_name == "Linear":
    model = SVC(kernel='linear')
    model.fit(X_pca, y)
elif kernel_name == "RBF":
    model = SVC(kernel='rbf')
    model.fit(X_pca, y)


fig, ax = plt.subplots()
ax.set_title(f"{kernel_name} Kernel")
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.4)
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=20, edgecolor="k")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)
