# Importação de pacotes
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Criando um título para nosso projeto
st.title("Análise de Datasets com Streamlit")

st.write('''
# Explorando diferentes classificadores
''')

# Criando os menus laterais para os tipos de datasets
dataset_name = st.sidebar.selectbox("Selecione o Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
st.write("Nome do Dataset = ", dataset_name)
# Criando os menus laterais para o tipo de Modelo
classifier_name = st.sidebar.selectbox("Selecione o Classificador", ("KNN", "SVM", "Random Forest"))
'''
# Função para carregar os datasets
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
# Mostrar o tamanho do dataset
st.write("Tamanho do Dataset", X.shape)
# Mostrar o número de classes do dataset
st.write("Número de Classes", len(np.unique(y)))

# Função para criar os parâmetros do classificador
def add_paramter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_paramter_ui(classifier_name)

# Função para carregar os modelos
def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"],
                                     n_estimators=params["n_estimators"], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

#Classification
# Vamos usar a função train_test_split para separar nosso dataset em treino e teste.
# Aqui vamos usar 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# Realizando o treinamento do Classificador
clf.fit(X_train, y_train)
# Gerando a predição no dataset de teste
y_pred = clf.predict(X_test)
# Verificando a acurácia
acc = accuracy_score(y_test, y_pred)
# Escrevendo na página o nome do classificador
st.write(f"Classificador = {classifier_name}")
# Escrevendo na página o resultado da acurácia
st.write(f"Acurácia = {acc}")

# PCA
# Para que possamos criar um gráfico de dispersão(scatter plot), é necessário usar um algoritimo não supervisionado
# chamado PCA (Principal Component Analysis), com isso iremos reduzir a dimensão do conjunto de dados:
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

# Plotagem
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.pyplot(fig)

'''