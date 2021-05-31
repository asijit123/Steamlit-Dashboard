import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn import tree

st.title('Streamlit Dashboard')

st.write("""
# Explore two different classifier
To know which is more useful
""")

dataset_name=st.sidebar.selectbox('Select Dataset',('Iris','Diabetes','Wine Dataset'))
st.write(f'Data set =',dataset_name)

classifier_name= st.sidebar.selectbox('Select Dataset',('Random Forest','Gradient Boosting'))

def get_dataset(dataset_name):
    if dataset_name== 'Iris':
        data =datasets.load_iris()
    elif dataset_name=='Diabetes':
        data = datasets.load_diabetes()
    else:
        data = datasets.load_wine()
    X= data.data
    y= data.target
    return X, y
X, y =get_dataset(dataset_name)
st.write("shape of dataset= ",X.shape)
st.write("number of classes= ",len(np.unique(y)))

def add_parameter_ui(clf_name,X_train):
    params =dict()
    if clf_name=='Random Forest':
        st.sidebar.markdown('# Random Forest')
        max_depth = st.sidebar.slider('max_depth',2,15)
        params['max_depth']=max_depth
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] = n_estimators
        criterion = st.sidebar.selectbox('criterion',('gini', 'entropy'))
        params['criterion']=criterion
        random_state = st.sidebar.slider('random_state',1,100)
        params['random_state']=random_state
        min_samples_split = st.sidebar.slider('Min Samples Split',1, X_train.shape[0], 2,key=1234)
        params['min_samples_split'] = min_samples_split
        min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)
        params['min_samples_leaf'] = min_samples_leaf
        max_features = st.sidebar.slider('Max Features', 1, 2,2)
        params['max_features'] = max_features
        min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')
        params['min_impurity_decrease'] = min_impurity_decrease
    else:
        st.sidebar.markdown('# Gradient Boosting')
        max_depth = st.sidebar.slider('max_depth', 2, 15,3)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100,100)
        params['n_estimators'] = n_estimators
        criterion = st.sidebar.selectbox('criterion', ('friedman_mse', 'mse','mae'))
        params['criterion'] = criterion
        loss = st.sidebar.selectbox('loss', ('deviance', 'exponential'))
        params['loss'] = loss
        subsample=st.sidebar.slider('subsample',0.1,1.0,1.0)
        params['subsample']=subsample
        learning_rate=st.sidebar.slider('learning_rate',0.0,100.0,0.1)
        params['learning_rate']=learning_rate
        min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)
        params['min_samples_split'] = min_samples_split
        min_samples_leaf = st.sidebar.slider('Min Samples Leaf',  1, X_train.shape[0], 1,key=1235)
        params['min_samples_leaf'] = min_samples_leaf
        max_features = st.sidebar.slider('Max Features', 1, 2,2)
        params['max_features'] = max_features
        min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')
        params['min_impurity_decrease'] = min_impurity_decrease
        random_state = st.sidebar.slider('random_state', 1, 100)
        params['random_state'] = random_state
    return params
### CLASSIFICATION ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
params=add_parameter_ui(classifier_name,X_train)

def get_classifier(clf_name,params):
    clf = None
    if clf_name =='Random Forest':
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], criterion=params['criterion'],
                                     random_state=params['random_state'], min_samples_split=params['min_samples_split'],
                                     min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'],
                                     min_impurity_decrease=params['min_impurity_decrease'])
    else:
        clf= GradientBoostingClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], criterion=params['criterion'],
                                        random_state=params['random_state'], min_samples_split=params['min_samples_split'],
                                     min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'],
                                     min_impurity_decrease=params['min_impurity_decrease'],learning_rate=params['learning_rate'],subsample=params['subsample'],
                                     loss=params['loss'])
    return clf


clf= get_classifier(classifier_name,params)


clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

acc=accuracy_score(y_test,y_pred)
st.write(f'Classifer ={classifier_name}')
st.write(f'Accuracy=',acc)

st.write("""
# Graphs
helpful to visualize the dataset
""")
#load inital graph
fig, ax = plt.subplots()
#plot initial graph
ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
orig = st.pyplot(fig)

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)
st.write("""
 Data set Accuracy in percentage
""")
st.write(f'Accuracy= ', str((round(accuracy_score(y_test, y_pred), 2))*100),'%')
