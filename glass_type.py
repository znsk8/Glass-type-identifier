# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache_data()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data()

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
features = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
@st.cache_data()
def prediction(_model,features):
  glass_type = _model.predict([features])
  if glass_type[0] == 1:
   return "building windows float processed"
  elif glass_type[0] ==2:
    return "building windows non float processed"
  elif glass_type[0]== 3:
    return  "vehicle windows float processed"
  elif glass_type[0]== 4:
    return "vehicle windows non float processed"
  elif glass_type[0]== 5:
    return "containers"
  elif glass_type[0]== 6:
    return "tableware"
  else: 
    return "headlamp"

# S4.1: Add title on the main page and in the sidebar.
st.title("Glass Type Predictor")
st.sidebar.title("Exploratory Data Analysis")
#checkbox
if st.sidebar.checkbox('Show raw data'):
  st.subheader('Full Dataset')
  st.table(glass_df)

# S6.1: Scatter Plot between the features and the target variable.
# Add a subheader in the sidebar with label "Scatter Plot".
st.sidebar.subheader("Scatter Plot")
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
features_list = st.sidebar.multiselect("Select the X-axis values for scatterplot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

# S6.2: Create scatter plots between the features and the target variable.
# Remove deprecation warning.
st.set_option("deprecation.showPyplotGlobalUse",False)
for i in features_list:
  st.subheader(f"Scatter plot between the feature {i} & Glass Type.")
  plt.figure(figsize=(10,9))
  sns.scatterplot(x = glass_df[i], y = glass_df['GlassType'])
  st.pyplot()

st.sidebar.subheader("Visualisation Selector")
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# and with 6 options passed as a tuple ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot').
# Store the current value of this widget in a variable 'plot_types'.
plot_types = st.sidebar.multiselect("Select the charts/plots", ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))

# S1.2: Create histograms for the selected features using the 'selectbox' widget.
if 'Histogram' in plot_types:
  #histogram
  st.subheader("Histogram")
  features_list = st.sidebar.multiselect("Select the X-axis values for Histogram",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  for i in features_list:
    st.subheader(f"Histogram for the feature {i}.")
    plt.figure(figsize=(10,9))
    plt.hist(glass_df[i],bins = 'sturges', edgecolor = 'red')
    st.pyplot()

# S1.3: Create box plots for the selected column using the 'selectbox' widget.
if "Box Plot" in plot_types:
  #boxplot
  st.subheader("Boxplot")
  features_list = st.sidebar.multiselect("Select the X-axis values for Boxplot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  for i in features_list:
    st.subheader(f"Boxplot for the feature {i}.")
    plt.figure(figsize=(10,9))
    sns.boxplot(glass_df[i])
    st.pyplot()
# Create count plot using the 'seaborn' module and the 'st.pyplot()' function.
if "Count Plot" in plot_types:
  #boxplot
  st.subheader("Count Plot")
  plt.figure(figsize=(10,9))
  sns.countplot(x = glass_df['GlassType'])
  st.pyplot()
# Create pie chart using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.
if "Pie Chart" in plot_types:
  #boxplot
  st.subheader("Pie Chart")
  plt.figure(figsize=(10,9))
  plt.pie(glass_df['GlassType'].value_counts())
  st.pyplot() 

# Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
if "Correlation Heatmap" in plot_types:
  #boxplot
  st.subheader("Correlation Heatmap")
  plt.figure(figsize=(10,9))
  sns.heatmap(glass_df.corr(),annot = True)
  st.pyplot() 

# Display pair plots using the the 'seaborn' module and the 'st.pyplot()' function.
if "Pair Plot" in plot_types:
  #boxplot
  st.subheader("Pair Plot")
  plt.figure(figsize=(10,9))
  sns.pairplot(glass_df)
  st.pyplot()

#slider for 9 widgets
ri = st.sidebar.slider("Ri",float(glass_df['RI'].min()), float(glass_df["RI"].max()))
na = st.sidebar.slider("Na",float(glass_df['Na'].min()), float(glass_df["Na"].max()))
mg = st.sidebar.slider("Mg",float(glass_df['Mg'].min()), float(glass_df["Mg"].max()))
al = st.sidebar.slider("Al",float(glass_df['Al'].min()), float(glass_df["Al"].max()))
si = st.sidebar.slider("Si",float(glass_df['Si'].min()), float(glass_df["Si"].max()))
k  = st.sidebar.slider("K",float(glass_df['K'].min()), float(glass_df["K"].max()))
ca = st.sidebar.slider("Ca",float(glass_df['Ca'].min()), float(glass_df["Ca"].max()))
ba = st.sidebar.slider("Ba",float(glass_df['Ba'].min()), float(glass_df["Ba"].max()))
fe = st.sidebar.slider("Fe",float(glass_df['Fe'].min()), float(glass_df["Fe"].max()))

# S3.1: Add a subheader and multiselect widget.
# Add a subheader in the sidebar with label "Choose Classifier"
st.sidebar.subheader("Choose Classifier")
# Add a selectbox in the sidebar with label 'Classifier'.
# and with 2 options passed as a tuple ('Support Vector Machine', 'Random Forest Classifier').
# Store the current value of this slider in a variable 'classifier'.
classifier = st.sidebar.selectbox("Classifier",('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression') )

#SVM
if classifier == 'Support Vector Machine':
  st.sidebar.subheader("Model Hyper Parameters")
  c = st.sidebar.number_input("C", 1, 100,step = 1)
  kernel = st.sidebar.radio("Kernel", ('linear','rbf','poly'))
  gamma = st.sidebar.number_input("Gamma", 1 , 100, step = 1)
  if st.sidebar.button("Classify"):
    st.subheader("Support Vector Machine")
    model = SVC(kernel = kernel,C = c,gamma = gamma).fit(X_train,y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    glass_type = prediction(model, [ri,na,mg,al,si,k,ca,ba,fe])
    st.write(f'The predicted Glass type is {glass_type}')
    st.write(f"The accuracy of the model is {accuracy}")
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    st.pyplot()

#RANDOM FOREST CLASSIFIER
if classifier == 'Random Forest Classifier':
  st.sidebar.subheader("Model Hyper Parameters")
  n = st.sidebar.number_input("n estimater", 100, 5000,step = 50)
  max_dept = st.sidebar.number_input("max. depth of the tree", 1 , 100, step = 1)
  if st.sidebar.button("Classify"):
    st.subheader("Random Forest Classifier")
    model = RandomForestClassifier(n_estimators = n,max_depth = max_dept).fit(X_train,y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    glass_type = prediction(model, [ri,na,mg,al,si,k,ca,ba,fe])
    st.write(f'The predicted Glass type is {glass_type}')
    st.write(f"The accuracy of the model is {accuracy}")
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    st.pyplot()

#Logistic Regression
if classifier == 'Logistic Regression':
  st.sidebar.subheader("Model Hyper Parameters")
  c = st.sidebar.number_input("c", 1, 10,step = 1)
  max_dept = st.sidebar.number_input("max. iteration", 10 , 1000, step = 10)
  if st.sidebar.button("Classify"):
    st.subheader("Logistic Regression")
    model = LogisticRegression(C = c,max_iter = max_dept).fit(X_train,y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    glass_type = prediction(model, [ri,na,mg,al,si,k,ca,ba,fe])
    st.write(f'The predicted Glass type is {glass_type}')
    st.write(f"The accuracy of the model is {accuracy}")
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
    st.pyplot()