import streamlit as st
import pandas as pd
import matplotlib as plt
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Load the data
df = pd.read_csv('diabetes_fix.csv')
st.title('Diabetes Prediction App')
st.write('Aplikasi ini memprediksi kemungkinan seseorang menderita diabetes berdasarkan beberapa fitur yang dimasukan')
st.write('Dataset yang digunakan adalah dataset diabetes dari kaggle')
st.write('Dataset : https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset')
st.write('')
st.write(df.head())

# Sidebar Data Visualization
st.sidebar.subheader('Data Visualization')
# Histogram
if st.sidebar.checkbox('Show Histogram'):
    st.header('Histogram')
    st.write('Pilih fitur yang ingin ditampilkan histogramnya')
    fitur = st.selectbox('Fitur', ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'))
    st.write('Histogram dari fitur', fitur)
    fig, ax = plt.subplots()
    plt.hist(df[fitur], bins=20)
    st.pyplot(fig)

# Scatter Plot
if st.sidebar.checkbox('Show Scatter Plot'):
    st.header('Scatter Plot')
    st.write('Pilih fitur yang ingin ditampilkan scatter plotnya')
    fitur1 = st.selectbox('Fitur 1', ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'))
    fitur2 = st.selectbox('Fitur 2', ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'))
    st.write('Scatter Plot dari fitur', fitur1, 'dan', fitur2)
    fig, ax = plt.subplots()
    plt.scatter(df[fitur1], df[fitur2])
    st.pyplot(fig)

# Density Plot
if st.sidebar.checkbox('Show Barplot'):
    st.subheader('Jumlah pasien diabetes')
    fig, ax = plt.subplots()
    a = sns.countplot(x='Outcome', data=df)
    for j in a.containers:
        a.bar_label(j, label_type='edge')
    a.set_xlabel('Outcome')
    st.pyplot(fig)

    st.subheader('Jumlah pasien diabetes berdasarkan usia')
    fig, ax = plt.subplots()
    a = sns.countplot(x='Outcome', hue='Age_grup', data=df)
    for j in a.containers:
        a.bar_label(j, label_type='edge')
    a.set_xlabel('Outcome')
    plt.legend(loc='upper right', title='Kelompok Umur')
    st.pyplot(fig)    

    st.subheader('Jumlah pasien diabetes berdasarkan BMI')
    fig, ax = plt.subplots()
    a = sns.countplot(x='Outcome', hue='BMI_grup', data=df)
    for j in a.containers:
        a.bar_label(j, label_type='edge')
    a.set_xlabel('Outcome')
    plt.legend(loc='upper right', title='Kelompok BMI')
    st.pyplot(fig)    

# Correlation Plot
if st.sidebar.checkbox('Show Correlation Plot'):
    st.subheader('Correlation Plot')
    fig, ax = plt.subplots()
    a = sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    st.pyplot(fig)

# Sidebar Prediction
#st.sidebar.header('Prediction')
# Split the data
X = df.drop(['Outcome', 'Age_grup', 'BMI_grup'], axis=1)
y = df['Outcome']
# Oversampling
# Imbalance data
from imblearn.over_sampling import SMOTE
from collections import Counter
smote = SMOTE()
X, y = smote.fit_resample(X, y)
#print(sorted(Counter(y).items()))
# Feature Scaling
# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)

# Probability 
y_train_pred = logreg.predict_proba(X_train)[:,1]

# User Input

st.sidebar.subheader('Prediction')
if st.sidebar.checkbox('Show Prediction'):
# Input
    st.subheader('Prediction Input')
    preganancies = st.number_input('Masukkan Jumlah Kehamilan     :',0)
    glucose = st.number_input('Masukkan Kadar Glukosa     :',0)
    bloodpressure = st.number_input('Masukkan Tekanan Darah     :',0)
    skinthickness = st.number_input('Masukkan Ketebalan Kulit     :',0)
    insulin = st.number_input('Masukkan Insulin     :',0)
    bmi = st.number_input('Masukkan BMI     :',0)
    dpf = st.number_input('Masukkan Diabetes Pedigree Function     :',0)
    age = st.number_input('Masukkan Usia     :',0)
    data = [[preganancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]]

# Button
    predict = st.button('Predict')

# Button

    #st.subheader('Prediction')
    #st.write(data)
    # Prediction
    pred = logreg.predict(data)
    #st.write(pred)
    if pred == 0:
        st.write('Pasien tidak memiliki diabetes')
    else:
        st.write('Pasien memiliki diabetes')

# Sidebar About 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
st.sidebar.subheader('About Model')
if st.sidebar.checkbox('Show About'):
    # Akurasi Model
    st.subheader('Akurasi Model')
    st.write('Akurasi Model     : ', (accuracy_score(y_test, pred_logreg)*100).round(2), '%')
    st.write('Precision Score   : ', (precision_score(y_test, pred_logreg)*100).round(2), '%')
    st.write('Recall Score      : ', (recall_score(y_test, pred_logreg)*100).round(2), '%')
    st.write('F1 Score          : ', (f1_score(y_test, pred_logreg)*100).round(2), '%')
    st.write('Mean Absolute Error : ', (mean_absolute_error(y_test, pred_logreg)*100).round(2), '%')

    st.subheader('Confusion Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, pred_logreg), annot=True, fmt='.2f', cmap='summer')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    
    # ROC Curve Train
    st.subheader('ROC Curve Train')
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(y_train, y_train_pred)
    roc_auc = roc_auc_score(y_train, y_train_pred)


    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    #plt.title('Receiver Operating Characteristic')
    plt.title('ROC Curve Train')
    plt.legend(loc="lower right")
    st.pyplot(fig)

    # Test
    st.subheader('ROC Curve Test')
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(y_test, pred_logreg)
    roc_auc = roc_auc_score(y_test, pred_logreg)


    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    #plt.title('Receiver Operating Characteristic')
    plt.title('ROC Curve Test')
    plt.legend(loc="lower right")
    st.pyplot(fig)


