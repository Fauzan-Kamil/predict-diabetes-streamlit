import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
import warnings
warnings.filterwarnings('ignore')
import pickle 
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Load the data
df = pd.read_csv('Data/diabetes_fix.csv')
st.title('Diabetes Prediction App')
st.write('Aplikasi ini memprediksi kemungkinan seseorang menderita diabetes berdasarkan beberapa fitur yang dimasukan')
st.write('Dataset yang digunakan adalah dataset diabetes dari kaggle')
st.write('Dataset : https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset')
st.write('')
st.write(df.head())
divider = st.container()
divider.markdown('---')
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
    fitur1 = st.selectbox('X', ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'))
    fitur2 = st.selectbox('Y', ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'))
    st.write('Scatter Plot dari fitur', fitur1, 'dan', fitur2)
    fig, ax = plt.subplots()
    plt.scatter(x=df[fitur1], y=df[fitur2], c=df['Outcome'], cmap='rainbow')
    st.pyplot(fig)

# Density Plot
if st.sidebar.checkbox('Show Data Vis'):
    st.subheader('Jumlah pasien diabetes')
    fig, ax = plt.subplots()
    a = sns.countplot(x='Outcome', data=df)
    for j in a.containers:
        a.bar_label(j, label_type='edge')
    a.set_xlabel('Outcome')
    st.pyplot(fig)
    st.write('Bisa dilihat dari grafik diatas bahwa banyak orang yang tidak terkena diabetes dan sedikit orang yang terkena diabetes yaitu 268 orang. ')

    divider = st.container()
    divider.markdown('---')
    st.subheader('Jumlah pasien diabetes berdasarkan usia')
    fig, ax = plt.subplots()
    a = sns.countplot(x='Outcome', hue='Age_grup', data=df)
    for j in a.containers:
        a.bar_label(j, label_type='edge')
    a.set_xlabel('Outcome')
    plt.legend(loc='upper right', title='Kelompok Umur')
    st.pyplot(fig) 
    st.write('Banyak pasien yang terkena diabetes adalah yang berumur 26-35 tahun atau dewasa awal dengan jumlah 86 orang lalu diikutu dengan dewasa akhir yaitu 46-55 tahun dengan jumlah 79 orang dan yang paling sedikit adalah manula dengan jumlah 4 orang.')

    divider = st.container()
    divider.markdown('---')
    st.subheader('Jumlah pasien diabetes berdasarkan BMI')
    fig, ax = plt.subplots()
    a = sns.countplot(x='Outcome', hue='BMI_grup', data=df)
    for j in a.containers:
        a.bar_label(j, label_type='edge')
    a.set_xlabel('Outcome')
    plt.legend(loc='upper right', title='Kelompok BMI')
    st.pyplot(fig)    
    st.write('Berdasarkan kelompok BMI yang paling banyak terkena diabetes adalah yang memiliki BMI lebih dari 30 (Obesitasa II) dengan jumlah 219 orang lalu diikuti dengan BMI 25 - 29.9 (Obesitas) dengan jumlah 40 orang.')

# Correlation Plot
if st.sidebar.checkbox('Show Correlation Plot'):
    st.subheader('Correlation Plot')
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Sidebar Prediction
model = pd.read_pickle('model_svm.pkl')
# User Input
st.sidebar.subheader('Prediction')
if st.sidebar.checkbox('Show Prediction'):
# Input
    st.subheader('Prediction Input')
    nama = st.text_input('Masukkan nama', 'Nama' )
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
    # Prediction
    pred = model.predict(data)
    #st.write(pred)
    if predict:
        #st.write(pred)
        if pred == 0:
            st.write('Hiiii, ',nama,'. Kamu aman dari diabetes, tetap jaga kesehatan ya!')
        else:
            st.write('Semoga cepat sembuh ya ',nama,'. Kamu memiliki resiko terkena diabetes, jangan lupa untuk ke dokter ya!')

# Sidebar About 
st.sidebar.subheader('About Model')
if st.sidebar.checkbox('Show About'):
    st.subheader('Jumlah Data')
    st.write('Jumlah data yang digunakan adalah', 1000)
    st.write('Jumlah data yang digunakan untuk training adalah', 640)
    st.write('Jumlah data yang digunakan untuk testing adalah', 200)
    st.write('Jumlah data yang digunakan untuk validation adalah', 160)

    divider = st.container()
    divider.markdown('---')
    st.subheader('Akurasi Model Support Vector Machine')
    st.write('Accuracy  : 0.78')
    st.write('Precision : 0.75')
    st.write('Recall    : 0.82')
    st.write('F1 Score  : 0.67')
    st.write('AUC       : 0.77')
    st.write('MAE       : 0.225')

    divider = st.container()
    divider.markdown('---')

# Load Gambar
    st.subheader('Classification Report')
    image = Image.open('img\classification_svm.png')
    st.image(image, caption='Model Support Vector Machine', use_column_width=True)

    divider = st.container()
    divider.markdown('---')
    st.subheader('Confusion Matrix Trainig')
    image = Image.open('img\cm-train.png')
    st.image(image, caption='Comfusion Matrix Training Model SVM', use_column_width=True)

    divider = st.container()
    divider.markdown('---')
    st.subheader('Confusion Matrix Testing')
    image = Image.open('img\cm-test.png')
    st.image(image, caption='Comfusion Matrix Testing Model SVM', use_column_width=True)

    divider = st.container()
    divider.markdown('---')
    st.subheader('Confusion Matrix Validation')
    image = Image.open('img\cm-val.png')
    st.image(image, caption='Comfusion Matrix Validation Model SVM', use_column_width=True)

    divider = st.container()
    divider.markdown('---')
    st.subheader('ROC Curve')
    image = Image.open('img\model_svm.png')
    st.image(image, caption='ROC Curve Model SVM', use_column_width=True)

    divider = st.container()
    divider.markdown('---')
    st.subheader('ROC Curve Testing')
    image = Image.open('img\model_svm-test.png')
    st.image(image, caption='ROC Curve Model SVM Test', use_column_width=True)

