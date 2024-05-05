import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
# Create menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Visualisation", "Prediction"],
    icons=["house", "book", "calculator"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

#row0_spacer1, row0_1, row0_spacer2= st.columns((0.1, 3.2, 0.1))
#row1_spacer1, row1_1, row1_spacer2, row1_2 = st.columns((0.1, 1.5, 0.1, 1.5))
#row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))
#row0_spacer3, row3_0, row0_spacer3= st.columns((0.1, 3.2, 0.1))

row0_spacer1, row0_1, row0_spacer2 = st.columns((0.1, 3.2, 0.1))
row1_spacer1, row1_1, row1_spacer2, row1_2 = st.columns((0.1, 1.5, 0.1, 1.5))
row0_spacer3, row3_0, row0_spacer4 = st.columns((0.1, 3.2, 0.1))

# Load dataset
df = pd.read_csv('Data/diabetes.csv')
# Kelompok usia
age_grup = []
for i in df['Age']:
    if i >= 17 and i <= 25:
        age_grup.append('Remaja Akhir')
    elif i >= 26 and i <= 35:
        age_grup.append('Dewasa Awal')
    elif i >= 36 and i <= 45:
        age_grup.append('Dewasa Akhir')
    elif i >= 46 and i <= 55:
        age_grup.append('Lansia Awal')
    elif i >= 56 and i <= 65:
        age_grup.append('Lansia Akhir')
    else:
        age_grup.append('Manula')
df['AgeGrup'] = age_grup
# Kelompok BMI
BMI_grup = []
for i in df['BMI']:
    if i >= 0 and i <= 18.5:
        BMI_grup.append('Kurus')
    elif i >= 18.6 and i <= 22.9:
        BMI_grup.append('Normal')
    elif i >= 23 and i <= 24.9:
        BMI_grup.append('Gemuk')
    elif i >= 25 and i <= 29.9:
        BMI_grup.append('Obesitas')
    else:
        BMI_grup.append('Obesitas II')            
df['BMIGrup'] = BMI_grup
# Model
model = pd.read_pickle('model_svm.pkl')
# Handle selected option
if selected == "Home":
    row0_1.title("Diabetes Prediction App")
    with row0_1:
        st.markdown(
            "Diabetes Prediction App adalah sebuah aplikasi yang berguna untuk memprediksi kemungkinan seseorang menderita diabetes berdasarkan beberapa fitur yang dimasukkan. Aplikasi ini menggunakan dataset diabetes dari Kaggle untuk melakukan prediksi. Dengan memasukkan fitur yang relevan, seperti kadar gula darah, tekanan darah, dan sebagainya, aplikasi ini dapat memberikan prediksi yang cukup akurat mengenai kemungkinan seseorang menderita diabetes. Aplikasi ini sangat bermanfaat bagi orang-orang yang ingin mengetahui apakah mereka berisiko terkena diabetes atau tidak, sehingga dapat memperbaiki pola makan dan gaya hidup mereka untuk mencegah terjadinya penyakit diabetes."
        )
        st.markdown('Dataset : https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset')
        st.write(df.head())
        st.markdown('Atribut Dataset :')
        st.markdown("1. Pregnancies: Merupakan jumlah kehamilan yang pernah dialami oleh seorang pasien ")
        st.markdown("2. Glucose: Merupakan kadar gula darah pasien")
        st.markdown("3. Blood Pressure: Merupakan tekanan darah pasien")
        st.markdown("4. Skin Thickness: Merupakan ketebalan kulit pasien")
        st.markdown("5. Insulin: Merupakan kadar insulin pasien")
        st.markdown("6. BMI: Merupakan Body Mass Index pasien")
        st.markdown("7. Diabetes Pedigree Function: Merupakan riwayat diabetes dalam keluarga pasien")
        st.markdown("8. Age: Merupakan usia pasien")
        st.markdown("9. Outcome: Merupakan hasil diagnosis pasien, 0 berarti tidak terkena diabetes, 1 berarti terkena diabetes")

elif selected == "Data Visualisation":
    # Data Visualisasi dengan plotly
    with row1_1:
        st.subheader('Pilih fitur yang ingin ditampilkan histogramnya')
        fitur = st.selectbox('Fitur', ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome'))
        fig = px.histogram(df, x=fitur, color='Outcome', marginal='box', hover_data=df.columns)
        st.plotly_chart(fig)
        fig = px.histogram(df, x='Outcome', color='Pregnancies', barmode='group', hover_data=df.columns)
        fig.update_layout(title='Jumlah pasien per kelompok kehamilan', xaxis_title='Outcome', yaxis_title='Jumlah', font=dict(size=15))
        st.plotly_chart(fig)
        st.markdown(
            'Data menunjukkan bahwa pasien dengan 0 kehamilan merupakan kelompok yang paling banyak terkena diabetes dengan jumlah 38 orang, diikuti oleh kelompok dengan 3 kehamilan yang jumlahnya sebanyak 27 orang. Sementara itu, kelompok dengan 17 kehamilan memiliki jumlah pasien diabetes yang paling sedikit, hanya 1 orang. Perlu diperhatikan bahwa risiko terkena diabetes tidak selalu berkaitan dengan jumlah kehamilan seseorang, sehingga pemeriksaan secara rutin dan gaya hidup sehat tetap menjadi hal yang penting untuk dilakukan.'
        )
        fig = px.histogram(df, x='Outcome', color='AgeGrup', barmode='group', hover_data=df.columns)
        fig.update_layout(title='Jumlah pasien per kelompok usia', xaxis_title='Outcome', yaxis_title='Jumlah', font=dict(size=15))
        st.plotly_chart(fig)
        st.markdown(
            "Dari data yang dianalisis, terlihat bahwa kelompok usia 26-35 tahun atau dewasa awal memiliki jumlah pasien diabetes terbanyak, mencapai 86 orang. Sementara itu, kelompok dewasa akhir yaitu 46-55 tahun memiliki jumlah pasien diabetes yang hampir setara, mencapai 79 orang. Namun, terdapat perbedaan yang cukup signifikan pada kelompok manula dengan hanya terdapat 4 orang yang terkena diabetes. Hal ini menunjukkan bahwa usia masih menjadi faktor penting dalam kejadian diabetes pada pasien."
        )
    with row1_2:
        st.subheader('Pilih fitur yang ingin ditampilkan scatter plotnya')
        fitur1 = st.selectbox('Fitur 1', ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'))
        fitur2 = st.selectbox('Fitur 2', ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'))
        fig = px.scatter(df, x=fitur1, y=fitur2, color='Outcome', hover_data=df.columns)
        st.plotly_chart(fig)
        fig = px.histogram(df, x='Outcome', color='Outcome', hover_data=df.columns)
        fig.update_layout(title='Jumlah Pasien Diabetes', xaxis_title='Outcome', yaxis_title='Jumlah', font=dict(size=15))
        st.plotly_chart(fig)
        st.markdown(
            'Mari kita lihat grafik di atas. Dari grafik tersebut, terlihat bahwa mayoritas orang yang dianalisis dalam studi ini tidak terkena diabetes. Namun, terdapat sejumlah kecil orang yang terdiagnosis dengan diabetes, yaitu hanya 268 orang dari keseluruhan. Ini menunjukkan bahwa diabetes mungkin masih merupakan masalah kesehatan yang signifikan, namun masih mempengaruhi sebagian kecil populasi.'
        )
        fig = px.histogram(df, x='Outcome', color='BMIGrup', barmode='group', hover_data=df.columns)
        fig.update_layout(title='Jumlah pasien per kelompok BMI', xaxis_title='Outcome', yaxis_title='Jumlah', font=dict(size=15))
        st.plotly_chart(fig)
        st.markdown(
            'Dari hasil analisis data, terlihat bahwa kondisi BMI memiliki korelasi yang kuat dengan kemungkinan seseorang terkena diabetes. Kelompok BMI yang paling berisiko adalah yang memiliki BMI lebih dari 30 (Obesitas II), dengan jumlah 219 orang yang terkena diabetes. Sementara itu, kelompok BMI 25-29,9 (Obesitas) juga memiliki risiko yang cukup tinggi dengan jumlah 40 orang yang terkena diabetes. Temuan ini menunjukkan betapa pentingnya menjaga kesehatan dan berat badan yang sehat untuk mencegah risiko terkena diabetes.'
        )
elif selected == "Prediction":
    with row0_1:
        st.subheader('Masukkan Data')
    with row1_1:
        pregnancies = st.number_input('Jumlah Kehamilan', min_value=0, max_value=20, value=0)
        glucose = st.number_input('Kadar Gula', min_value=0, max_value=200, value=0)
        blood_pressure = st.number_input('Tekanan Darah', min_value=0, max_value=200, value=0)
        skin_thickness = st.number_input('Ketebalan Kulit', min_value=0, max_value=100, value=0)
    with row1_2:
        insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0)
        bmi = st.number_input('Body Mass Index (BMI)')
        diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function (Resiko Genetik)' )
        age = st.number_input('Umur', min_value=0, max_value=100, value=0)
    with row3_0:
        button = st.button('Predict')
        if button:
            data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
            pred = model.predict(data)
            if pred == 1:
                st.write('Pasien terkena diabetes')
            else:
                st.write('Pasien tidak terkena diabetes')
