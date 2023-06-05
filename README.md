# Prediksi Diabetes Development Streamlit

## Tools

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)]()
[![Google Colab](https://img.shields.io/badge/Google%20Colab-black?style=for-the-badge&logo=google-colab&logoColor=golden)]()
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-white?style=for-the-badge&logo=jupyter&logoColor=golden)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)]()
[![Pandas](https://img.shields.io/badge/Pandas-356?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Numpy](https://img.shields.io/badge/Numpy-FFF?style=for-the-badge&logo=numpy&logoColor=blue)](https://numpy.org/)
[![Sklearn](https://img.shields.io/badge/Sklearn-white?style=for-the-badge&logo=scikit-learn&logoColor=orange)](https://scikit-learn.org/stable/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-white?style=for-the-badge&logo=https://matplotlib.org/&logoColor=blue)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-blue?style=for-the-badge&logo=seaborn.pydata&logoColor=white)](https://seaborn.pydata.org/)
[![Kagle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

## Algorithms

[![Support Vector Machine](https://img.shields.io/badge/Support%20Vector%20Machine-ff69b4.svg?style=for-the-badge&logo=Support-Vector-Machines&logoColor=white)]()
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-blueviolet.svg?style=for-the-badge&logo=Logistic-Regression&logoColor=white)]()

## Link to Streamlit App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://prediksi-diabetes.streamlit.app/](https://fauzan-kamil-predict-diabetes-streamlit-main-9w6qpd.streamlit.app/))

## Introduction

Prediksi Diabetes di deployment Streamlit dengan algoritma SVM adalah sebuah aplikasi yang dibuat untuk memprediksi kemungkinan seseorang mengalami diabetes berdasarkan beberapa faktor seperti usia, BMI, tekanan darah, kadar insulin, dan lain-lain. SVM (Support Vector Machine) adalah salah satu algoritma machine learning yang digunakan dalam aplikasi ini untuk membangun model prediksi berdasarkan data yang telah diolah sebelumnya. Aplikasi ini memudahkan pengguna untuk menginput data dan langsung mendapatkan hasil prediksi apakah ia berisiko terkena diabetes atau tidak. Dengan begitu, aplikasi ini dapat membantu dalam mencegah dan mengatasi penyakit diabetes dengan lebih efektif.

## Requirements

- Python 3.9
- Streamlit 1.14.1
- Pandas 1.3.3
- Numpy 1.22.4
- Scikit-learn 1.0.2
- Matplotlib 3.5.2
- Seaborn 0.11.2

## How to Use

**Jika dijalankan di local machine VSCode/Jupyter Notebook**

1. Clone repository ini
2. Buka terminal dan arahkan ke folder repository
3. Install requirements dengan perintah `pip install -r requirements.txt`
4. Jalankan perintah `streamlit run main.py`

**Jika dijalankan di Google Colab**

1. Buka Google Colab
2. Buat notebook baru
3. Jalankan perintah berikut pada cell pertama
   ```
   !pip install -r requirements.txt
   ```
4. Bisa jalankan file `diabetes.ipynb` atau `main.py`

> _Insight_ dari dataset diabetes dapat dilihat di file [diabetes.ipynb](https://github.com/Fauzan-Kamil/predict-diabetes-streamlit/blob/master/diabetes.ipynb)

```
Bagi yang penasaran dengan insights dari dataset diabetes, yuk cek file diabetes.ipynb!
Di sana tersedia informasi menarik yang dapat meningkatkan pemahaman kita tentang dataset tersebut.
```

## Struktur Folder

```
📦predict-diabetes-streamlit
 ┣ 📂.ipynb_checkpoints
 ┃ ┗ 📜diabetes-checkpoint.ipynb
 ┣ 📂Data
 ┃ ┣ 📜diabetes.csv
 ┃ ┗ 📜diabetes_fix.csv
 ┣ 📂img
 ┃ ┣ 📜classification_svm.png
 ┃ ┣ 📜cm-test.png
 ┃ ┣ 📜cm-train.png
 ┃ ┣ 📜cm-val.png
 ┃ ┣ 📜model_svm-test.png
 ┃ ┗ 📜model_svm.png
 ┣ 📜app.py
 ┣ 📜diabetes - Jupyter Notebook.pdf
 ┣ 📜diabetes.ipynb
 ┣ 📜main.py
 ┣ 📜model_svm.pkl
 ┣ 📜README.md
 ┗ 📜requirements.txt
```

## Create a

Author : [Fauzan Kamil](https://github.com/Fauzan-Kamil)
