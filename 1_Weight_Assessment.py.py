import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("Weight Assessment")

df = pd.read_csv("weight-height.csv")

df1 = df[['Height', 'Weight']]

def filter_outlier(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]

df_c = filter_outlier(df1, 'Weight')
df_c = filter_outlier(df_c, 'Height')

x = df_c[['Weight']]
y = df_c['Height']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=42
)

scaler = StandardScaler()

x_train_s = scaler.fit_transform(x_train)

model = LinearRegression()
model.fit(x_train_s, y_train)

patient_name = st.text_input("Patient Name")

weight = st.number_input(
    "Enter Weight (kg)",
    min_value=40,
    max_value=200
)

if st.button("Predict Height"):

    prediction = model.predict(
        scaler.transform([[weight]])
    )[0]

    st.success(f"""
    Dear ***{patient_name}***, your estimated height is: {prediction:.2f} cm
    """)