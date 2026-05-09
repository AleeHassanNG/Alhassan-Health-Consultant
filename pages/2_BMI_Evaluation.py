import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("BMI Evaluation")

df = pd.read_csv("weight-height.csv")

df2 = df[['Height', 'Weight', 'Age', 'Gender', 'BMI']]

df2['Gender'] = df2['Gender'].map({
    'Male': 0,
    'Female': 1
})

def filter_outlier(df2, column):
    Q1 = df2[column].quantile(0.25)
    Q3 = df2[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    return df2[(df2[column] >= lower_bound) &(df2[column] <= upper_bound)]
df_s  = filter_outlier(df2, 'Weight')
df_s  = filter_outlier(df_s, 'Height')
df_s  = filter_outlier(df_s, 'BMI')

x = df_s[['Height', 'Weight', 'Age', 'Gender']]
y = df_s['BMI']

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

height = st.slider("Height (cm)", 50, 220)
weight = st.slider("Weight (kg)", 40, 200)
age = st.slider("Age", 18, 80)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

gender_value = 0 if gender == "Male" else 1

if st.button("Predict BMI"):

    bmi = model.predict(
        scaler.transform(
            [[height, weight, age, gender_value]]
        )
    )[0]

    if bmi < 18.5:
        category = "Underweight"
        advise = f'You are {category}, Nutritional improvement is recommened'

    elif bmi < 25:
        category = "Normal"
        advise = f'Your BMI category is {category}, your weight falls withing the healthy range, keep it up'

    elif bmi < 30:
        category = "Overweight"
        advise = f'You are {category}, Lifestyle and dietary adjustments are advised.'

    else:
        category = "Obese"
        advise = f'Your BMI category is {category}, Clinical weight management is strongly recommended.'


    st.success(f"""
    Patient: {patient_name}

    Your Predicted BMI is: {bmi:.2f}

    Classification: {category}
    """)
    st.warning(f'Clinical Advice:{advise}')


    
