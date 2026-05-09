import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title("BP Risk Assessment")

st.write("""
This module evaluates and approximates your Blood Pressure (BP) risk level.
""")

# Load Dataset
df = pd.read_csv("weight-height.csv")

# -----------------------------
# DATA PREPROCESSING
# -----------------------------

def cc_score(cc):

    if cc == 'Normal':
        return 0

    elif cc == 'Elevated':
        return 1

    else:
        return 2

df['BP'] = df['BP_Category'].apply(cc_score)

df['Smoker_'] = df['Smoker'].apply(
    lambda x: 1 if x.lower() == 'yes' else 0
)

df['Gender_'] = df['Gender'].apply(
    lambda x: 1 if x.lower() == 'male' else 0
)

def activity_score(act):

    if act == 'Very Active':
        return 0

    elif act == 'Moderately Active':
        return 1

    elif act == 'Lightly Active':
        return 2

    else:
        return 3

df['Activity'] = df['Activity_Level'].apply(activity_score)

# -----------------------------
# REMOVE OUTLIERS
# -----------------------------

def filter_outlier(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[
        (df[column] >= lower_bound) &
        (df[column] <= upper_bound)
    ]

df_l = filter_outlier(df, 'Diastolic_BP')
df_l = filter_outlier(df_l, 'Systolic_BP')
df_l = filter_outlier(df_l, 'BMI')

# -----------------------------
# FEATURES
# -----------------------------

X = df_l[
    [
        'Diastolic_BP',
        'BMI',
        'Systolic_BP',
        'Smoker_',
        'Activity',
        'Age',
        'Gender_'
    ]
]

Y = df_l['BP']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.30,
    random_state=42
)

# -----------------------------
# STANDARDIZATION
# -----------------------------

scaler = StandardScaler()

x_train_s = scaler.fit_transform(x_train)

# -----------------------------
# MODEL
# -----------------------------

@st.cache_resource
def train_model():

    model = LogisticRegression(
        C=0.1,
        penalty='l1',
        solver='liblinear',
        max_iter=1000
    )

    model.fit(x_train_s, y_train)

    return model

model = train_model()

# -----------------------------
# USER INPUTS
# -----------------------------

patient_name = st.text_input("Patient Name")

Diastolic_BP = st.slider(
    "Enter Your Diastolic BP",
    50,
    150
)

BMI = st.slider(
    "Enter Your BMI",
    15,
    50
)

Systolic_BP = st.slider(
    "Enter Your Systolic BP",
    80,
    250
)

Smoker = st.selectbox(
    "Are you a Smoker?",
    ["No", "Yes"]
)

Activity = st.selectbox(
    "Select Your Activity Level",
    [
        "Very Active",
        "Moderately Active",
        "Lightly Active",
        "Sedentary"
    ]
)

Age = st.slider(
    "Select Your Age",
    18,
    80
)

Gender = st.selectbox(
    "Select Your Gender",
    ["Male", "Female"]
)

# -----------------------------
# ENCODING INPUTS
# -----------------------------

Smoker_value = 1 if Smoker == "Yes" else 0

Gender_value = 1 if Gender == "Male" else 0

activity_map = {
    "Very Active": 0,
    "Moderately Active": 1,
    "Lightly Active": 2,
    "Sedentary": 3
}

Activity_value = activity_map[Activity]

# -----------------------------
# PREDICTION
# -----------------------------

if st.button('Predict BP'):

    predicted_BP = model.predict(
        scaler.transform(
            [[
                Diastolic_BP,
                BMI,
                Systolic_BP,
                Smoker_value,
                Activity_value,
                Age,
                Gender_value
            ]]
        )
    )[0]

    # BP Categories
    if predicted_BP == 0:

        category = 'Normal'

        advise = """
        Your BP level appears normal.
        Maintain a healthy lifestyle,
        regular exercise, and balanced diet.
        """

    elif predicted_BP == 1:

        category = 'Elevated'

        advise = """
        Your BP level is elevated.
        Reduce salt intake, improve diet,
        and increase physical activity.
        """

    else:

        category = 'High'

        advise = """
        Your BP level is high.
        Medical consultation and lifestyle
        modification are strongly recommended.
        """

    # Display Results
    st.success(f"""
    Patient: {patient_name}

    Predicted BP Category: {category}
    """)

    st.warning(f"Clinical Advice: {advise}")
