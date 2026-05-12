import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("BP Risk Assessment")

st.write("""
This module evaluates and predicts your Blood Pressure (BP) risk category.
""")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("weight-height.csv")

# ---------------------------------------------------
# TARGET ENCODING
# ---------------------------------------------------

def bp_score(bp):

    if bp == 'Normal':
        return 0

    elif bp == 'Elevated':
        return 1

    else:
        return 2

df['BP'] = df['BP_Category'].apply(bp_score)

# ---------------------------------------------------
# FEATURE ENCODING
# ---------------------------------------------------

df['Smoker_'] = df['Smoker'].apply(
    lambda x: 1 if x.lower() == 'yes' else 0
)

df['Gender_'] = df['Gender'].apply(
    lambda x: 1 if x.lower() == 'male' else 0
)

def activity_score(activity):

    if activity == 'Very Active':
        return 3

    elif activity == 'Moderately Active':
        return 2

    elif activity == 'Lightly Active':
        return 1

    else:
        return 0

df['Activity'] = df['Activity_Level'].apply(activity_score)

# ---------------------------------------------------
# OUTLIER REMOVAL
# ---------------------------------------------------

def filter_outlier(dataframe, column):

    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return dataframe[
        (dataframe[column] >= lower) &
        (dataframe[column] <= upper)
    ]

df_clean = filter_outlier(df, 'Diastolic_BP')
df_clean = filter_outlier(df_clean, 'Systolic_BP')
df_clean = filter_outlier(df_clean, 'BMI')

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------

X = df_clean[
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

y = df_clean['BP']

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

@st.cache_resource
def train_model():

    model = LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000
    )

    model.fit(x_train_scaled, y_train)

    return model

model = train_model()

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

patient_name = st.text_input("Patient Name")

Diastolic_BP = st.slider(
    "Enter Diastolic BP",
    50,
    150
)

BMI = st.slider(
    "Enter BMI",
    15,
    50
)

Systolic_BP = st.slider(
    "Enter Systolic BP",
    80,
    250
)

Smoker = st.selectbox(
    "Are You a Smoker?",
    ["No", "Yes"]
)

Activity = st.selectbox(
    "Select Activity Level",
    [
        "Very Active",
        "Moderately Active",
        "Lightly Active",
        "Sedentary"
    ]
)

Age = st.slider(
    "Enter Age",
    18,
    80
)

Gender = st.selectbox(
    "Select Gender",
    ["Male", "Female"]
)

# ---------------------------------------------------
# INPUT ENCODING
# ---------------------------------------------------

Smoker_value = 1 if Smoker == "Yes" else 0

Gender_value = 1 if Gender == "Male" else 0

activity_map = {
    "Very Active": 0,
    "Moderately Active": 1,
    "Lightly Active": 2,
    "Sedentary": 3
}

Activity_value = activity_map[Activity]

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("Predict BP"):

    prediction = model.predict(
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

    # -----------------------------------------------
    # CATEGORY & ADVICE
    # -----------------------------------------------

    if prediction == 0:

        category = "Normal"

        advice = """
        Your blood pressure appears normal.
        Maintain healthy eating habits,
        regular exercise, and proper hydration.
        """

    elif prediction == 1:

        category = "Elevated"

        advice = """
        Your blood pressure is elevated.
        Reduce salt intake, improve diet,
        and increase physical activity.
        """

    else:

        category = "High"

        advice = """
        Your blood pressure is high.
        Clinical consultation and lifestyle
        modification are strongly recommended.
        """

    # -----------------------------------------------
    # OUTPUT
    # -----------------------------------------------

    st.success(f"""
    Patient: {patient_name}

    BP Classification: {category}
    """)

    st.warning(f"Clinical Advice: {advice}")
