from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# venv\Scripts\activate.bat  
# pip install -r requirements.txt
# uvicorn app:app --reload 

# http://127.0.0.1:8000/docs

app = FastAPI()

# Load pre-trained model
model = joblib.load('C:/Coding1/Coding/python/Projects/Credit card approval/credit_card_model.pkl')

# Define essential features (5-6 features)
# You should ensure these match your model's expected features
features = [
    'Children count', 'Income', 'Age', 'Employment length', 'Has a mobile phone', 
    'Has a work phone', 'Has a phone', 'Has an email', 'Family member count', 
    'Account age', 'Job title_Cleaning staff', 'Job title_Cooking staff', 
    'Job title_Core staff', 'Job title_Drivers', 'Job title_HR staff', 
    'Job title_High skill tech staff', 'Job title_IT staff', 'Job title_Laborers', 
    'Job title_Low-skill Laborers', 'Job title_Managers', 'Job title_Medicine staff', 
    'Job title_No Job', 'Job title_Private service staff', 'Job title_Realty agents', 
    'Job title_Sales staff', 'Job title_Secretaries', 'Job title_Security staff', 
    'Job title_Waiters/barmen staff', 'Employment status_Pensioner', 
    'Employment status_State servant', 'Employment status_Student', 
    'Employment status_Working', 'Gender_M', 'Has a car_Y', 'Has a property_Y', 
    'Education level_Higher education', 'Education level_Incomplete higher', 
    'Education level_Lower secondary', 'Education level_Secondary / secondary special', 
    'Marital status_Married', 'Marital status_Separated', 'Marital status_Single / not married', 
    'Marital status_Widow', 'Dwelling_House / apartment', 'Dwelling_Municipal apartment', 
    'Dwelling_Office apartment', 'Dwelling_Rented apartment', 'Dwelling_With parents'
]

# Input model schema using Pydantic for data validation
class InputData(BaseModel):
    Children_count: float
    Income: float
    Age: float
    Employment_length: float
    Has_a_mobile_phone: float
    Has_a_work_phone: float
    Has_a_phone: float
    Has_an_email: float
    Family_member_count: float
    Account_age: float
    Job_title_Cleaning_staff: bool
    Job_title_Cooking_staff: bool
    Job_title_Core_staff: bool
    Job_title_Drivers: bool
    Job_title_HR_staff: bool
    Job_title_High_skill_tech_staff: bool
    Job_title_IT_staff: bool
    Job_title_Laborers: bool
    Job_title_Low_skill_Laborers: bool
    Job_title_Managers: bool
    Job_title_Medicine_staff: bool
    Job_title_No_Job: bool
    Job_title_Private_service_staff: bool
    Job_title_Realty_agents: bool
    Job_title_Sales_staff: bool
    Job_title_Secretaries: bool
    Job_title_Security_staff: bool
    Job_title_Waiters_barmen_staff: bool
    Employment_status_Pensioner: bool
    Employment_status_State_servant: bool
    Employment_status_Student: bool
    Employment_status_Working: bool
    Gender_M: bool
    Has_a_car_Y: bool
    Has_a_property_Y: bool
    Education_level_Higher_education: bool
    Education_level_Incomplete_higher: bool
    Education_level_Lower_secondary: bool
    Education_level_Secondary_secondary_special: bool
    Marital_status_Married: bool
    Marital_status_Separated: bool
    Marital_status_Single_not_married: bool
    Marital_status_Widow: bool
    Dwelling_House_apartment: bool
    Dwelling_Municipal_apartment: bool
    Dwelling_Office_apartment: bool
    Dwelling_Rented_apartment: bool
    Dwelling_With_parents: bool

# Helper function to preprocess the input
def preprocess_input(data: InputData):
    # Convert the InputData to a numpy array with the same order as the model expects
    input_array = np.array([
        data.Children_count, data.Income, data.Age, data.Employment_length, 
        data.Has_a_mobile_phone, data.Has_a_work_phone, data.Has_a_phone, 
        data.Has_an_email, data.Family_member_count, data.Account_age, 
        data.Job_title_Cleaning_staff, data.Job_title_Cooking_staff, 
        data.Job_title_Core_staff, data.Job_title_Drivers, data.Job_title_HR_staff, 
        data.Job_title_High_skill_tech_staff, data.Job_title_IT_staff, 
        data.Job_title_Laborers, data.Job_title_Low_skill_Laborers, 
        data.Job_title_Managers, data.Job_title_Medicine_staff, 
        data.Job_title_No_Job, data.Job_title_Private_service_staff, 
        data.Job_title_Realty_agents, data.Job_title_Sales_staff, 
        data.Job_title_Secretaries, data.Job_title_Security_staff, 
        data.Job_title_Waiters_barmen_staff, data.Employment_status_Pensioner, 
        data.Employment_status_State_servant, data.Employment_status_Student, 
        data.Employment_status_Working, data.Gender_M, data.Has_a_car_Y, 
        data.Has_a_property_Y, data.Education_level_Higher_education, 
        data.Education_level_Incomplete_higher, data.Education_level_Lower_secondary, 
        data.Education_level_Secondary_secondary_special, data.Marital_status_Married, 
        data.Marital_status_Separated, data.Marital_status_Single_not_married, 
        data.Marital_status_Widow, data.Dwelling_House_apartment, 
        data.Dwelling_Municipal_apartment, data.Dwelling_Office_apartment, 
        data.Dwelling_Rented_apartment, data.Dwelling_With_parents
    ]).reshape(1, -1)  # Ensure it's a 2D array
    return input_array

# Prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    input_data = preprocess_input(data)
    prediction = model.predict(input_data)
    result = 'Approved' if prediction[0] == 1 else 'Not Approved'
    return {"approval_status": result}
    