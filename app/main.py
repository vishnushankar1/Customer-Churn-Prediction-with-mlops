# app/main.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import numpy as np
from app.model_loader import make_prediction
from app.schemas import CustomerData

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  SeniorCitizen: float = Form(...),
                  tenure: float = Form(...),
                  MonthlyCharges: float = Form(...),
                  TotalCharges: float = Form(...),
                  gender_Male: int = Form(...),
                  Partner_Yes: int = Form(...),
                  Dependents_Yes: int = Form(...),
                  PhoneService_Yes: int = Form(...),
                  MultipleLines_No_phone_service: int = Form(...),
                  MultipleLines_Yes: int = Form(...),
                  InternetService_Fiber_optic: int = Form(...),
                  InternetService_No: int = Form(...),
                  OnlineSecurity_No_internet_service: int = Form(...),
                  OnlineSecurity_Yes: int = Form(...),
                  OnlineBackup_No_internet_service: int = Form(...),
                  OnlineBackup_Yes: int = Form(...),
                  DeviceProtection_No_internet_service: int = Form(...),
                  DeviceProtection_Yes: int = Form(...),
                  TechSupport_No_internet_service: int = Form(...),
                  TechSupport_Yes: int = Form(...),
                  StreamingTV_No_internet_service: int = Form(...),
                  StreamingTV_Yes: int = Form(...),
                  StreamingMovies_No_internet_service: int = Form(...),
                  StreamingMovies_Yes: int = Form(...),
                  Contract_One_year: int = Form(...),
                  Contract_Two_year: int = Form(...),
                  PaperlessBilling_Yes: int = Form(...),
                  PaymentMethod_Credit_card_automatic: int = Form(...),
                  PaymentMethod_Electronic_check: int = Form(...),
                  PaymentMethod_Mailed_check: int = Form(...)):

    input_array = np.array([
        SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
        gender_Male, Partner_Yes, Dependents_Yes, PhoneService_Yes,
        MultipleLines_No_phone_service, MultipleLines_Yes,
        InternetService_Fiber_optic, InternetService_No,
        OnlineSecurity_No_internet_service, OnlineSecurity_Yes,
        OnlineBackup_No_internet_service, OnlineBackup_Yes,
        DeviceProtection_No_internet_service, DeviceProtection_Yes,
        TechSupport_No_internet_service, TechSupport_Yes,
        StreamingTV_No_internet_service, StreamingTV_Yes,
        StreamingMovies_No_internet_service, StreamingMovies_Yes,
        Contract_One_year, Contract_Two_year,
        PaperlessBilling_Yes,
        PaymentMethod_Credit_card_automatic,
        PaymentMethod_Electronic_check,
        PaymentMethod_Mailed_check
    ])

    result = make_prediction(input_array)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
