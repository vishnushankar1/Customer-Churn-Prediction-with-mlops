from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import numpy as np
from app.model_loader import make_prediction

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
                  gender: str = Form(...),
                  partner: str = Form(...),
                  dependents: str = Form(...),
                  phone_service: str = Form(...),
                  multiple_lines: str = Form(...),
                  internet_service: str = Form(...),
                  online_security: str = Form(...),
                  online_backup: str = Form(...),
                  device_protection: str = Form(...),
                  tech_support: str = Form(...),
                  streaming_tv: str = Form(...),
                  streaming_movies: str = Form(...),
                  contract: str = Form(...),
                  paperless_billing: str = Form(...),
                  payment_method: str = Form(...)):

    # One-hot encode categorical features manually (must match training order!)
    gender_male = 1 if gender == "Male" else 0
    partner_yes = 1 if partner == "Yes" else 0
    dependents_yes = 1 if dependents == "Yes" else 0
    phone_service_yes = 1 if phone_service == "Yes" else 0

    multiple_lines_no_phone = 1 if multiple_lines == "No phone service" else 0
    multiple_lines_yes = 1 if multiple_lines == "Yes" else 0

    internet_service_fiber = 1 if internet_service == "Fiber optic" else 0
    internet_service_no = 1 if internet_service == "No" else 0

    online_security_no_internet = 1 if online_security == "No internet service" else 0
    online_security_yes = 1 if online_security == "Yes" else 0

    online_backup_no_internet = 1 if online_backup == "No internet service" else 0
    online_backup_yes = 1 if online_backup == "Yes" else 0

    device_protection_no_internet = 1 if device_protection == "No internet service" else 0
    device_protection_yes = 1 if device_protection == "Yes" else 0

    tech_support_no_internet = 1 if tech_support == "No internet service" else 0
    tech_support_yes = 1 if tech_support == "Yes" else 0

    streaming_tv_no_internet = 1 if streaming_tv == "No internet service" else 0
    streaming_tv_yes = 1 if streaming_tv == "Yes" else 0

    streaming_movies_no_internet = 1 if streaming_movies == "No internet service" else 0
    streaming_movies_yes = 1 if streaming_movies == "Yes" else 0

    contract_one_year = 1 if contract == "One year" else 0
    contract_two_year = 1 if contract == "Two year" else 0

    paperless_billing_yes = 1 if paperless_billing == "Yes" else 0

    payment_credit_card_auto = 1 if payment_method == "Credit card (automatic)" else 0
    payment_electronic_check = 1 if payment_method == "Electronic check" else 0
    payment_mailed_check = 1 if payment_method == "Mailed check" else 0

    input_array = np.array([
        SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
        gender_male, partner_yes, dependents_yes, phone_service_yes,
        multiple_lines_no_phone, multiple_lines_yes,
        internet_service_fiber, internet_service_no,
        online_security_no_internet, online_security_yes,
        online_backup_no_internet, online_backup_yes,
        device_protection_no_internet, device_protection_yes,
        tech_support_no_internet, tech_support_yes,
        streaming_tv_no_internet, streaming_tv_yes,
        streaming_movies_no_internet, streaming_movies_yes,
        contract_one_year, contract_two_year,
        paperless_billing_yes,
        payment_credit_card_auto, payment_electronic_check, payment_mailed_check
    ]).reshape(1, -1)  # Ensure it's 2D

    result = make_prediction(input_array)
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
