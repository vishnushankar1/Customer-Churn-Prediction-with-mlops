# 🧠 Customer Churn Prediction – MLOps Project

This is a customer churn prediction system built using Machine Learning (ANN), FastAPI, and MLOps tools like DVC, MLflow, and Docker. The model predicts whether a customer is likely to leave a telecom service.

---

## 📂 What this project contains

- `src/` – Code for data ingestion, validation, transformation, and training  
- `artifacts/` – Final model and scaler files  
- `app/` – FastAPI app for serving predictions  
- `Dockerfile` – To build the container  
- `dvc.yaml` – DVC pipeline definition  
- `requirements.txt` – Python dependencies  

---

## 📦 How to run from GitHub Container Registry

This image is already built and pushed to GHCR.

### 🔧 Step 1: Pull the Docker image

```bash
docker pull ghcr.io/vishnushankar1/churn-fastapi:latest
```

### ▶️ Step 2: Run the container

```bash
docker run -p 8000:8000 ghcr.io/vishnushankar1/churn-fastapi:latest
```

### 🌐 Step 3: Open your browser

Go to:  
[http://localhost:8000/docs](http://localhost:8000/docs)  
Use Swagger UI to test the API.

---

## 🔗 Useful Links

- 🔗 GitHub Repo: [Customer Churn Prediction](https://github.com/vishnushankar1/Customer-Churn-Prediction-with-mlops)  
- 🔗 GHCR Image: `ghcr.io/vishnushankar1/churn-fastapi:latest`  
- 🔗 API Docs (local): `http://localhost:8000/docs`  

---

## ✍️ Author

**Vishnu Shankar**  
GitHub: [@vishnushankar1](https://github.com/vishnushankar1)

---

> ⭐ Don't forget to ⭐ the repo if you find this useful!