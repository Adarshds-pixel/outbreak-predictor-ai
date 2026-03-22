from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import joblib
from collections import deque

app = FastAPI()

# Static + Template Mount
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load ML Assets
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_map = joblib.load("models/label_map.pkl")
reverse_map = {v: k for k, v in label_map.items()}

# Weekly Trend Storage
weekly_risk = deque(maxlen=7)

last_stats = {
    "prediction": "Low",
    "risk_counts": [],
    "feature_importance": {},
    "weekly_data": [],
}

# -----------------------------
#  SAFETY MEASURES FUNCTION
# -----------------------------
def get_safety_measures(risk):
    if risk == "Low":
        return [
            "Maintain normal hygiene practices.",
            "Ensure clean storage of drinking water.",
            "Avoid stagnant water near your home.",
        ]
    elif risk == "Medium":
        return [
            "Boil drinking water for at least 10 minutes.",
            "Use chlorine tablets if water seems cloudy.",
            "Increase sanitation and community cleaning drives.",
            "Monitor children and elderly for early symptoms.",
        ]
    else:  # High
        return [
            "Do NOT drink tap or ground water—use purified/tanker water only.",
            "Immediate chlorination of community water sources.",
            "Distribute ORS, zinc tablets, and safe water packets.",
            "Alert nearest PHC/CHC for emergency response.",
        ]


# -----------------------------
#  EXPLANATION ENGINE
# -----------------------------
def generate_explanation(pred_idx, feature_values):
    coefs = model.coef_[pred_idx]
    top_idx = np.argsort(np.abs(coefs))[::-1][:3]  # top 3 factors

    feature_names = [
        "Temperature","Humidity","Rainfall","Population Density",
        "Cases Last Week","Social Sentiment","Resource Utilization",
        "pH Level","Turbidity","Contamination Index","TDS","Water Temperature"
    ]

    explanation = []
    for i in top_idx:
        explanation.append(f"{feature_names[i]} contributed significantly (value: {feature_values[i]})")

    return explanation


# -----------------------------
#  Pydantic Input Schema
# -----------------------------
class InputData(BaseModel):
    temperature: float
    humidity: float
    rainfall: float
    population_density: float
    cases_last_week: float
    social_sentiment: float
    resource_utilization: float

    ph_level: float
    turbidity: float
    contamination_index: float
    tds: float
    water_temperature: float


# -----------------------------
#  ROUTES
# -----------------------------
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/how-it-works")
def how_it_works(request: Request):
    return templates.TemplateResponse("how-it-works.html", {"request": request})


@app.get("/stats")
def stats():
    return last_stats


# -----------------------------
#  PREDICTION API
# -----------------------------
@app.post("/predict")
def predict(data: InputData):

    # Convert to array for model
    features = [
        data.temperature, data.humidity, data.rainfall,
        data.population_density, data.cases_last_week,
        data.social_sentiment, data.resource_utilization,
        data.ph_level, data.turbidity, data.contamination_index,
        data.tds, data.water_temperature
    ]

    arr = np.array([features])
    arr_scaled = scaler.transform(arr)

    pred_class = int(model.predict(arr_scaled)[0])
    predicted_label = reverse_map[pred_class]

    # Update Weekly Trend
    weekly_map = {"Low": 1, "Medium": 2, "High": 3}
    weekly_risk.append(weekly_map[predicted_label])

    # Feature Importance
    feature_names = [
        "Temperature","Humidity","Rainfall","Population Density",
        "Cases Last Week","Social Sentiment","Resource Utilization",
        "pH Level","Turbidity","Contamination Index","TDS","Water Temperature"
    ]

    importance = np.abs(model.coef_[pred_class]).tolist()
    feature_importance = dict(zip(feature_names, importance))

    # Explanation Engine
    explanation_list = generate_explanation(pred_class, features)

    # Safety Instructions
    safety_list = get_safety_measures(predicted_label)

    # Update stats for dashboard
    last_stats.update({
        "prediction": predicted_label,
        "feature_importance": feature_importance,
        "weekly_data": list(weekly_risk),
        "safety": safety_list,
        "explanation": explanation_list
    })

    # Response for frontend
    return {
        "prediction": predicted_label,
        "feature_importance": feature_importance,
        "weekly_data": list(weekly_risk),
        "safety": safety_list,
        "explanation": explanation_list
    }


# -----------------------------
#  RUN SERVER
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
