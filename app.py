from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import pandas as pd

app = FastAPI()

model = joblib.load("linear_model.pkl")

class ModelInput(BaseModel):
        daromad: float
        uy_yoshi: float
        xonalar_soni: float
        yotoqxonalar_soni: float
        aholi_soni: float 
        xonadon_aholisi: float 
        kenglik: float 
        uzunlik: float 

@app.get("/")
def hello():
        return "Salom bratan ishlayapti!"

@app.post("/predict")
def model_predict(data: ModelInput):

        input_data = pd.DataFrame([{
        "id": 1,
        "MedInc": data.daromad,
        "HouseAge": data.uy_yoshi,
        "AveRooms": data.xonalar_soni,
        "AveBedrms": data.yotoqxonalar_soni,
        "Population": data.aholi_soni,
        "AveOccup": data.xonadon_aholisi,
        "Latitude": data.kenglik,
        "Longitude": data.uzunlik
    }])
        
        result = model.predict(input_data)

        return f"Uy narxi ---> {result[0]}"

if __name__ == "__main__":
        uvicorn.run(app)