from fastapi import APIRouter, status
import joblib
import polars as pl
from routers.utils import Vin

router = APIRouter(
    prefix="/api/predict",
    tags=["predict"]
)

@router.post("/")
async def predict_wine_quality(vin: Vin):
    """ predict wine quality """
    data = {
        'fixed acidity': [vin.fixed_acidity],
        'volatile acidity': [vin.volatile_acidity],
        'citric acid': [vin.citric_acid],
        'residual sugar': [vin.residual_sugar],
        'chlorides': [vin.chlorides],
        'free sulfur dioxide': [vin.free_sulfur_dioxide],
        'total sulfur dioxide': [vin.total_sulfur_dioxide],
        'density': [vin.density],
        'pH': [vin.pH],
        'sulphates': [vin.sulphates],
        'alcohol': [vin.alcohol]
    }
    # Créer le DataFrame polars
    vin_df = pl.DataFrame(data)
    # Loading models
    model = joblib.load("../models/model_wine_recommendation.pkl")
    # Predict
    quality = int(model.predict(vin_df)[0])
    return {"quality": quality}

@router.get("/")
async def get_perfect_wine():
    """ Renvoie les caractéristique du vin parfait statistiquement. 
    Ici, renvoi le vin le plus chargé en alcool.
    """
    # Read csv
    df_csv = pl.read_csv("../data/Wines.csv")
    # Garder la ligne avec la valeur maximale dans la colonne 'alcohol'
    max_alcohol_row = df_csv.filter(pl.col("alcohol") == pl.col("alcohol").max())
    
    vin_parfait = Vin(
        fixed_acidity = max_alcohol_row['fixed acidity'][0],
        volatile_acidity = max_alcohol_row['volatile acidity'][0],
        citric_acid = max_alcohol_row['citric acid'][0],
        residual_sugar = max_alcohol_row['residual sugar'][0],
        chlorides = max_alcohol_row['chlorides'][0],
        free_sulfur_dioxide = max_alcohol_row['free sulfur dioxide'][0],
        total_sulfur_dioxide = max_alcohol_row['total sulfur dioxide'][0],
        density = max_alcohol_row['density'][0],
        pH = max_alcohol_row['pH'][0],
        sulphates = max_alcohol_row['sulphates'][0],
        alcohol = max_alcohol_row['alcohol'][0],
        quality = max_alcohol_row['quality'][0]
    )
    return {"vin": vin_parfait}