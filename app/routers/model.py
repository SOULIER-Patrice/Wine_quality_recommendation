from fastapi import APIRouter, status
import joblib
from routers.utils import Vin
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


router = APIRouter(
    prefix="/api/model",
    tags=["model"]
)


@router.get("/", tags=["model"])
async def get_model():
    """ Renvoie le modèle sérialisé """
    model = joblib.load("../models/model_wine_recommendation.pkl")
    model_params = model.get_params()
    return {"model_type": model.__class__.__name__,
            "model_params": model_params}


@router.get("/description", tags=["model"])
async def get_model_description():
    """ Renvoie l'accuracy du modèle """
    # Load data
    df_csv = pl.read_csv("../data/Wines.csv")
    X = df_csv.select([col for col in df_csv.columns if col not in ["Id", "quality"]])
    y = df_csv.select("quality")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model = joblib.load("../models/model_wine_recommendation.pkl")

    # Calcul accuracy train
    y_head_train = model.predict(X_train)
    accuracy_score_train = accuracy_score(y_train, y_head_train)
    # Calcul accuracy test
    y_head_test = model.predict(X_test)
    accuracy_score_test = accuracy_score(y_test, y_head_test)

    return {"model_accuracy_train": accuracy_score_train,
            "model_accuracy_test": accuracy_score_test}


@router.put("/", tags=["model"], status_code=status.HTTP_201_CREATED)
async def add_data(wine_to_add: Vin):
    """ Ajoute une donnée au dataset """
    # Read CSV
    df_csv = pl.read_csv("../data/Wines.csv")
    
    # Prend l'id le plus grand et ajoute 1
    new_id = df_csv['Id'].max() + 1

    # Add data
    new_wine = pl.DataFrame({
        'fixed acidity' : [wine_to_add.fixed_acidity],
        'volatile acidity' : [wine_to_add.volatile_acidity],
        'citric acid' : [wine_to_add.citric_acid],
        'residual sugar' : [wine_to_add.residual_sugar],
        'chlorides' : [wine_to_add.chlorides],
        'free sulfur dioxide' : [wine_to_add.free_sulfur_dioxide],
        'total sulfur dioxide' : [wine_to_add.total_sulfur_dioxide],
        'density' : [wine_to_add.density],
        'pH' : [wine_to_add.pH],
        'sulphates' : [wine_to_add.sulphates],
        'alcohol' : [wine_to_add.alcohol],
        'quality' : [wine_to_add.quality],
        'Id' : [new_id]
    })

    # Concat
    df_final = pl.concat([df_csv, new_wine])
    
    # Save new csv
    df_final.write_csv("../data/Wines.csv")

    return {"Id": new_id,
            "characteristics":wine_to_add}


@router.post("/retrain", tags=["model"])
async def post_retrain():
    """ Retrain model """
    # Load data
    df_csv = pl.read_csv("../data/Wines.csv")
    X = df_csv.select([col for col in df_csv.columns if col not in ["Id", "quality"]])
    y = df_csv.select("quality")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_head_test_rf = rf.predict(X_test)

    joblib.dump(rf, "../models/model_wine_recommendation.pkl")

    return {"new_accuracy": accuracy_score(y_test, y_head_test_rf)}