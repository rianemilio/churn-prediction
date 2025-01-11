from fastapi import FastAPI
import joblib
import pandas as pd
import sys
import os

# Adicionar o diretório raiz ao path para que possamos importar 'config'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from api.schemas import ChurnFeatures, PredictionOut

# Inicializar o aplicativo FastAPI
app = FastAPI(
    title="API de Previsão de Churn",
    description="Uma API para prever a probabilidade de um cliente cancelar um serviço.",
    version="1.0.0"
)

# Carregar o modelo e o pré-processador na inicialização da API
try:
    model = joblib.load(config.MODEL_PATH)
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    print("Modelo e pré-processador carregados com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivos de modelo ou pré-processador não encontrados.")
    model = None
    preprocessor = None

# Definir um endpoint raiz para verificação de saúde (health check)
@app.get("/", tags=["Health Check"])
def read_root():
    """Endpoint raiz para verificar se a API está online."""
    return {"status": "API está online e funcionando!"}

# Definir o endpoint de previsão
@app.post("/predict/", response_model=PredictionOut, tags=["Prediction"])
def predict_churn(features: ChurnFeatures) -> PredictionOut:
    """
    Recebe os dados de um cliente e retorna a previsão de churn e a probabilidade.
    """
    if not model or not preprocessor:
        return {"error": "Modelo ou pré-processador não foram carregados."}

    # Converter os dados de entrada (Pydantic model) para um DataFrame do Pandas
    input_df = pd.DataFrame([features.model_dump()])

    # Aplicar o mesmo pré-processamento usado no treinamento
    processed_features = preprocessor.transform(input_df)

    # Fazer a previsão com o modelo
    prediction_array = model.predict(processed_features)
    prediction = int(prediction_array[0])

    # Obter a probabilidade de churn (probabilidade da classe 1)
    probability_array = model.predict_proba(processed_features)
    probability = float(probability_array[0][1])

    return PredictionOut(prediction=prediction, probability=probability)
