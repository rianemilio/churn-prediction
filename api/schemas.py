from pydantic import BaseModel, Field
from typing import Literal

# Este schema define os dados que esperamos receber no corpo da requisição POST
class ChurnFeatures(BaseModel):
    """
    Define as features esperadas para a previsão de churn.
    Os nomes dos campos devem corresponder exatamente às colunas do DataFrame
    usado no treinamento, antes do pré-processamento.
    """
    gender: Literal['Male', 'Female']
    SeniorCitizen: int = Field(..., ge=0, le=1) # ge=greater or equal, le=less or equal
    Partner: Literal['Yes', 'No']
    Dependents: Literal['Yes', 'No']
    tenure: int = Field(..., ge=0)
    PhoneService: Literal['Yes', 'No']
    MultipleLines: Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['Yes', 'No', 'No internet service']
    OnlineBackup: Literal['Yes', 'No', 'No internet service']
    DeviceProtection: Literal['Yes', 'No', 'No internet service']
    TechSupport: Literal['Yes', 'No', 'No internet service']
    StreamingTV: Literal['Yes', 'No', 'No internet service']
    StreamingMovies: Literal['Yes', 'No', 'No internet service']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: Literal['Yes', 'No']
    PaymentMethod: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    class Config:
        # Exemplo de payload para a documentação automática do FastAPI
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }


# Este schema define o formato da nossa resposta de previsão
class PredictionOut(BaseModel):
    """
    Define o formato da resposta da API.
    """
    prediction: int
    probability: float
