import config
from src.data_ingestion import load_data
from src.exploratory_analysis import perform_eda
from src.data_preprocessing import preprocess_data
from src.model_training import train_and_evaluate_model
import mlflow
import numpy as np

def main():
    """Função principal para executar o pipeline de previsão de churn com MLflow."""
    
    # Configurar e iniciar o rastreamento do MLflow
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        print("Executando pipeline com rastreamento MLflow...")
        mlflow.log_param("data_path", config.DATA_PATH)

        # 1. Carregamento de Dados
        df_raw = load_data(config.DATA_PATH)
        
        # 2. Análise Exploratória de Dados (EDA)
        df_eda = perform_eda(df_raw.copy(), config.IMAGE_PATH)
        # Logar os gráficos da EDA como artefatos no MLflow
        mlflow.log_artifacts(config.IMAGE_PATH, artifact_path="images")
        
        # 3. Pré-processamento e Engenharia de Features
        X_processed, y_series, preprocessor = preprocess_data(df_eda, config.PREPROCESSOR_PATH)
        mlflow.log_artifact(config.PREPROCESSOR_PATH, "preprocessor")
        
        y_numpy = y_series.to_numpy()

        # 4. Treinamento e Avaliação do Modelo
        model = train_and_evaluate_model(X_processed, y_numpy, preprocessor, config.MODEL_PATH, config.IMAGE_PATH)


        print("\nPIPELINE DE MACHINE LEARNING RASTREÁVEL EXECUTADO COM SUCESSO!")

if __name__ == "__main__":
    main()
