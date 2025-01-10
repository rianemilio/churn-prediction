import config
from src.data_ingestion import load_data
from src.exploratory_analysis import perform_eda
from src.data_preprocessing import preprocess_data
from src.model_training import train_and_evaluate_model

def main():
    """Função principal para executar o pipeline de previsão de churn."""
    
    # 1. Carregamento de Dados
    df_raw = load_data(config.DATA_PATH)
    
    # 2. Análise Exploratória de Dados (EDA)
    df_eda = perform_eda(df_raw.copy(), config.IMAGE_PATH)
    
    # 3. Pré-processamento e Engenharia de Features
    X, y, preprocessor = preprocess_data(df_eda)
    
    # 4. Treinamento e Avaliação do Modelo
    model = train_and_evaluate_model(X, y, config.MODEL_PATH)

    print("\nPIPELINE DE MACHINE LEARNING EXECUTADO COM SUCESSO!")

if __name__ == "__main__":
    main()
