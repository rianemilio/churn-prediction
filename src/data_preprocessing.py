import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """
    Prepara os dados para o treinamento do modelo de machine learning.
    - Trata valores faltantes.
    - Remove colunas desnecessárias.
    - Aplica One-Hot Encoding em features categóricas.
    - Aplica Scaling em features numéricas.
    """
    print("\nIniciando o Pré-processamento dos Dados...")

    # Remover customerID, pois não é uma feature preditiva
    df = df.drop('customerID', axis=1)

    # Corrigir a coluna 'TotalCharges' que pode ter espaços em branco
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Preencher valores nulos com a mediana da coluna
    median_total_charges = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Converter a variável alvo 'Churn' para formato numérico (0 ou 1)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Separar as features (X) da variável alvo (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Identificar colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    print(f"Features numéricas: {list(numeric_features)}")
    print(f"Features categóricas: {list(categorical_features)}")

    # Criar um pipeline de pré-processamento
    # Para features numéricas, aplicaremos o StandardScaler
    # Para features categóricas, aplicaremos o OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # O pipeline completo aplica o pré-processamento e retorna os dados prontos
    X_processed = preprocessor.fit_transform(X)
    
    print("Pré-processamento concluído.")
    
    # Retornamos o objeto do pré-processador para uso futuro
    # e os dados processados.
    return X_processed, y, preprocessor
