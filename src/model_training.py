from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

def train_and_evaluate_model(X, y, model_path):
    """
    Treina, avalia e salva o modelo de machine learning.
    """
    print("\nIniciando o Treinamento e Avaliação do Modelo...")

    # Dividir os dados em conjuntos de treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

    # Inicializar e treinar o modelo RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\n--- Resultados da Avaliação ---")
    print(f"Acurácia do Modelo: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(report)
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------")

    # Salvar o modelo treinado
    # Garante que o diretório de saída exista
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModelo salvo com sucesso em: {model_path}")

    return model

