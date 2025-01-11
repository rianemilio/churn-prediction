import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import mlflow
import mlflow.sklearn

def plot_feature_importance(model, preprocessor, image_path):
    """
    Gera e salva o gráfico de importância das features e o loga no MLflow.
    """
    # Obter os nomes das features após o OneHotEncoding
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        print(f"Não foi possível obter nomes das features automaticamente: {e}")
        # Plano B, se get_feature_names_out falhar
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis', hue='feature', legend=False)
    plt.title('Top 15 Features Mais Importantes para Previsão de Churn')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Salvar e logar o gráfico
    save_path = os.path.join(image_path, 'feature_importance.png')
    plt.savefig(save_path)
    plt.close()
    
    mlflow.log_artifact(save_path, "images")
    print(f"Gráfico de Feature Importance salvo e logado no MLflow.")


def train_and_evaluate_model(X, y, preprocessor, model_path, image_path):
    """
    Treina, avalia, salva o modelo e loga tudo com MLflow.
    """
    print("\nIniciando o Treinamento e Avaliação do Modelo...")

    # Habilita o log automático do MLflow para Scikit-learn
    mlflow.autolog(log_model_signatures=True, log_input_examples=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)

    # Avaliação (as métricas serão logadas automaticamente pelo autolog)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia do Modelo: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Salvar o modelo (redundante com autolog, mas bom para o nosso fluxo de API)
    joblib.dump(model, model_path)
    print(f"\nModelo salvo com sucesso em: {model_path}")
    
    # Gerar e salvar o gráfico de importância das features
    plot_feature_importance(model, preprocessor, image_path)

    return model
