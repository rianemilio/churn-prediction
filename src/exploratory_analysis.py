import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def perform_eda(df, output_path):
    """
    Realiza a análise exploratória dos dados e salva os gráficos gerados.
    """
    print("\nIniciando a Análise Exploratória de Dados (EDA)...")
    
    # Garante que o diretório de saída exista
    os.makedirs(output_path, exist_ok=True)
    
    # -- Análise da Variável Alvo (Churn) --
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df)
    plt.title('Distribuição de Churn (Cancelamento)')
    plt.xlabel('Churn')
    plt.ylabel('Contagem de Clientes')
    save_path = os.path.join(output_path, 'churn_distribution.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de distribuição de churn salvo em: {save_path}")

    # -- Análise de Features Categóricas vs Churn --
    # Exemplo: Tipo de Contrato vs Churn
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Contract', hue='Churn', data=df)
    plt.title('Churn por Tipo de Contrato')
    plt.xlabel('Tipo de Contrato')
    plt.ylabel('Contagem de Clientes')
    save_path = os.path.join(output_path, 'churn_by_contract.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de churn por contrato salvo em: {save_path}")

    # -- Análise de Features Numéricas vs Churn --
    # Corrigindo a coluna 'TotalCharges' que pode ter espaços em branco
    # 'coerce' transforma valores inválidos (como espaços) em NaN (Not a Number)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Preenchemos os valores NaN com a mediana para a plotagem
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Exemplo: Fatura Mensal vs Churn
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', kde=True)
    plt.title('Distribuição da Fatura Mensal por Churn')
    plt.xlabel('Fatura Mensal')
    plt.ylabel('Contagem de Clientes')
    save_path = os.path.join(output_path, 'churn_by_monthly_charges.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de churn por fatura mensal salvo em: {save_path}")
    
    print("\nAnálise Exploratória de Dados (EDA) concluída.")
    return df # Retornamos o df pois fizemos uma pequena modificação nele
