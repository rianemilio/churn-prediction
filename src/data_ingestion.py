import pandas as pd

def load_data(path):
    """Carrega os dados de um arquivo CSV."""
    print(f"Carregando dados de: {path}")
    return pd.read_csv(path)
