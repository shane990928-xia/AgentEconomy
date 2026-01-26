import pandas as pd
from pathlib import Path

def load_products():
    data_path = Path(__file__).resolve().parents[1] / "data" / "products_with_supply_chain_prices.csv"
    return pd.read_csv(data_path)
