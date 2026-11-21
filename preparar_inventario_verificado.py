"""Retoma o inventário já decodificado nesta sessão sem reler todas as imagens.

Use verificar_ambiente.py para uma nova auditoria completa se o dataset mudou.
"""
from pathlib import Path
import socket
import pandas as pd
from src import pipeline as p

root, device = p.setup(Path(__file__).resolve().parent)
raw_inventory = pd.read_csv(root / 'data/processed/inventory.csv')
invalid = pd.read_csv(root / 'data/processed/invalid.csv')
frame = p.clean_inventory(root, raw_inventory, len(invalid))
parts = p.split_data(root, frame)
p.analysis(root, parts)
print('DIVISÕES PREPARADAS', flush=True)
socket.setdefaulttimeout(15)
try:
    model = p.build()
except Exception as exc:
    print('Pesos ImageNet pendentes:', type(exc).__name__, str(exc), flush=True)
    raise SystemExit(2)
print('Pesos ImageNet disponíveis.', flush=True)
