"""Confere dados e pesos antes de iniciar os treinamentos."""
from pathlib import Path
import json
import socket
import sys
import torch
from src import pipeline as p

root = Path(__file__).resolve().parent
torch.set_num_threads(min(8, torch.get_num_threads()))
root, device = p.setup(root)
print('Analisando as imagens; aguarde...', flush=True)
frame = p.inventory(root)
parts = p.split_data(root, frame)
print('DADOS VALIDADOS:', len(frame), 'imagens únicas', flush=True)
print('Verificando pesos ImageNet...', flush=True)
socket.setdefaulttimeout(20)
try:
    model = p.build()
except Exception as exc:
    print('PESOS PENDENTES:', type(exc).__name__, str(exc), flush=True)
    sys.exit(2)
model.eval()
with torch.no_grad():
    output = model(torch.zeros(1, 3, 224, 224))
assert output.shape == (1, 8)
print('MODELO PRONTO: saída', tuple(output.shape), flush=True)
