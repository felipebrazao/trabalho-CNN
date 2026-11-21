"""Treina as três condições; não abre o teste nem inventa análises.

Retoma condições concluídas. Uma condição interrompida é reiniciada do seu
estado inicial: ImageNet/classificador novo ou checkpoint congelado.
"""
from pathlib import Path
import json
import time
import traceback
import os

import pandas as pd
import torch
from src import pipeline as p

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / 'results/training_status.json'


def status(state, **extra):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state=state, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(STATUS)


def main():
    torch.set_num_threads(4)
    root, device = p.setup(ROOT)
    inventory_path = root / 'data/processed/inventory.csv'
    if not inventory_path.exists():
        raise FileNotFoundError('Execute verificar_ambiente.py para auditar a base primeiro.')
    # Retoma a auditoria atual; não lê imagens do teste para escolher modelos.
    raw = pd.read_csv(inventory_path)
    audit_time = inventory_path.stat().st_mtime_ns
    for relative in raw.path:
        path = root / relative
        if not path.is_file() or path.stat().st_mtime_ns > audit_time:
            raise RuntimeError('Dados mudaram após a auditoria. Execute verificar_ambiente.py novamente.')
    invalid = pd.read_csv(root / 'data/processed/invalid.csv')
    frame = p.clean_inventory(root, raw, len(invalid))
    parts = p.split_data(root, frame)
    for name in ['frozen', 'ft_a', 'ft_b']:
        completed = root / f'results/metrics/config_{name}.json'
        if completed.exists() and (root / f'models/{name}.pt').exists():
            print('Condição já concluída; preservando:', name, flush=True)
            continue
        max_epochs = 15 if name == 'frozen' else 12
        status('training', condition=name, device=str(device), max_epochs=max_epochs, threads=4)
        print('INICIANDO', name, time.strftime('%H:%M:%S'), flush=True)
        p.train(root, parts, device, name, epochs=max_epochs, batch_size=32)
    status('validation', device=str(device))
    p.curves(root)
    table, predictions = p.compare(root, parts, device, 'val')
    print(table.to_string(index=False), flush=True)
    p.visual_errors(root, predictions)
    p.explain(root, predictions, device)
    status('awaiting_analysis', message='Treinos e validação concluídos. Revisar resultados e registrar recomendação antes do teste.')
    print('TREINOS E VALIDAÇÃO CONCLUÍDOS. Teste permanece reservado.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('error', error=repr(exc))
        traceback.print_exc()
        raise
