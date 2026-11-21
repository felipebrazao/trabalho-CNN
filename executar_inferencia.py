"""Executa apenas Inferencia.ipynb, sem treinamento, e salva suas saídas."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parent
path = root / 'Inferencia.ipynb'
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, km=manager, timeout=300,
                        resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
print('Inferência executada e saídas salvas no notebook.')
