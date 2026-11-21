"""Execute no ambiente do projeto; saídas parciais são salvas em caso de erro."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / 'CNN.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=None, kernel_name='python3',
                        resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
    print(f'Notebook e saídas disponíveis em {path}')
