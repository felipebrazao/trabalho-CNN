"""Empacota código, resultados, modelo selecionado e imagem externa; exclui a base bruta."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import os

root = Path(__file__).resolve().parent
archive = root.parent / 'Trabalho-CNN.zip'
excluded = {'.venv', '__pycache__', '.git', '.ipynb_checkpoints', 'PBC_dataset_normal_DIB'}
with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
    for directory, folders, names in os.walk(root):
        folders[:] = [name for name in folders if name not in excluded]
        directory = Path(directory)
        relative_directory = directory.relative_to(root)
        if relative_directory.parts[:1] == ('models',) or relative_directory.parts[:2] in [('data', 'raw'), ('data', 'external')]:
            folders[:] = []
        for name in names:
            path = directory / name
            rel = path.relative_to(root)
            if path.suffix == '.zip' or name in {'training.log', 'training_errors.log', 'training_status.json'}:
                continue
            if rel.parts[0] == 'models' and name not in {'.gitkeep', 'selected.pt'}:
                continue
            if rel.parts[:2] == ('data', 'raw') and name != '.gitkeep':
                continue
            if rel.parts[:2] == ('data', 'external') and name not in {'.gitkeep', 'fotografia-original.jpg'}:
                continue
            bundle.write(path, Path(root.name) / rel)
with ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    print(archive, '-', len(bundle.namelist()), 'arquivos,', archive.stat().st_size, 'bytes')
