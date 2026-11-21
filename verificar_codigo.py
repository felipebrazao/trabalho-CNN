"""Checagem técnica sintética; não gera resultados acadêmicos nem baixa pesos."""
from pathlib import Path
import ast
import json
import numpy as np
from PIL import Image
import torch
from src import pipeline as p

root = Path(__file__).resolve().parent
torch.set_num_threads(2)
for notebook_path in root.glob('*.ipynb'):
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    print('Notebook válido:', notebook_path.name, flush=True)

model = p.build(pretrained=False).eval()
with torch.no_grad():
    output = model(torch.zeros(1, 3, 224, 224))
assert output.shape == (1, 8) and torch.isfinite(output).all()
counts = {}
for strategy in ['frozen', 'ft_a', 'ft_b']:
    counts[strategy] = p.configure(model, strategy)
assert counts['frozen'] < counts['ft_a'] < counts['ft_b']
p.configure(model, 'frozen')
before = model.features[0][0].weight.detach().clone()
optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-3)
optimizer.zero_grad()
loss = torch.nn.functional.cross_entropy(model(torch.rand(2, 3, 224, 224)), torch.tensor([0, 1]))
loss.backward()
optimizer.step()
assert torch.equal(before, model.features[0][0].weight.detach())
image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8))
cam = p.gradcam(model, image, 0, torch.device('cpu'))
assert cam.shape == (224, 224) and np.isfinite(cam).all()
print('Forward, passo do classificador, base congelada e Grad-CAM: OK', flush=True)
print('Parâmetros treináveis:', counts, flush=True)
print('Checagem sintética: não mede desempenho no PBC.', flush=True)
