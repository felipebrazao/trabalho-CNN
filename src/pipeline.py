"""Experimento PBC/EfficientNetB0. Execute pelo notebook CNN.ipynb."""
from pathlib import Path
import copy
import hashlib
import json
import random
import time
import platform
import importlib.metadata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

SEED = 42
CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte',
           'monocyte', 'neutrophil', 'platelet']
LABELS_PT = ['Basófilo', 'Eosinófilo', 'Eritroblasto', 'Granulócito imaturo',
             'Linfócito', 'Monócito', 'Neutrófilo', 'Plaqueta']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup(root):
    root = Path(root).resolve()
    for name in ['data/raw', 'data/processed', 'data/splits', 'models',
                 'results/figures', 'results/metrics', 'results/gradcam', 'data/external']:
        (root / name).mkdir(parents=True, exist_ok=True)
    # Cache dentro do projeto, acessível também em ambiente restrito.
    torch.hub.set_dir(str(root / 'models/pretrained'))
    seed_all()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    versions = {p: importlib.metadata.version(p) for p in
                ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow', 'matplotlib']}
    versions.update(python=platform.python_version(), device=str(device), seed=SEED)
    if device.type == 'cuda':
        versions['gpu'] = torch.cuda.get_device_name(0)
    (root / 'results/environment.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
    print(versions)
    return root, device


def inventory(root):
    """Valida imagens; preserva inventário bruto e limpa antes do holdout."""
    rows, invalid = [], []
    examined = 0
    for path in sorted((root / 'data/raw').rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        if '__MACOSX' in path.parts or path.name.startswith('._'):
            continue
        label = path.parent.name.lower()
        if label not in CLASSES:
            continue
        examined += 1
        if examined % 1000 == 0:
            print(f'Inventário: {examined} arquivos examinados...', flush=True)
        try:
            with Image.open(path) as img:
                img.load()
                rgb = img.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                rows.append(dict(path=path.relative_to(root).as_posix(), label=label,
                                 target=CLASSES.index(label), width=img.width, height=img.height,
                                 mode=img.mode, sha256=digest))
        except (OSError, ValueError) as exc:
            invalid.append(dict(path=str(path), error=str(exc)))
    if not rows:
        raise FileNotFoundError('Extraia o dataset em data/raw, preservando as oito pastas de classes. Consulte README.md.')
    df = pd.DataFrame(rows)
    df.to_csv(root / 'data/processed/inventory.csv', index=False)
    pd.DataFrame(invalid, columns=['path', 'error']).to_csv(root / 'data/processed/invalid.csv', index=False)
    return clean_inventory(root, df, len(invalid))


def clean_inventory(root, df, invalid_count=0):
    """Exclui todo grupo com rótulo ambíguo; nunca escolhe rótulo arbitrário."""
    conflicts = df.groupby('sha256').label.nunique()
    conflict_hashes = conflicts.index[conflicts > 1]
    ambiguous = df[df.sha256.isin(conflict_hashes)]
    ambiguous.to_csv(root / 'data/processed/conflicting_labels.csv', index=False)
    df = df[~df.sha256.isin(conflict_hashes)].copy()
    duplicate_count = int(df.duplicated('sha256').sum())
    df[df.duplicated('sha256', keep='first')].to_csv(root / 'data/processed/duplicates_removed.csv', index=False)
    df = df.drop_duplicates('sha256').reset_index(drop=True)
    counts = df.label.value_counts().reindex(CLASSES, fill_value=0)
    if counts.min() < 100:
        raise ValueError(f'Base incompleta: são necessárias >=100 imagens por classe.\n{counts}')
    print(f'Imagens válidas únicas: {len(df)}; duplicatas removidas: {duplicate_count}; '
          f'arquivos com rótulos conflitantes excluídos: {len(ambiguous)}; inválidas: {invalid_count}', flush=True)
    return df


def split_data(root, df):
    """Reutiliza a separação existente apenas se a identidade da base for igual."""
    identity = hashlib.sha256(df[['path', 'sha256', 'label']].to_csv(index=False).encode()).hexdigest()
    meta = root / 'data/splits/metadata.json'
    if meta.exists():
        old = json.loads(meta.read_text(encoding='utf-8'))
        parts = {k: pd.read_csv(root / f'data/splits/{k}.csv') for k in ['train', 'val', 'test']}
        if old['fingerprint'] != identity:
            # A ordem de listagem muda entre sistemas (Windows x Linux); compara o conteúdo.
            key = lambda frame: set(zip(frame.path, frame.sha256, frame.label))
            if key(df) != key(pd.concat(parts.values())):
                raise ValueError('Dataset mudou. Preserve a separação anterior e revise a mudança antes de criar outro experimento.')
    else:
        train, rest = train_test_split(df, test_size=.30, stratify=df.target, random_state=SEED)
        val, test = train_test_split(rest, test_size=.50, stratify=rest.target, random_state=SEED)
        parts = dict(train=train, val=val, test=test)
        for k, part in parts.items():
            part.to_csv(root / f'data/splits/{k}.csv', index=False)
        meta.write_text(json.dumps(dict(fingerprint=identity, seed=SEED, fractions=[.70, .15, .15]), indent=2), encoding='utf-8')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not set(parts[a].sha256) & set(parts[b].sha256)
    print(pd.DataFrame({k: p.label.value_counts() for k, p in parts.items()}).reindex(CLASSES))
    return parts


def analysis(root, parts):
    train = parts['train']
    display_table = pd.DataFrame({k: p.label.value_counts() for k, p in parts.items()}).reindex(CLASSES)
    display_table.plot.bar(figsize=(11, 4), rot=30, title='Distribuição por classe e divisão')
    plt.tight_layout()
    plt.savefig(root / 'results/figures/distribution.png', dpi=160)
    plt.show()
    print('Dimensões do treino:\n', train.groupby(['width', 'height']).size())
    print('Canais originais no treino:\n', train['mode'].value_counts())
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, cls, title in zip(axes.flat, CLASSES, LABELS_PT):
        row = train[train.label == cls].iloc[0]
        with Image.open(root / row.path) as image:
            ax.imshow(image.convert('RGB'))
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(root / 'results/figures/train_samples.png', dpi=160)
    plt.show()


def preprocessing(training=False):
    # Resize direto preserva toda a célula; a distorção de 360x363 é pequena.
    ops = [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Cells(Dataset):
    def __init__(self, root, frame, training=False):
        self.root, self.frame, self.transform = root, frame.reset_index(drop=True), preprocessing(training)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(self.root / row.path) as image:
            x = self.transform(ImageOps.exif_transpose(image).convert('RGB'))
        return x, int(row.target)


def loader(root, frame, training=False, batch_size=32):
    # num_workers=0 é compatível com notebooks no Windows.
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(Cells(root, frame, training), batch_size=batch_size,
                      shuffle=training, num_workers=0, generator=generator,
                      pin_memory=torch.cuda.is_available())


def build(pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    return model


def configure(model, strategy):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    # features[6:]: estágios finais + convolução 1x1; features[4:]: parte maior.
    start = {'frozen': 9, 'ft_a': 6, 'ft_b': 4}[strategy]
    for block in list(model.features.children())[start:]:
        for p in block.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scores(y, pred):
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, labels=range(8), average='macro', zero_division=0),
                weighted_f1=f1_score(y, pred, labels=range(8), average='weighted', zero_division=0))


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    loss_sum, ys, probs = 0., [], []
    for x, y in batches:
        logits = model(x.to(device))
        loss_sum += nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
        probs.append(logits.softmax(1).cpu().numpy())
        ys.extend(y.tolist())
    probs = np.concatenate(probs)
    pred = probs.argmax(1)
    return dict(loss=loss_sum / len(ys), **scores(ys, pred)), np.array(ys), probs


def save_checkpoint(root, model, name, **extra):
    torch.save(dict(state_dict=model.state_dict(), classes=CLASSES, mean=MEAN, std=STD,
                    size=[224, 224], architecture='efficientnet_b0', **extra), root / f'models/{name}.pt')


def load_checkpoint(root, name, device):
    checkpoint = torch.load(root / f'models/{name}.pt', map_location='cpu', weights_only=True)
    if checkpoint['classes'] != CLASSES:
        raise ValueError('Mapeamento de classes incompatível.')
    model = build(pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval()


def prepare_b_context(root):
    """Restaura o contexto após reiniciar o PC sem repetir treinamentos."""
    root, device = setup(root)
    for name in ['frozen', 'ft_a']:
        config_path = root / f'results/metrics/config_{name}.json'
        checkpoint_path = root / f'models/{name}.pt'
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f'Condição {name} incompleta; restaure seus arquivos antes de iniciar B.')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        expected = config.get('checkpoint_sha256')
        if expected and hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checkpoint {name} mudou desde a consolidação.')
        print(f'{name}: {config["epochs"]} épocas preservadas; sem novo treinamento.', flush=True)
    inventory_path = root / 'data/processed/inventory.csv'
    raw = pd.read_csv(inventory_path)
    audit_time = inventory_path.stat().st_mtime_ns
    for relative in raw.path:
        path = root / relative
        if not path.is_file() or path.stat().st_mtime_ns > audit_time:
            raise RuntimeError('As imagens mudaram após a auditoria; conferir antes de iniciar B.')
    invalid = pd.read_csv(root / 'data/processed/invalid.csv')
    clean = clean_inventory(root, raw, len(invalid))
    parts = split_data(root, clean)
    print('Contexto restaurado. B partirá de frozen.pt; teste permanece reservado.', flush=True)
    return root, device, parts


def train(root, parts, device, strategy, epochs=15, batch_size=32):
    accepted_path = root / f'results/metrics/config_{strategy}.json'
    # Condição já treinada: reutiliza checkpoint e configuração, sem retreinar (A e B dependem de frozen.pt).
    if accepted_path.exists() and (root / f'models/{strategy}.pt').exists():
        accepted = json.loads(accepted_path.read_text(encoding='utf-8'))
        print(f'{strategy}: reutilizando {accepted["epochs"]} épocas salvas; nenhum treinamento iniciado.', flush=True)
        return accepted
    seed_all()
    model = build() if strategy == 'frozen' else load_checkpoint(root, 'frozen', device)
    n_trainable = configure(model, strategy)
    model.to(device)
    lr = dict(frozen=1e-3, ft_a=1e-4, ft_b=1e-5)[strategy]
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    train_batches = loader(root, parts['train'], True, batch_size)
    val_batches = loader(root, parts['val'], False, batch_size)
    best_score, stale, history = -1., 0, []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        # Base congelada não atualiza BatchNorm nem usa stochastic depth.
        if strategy == 'frozen':
            model.features.eval()
        # Estatísticas BN fixas em todas as condições; pesos afins podem ser ajustados.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        total, true, pred = 0., [], []
        for batch_index, (x, y) in enumerate(train_batches, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            true.extend(y.detach().cpu().tolist())
            pred.extend(logits.detach().argmax(1).cpu().tolist())
            if batch_index == 1 or batch_index % 50 == 0:
                elapsed = time.perf_counter() - started
                print(f'{strategy} | época {epoch}/{epochs} | lote {batch_index}/{len(train_batches)} | '
                      f'loss do lote {loss.item():.4f} | decorrido {elapsed:.0f}s', flush=True)
        val, _, _ = evaluate(model, val_batches, device)
        row = dict(epoch=epoch, train_loss=total / len(true), train_macro_f1=scores(true, pred)['macro_f1'],
                   val_loss=val['loss'], val_macro_f1=val['macro_f1'])
        history.append(row)
        pd.DataFrame(history).to_csv(root / f'results/metrics/history_{strategy}.csv', index=False)
        print(strategy, {k: round(v, 4) for k, v in row.items()}, flush=True)
        if val['macro_f1'] > best_score + 1e-6:
            best_score, stale = val['macro_f1'], 0
            save_checkpoint(root, model, strategy, best_epoch=epoch)
        else:
            stale += 1
        if stale >= 4:
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    hist = pd.DataFrame(history)
    hist.to_csv(root / f'results/metrics/history_{strategy}.csv', index=False)
    metadata = dict(strategy=strategy, trainable=n_trainable, total=sum(p.numel() for p in model.parameters()),
                    epochs=len(hist), seconds=seconds, learning_rate=lr, batch_size=batch_size,
                    optimizer='AdamW', weight_decay=1e-4, patience=4,
                    selection='validation macro F1', max_epochs=epochs)
    (root / f'results/metrics/config_{strategy}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    del model
    return metadata


def compare(root, parts, device, split='val'):
    rows, predictions = [], {}
    for name in ['frozen', 'ft_a', 'ft_b']:
        model = load_checkpoint(root, name, device)
        metrics, y, prob = evaluate(model, loader(root, parts[split]), device)
        pred = prob.argmax(1)
        config = json.loads((root / f'results/metrics/config_{name}.json').read_text())
        rows.append(dict(model=name, **metrics, **{k: config[k] for k in ['trainable', 'epochs', 'seconds']},
                         time_measurement=config.get('time_measurement', 'measured'),
                         time_note=config.get('time_note', 'Tempo medido durante a execução.')))
        report = classification_report(y, pred, labels=list(range(8)), target_names=CLASSES,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(root / f'results/metrics/report_{split}_{name}.csv')
        out = parts[split][['path', 'label', 'target']].reset_index(drop=True).copy()
        out['prediction'] = pred
        out['score'] = prob.max(1)
        for i, cls in enumerate(CLASSES):
            out[f'p_{cls}'] = prob[:, i]
        out.to_csv(root / f'results/metrics/predictions_{split}_{name}.csv', index=False)
        predictions[name] = out
        fig, ax = plt.subplots(figsize=(9, 8))
        ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=list(range(8))), display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f'{name} — {split}: contagens')
        fig.tight_layout()
        fig.savefig(root / f'results/figures/cm_{split}_{name}.png', dpi=140)
        plt.show()
        del model
    table = pd.DataFrame(rows)
    table.to_csv(root / f'results/metrics/comparison_{split}.csv', index=False)
    return table, predictions


def curves(root):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for col, name in enumerate(['frozen', 'ft_a', 'ft_b']):
        h = pd.read_csv(root / f'results/metrics/history_{name}.csv')
        for row, metric in enumerate(['loss', 'macro_f1']):
            for part in ['train', 'val']:
                axes[row, col].plot(h.epoch, h[f'{part}_{metric}'], label=part)
            axes[row, col].set(title=f'{name}: {metric}', xlabel='Época')
            axes[row, col].legend()
    fig.tight_layout()
    fig.savefig(root / 'results/figures/learning_curves.png', dpi=160)
    plt.show()


def visual_errors(root, predictions):
    base = predictions['frozen']
    chosen = []
    # Erros das três condições + um exemplo de cada classe + casos incertos.
    for frame in predictions.values():
        chosen.extend(frame.index[frame.prediction != frame.target][:3].tolist())
    for cls in range(8):
        chosen.append(int(base.index[base.target == cls][0]))
    chosen.extend(base.nsmallest(2, 'score').index.tolist())
    chosen = list(dict.fromkeys(chosen))
    fig, axes = plt.subplots(len(chosen), len(predictions),
                             figsize=(4 * len(predictions), 3 * len(chosen)), squeeze=False)
    records = []
    for r, idx in enumerate(chosen):
        record = dict(path=base.loc[idx, 'path'], real=base.loc[idx, 'label'])
        for c, (name, frame) in enumerate(predictions.items()):
            item = frame.loc[idx]
            with Image.open(root / item.path) as img:
                axes[r, c].imshow(img.convert('RGB'))
            axes[r, c].set_title(f'{name}: {CLASSES[int(item.prediction)]} ({item.score:.3f})\nReal: {item.label}')
            axes[r, c].axis('off')
            record[name] = CLASSES[int(item.prediction)]
        records.append(record)
    fig.tight_layout()
    fig.savefig(root / 'results/figures/same_images_comparison.png', dpi=100)
    plt.show()
    out = pd.DataFrame(records)
    out.to_csv(root / 'results/metrics/visual_cases.csv', index=False)
    return out


def gradcam(model, image, class_idx, device):
    model.eval()
    captured = {}
    def capture(module, inputs, output):
        captured['activation'] = output
    handle = model.features[-1].register_forward_hook(capture)
    try:
        x = preprocessing()(image).unsqueeze(0).to(device).requires_grad_(True)
        logits = model(x)
        activations = captured['activation']
        gradients = torch.autograd.grad(logits[0, class_idx], activations)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True).relu()
        cam = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        return cam / max(float(cam.max()), 1e-8)
    finally:
        handle.remove()


def gradcam_cases(predictions):
    """Índices por categoria: erros corrigidos, erros novos, erro persistente e acerto comum."""
    ok = {name: frame.prediction == frame.target for name, frame in predictions.items()}
    base, ft = ok['frozen'], ok['ft_a'] & ok['ft_b']
    groups = [(~base & ft, 2), (base & ~(ok['ft_a'] & ok['ft_b']), 2),
              (~base & ~ok['ft_a'] & ~ok['ft_b'], 1), (base & ft, 1)]
    return list(dict.fromkeys(i for mask, n in groups for i in mask[mask].index[:n].tolist()))


def explain(root, predictions, device):
    base = predictions['frozen']
    indices = gradcam_cases(predictions)
    fig, axes = plt.subplots(len(indices), 4, figsize=(14, len(indices) * 3), squeeze=False)
    records = []
    for row, idx in enumerate(indices):
        item = base.loc[idx]
        with Image.open(root / item.path) as im:
            image = ImageOps.exif_transpose(im).convert('RGB')
        resized = image.resize((224, 224))
        axes[row, 0].imshow(resized)
        axes[row, 0].set_title(f'Real: {item.label}')
        for col, name in enumerate(['frozen', 'ft_a', 'ft_b'], 1):
            model = load_checkpoint(root, name, device)
            # Classe real fixa para comparabilidade; prevista está no título.
            cam = gradcam(model, image, int(item.target), device)
            axes[row, col].imshow(resized)
            axes[row, col].imshow(cam, cmap='jet', alpha=.45, vmin=0, vmax=1)
            predicted = predictions[name].loc[idx]
            axes[row, col].set_title(f'{name}: {CLASSES[int(predicted.prediction)]}\nGrad-CAM da classe real')
            records.append(dict(path=item.path, model=name, investigated_class=item.label,
                                layer='features[-1]', map_max=float(cam.max())))
            del model
    for ax in axes.flat:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(root / 'results/gradcam/comparison.png', dpi=140)
    plt.show()
    pd.DataFrame(records).to_csv(root / 'results/gradcam/cases.csv', index=False)


def recommend(root, table, rationale):
    """Decisão persistida ANTES do teste; empates resolvidos por custo."""
    if len(rationale.strip()) < 80 or 'PREENCHER' in rationale:
        raise ValueError('Escreva a justificativa com base na validação, curvas, classes, erros e Grad-CAM.')
    rank = table.sort_values(['macro_f1', 'seconds', 'trainable'], ascending=[False, True, True])
    winner = rank.iloc[0]['model']
    decision = dict(model=winner, rule='Maior macro F1 na validação; empate: menor tempo, depois menos parâmetros',
                    rationale=rationale, validation=table.to_dict(orient='records'))
    path = root / 'results/recommendation.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous['model'] != winner:
            raise RuntimeError('Já existe uma recomendação. Não altere a escolha depois de consultar o teste.')
        # Reexecução: preserva o registro original feito antes do teste.
        print('Recomendação já registrada antes do teste:', winner)
        return winner
    path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding='utf-8')
    checkpoint = torch.load(root / f'models/{winner}.pt', map_location='cpu', weights_only=True)
    torch.save(checkpoint, root / 'models/selected.pt')
    print('Modelo recomendado:', winner)
    return winner


def final_test(root, parts, device):
    if not (root / 'results/recommendation.json').exists():
        raise RuntimeError('Registre a recomendação pela validação antes de abrir o teste.')
    print('Teste final: não use os resultados para ajustar ou trocar o modelo.')
    return compare(root, parts, device, 'test')


@torch.no_grad()
def predict(root, image_path, device):
    model = load_checkpoint(root, 'selected', device)
    with Image.open(image_path) as im:
        image = ImageOps.exif_transpose(im).convert('RGB')
    p = model(preprocessing()(image).unsqueeze(0).to(device)).softmax(1)[0].cpu().numpy()
    idx = int(p.argmax())
    plt.imshow(image)
    plt.title(f'{CLASSES[idx]} — score softmax: {p[idx]:.3f}')
    plt.axis('off')
    plt.show()
    return pd.Series(p, index=CLASSES).sort_values(ascending=False)
