# Auditoria do dataset local

Auditoria executada em 20/11/2026 sobre as imagens extraídas pelo usuário.

| Etapa | Quantidade |
|---|---:|
| Arquivos JPG na pasta principal | 17.093 |
| Imagens decodificadas corretamente | 17.092 |
| Arquivo ilegível `.DS_169665.jpg` | 1 |
| Recursos adicionais de macOS `__MACOSX` (não são imagens utilizáveis) | 20 |
| Imagens excluídas por rótulos conflitantes | 2 |
| Cópias exatas restantes removidas | 16 |
| Imagens únicas usadas no experimento | 17.074 |
| Treino | 11.951 |
| Validação | 2.561 |
| Teste reservado | 2.562 |

As imagens `eosinophil/EO_225902.jpg` e `neutrophil/BNE_191112.jpg` têm os mesmos pixels RGB, mas rótulos diferentes. Ambas foram excluídas do experimento, sem escolher uma classe arbitrária. A exclusão ocorreu antes do holdout e os originais foram preservados.

| Classe | Treino | Validação | Teste |
|---|---:|---:|---:|
| Basófilo | 848 | 182 | 182 |
| Eosinófilo | 2.177 | 466 | 467 |
| Eritroblasto | 1.086 | 233 | 232 |
| Granulócito imaturo | 2.024 | 434 | 434 |
| Linfócito | 850 | 182 | 182 |
| Monócito | 993 | 213 | 213 |
| Neutrófilo | 2.329 | 499 | 500 |
| Plaqueta | 1.644 | 352 | 352 |

Holdout estratificado 70/15/15, seed 42, sem hashes iguais entre divisões. Não foram consultadas previsões ou métricas de teste. A deduplicação considera os pixels RGB decodificados e dimensões, não apenas os bytes do JPG. Não elimina imagens similares nem substitui separação por paciente.

No treino, todas as imagens são RGB. A dimensão predominante é 360×363 (11.627 imagens), mas há 173 imagens 366×369, 147 imagens 360×360 e quatro imagens em outras dimensões. Portanto, o tamanho declarado na página descreve a maioria, não todos os arquivos observados. Resize e conversão RGB permanecem explícitos.

As amostras de treino mostram diferenças na proporção núcleo/citoplasma, granulação e tamanho relativo da célula; também há hemácias e elementos de fundo. Isso motiva inspecionar se o Grad-CAM destaca a célula de interesse ou o contexto. As observações visuais são qualitativas e não validam os rótulos sem especialistas.

Arquivos rastreáveis: `data/processed/inventory.csv`, `conflicting_labels.csv`, `duplicates_removed.csv`, `invalid.csv` e os manifestos em `data/splits`. A primeira auditoria registrou 20 recursos macOS como ilegíveis; execuções seguintes já os ignoram, mantendo a mesma base limpa.
