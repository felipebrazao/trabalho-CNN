# Classificação de células sanguíneas com EfficientNetB0

**Equipe:** Felipe Brazão, Felipe Libório, Gabriel Góes e João Ricardo Almeida.

**Disciplina:** Inteligência Computacional / Machine Learning.

**Professora:** Polyana Santos Fonseca Nascimento.

**Estado:** experimento executado; revisão final e slides pendentes. Veja [a passagem para revisão e apresentação](docs/REVISAO_E_SLIDES.md).

Projeto acadêmico com oito classes do PBC. Compara base congelada e duas estratégias de fine-tuning, com explicabilidade Grad-CAM. A base congelada concluiu 15 épocas; A e B foram avaliadas com 12 épocas completas. Comparação, Grad-CAM, recomendação anterior ao teste e avaliação final foram realizados. A discussão está no notebook e em docs/discussao_final.md. Ainda faltam a demonstração com imagem externa, os slides e a revisão final da entrega.

## Preparar no Windows

Abra um terminal nesta pasta. Recomenda-se Python 3.12 e espaço livre para o dataset, ambiente e checkpoints.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name cnn-pbc --display-name "Python (CNN PBC)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Abra `CNN.ipynb` e selecione o kernel Python (CNN PBC). O código utiliza CUDA quando disponível. Para instalar uma edição do PyTorch compatível com sua GPU/driver, consulte o seletor oficial: https://pytorch.org/get-started/locally/ . A instalação genérica pode usar apenas CPU, dependendo da plataforma e dos pacotes disponíveis. Não altere configurações entre os três experimentos sem registrar a mudança.

## Dataset

1. Acesse https://data.mendeley.com/datasets/snkd93bnjr/1 e baixe os arquivos pela opção Download All.
2. Extraia em `data/raw`. É permitido haver uma pasta intermediária como `PBC_dataset_normal_DIB`.
3. As imagens devem permanecer dentro das pastas `basophil`, `eosinophil`, `erythroblast`, `ig`, `lymphocyte`, `monocyte`, `neutrophil` e `platelet`.
4. Execute o notebook na ordem. O inventário registra arquivos inválidos, canais e dimensões; cópias de pixels idênticos são removidas antes da divisão. A distribuição é conferida a partir dos arquivos reais.

O download e a extração são manuais para evitar dependência de endpoints internos do repositório. Não coloque o dataset no GitHub. Cite Acevedo et al. e preserve a atribuição CC BY 4.0.

O dataset extraído nesta sessão foi movido para `data/raw/PBC_dataset_normal_DIB`. Arquivos de metadados do macOS são ignorados. Para obter antecipadamente os pesos oficiais (cerca de 20 MB), execute no PowerShell da pasta do projeto:

```powershell
.\baixar_pesos.ps1
```

O script é código local deste projeto, baixa apenas o checkpoint oficial e confere o prefixo SHA-256 informado pelo Torchvision. Se a política de execução da máquina bloquear scripts, abra `baixar_pesos.ps1` e execute seu conteúdo no terminal. O cache fica em `models/pretrained`; não substitua transfer learning por pesos aleatórios quando faltar acesso à internet.

## Execução

Quando decidir executar a B e gerar os resultados de validação sem abrir o teste:

```powershell
.\.venv\Scripts\python.exe -u executar_treinamentos.py
```

O executor preserva condições já concluídas, reinicia uma condição interrompida a partir do estado inicial correto e salva seu estado em `results/training_status.json`. Não inicie duas execuções simultâneas. A execução iniciada nesta sessão registra a saída em `results/training.log`. Ao terminar os treinos, o estado será `awaiting_analysis`: é necessário interpretar a validação, registrar a recomendação e só depois avaliar o teste.

`CNN.ipynb` importa `src/pipeline.py`; mantenha a estrutura de pastas. O notebook contém treinamento, análise, recomendação e teste. Leia e complete as células Markdown à medida que os resultados aparecerem. A célula de recomendação pede uma justificativa real antes de liberar o teste. Essa pausa é intencional: o teste só pode ser consultado depois da escolha pela validação.

Para executar pelo terminal e salvar as saídas no notebook:

```powershell
.\.venv\Scripts\python.exe executar_notebook.py
```

O executor interrompe com erro quando encontrar uma etapa que dependa de preenchimento. Nesse caso, abra o notebook e complete a análise correspondente. Uma nova execução completa repete os treinamentos; para apenas inferência, use `Inferencia.ipynb`.

No Colab, envie o projeto completo ou clone seu futuro repositório, instale `requirements.txt`, selecione GPU e ajuste `ROOT` na primeira célula. Mantenha dados e resultados em armazenamento persistente antes de encerrar a sessão. O notebook não monta contas automaticamente.

## Protocolo

| Condição | Base ajustada | Learning rate | Máximo de épocas |
|---|---|---:|---:|
| `frozen` | Nenhuma; somente classificador | 0,001 | 15 |
| `ft_a` | `features[6:]` e classificador | 0,0001 | 12 completas preservadas |
| `ft_b` | `features[4:]` e classificador | 0,00001 | 12 completas |

AdamW, batch 32, weight decay 0,0001, paciência 4; checkpoint por macro F1 de validação. A e B recarregam separadamente o mesmo checkpoint congelado. As estatísticas BatchNorm ficam fixas nas três condições; parâmetros afins só são treináveis nos blocos descongelados. Cada condição usa otimizador novo. O seed é reiniciado para controlar o embaralhamento e as transformações.

Holdout estratificado 70/15/15 com semente 42. RGB, resize 224×224 bicúbico, escala [0,1] e normalização ImageNet. Resize direto preserva toda a célula; difere do resize+crop padrão do preset oficial e é uma escolha explícita. Flip e rotação de até 20° somente no treino. Sem pesos de classe na configuração inicial; macro F1 e balanced accuracy acompanham o desbalanceamento.

Seleção: maior macro F1 na validação; empate exato resolvido por menor tempo, depois menos parâmetros. A equipe deve discutir se diferenças pequenas justificam o custo. São medições em uma única divisão e uma semente; não demonstram significância estatística. A análise final de teste não altera a decisão.

## Organização e entrega

| Arquivo/pasta | Finalidade |
|---|---|
| `CNN.ipynb` | Notebook principal com explicações e etapas executáveis |
| `Inferencia.ipynb` | Reutilização sem treinamento e imagem externa |
| `src/pipeline.py` | Preparação, modelos, treinamento, avaliação e Grad-CAM |
| `executar_notebook.py` | Executor que salva saídas mesmo se houver erro |
| `data/splits` | Manifestos de treino, validação e teste |
| `models` | Checkpoints e `selected.pt` após recomendação |
| `results` | Curvas, tabelas, previsões, configurações e mapas |
| `slides/roteiro.md` | Roteiro de apresentação de até 12 minutos |
| `docs/defesa.md` | Perguntas e explicações para preparação oral |

Antes da entrega: executar tudo, completar análises, registrar a contribuição real da IA, preencher fonte e classe da imagem externa, exportar os slides e anexar o modelo. Os slides finais precisam incorporar os resultados reais. O Git inclui apenas o modelo selecionado e a fotografia externa; dataset bruto, ambiente e checkpoints auxiliares são ignorados. O checkpoint recomendado está em `models/selected.pt`.

## Limitações

Não há separação por paciente implementada, pois o projeto não assume identificadores de pacientes disponíveis. A equipe deve verificar a documentação original: imagens do mesmo paciente em divisões distintas podem inflar o desempenho. Deduplicação exata não remove imagens semelhantes. Dados de um único contexto de aquisição não garantem generalização. Score softmax não é probabilidade clínica calibrada. O projeto classifica tipos celulares e não diagnostica doenças. Grad-CAM fornece localização aproximada e hipóteses, não prova causalidade.

## Referências

- Acevedo et al. Dataset PBC, versão 1 (2020). https://doi.org/10.17632/snkd93bnjr.1
- Artigo associado ao dataset: https://doi.org/10.1016/j.cmpb.2019.105020
- Tan, M.; Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019. https://arxiv.org/abs/1905.11946
- Torchvision EfficientNetB0: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
- Selvaraju et al. Grad-CAM. ICCV, 2017. https://arxiv.org/abs/1610.02391

## Estado consolidado

A conserva 12 épocas completas e o checkpoint salvo. Seu tempo de 5.365 s é uma estimativa baseada no primeiro lote da época 13; a tabela identifica essa limitação. A execução original foi preservada. A B concluiu as 12 épocas e seus resultados estão registrados. A célula B inicia o treinamento quando executada; veja `docs/proxima_execucao.md` para executar em outro dia. O histórico das etapas está preservado nos resultados.

## Continuar pela B após reiniciar o PC

Abra CNN.ipynb, selecione o kernel .venv e execute diretamente a célula da B, na seção 6. Ela é autossuficiente: restaura imports, diretório, dispositivo e os conjuntos salvos; preserva a base congelada e A; inicia B com máximo de 12 épocas. Não é necessário executar as células de treinamento anteriores. O treinamento deve permanecer no VS Code ou no terminal aberto pelo usuário, pois a sessão de comandos do assistente não permaneceu ativa entre turnos.

## Resultados finais já avaliados

A recomendação anterior ao teste foi mantida: fine-tuning B. Macro F1 no teste: base 89,69%, A 98,01%, B 98,25%, em 2.562 imagens. Consulte a discussão final para os custos, erros por classe e limitações. Não use o teste para novos ajustes. As instruções de fases anteriores são histórico de execução; os três checkpoints já existem, portanto não é necessário repetir os treinos.

## Demonstração externa concluída

Inferencia.ipynb foi executado e contém as saídas salvas. O arquivo real é data/external/fotografia-original.jpg. Classe esperada pela fonte: neutrophil; previsão: platelet; score: 86,45%. O erro foi preservado e discutido, sem alterar o modelo. Fonte e atribuição estão no notebook e em docs/demonstracao_externa.md. O ZIP agora inclui models/selected.pt e a imagem externa para permitir a reutilização; os demais pesos, o dataset bruto e o ambiente continuam excluídos. Restam os slides e a revisão final da entrega.
