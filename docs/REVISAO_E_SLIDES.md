# Passagem para revisão e apresentação

Este primeiro commit reúne o experimento e os resultados disponíveis. A revisão final e os slides serão feitos por outro integrante; ainda não é a entrega final revisada.

## Material disponível

- `CNN.ipynb`: arquitetura, preparação, três condições, validação, Grad-CAM, recomendação e teste. A revisão deve conferir ordem das células, coerência das explicações e saídas salvas.
- `Inferencia.ipynb`: demonstração sem novo treinamento, com imagem externa, fonte e análise da previsão incorreta.
- `src/pipeline.py`: implementação dos experimentos.
- `results/`: históricos, configurações, relatórios por classe, previsões, gráficos, recomendação e demonstração externa.
- `data/splits/`: manifestos estratificados persistidos. As imagens brutas não entram no Git.
- `models/selected.pt`: checkpoint recomendado B; inclui os pesos, a arquitetura e o mapeamento das classes para reutilização.
- `data/external/fotografia-original.jpg`: imagem externa em domínio público; atribuição no notebook e em `docs/demonstracao_externa.md`.
- `slides/roteiro.md`: roteiro inicial, ainda sem slides finais.

## Estado dos resultados

| Condição | Épocas concluídas usadas | Macro F1 de validação | Macro F1 de teste |
|---|---:|---:|---:|
| Base congelada | 15 | 89,31% | 89,69% |
| Fine-tuning A | 12 | 97,38% | 98,01% |
| Fine-tuning B | 12 | 98,19% | 98,25% |

A recomendação de B foi registrada antes do teste e está preservada em `results/recommendation.json`. A base foi treinada por 15 épocas; o checkpoint escolhido pela validação, da época 12, inicializou ambos os fine-tunings.

A foi interrompida durante a época 13; a comparação usa as 12 épocas completas e seu melhor checkpoint. Seu tempo de 5.365 s é uma estimativa obtida do primeiro lote da época seguinte. O registro original está em `results/metrics/original_ft_a_execution.json`. Não apresentar o tempo da A como medição exata nem substituir resultados existentes.

Na demonstração externa, a fonte identifica um neutrófilo; o modelo previu plaqueta com score de 86,45%. O erro foi documentado, sem novo ajuste. Essa demonstração não exige acerto e não substitui o teste.

## Revisão pendente

1. Confrontar o material com a lauda e incorporar orientações recebidas na defesa intermediária.
2. Revisar as explicações, a declaração real de uso de IA, as referências e as saídas dos notebooks. A declaração deve distinguir o apoio da IA das verificações efetivamente feitas pela equipe.
3. Conferir métricas, exemplos e mapas já gerados, preservando a recomendação anterior ao teste e suas limitações.
4. Preparar slides com resultados reais, referências e narrativa centrada nas decisões e descobertas. A apresentação tem limite de 12 minutos e todos devem participar.
5. Conferir o carregamento do modelo e os arquivos/links solicitados para a entrega.

## Para abrir em outro computador

Crie um `.venv` e instale `requirements.txt`, conforme o README. O repositório contém o modelo escolhido e a foto para executar apenas `Inferencia.ipynb`, sem baixar o dataset nem treinar novamente. A instalação local usada foi PyTorch CPU; as versões efetivas estão em `results/environment.json`.

Para conferir o material existente, comece pelas saídas salvas. **Executar todas as células de CNN.ipynb pode repetir o treinamento da base congelada.** Baixar o dataset só é necessário para reproduzir os treinos ou avaliações; os checkpoints congelado/A/B e `selected.pt` estão versionados em `models/`.

O notebook principal exige revisão para uma reprodução completa de ponta a ponta em um clone limpo, considerando os artefatos externos e o histórico da execução. Não utilizar o teste para escolher novos ajustes.
