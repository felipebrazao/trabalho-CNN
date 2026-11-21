# Próxima execução — B pendente

Nenhum treinamento foi iniciado durante este ajuste.

- Base congelada: 15 épocas completas e checkpoint preservado.
- A: consolidada com as 12 épocas completas, melhor macro F1 de validação 0,9737718886 na época 12. O checkpoint não foi alterado.
- B: máximo de 12 épocas, paciência 4, LR 1e-5 e `features[4:]`; ainda não iniciada.

O tempo da A é uma estimativa de 5.365 segundos, cerca de 89,4 minutos. O valor provém do primeiro lote da época 13 e inclui o custo desse lote; não é uma medição exata das 12 épocas. O histórico da época 13 incompleta não é incluído nas curvas. A execução original foi preservada em `results/metrics/original_ft_a_execution.json`.

## Pelo notebook, em outro dia

1. Se o VS Code perguntar sobre a versão alterada em disco, carregue a versão atualizada, preservando antes qualquer edição própria não salva.
2. Selecione o kernel `.venv`.
3. Execute diretamente a célula da **B**, na seção 6. Ela agora restaura os imports, ROOT, DEVICE e `parts` a partir dos arquivos auditados, sem repetir a base congelada nem A.
4. A B partirá do mesmo `frozen.pt` usado pela A, com máximo de 12 épocas e paciência 4.
5. Se B já estiver concluída, a célula apenas lê a configuração salva para evitar repetição acidental.
6. Mantenha o VS Code aberto enquanto a célula estiver executando; salve o notebook para preservar as saídas.
7. Após a B, execute a comparação e o Grad-CAM; preencha a recomendação antes de abrir o teste.

Não use o comando para executar as células acima da B indiscriminadamente: isso também executaria o treinamento da base congelada. Não execute notebook e script simultaneamente.

## Pelo terminal, quando decidir iniciar B

```powershell
.\.venv\Scripts\python.exe -u executar_treinamentos.py
```

O script preserva as condições congelada e A já concluídas, executa a B e gera a validação. Não há uma trava adicional: executar a célula ou o script inicia a B. Não foi adicionada retomada de otimizador: não é necessária para a opção escolhida de aproveitar a A existente.
