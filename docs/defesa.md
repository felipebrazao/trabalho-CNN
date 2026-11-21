# Preparação para a defesa

Todos devem conseguir explicar as escolhas e interpretar as próprias saídas. Leiam o artigo e o notebook; não basta decorar este roteiro.

1. **Transfer learning:** reaproveita filtros aprendidos no ImageNet, substituindo a saída por oito classes. O domínio é diferente, portanto a transferência precisa ser avaliada.
2. **Fine-tuning:** atualiza pesos da base. A descongela menos blocos; B descongela mais blocos com taxa menor. Alteramos duas decisões em conjunto, por isso não atribuímos diferenças exclusivamente a uma delas.
3. **EfficientNet:** Tan e Le (2019) propõem escalar profundidade, largura e resolução de forma coordenada. B0 é a arquitetura base; não é uma B7 reduzida após o treinamento.
4. **MBConv:** expansão dos canais por convolução 1×1, convolução depthwise espacial, atenção squeeze-and-excitation e projeção 1×1. Atalho residual somente quando dimensões e stride permitem. O bloco com expansão 1 dispensa expansão adicional.
5. **Squeeze-and-excitation:** resume canais espacialmente e aprende pesos para recalibrá-los. Não é Grad-CAM.
6. **Global average pooling:** reduz cada canal a uma média, antes de dropout e camada linear de oito saídas.
7. **Macro F1:** média dos F1 por classe com pesos iguais; ajuda a evitar que classes maiores dominem a comparação. Balanced accuracy é média dos recalls por classe.
8. **Teste:** reservado no início e consultado após recomendação. Escolher a rede pelo teste causaria vazamento na seleção.
9. **Erros:** examine especialmente a distinção granulócitos imaturos/neutrófilos e as confusões reais observadas. A importância biológica deve ser contextualizada com fontes e sem alegar diagnóstico.
10. **BatchNorm:** `requires_grad=False` não impede atualização das médias e variâncias. O código também controla modo de avaliação para manter as estatísticas fixas.
11. **Grad-CAM:** gradientes do logit da classe investigada ponderam os mapas da última convolução; ReLU mantém contribuições positivas. Comparamos a classe real nas mesmas imagens e registramos a camada.
12. **Overfitting:** treino melhora enquanto validação piora; interpretar perda e macro F1, não só acurácia. Underfitting exige considerar o domínio e as curvas, não um limiar arbitrário.
13. **Score:** softmax expressa confiança relativa entre as oito classes; pode ser alto mesmo em imagem fora do domínio.
14. **Custo e limites:** comparar tempo, parâmetros treináveis, recalls, macro F1, uma única divisão, possíveis pacientes compartilhados e diferenças de microscópio.

## Acompanhamento intermediário

- Pesquisa e diagrama da arquitetura.
- Dataset analisado, divisões persistidas e teste reservado.
- Base congelada treinada, curvas e validação.
- Exemplos preliminares e estratégias A/B planejadas.
- Incorporar posteriormente as orientações da professora.
