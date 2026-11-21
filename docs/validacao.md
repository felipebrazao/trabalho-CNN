# Estado de validação

Implementação inicial preparada em 20/11/2026.

- Código Python e células dos notebooks: verificação de sintaxe com AST.
- Estrutura JSON, IDs únicos e ausência de saídas fictícias: verificados.
- Pacote ZIP: verificação de integridade.
- Revisão estática: A e B carregam o checkpoint congelado de forma independente; teste exige recomendação; manifestos persistem a divisão; Grad-CAM habilita gradientes de entrada para a base congelada.

**Atualização:** dependências instaladas e `pip check` sem incompatibilidades. Os dois notebooks passaram na validação nbformat. A checagem `verificar_codigo.py` executou a EfficientNetB0 com pesos aleatórios, confirmou saída 1×8, realizou um passo de treinamento do classificador sem alterar a convolução congelada e gerou Grad-CAM finito 224×224. Os parâmetros treináveis foram 10.248 (congelada), 3.165.988 (A) e 3.952.066 (B). Esses números são estruturais; não são métricas de desempenho.

**Ainda não executado:** treinamento com ImageNet/PBC, comparação das estratégias e inferência com imagem externa. Os pesos pré-treinados ainda precisam ser baixados. O terminal da sessão falha na resolução de `download.pytorch.org`. Há uma RTX 3060 Ti no PC, mas a instalação atual do PyTorch é CPU. Não há resultados experimentais de desempenho.

Próximo passo: instalar requirements.txt em um ambiente com acesso à internet, baixar o dataset, executar as células em ordem e revisar todas as saídas. Os pesos pré-treinados também exigem acesso à internet na primeira carga. As dependências têm intervalos de versão; environment.json registra as versões efetivas da execução, que devem ser preservadas na entrega.

## Consolidação posterior dos resultados

Pesos ImageNet baixados e conferidos. A base congelada completou 15 épocas. A foi consolidada com as 12 épocas completas e checkpoint da época 12, sem alteração de pesos; macro F1 de validação 0,9737718886. B permanece pendente, configurada para no máximo 12 épocas. O tempo da A é uma estimativa explicitamente identificada. Os estados anteriores deste documento descrevem o histórico da preparação.
