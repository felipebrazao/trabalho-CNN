### Discussão final após a avaliação no teste

#### Resultados e compatibilidade com a validação

A avaliação utilizou as mesmas **2.562 imagens de teste**, reservadas desde o início. O fine-tuning B já havia sido recomendado pelo maior macro F1 de validação e essa decisão foi mantida. O teste não orientou novos ajustes, treinamentos ou troca do modelo.

| Condição | Macro F1 na validação | Macro F1 no teste | Acurácia no teste | Balanced accuracy no teste | Acertos / erros no teste |
|---|---:|---:|---:|---:|---:|
| Base congelada | 89,31% | 89,69% | 90,28% | 89,47% | 2.313 / 249 |
| Fine-tuning A | 97,38% | 98,01% | 98,09% | 98,28% | 2.513 / 49 |
| Fine-tuning B | 98,19% | 98,25% | 98,28% | 98,33% | 2.518 / 44 |

O ranking observado na validação se manteve. O macro F1 de B no teste difere apenas **+0,06 ponto percentual** da validação; A apresenta +0,63 ponto e a base +0,38 ponto. Essas diferenças são compatíveis com variação entre as amostras dos dois holdouts, mas não comprovam generalização para outros pacientes, laboratórios ou microscópios.

O ganho de B sobre a base congelada foi de **8,56 pontos percentuais de macro F1**. Sobre A, o ganho foi menor: **0,24 ponto**, contra 0,81 ponto na validação. B teve cinco acertos líquidos adicionais, mas não acertou simplesmente todos os casos corretos de A: na comparação das mesmas imagens, corrigiu **27 erros de A** e introduziu **22 novos erros**. Logo, a vantagem global é pequena entre os fine-tunings e envolve trocas entre classes.

#### Classes e confusões mais relevantes

No modelo B, granulócitos imaturos mantiveram o menor F1 entre as oito classes (**96,61%**) e recall de **95,16%**: 21 das 434 imagens dessa classe foram atribuídas a outras classes. A confusão mais frequente foi granulócito imaturo → neutrófilo, com nove casos; o sentido inverso ocorreu cinco vezes. A semelhança morfológica é uma hipótese explicativa que exigiria revisão especializada, não uma conclusão obtida apenas pela matriz.

Os fine-tunings reduziram confusões expressivas da base. Por exemplo, eosinófilo → neutrófilo passou de 33 casos na base para zero em A e B. Granulócito imaturo → neutrófilo caiu de 27 casos na base para 18 em A e nove em B.

B não foi superior em todos os indicadores por classe. O recall de monócitos foi **97,18% em B**, contra **99,06% em A**; o recall de neutrófilos foi **98,40% em B**, contra **99,40% em A**. O F1 de eosinófilos e eritroblastos foi ligeiramente maior em A, e o de plaquetas empatou. Esses resultados impedem afirmar que B domina A em todas as situações. Em uma aplicação com prioridades diferentes por classe, seria necessário definir o custo dos erros previamente; neste estudo mantemos o critério de macro F1 estabelecido antes do teste.

#### Aprendizagem e explicabilidade

As curvas de B mostraram redução de loss e aproximação entre os F1 de treino e validação no final. O desempenho do teste não revelou uma queda agregada frente à validação. Assim, não há evidência forte de sobreajuste nessa divisão, embora isso não prove sua ausência. A oscilação de validação da A e o platô da base devem ser interpretados junto às métricas; a comparação de treino e validação também envolve augmentation e dropout somente no treino.

A análise de Grad-CAM foi realizada anteriormente com imagens de validação, sem selecionar mapas de teste para orientar a escolha. Nos exemplos corrigidos de linfócito e eritroblasto, os fine-tunings concentraram mais os destaques na célula de interesse. Nos granulócitos imaturos classificados incorretamente pelos fine-tunings, a célula também foi destacada. Portanto, localizar a célula não basta para distinguir seu tipo. Os mapas da classe real, com resolução original 7×7 e normalização individual, sustentam hipóteses qualitativas; não demonstram causalidade, precisão clínica ou superioridade de B em todos os casos.

#### Custo e condições de comparação

A base foi treinada por 15 épocas e seu melhor checkpoint de validação, da época 12, serviu como ponto de partida comum de A e B. A comparação utiliza as 12 épocas completas da A e as 12 realizadas pela B. Isso avalia o benefício de continuar o treinamento descongelando diferentes partes da rede; não isola o efeito de mais épocas do efeito de descongelar, pois não houve uma quarta condição com 12 épocas adicionais de base congelada. As estratégias A/B também variam simultaneamente blocos descongelados e learning rate, portanto comparamos estratégias completas.

O custo medido da etapa congelada foi **91,7 minutos**, comum às duas estratégias. A etapa B levou **133,2 minutos** e ajustou 3.952.066 parâmetros. A etapa A ajustou 3.165.988 parâmetros e tem tempo **estimado em 89,4 minutos**, recuperado do primeiro lote da época 13; não é uma medição exata das 12 épocas. A diferença aproximada de 43,8 minutos entre as etapas adicionais deve ser tratada com essa limitação, não como benchmark rigoroso. Os tempos representam treinamento com validações e salvamento dos melhores pesos, não latência de inferência.

Na validação, o maior macro F1 justificou a recomendação de B segundo o critério definido. O teste é compatível com essa expectativa, mas o ganho menor sobre A reforça que A é uma alternativa competitiva de menor custo de ajuste. Nenhuma preferência foi redefinida com base no teste.

#### Limitações e conclusão

O experimento utiliza uma única divisão estratificada e uma semente. Não foi realizada análise estatística para afirmar significância da pequena diferença entre A e B. A deduplicação de pixels evitou cópias exatas entre divisões, mas não elimina imagens semelhantes nem garante independência entre pacientes: não implementamos um holdout por paciente por ausência de identificação utilizada para esse fim. A validação foi usada repetidamente para selecionar checkpoints, e o teste confirma desempenho apenas dentro do protocolo e contexto de aquisição estudados.

As amostras são células normais provenientes de um contexto de aquisição específico. Mudanças de coloração, equipamento e população podem alterar o desempenho. O score softmax não foi calibrado e não deve ser interpretado como certeza clínica. O modelo classifica tipos celulares; o experimento não valida diagnóstico de doenças. A demonstração externa, registrada na seção 12 de CNN.ipynb, utilizou uma fotografia de neutrófilo de Tommaso Leonardi (Wikimedia Commons, domínio público). O modelo previu plaqueta com score de 86,45%, um erro que reforça a limitação frente a mudanças de enquadramento, escala e aquisição. Essas são hipóteses, não causas comprovadas. A demonstração não substitui o teste e não orientou novos ajustes.

Concluímos que o transfer learning com fine-tuning melhorou o desempenho em relação à base congelada neste conjunto. **Mantemos a recomendação prévia do fine-tuning B**, salvo como `models/selected.pt`, com macro F1 de teste de **98,25%**, reconhecendo a pequena vantagem sobre A, o maior custo e as limitações de generalização. Não foram realizados novos ajustes após consultar o teste.
