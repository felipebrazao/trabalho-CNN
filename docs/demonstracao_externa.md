## Comentário do resultado
A fonte identifica a célula como **neutrófilo**, mas o modelo selecionado previu **plaqueta**, com score softmax de **0,8645 (86,45%)**. Portanto, a execução funcionou, mas a classificação da imagem externa foi incorreta. Não houve novo treinamento nem alteração do modelo após observar esse resultado.

A foto de Tommaso Leonardi, disponibilizada em domínio público no Wikimedia Commons, tem 800×600 pixels, campo visual amplo, muitas hemácias e texto na borda inferior. A célula de interesse ocupa uma fração menor da imagem do que nas amostras PBC. A coloração também difere visualmente. Enquadramento, escala, fundo, texto e aquisição são hipóteses para a mudança de domínio; este caso não permite determinar qual fator causou o erro.

Foi preservado o pré-processamento da avaliação: RGB, resize direto 224×224 e normalização ImageNet. O score alto em uma previsão errada demonstra que softmax não equivale a uma probabilidade clínica calibrada. Este exemplo único demonstra a reutilização do checkpoint, mas não mede acurácia externa, não substitui o teste e não autoriza conclusões gerais sobre outras fontes. A lauda não exige acerto na imagem externa.

Fonte: https://commons.wikimedia.org/wiki/File:Neutrophil.jpg

Atribuição: Tommaso Leonardi — Neutrophil.jpg — Wikimedia Commons — domínio público.
