# Resumo relatório

1. Foram obtidos os dados pelo RCSB PDB
   1. Arquivos PDB
      1. Proteína alvo 6B1T
      2. Proteínas de teste
         1. 4HHB
         2. 6MID
         3. 1AON
   2. Anotações
      1. Manual
      2. Em formato JSON
      3. Mapeia dominio/familia para cadeias
2. PDBs convertidos em dataframes do pandas, utilizando BioPandas
3. Feito o tratamento do dataframe
   1. Remoção de colunas não utilizadas
   2. Remoção de atómos com alt-loc
   3. Criação de um id único para cada nó
4. De inicio foram implementados redes de contato, sendo elas:
   1. alpha-carbon (nós são átomos CA)
      1. Conecta pares de átomos com distância dentro de um limiar
      2. Opção ponderada: peso igual ao inverso da distância
   2. beta-carbon (nós são átomos CB, ou CA para GLY)
   3. cadeia (cada nó é uma cadeia)
      1. Monta um grafo atomico por distancia
      2. Conecta pares de cadeias que tem atomos se conectando
      3. Peso é o número de conexões de atomos entre as cadeias
   4. residuo (cada nó é um residuo)
   5. similaridade de cadeia (cada nó é uma cadeia)
      1. Conecta cadeias que estão dentro de um limiar de similaridade
      2. Peso é a similaridade
5. Algoritmos de detecção de comunidade:
   1. Louvain
   2. Infomap
   3. Greedy modularity
   4. Label propagation
   5. Bipartição espectral
6. Implementado uma pipeline para testar todas as combinações desses modelos com parâmetros diferentes e com todos algoritmos de detecção de comunidades.
7. Validação da partição comparando com a anotação, medido por ARI, NMI e pureza.
   1. Para 4HHB e 6B1T, esses modelos foram capazes apenas de detectar as cadeias (para a-carbon e b-carbon)
   2. Para a 6MID conseguiu separar as familias
   3. Percebi que, mesmo para a 6MID, as comunidades eram sempre baseadas na estrutura da proteina, ou seja, não era capaz de detectar grupos funcionais que muitas vezes estavam dispersos ou próximos de outros grupos.
8. Foi então implementado um modelo por similaridade de cadeias
   1. Nós são cadeias
   2. Conectam par de cadeias com similaridade dentro de um limiar
   3. Com esse modelo foi possível obter a separação desejada em todos os modelos

OBS: Não sei como fazer as análises de centralidade e distribuição de graus para as proteínas. Não sei como interpretar essas informações no grafo de similaridade.
