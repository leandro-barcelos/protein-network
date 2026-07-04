# Rede de Similaridade de Cadeias: método e justificativa

**Projeto:** Detecção de Domínios Funcionais em Redes de Proteínas
**Módulo:** `ChainSimilarityNetwork` (`src/protein_network.py`)
**Alvo:** 6B1T (capsídeo de adenovírus humano)

---

## 1. Motivação: por que as redes de contato não recuperam as famílias

A pipeline original modela a proteína como uma **rede de contatos espaciais** — nós são átomos (Cα/Cβ), resíduos ou cadeias, e as arestas ligam elementos que estão **fisicamente próximos** (abaixo de um limiar de distância). Sobre esse grafo aplicamos detecção de comunidades (Louvain, Infomap, greedy modularity, label propagation, bipartição espectral) e comparamos a partição com as anotações funcionais do RCSB PDB via ARI e NMI.

O ponto central, confirmado experimentalmente neste projeto, é:

> **Detecção de comunidades sobre uma rede de contatos recupera módulos *geométricos* (o que está fisicamente agrupado), e não famílias de proteína definidas por *identidade de sequência*.**

Esses dois agrupamentos são **ortogonais**, e isso explica os três resultados observados de uma só vez:

| Estrutura | O que o método de contato recupera | O que a anotação pede | Resultado |
|-----------|-----------------------------------|----------------------|-----------|
| **4HHB** (hemoglobina) | dímeros α₁β₁ / α₂β₂ e domínios internos (as unidades espacialmente coesas) | "todos os α" vs "todos os β" | ARI ≈ 0,15 — nunca separa por família |
| **6MID** (poliproteína + anticorpo) | anticorpo destacado espacialmente da poliproteína | anticorpo vs poliproteína | ARI ≈ 1,0 — **coincidência**: identidade e geometria coincidem |
| **6B1T** (adenovírus) | os 4 *clusters* de hexon (organização *group-of-nine*), cada trímero + proteínas de cimento vizinhas | agrupar todos os 12 hexons; isolar o penton | só separa cadeias/trímeros — nunca a família |

No 6B1T, os 12 hexons (cadeias A–L) estão espalhados em **quatro locais distintos** da faceta viral; como não formam um único bloco em contato, nenhum método de contato os agrupa numa só comunidade. Além disso, na unidade assimétrica cristalográfica existe apenas **uma subunidade** do penton base (cadeia M), encravada entre os hexons (1374 contatos atômicos com o hexon B), de modo que ela é sempre absorvida no *cluster* do hexon vizinho.

**Conclusão do diagnóstico:** o "fracasso" no 4HHB e no 6B1T não é um defeito da pipeline nem uma questão de ajuste de hiperparâmetro — é uma consequência esperada de se usar geometria para responder uma pergunta de identidade. Para recuperar as **famílias funcionais**, precisamos de um grafo cuja noção de aresta seja **homologia/identidade de sequência**, não proximidade espacial.

---

## 2. O método: rede de similaridade de cadeias

A ideia é trocar a semântica das arestas mantendo o mesmo arcabouço de detecção de comunidades:

- **Nós:** cada cadeia da estrutura é um nó (25 nós para o 6B1T).
- **Arestas:** ligam cadeias com **sequências semelhantes**; o peso é a similaridade.
- **Comunidades:** como cadeias da mesma proteína têm sequência (quase) idêntica, elas formam agrupamentos densos → a detecção de comunidades recupera as **famílias de proteína**.

### 2.1 Extração da sequência por cadeia

Para cada cadeia, obtém-se um resíduo por posição (ordenados por número de resíduo e código de inserção) e converte-se o nome de três letras para o código de uma letra (`_chain_sequence` + `THREE_TO_ONE`, com `MSE→M` etc. e `X` para resíduos não-padrão). Resulta numa *string* de aminoácidos por cadeia.

### 2.2 Medida de similaridade

Duas medidas estão implementadas (parâmetro `--sim-method`):

**(a) Cosseno de perfis de k-mers — padrão (`kmer`).**
Cada sequência é representada pelo vetor de contagens de todos os seus *k*-mers (subcadeias de comprimento *k*, padrão *k*=3). A similaridade entre duas cadeias é o **cosseno** entre seus vetores:

$$\text{sim}(a,b)=\frac{\sum_{w} n_a(w)\,n_b(w)}{\lVert n_a\rVert\,\lVert n_b\rVert}$$

onde $n_a(w)$ é a contagem do k-mer $w$ na cadeia $a$. É um método **livre de alinhamento** (*alignment-free*): sequências idênticas têm cosseno 1,0; proteínas diferentes têm cosseno baixo. Barato ($O(L)$ por cadeia) e robusto.

**(b) Identidade por alinhamento (`identity`).**
Alinhamento global (Needleman–Wunsch, via `Bio.Align.PairwiseAligner`), contando colunas idênticas. A identidade é normalizada pelo **maior** comprimento das duas sequências:

$$\text{id}(a,b)=\frac{\text{colunas idênticas}}{\max(|a|,|b|)}$$

### 2.3 Construção do grafo e detecção de comunidades

Calcula-se a similaridade para todos os pares de cadeias; adiciona-se uma aresta ponderada quando a similaridade $\ge$ **limiar** (`--sim-threshold`, padrão 0,5). Sobre esse grafo aplicam-se **os mesmos algoritmos** já usados na pipeline (Louvain, greedy, Infomap, label propagation, espectral), sem qualquer modificação — todo o restante (validação por ARI/NMI/pureza, exportação de composição, `membership.csv`, *script* ChimeraX, figuras) funciona igual aos demais grafos de cadeia.

---

## 3. Justificativa detalhada das escolhas

### 3.1 Por que uma rede de similaridade (e não de contato)
Como demonstrado na Seção 1, o objetivo declarado — reproduzir os "grupos funcionais" do `6B1T-validation.json` — é um agrupamento por **identidade de proteína**. A ferramenta correta para esse objetivo é uma rede cuja aresta codifique semelhança de sequência. A rede de contato continua válida e útil, mas responde uma pergunta **complementar** (organização geométrica: os *clusters* de hexon), e as duas devem ser apresentadas como resultados que se complementam, não como concorrentes.

### 3.2 Por que k-mers (cosseno) como padrão
- **Livre de alinhamento:** dispensa alinhar pares de sequências longas (~910 resíduos nos hexons), com custo linear e sem dependência de parâmetros de *gap*.
- **Robusto a lacunas do modelo:** estruturas cristalográficas têm resíduos/laços faltantes; cópias da mesma proteína raramente têm *strings* exatamente iguais. O cosseno de k-mers ainda dá similaridade ~1,0 nesses casos, ao passo que comparação por igualdade exata falharia.
- **Empiricamente ótimo aqui:** recupera as famílias com ARI 0,974 / NMI 0,971 / pureza 1,0 (Seção 4).

### 3.3 Por que normalizar a identidade pelo comprimento máximo
A primeira versão do método de alinhamento normalizava pelo **menor** comprimento. Isso fez fragmentos curtos (por exemplo, cadeias de ~30 resíduos das proteínas menores) alinharem-se *dentro* de proteínas grandes e pontuarem identidade ~1,0, criando **arestas espúrias** entre famílias distintas (260 arestas, Louvain colapsando em 2 comunidades, ARI 0,37). Normalizar por $\max(|a|,|b|)$ penaliza diferenças de comprimento: um fragmento de 30 resíduos contra um hexon de 910 pontua no máximo $30/910\approx0{,}03$, ficando abaixo do limiar. Com essa correção, o método de identidade passou a coincidir com o de k-mers (ARI 0,974).

### 3.4 Por que a escolha do limiar é segura
As similaridades intra-família são altas (0,82–1,0) e as inter-família são baixas (≤ 0,12 no 6B1T). Há, portanto, uma **grande separação (*gap*)** entre os dois regimes, e qualquer limiar em ampla faixa produz o mesmo grafo. O *sweep* confirma um **platô estável de 0,2 a 0,8** em todos os cinco algoritmos (Seção 4). Isso indica que o resultado **não depende de um ajuste fino** — uma propriedade desejável de robustez.

### 3.5 Por que reutilizar os mesmos algoritmos de comunidade
Manter Louvain/greedy/Infomap/label-prop/espectral permite comparar diretamente os dois tipos de rede sob o mesmo protocolo de avaliação e mostra que o resultado **independe do algoritmo** — todos convergem para a mesma partição, o que reforça que a estrutura de comunidades está no *grafo*, não num algoritmo específico.

---

## 4. Resultados no 6B1T

Todos os algoritmos (Louvain, greedy, Infomap, label propagation, espectral) produzem a mesma partição.

**Validação contra as 7 famílias funcionais** (`6B1T-validation.json`):

| Métrica | Valor |
|--------|-------|
| ARI | **0,974** |
| NMI | **0,971** |
| Pureza | **1,000** |
| Nº de comunidades | 8 |

Partição obtida: `{A–L}` (12 hexons), `{M}` (penton), `{N}` (IIIa), `{O,P}` (VIII), `{Q,R,S,T}` (IX), `{U,V,Y}` + `{X}` (VI), `{W}` (VII). A pureza 1,0 significa que **nenhuma comunidade mistura famílias**. A única divergência em relação à anotação é a cadeia **X** (fragmento de 33 resíduos da proteína VI) que se separa de U,V,Y — trata-se de baixa similaridade de sequência real, não de erro do método, e produz 8 comunidades em vez de 7 (por isso ARI 0,974 e não 1,0).

**Validação mínima penton × hexon** (`6B1T-validation-min.json`, 3 classes): hexon `{A–L}` e penton `{M}` são separados de forma limpa, **pureza 1,000** (ARI 0,60 apenas porque o método é *mais* granular que o rótulo, distinguindo IIIa, VI, VIII, IX e VII em vez de agrupá-los todos como "hexon-associated"). **O objetivo mínimo é plenamente atingido.**

**Robustez (sweep de limiar, `exports/model_selection_6b1t_chainsim.csv`):** ARI 0,974 / NMI 0,971 / pureza 1,0 constantes para todo limiar em [0,2; 0,8], nos cinco algoritmos.

---

## 5. Limitações

- **Cadeias muito curtas** (< ~40 resíduos) têm perfis de k-mers instáveis e podem se separar da sua família (caso da cadeia X). Para essas, o alinhamento local ou anotação manual é mais confiável.
- A rede de similaridade **ignora a geometria**: ela não distingue os quatro hexons individuais nem localiza o penton no vértice. Isso é intencional (é a rede de contato que fornece essa informação), mas significa que as duas redes precisam ser usadas em conjunto para uma leitura estrutural completa.
- Isolar o **pentâmero** do penton como um módulo espacial exigiria reconstruir a montagem biológica aplicando os operadores de simetria icosaédrica (registros BIOMT), o que está fora do arquivo cristalográfico da unidade assimétrica.

---

## 6. Reprodução

```bash
cd src

# Recuperar as 7 famílias funcionais
python3 run_model.py ../pdb/6b1t-pdb-bundle.tar.gz -g chain-sim \
  --sim-threshold 0.5 --louvain --greedy --spectral --groups 7 \
  --validate ../pdb/6B1T-validation.json --chimerax -o ../exports/6B1T-chainsim

# Validação mínima penton x hexon (com figuras)
python3 run_model.py ../pdb/6b1t-pdb-bundle.tar.gz -g chain-sim \
  --sim-threshold 0.5 --louvain --plot \
  --validate ../pdb/6B1T-validation-min.json -o ../exports/6B1T-chainsim-min

# Sweep de limiar (robustez)
python3 model_selection.py ../pdb/6b1t-pdb-bundle.tar.gz --graphs chain-sim \
  --cutoff-start 0.1 --cutoff-stop 0.9 --cutoff-step 0.1 \
  --validate ../pdb/6B1T-validation.json -o ../exports
```

Parâmetros: `--sim-method {kmer|identity}` (padrão `kmer`), `--sim-threshold` (padrão 0,5), `--kmer` (ordem *k*, padrão 3). No `model_selection.py`, o grafo `chain-sim` reinterpreta a grade de `--cutoff` como grade de **limiar** (use valores em 0..1).

---

## 7. Fontes

**Estruturas (RCSB PDB).** Confirmar a citação primária de cada entrada na respectiva página do RCSB.
- 6B1T — capsídeo de adenovírus humano. RCSB PDB: https://www.rcsb.org/structure/6B1T
- 4HHB — Fermi, G.; Perutz, M. F.; Shaanan, B.; Fourme, R. *The crystal structure of human deoxyhaemoglobin at 1.74 Å resolution.* Journal of Molecular Biology, 1984.
- 6MID — complexo poliproteína de flavivírus + fragmento de anticorpo. RCSB PDB: https://www.rcsb.org/structure/6MID
- Berman, H. M. et al. *The Protein Data Bank.* Nucleic Acids Research, 2000.

**Biologia estrutural do adenovírus (organização hexon/penton, *group-of-nine*).**
- Reddy, V. S.; Natchiar, S. K.; Stewart, P. L.; Nemerow, G. R. *Crystal structure of human adenovirus at 3.5 Å resolution.* Science, 2010.
- Liu, H. et al. *Atomic structure of human adenovirus by cryo-EM reveals interactions among protein networks.* Science, 2010.
- Reddy, V. S.; Nemerow, G. R. *Structures and organization of adenovirus cement proteins provide insights into the role of capsid maturation in virus entry and infection.* PNAS, 2014.

**Redes de proteínas (redes de contato de resíduos).**
- Vishveshwara, S.; Brinda, K. V.; Kannan, N. *Protein structure: insights from graph theory.* Journal of Theoretical and Computational Chemistry, 2002.
- Greene, L. H.; Higman, V. A. *Uncovering network systems within protein structures.* Journal of Molecular Biology, 2003.
- Di Paola, L. et al. *Protein contact networks: an emerging paradigm in chemistry.* Chemical Reviews, 2013.

**Detecção de comunidades e avaliação.**
- Blondel, V. D.; Guillaume, J.-L.; Lambiotte, R.; Lefebvre, E. *Fast unfolding of communities in large networks* (Louvain). J. Stat. Mech., 2008.
- Rosvall, M.; Bergstrom, C. T. *Maps of random walks on complex networks reveal community structure* (Infomap). PNAS, 2008.
- Clauset, A.; Newman, M. E. J.; Moore, C. *Finding community structure in very large networks* (greedy modularity). Physical Review E, 2004.
- Raghavan, U. N.; Albert, R.; Kumara, S. *Near linear time algorithm to detect community structures in large-scale networks* (label propagation). Physical Review E, 2007.
- Newman, M. E. J.; Girvan, M. *Finding and evaluating community structure in networks* (modularidade). Physical Review E, 2004.
- Fiedler, M. *Algebraic connectivity of graphs* (bipartição espectral). Czechoslovak Math. Journal, 1973.
- von Luxburg, U. *A tutorial on spectral clustering.* Statistics and Computing, 2007.
- Hubert, L.; Arabie, P. *Comparing partitions* (ARI). Journal of Classification, 1985.
- Danon, L.; Díaz-Guilera, A.; Duch, J.; Arenas, A. *Comparing community structure identification* (NMI). J. Stat. Mech., 2005.

**Comparação de sequências (alinhamento e livre de alinhamento).**
- Needleman, S. B.; Wunsch, C. D. *A general method applicable to the search for similarities in the amino acid sequence of two proteins* (alinhamento global). Journal of Molecular Biology, 1970.
- Vinga, S.; Almeida, J. *Alignment-free sequence comparison — a review.* Bioinformatics, 2003.
- Zielezinski, A. et al. *Alignment-free sequence comparison: benefits, applications, and tools.* Genome Biology, 2017.

**Ferramentas.**
- Cock, P. J. A. et al. *Biopython: freely available Python tools for computational molecular biology and bioinformatics.* Bioinformatics, 2009.
- Raschka, S. *BioPandas: Working with molecular structures in pandas DataFrames.* Journal of Open Source Software, 2017.
- Hagberg, A.; Schult, D.; Swart, P. *Exploring network structure, dynamics, and function using NetworkX.* SciPy Conf., 2008.
- Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR, 2011.
- Virtanen, P. et al. *SciPy 1.0: fundamental algorithms for scientific computing in Python.* Nature Methods, 2020.

> **Nota sobre as referências:** as citações acima trazem autores, título, veículo e ano para localização das obras originais; recomenda-se conferir volume, páginas e DOI nas fontes primárias antes da entrega. As citações primárias de 6B1T e 6MID devem ser confirmadas diretamente nas páginas do RCSB PDB indicadas.
