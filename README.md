# Projeto de Redes Complexas - Detecção de domínios funcionais

O projeto constrói grafos a partir de estruturas de proteínas (arquivos `.pdb`) — redes de contato entre átomos/resíduos/cadeias ou uma rede de similaridade entre cadeias — e executa sobre eles centralidades, detecção de comunidades (Louvain, Infomap, greedy modularity, label propagation, bipartição espectral) e validação contra famílias conhecidas (ARI/NMI).

## Requisitos

- Python 3.10 ou superior
- pip

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/leandro-barcelos/protein-network.git
cd projeto

# 2. Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Instalar as dependências
pip install -r requirements.txt
```

## Dados de entrada

As estruturas `.pdb` ou `.tar.gz` usadas para testar o projeto precisam ser baixadas manualmente do [RCSB PDB](https://www.rcsb.org) para a pasta `pdb/`. Os arquivos JSON de validação de família para as proteínas analisadas já estão incluídos no repositório.

## Uso

### Comparar parâmetros e algoritmos (`model_sweep.py`)

Varre tipos de rede, limiares e algoritmos de comunidade, avaliando cada combinação contra famílias conhecidas e salvando o resultado em um CSV:

```bash
# Exemplo para o 6B1T
python src/model_sweep.py pdb/6b1t-pdb-bundle.tar.gz --validate pdb/6B1T-validation.json
```

Veja `python src/model_sweep.py -h` para uma lista com todas as opções.

### Construir uma rede e executar análises (`run_model.py`)

```bash
# Exemplo para o 4HHB, rede de contato entre carbonos-alfa a 8 Å, com Louvain e estatísticas
python src/run_model.py pdb/4HHB.pdb -g a-carbon --cutoff 8 -l -s

# Exemplo para o 6B1T, rede de similaridade entre cadeias, validada contra famílias conhecidas, com plots
python src/run_model.py pdb/6b1t-pdb-bundle.tar.gz -g chain-sim --sim-threshold 0.5 \
    -l -p --validate pdb/6B1T-validation.json
```

Principais opções (veja `python src/run_model.py -h` para a lista completa):

| Opção | Descrição |
| --- | --- |
| `-g, --graph` | Tipo de rede: `a-carbon`, `b-carbon`, `residue`, `chain`, `chain-sim` |
| `--cutoff` | Distância de corte (Å) para redes de contato (padrão: 8.0) |
| `--weighted` | Pondera arestas de `a-carbon`/`b-carbon` pelo inverso da distância |
| `--sim-threshold` | Limiar mínimo de similaridade para `chain-sim` (padrão: 0.5) |
| `--sim-method` | Medida de similaridade: `kmer` (padrão) ou `identity` |
| `-l, -i, -c, --labelprop, -b` | Detecção de comunidades: Louvain, Infomap, greedy (CNM), label propagation, espectral |
| `-k, --groups` | Número de grupos na bipartição espectral (padrão: 2) |
| `--validate JSON` | Valida comunidades contra `{família: [cadeias]}` (ARI/NMI) |
| `-s, --statistics` | Imprime estatísticas da rede |
| `-p, --plot` | Salva gráficos (distribuição de grau, centralidades, comunidades) |
| `--chimerax` | Exporta script ChimeraX para visualização 3D das comunidades |
| `-o, --out` | Diretório de saída (padrão: `exports`) |

## Saída

Por padrão, os resultados são salvos em `exports/`: estatísticas e centralidades em CSV, gráficos (`.jpg`) e, para cada algoritmo de comunidade executado, uma subpasta com composição das comunidades, validação e (opcionalmente) script ChimeraX.
