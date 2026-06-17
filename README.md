# TCC MBA Data Science e Analytics

Este repositório contém os algoritmos, tabelas e artefatos gerados para o Trabalho de Conclusão de Curso do MBA em Data Science e Analytics.

## Tema

Fatores associados a itens desertos e fracassados em licitações eletrônicas.

## Objetivo

Identificar fatores associados ao insucesso de itens em licitações eletrônicas, considerando como insucesso principal os itens desertos e fracassados.

## Fonte dos dados

Os dados utilizados foram extraídos dos sistemas das plataformas Bolsa de Licitações e Leilões do Brasil [BLL] e Bolsa Nacional de Compras [BNC], disponibilizados pela T I PRO-DESENVOLVIMENTO DE SISTEMAS LTDA mediante Termo de Anuência.

Por se tratarem de dados operacionais privados, as bases completas de entrada podem não estar disponíveis publicamente neste repositório. Os resultados são apresentados de forma agregada.

## Estrutura

- `codigo/`: scripts Python utilizados no processamento.
- `resultados/tabelas/`: tabelas agregadas geradas pelo algoritmo.
- `resultados/graficos/`: gráficos gerados para análise.
- `resultados/textos/`: textos-base gerados para interpretação dos resultados.
- `documentos/`: documentos acadêmicos relacionados ao TCC.

## Execução

 1. Criar ambiente

```bash
python -m venv .venv

2. Ativar ambiente no Windows

.venv\Scripts\activate

3. Instalar dependências
pip install -r requirements.txt

4. Gerar base integrada
python codigo/01_integrar_csvs_gerar_base.py --input-dir entrada --output-dir saida_integracao --chunksize 300000 --max-model-rows 0

5. Gerar comparativos e modelos
python codigo/02_comparativos_estatistica_ml.py --base saida_integracao/bases/base_analitica_itens_completa.csv.gz --output-dir saida_tcc_comparativos --max-model-rows 0 --max-text-fit-rows 40000 --n-text-clusters 20

### Observação metodológica

O número de propostas, o valor homologado e o desconto foram tratados como variáveis diagnósticas ou posteriores ao certame, não como preditores do modelo pré-disputa, para evitar vazamento de informação e interpretações tautológicas.

O insucesso principal foi definido como item deserto ou fracassado. A baixa competitividade foi mantida como indicador complementar.