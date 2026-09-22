# TCC MBA Data Science e Analytics

Fatores associados a itens desertos e fracassados em licitações eletrônicas.
Este repositório reúne os códigos e resultados agregados do trabalho, com três
abordagens complementares e finalidades distintas.

## Dados e definição do desfecho

Os dados operacionais das plataformas Bolsa de Licitações e Leilões do Brasil
(BLL) e Bolsa Nacional de Compras (BNC) foram disponibilizados pela
T I PRO-DESENVOLVIMENTO DE SISTEMAS LTDA mediante Termo de Anuência.
Os dados privados originais e os modelos binários não são disponibilizados no GitHub.
A reprodução integral exige acesso autorizado a esses dados.

A unidade de análise é o item. Insucesso corresponde a **DESERTO ou FRACASSADO**;
sucesso, a **HOMOLOGADO, ADJUDICADO ou RESULTADO FINAL**. Os demais estados ficam
fora do desfecho binário. Baixa competitividade é um indicador complementar.

## Três abordagens metodológicas

| Abordagem | Finalidade | Implementação e interpretação |
| --- | --- | --- |
| Descritivas, qui-quadrado e V de Cramér | Avaliar associações bivariadas entre cada fator e o insucesso | Tabelas de frequência e contingência; qui-quadrado de Pearson e V de Cramér para a intensidade da associação. Não ajustam pelos demais fatores. |
| Regressão logística inferencial | Estimar associações ajustadas, considerando simultaneamente os demais fatores do modelo | `statsmodels.Logit`, sem regularização, sem balanceamento de classes e sem divisão treino/teste; erros-padrão robustos agrupados por `idProcess`. Modelo principal por Região e sensibilidade substituindo Região por UF. |
| Regressão logística preditiva | Avaliar a capacidade de antecipar insucessos | `scikit-learn`, penalização L2, pesos de classe balanceados e divisão estratificada de 75% para treino e 25% para teste. |

As análises identificam **associações, não causalidade**. O número de propostas,
o valor homologado e o desconto não são preditores do modelo pré-disputa.
O modelo diagnóstico posterior ao certame, preservado no repositório, tem finalidade
distinta e não representa desempenho de previsão antecipada.

### Texto e especificação inferencial

TF-IDF foi usado para representar numericamente as descrições; MiniBatchKMeans
foi utilizado anteriormente para formar 20 famílias textuais. O modelo inferencial
reutiliza essas famílias, além das medidas de tamanho das descrições. **Os até
1.800 termos TF-IDF individuais do classificador preditivo não foram inseridos.**
Essa representação reduz a dimensionalidade e resume parte da heterogeneidade
dos objetos, sem controlá-la completamente. Uma família concentra aproximadamente
98,54% da amostra inferencial, limitando o detalhamento desse controle.

Famílias com menos de 100 itens foram reunidas em uma categoria residual para
reduzir problemas de separação e estimativas não finitas. Região e UF são usadas
em modelos separados; os indicadores sobrepostos de participação foram
representados pelo grupo consolidado de exclusividade/regionalidade.
Portanto, a especificação inferencial é reduzida em relação ao classificador
preditivo, e não apenas uma retirada da penalização L2.

Os demais controles incluem modalidade, tipo de disputa, tipo de encerramento,
tipo de lance, estrutura e quantidade de itens do lote, logaritmo do valor cotado,
tamanho das descrições e exigências de informação/arquivo. O código calcula
β (log-odds), odds ratio, IC95%, p-value e testes conjuntos de Wald por variável/bloco.
As referências, escalas, imputações e diagnósticos estão documentados nos
[resultados inferenciais](tabelas/inferencia/README.md).

## Resultados da regressão inferencial

Dos 952.838 itens elegíveis e 79.068 insucessos do trabalho, houve exclusão adicional
de **3.509 itens sem vínculo válido com os dados do processo**, incluindo 126
insucessos. A regressão utilizou **949.329 itens, 33.181 processos e 78.942 insucessos**.
Os dois modelos convergiram. A [auditoria da amostra](tabelas/inferencia/auditoria_base.json)
registra os denominadores e as exclusões.

Principais contrastes no modelo por Região:

| Contraste | OR ajustada | IC95% | p-value |
| --- | --- | --- | --- |
| Dispensa eletrônica vs. Pregão eletrônico | 1,996 | 1,677–2,375 | < 0,001 |
| Somente ME/EPP vs. sem critério | 1,199 | 1,050–1,368 | 0,007 |
| Lote global vs. unitário (estrutura calculada) | 0,527 | 0,395–0,701 | < 0,001 |
| Valor cotado em escala logarítmica | 1,005 | 0,987–1,022 | 0,610 |

Fontes: [coeficientes por Região](tabelas/inferencia/coeficientes_Regiao.csv) e
[coeficientes por UF](tabelas/inferencia/coeficientes_UF.csv).
OR refere-se à razão de odds, não a uma diferença de probabilidade. O coeficiente
do valor cotado corresponde a uma unidade de `log(1 + valor)`, não a R$1.
Essa variável é diferente da classificação de preço em relação à referência
interna examinada nas análises bivariadas do TCC.

O **bloco de critérios de participação** apresentou **p global = 0,051 no modelo
com Região e p = 0,007 no modelo com UF**, mostrando sensibilidade à especificação
territorial. O contraste individual de ME/EPP e o teste conjunto do bloco
respondem a hipóteses diferentes. Não se deve afirmar confirmação universal das
associações bivariadas. Consulte os testes de Wald
[por Região](tabelas/inferencia/testes_conjuntos_Regiao.csv) e
[por UF](tabelas/inferencia/testes_conjuntos_UF.csv).

## Estrutura do repositório

- [codigo/](codigo/): integração (`01_integrar_csvs_gerar_base.py`), comparativos e
  modelos preditivos (`02_comparativos_estatistica_ml.py`) e análise inferencial
  (`03_regressao_logistica_inferencial.py`). O script anterior
  `algoritmo_tcc_comparativos_ml.py` foi preservado.
- [tabelas/](tabelas/): resultados descritivos agregados anteriores.
- [tabelas/inferencia/](tabelas/inferencia/): cópias revisadas dos resultados
  inferenciais, auditoria, diagnósticos e versões do ambiente.
- [modelos/](modelos/): métricas, matrizes de confusão e coeficientes dos modelos
  preditivos anteriores. Modelos binários locais são ignorados pelo Git.
- [graficos/](graficos/): gráficos descritivos já produzidos.
- [texto/](texto/): texto-base preliminar preservado como registro da etapa anterior;
  não substitui a interpretação inferencial final.

## Reprodução em ambiente autorizado

Execute os comandos na raiz do projeto. Para criar e ativar o ambiente no PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

A análise inferencial requer localmente os arquivos `processos (2).csv`, `lotes.csv`
e `itens.csv` na pasta de entrada, além dos artefatos de famílias textuais
`tfidf_vectorizer_familias_textuais.joblib` e `kmeans_familias_textuais.joblib`
na pasta de modelos. Esses arquivos privados não são versionados.

Com os dados e os artefatos originais já disponíveis localmente:

```powershell
python codigo/03_regressao_logistica_inferencial.py
```

Caso a base local da análise inferencial já esteja preparada:

```powershell
python codigo/03_regressao_logistica_inferencial.py --reuse-base
```

O script grava em `resultados_inferencia/` (pasta local ignorada pelo Git), incluindo
uma base privada de trabalho. **Não use `tabelas/inferencia/` como `--output-dir`**:
essa pasta pública contém apenas os agregados selecionados após revisão.
Executar a análise não atualiza automaticamente as cópias públicas.

### Preparação das etapas anteriores, quando necessária

As pastas de saída abaixo são geradas pelos comandos e não fazem parte dos
arquivos públicos. A integração completa também exige `resultado-item.csv` e
`propostas-lote.csv` na entrada privada.

```powershell
python codigo/01_integrar_csvs_gerar_base.py --input-dir entrada --output-dir saida_integracao --chunksize 300000 --max-model-rows 0
python codigo/02_comparativos_estatistica_ml.py --base saida_integracao/bases/base_analitica_itens_completa.csv.gz --output-dir saida_tcc_comparativos --max-model-rows 0 --max-text-fit-rows 40000 --n-text-clusters 20
```

O segundo comando gera os artefatos textuais na subpasta `modelos` de sua saída.
Para usar essa localização em uma nova execução inferencial:

```powershell
python codigo/03_regressao_logistica_inferencial.py --model-dir saida_tcc_comparativos/modelos
```

Uma nova preparação não substitui os resultados publicados. Para reproduzir
exatamente o ajuste arquivado, são necessários os mesmos dados, artefatos textuais
originais e ambiente registrado em [versoes.json](tabelas/inferencia/versoes.json).
As dependências não tiveram versões fixadas retroativamente.
