# Resultados agregados da regressão logística inferencial

Os CSVs e JSONs desta pasta são cópias dos resultados já calculados, sem nova
estimação ou alteração dos números. O código responsável pelo ajuste é
[`03_regressao_logistica_inferencial.py`](../../codigo/03_regressao_logistica_inferencial.py).
O SHA-256 do script e as versões usadas estão em [versoes.json](versoes.json).

## Arquivos e finalidade

| Arquivos | Conteúdo |
| --- | --- |
| [coeficientes_Regiao.csv](coeficientes_Regiao.csv), [coeficientes_UF.csv](coeficientes_UF.csv) | β, erro-padrão robusto agrupado, z, p-value, OR e IC95%; categoria de referência e escala de cada contraste. |
| [testes_conjuntos_Regiao.csv](testes_conjuntos_Regiao.csv), [testes_conjuntos_UF.csv](testes_conjuntos_UF.csv) | Estatística de Wald, graus de liberdade e p-value por variável/bloco. |
| [covariancia_cluster_Regiao.csv](covariancia_cluster_Regiao.csv), [covariancia_cluster_UF.csv](covariancia_cluster_UF.csv) | Matrizes de covariância dos coeficientes para auditoria dos testes conjuntos; não contêm registros por processo. |
| [diagnostico_Regiao.json](diagnostico_Regiao.json), [diagnostico_UF.json](diagnostico_UF.json) | Convergência, tamanho da amostra, parâmetros, log-verossimilhança e diagnósticos numéricos. |
| [auditoria_base.json](auditoria_base.json) | Contagens da base, elegibilidade, exclusões e amostra final. |
| [dicionario_coeficientes_Regiao.csv](dicionario_coeficientes_Regiao.csv), [dicionario_coeficientes_UF.csv](dicionario_coeficientes_UF.csv) | Referências, centralização, escalas, medianas e contagens de imputações. |
| [contagens_Regiao.csv](contagens_Regiao.csv), [contagens_UF.csv](contagens_UF.csv) | Quantidade de itens, sucessos, insucessos e processos por categoria, sem identificadores. |
| [familias_textuais.csv](familias_textuais.csv), [agrupamento_familias_raras.csv](agrupamento_familias_raras.csv) | Termos centrais agregados das famílias e correspondência das famílias com menos de 100 itens para a categoria residual. |
| [associacoes_bivariadas_mesma_amostra.csv](associacoes_bivariadas_mesma_amostra.csv) | Auditoria complementar já calculada: Pearson e V de Cramér sobre as mesmas contagens categóricas da amostra inferencial. |
| [contagens_familias_antes_agrupamento.csv](contagens_familias_antes_agrupamento.csv) | Auditoria complementar das frequências e eventos antes da reunião das famílias raras. |
| [versoes.json](versoes.json) | Ambiente da estimação e assinatura do script. |

Formato: CSV com separador `;`, codificação UTF-8 com BOM e ponto decimal;
JSON em UTF-8. Os valores completos foram preservados; o README principal
apresenta apenas arredondamentos para leitura.

## Amostra e especificação

A amostra final contém **949.329 itens, 78.942 insucessos e 33.181 processos**.
Foram excluídos adicionalmente 3.509 dos 952.838 itens elegíveis por ausência
de vínculo válido com os dados do processo. Esses itens incluíam 126 insucessos.

A auditoria reconstruída registra 36.383 identificadores de processo na base
completa; o total anteriormente informado no trabalho é 36.386. A diferença
de três identificadores permanece não reconciliada. Os totais de itens e desfechos
coincidiram, e os 33.181 processos da amostra inferencial foram contados diretamente
após a exclusão. Os resultados e as contagens anteriores não foram sobrescritos.

O modelo principal usa Região (referência: Sudeste); a sensibilidade substitui
Região por UF (referência: SP). Ambos usam máxima verossimilhança sem regularização,
sem pesos de balanceamento e sem treino/teste. Os erros-padrão são agrupados por
processo, com correção para número finito de grupos; a inferência usa testes
z/Wald assintóticos. As duas matrizes de desenho têm posto completo e os ajustes
convergiram, respectivamente, em oito e nove iterações, sem avisos de estimação.

O preço entra como `log(1 + valor cotado)`; a descrição, por 100 caracteres e
10 palavras; a quantidade do lote, por 10 itens. As numéricas são centradas e
ausências são imputadas pela mediana (16 imputações no valor cotado). Os indicadores
ausentes seguem o tratamento como zero do processamento original. Um valor cotado
negativo foi tratado como zero antes da transformação logarítmica.

As famílias textuais reutilizam os modelos originais. Oito famílias com menos de
100 itens foram reunidas em uma categoria residual; quatro não tinham insucessos
antes desse agrupamento. A decisão é uma adaptação exploratória informada pela
auditoria. A família 13 concentra cerca de 98,54% dos itens, de modo que o controle
da heterogeneidade dos objetos é limitado. Os 1.800 termos TF-IDF individuais do
classificador preditivo não integram este ajuste.

## Interpretação e auditoria

O bloco de critérios de participação tem p global de aproximadamente 0,051 no
modelo por Região e 0,007 no modelo por UF. Isso deve ser apresentado juntamente
com o contraste individual de somente ME/EPP, que tem p de aproximadamente 0,007
no modelo principal. O resultado global é sensível ao controle territorial.

Tipo de lance GLOBAL e estrutura calculada Lote global são variáveis distintas,
incluídas simultaneamente: seus coeficientes representam contrastes condicionais.
O preço logarítmico não é a classificação relativa à referência interna de preço
utilizada nos testes bivariados do TCC. Não se deve transportar a conclusão de
uma variável diretamente para a outra.

Os testes bivariados desta pasta foram calculados posteriormente sobre a amostra
inferencial, não substituem as tabelas anteriores e não reproduzem todos os recortes
do TCC (por exemplo, a classificação relativa de preço não está incluída).
Podem ser auditados pelas contagens de sucessos/insucessos em `contagens_Regiao.csv`:
Pearson sem correção de continuidade, e `V = sqrt(qui_quadrado / n)` para desfecho
binário. Seus p-values não ajustam a dependência entre itens de um processo.
Já os testes de Wald usam a covariância agrupada. Para um bloco de coeficientes β
e sua submatriz de covariância C, a estatística é `β' C^(-1) β`, com distribuição
qui-quadrado assintótica e graus de liberdade iguais ao número de coeficientes.

Não há interpretação causal. Os p-values são exploratórios e não ajustados por
multiplicidade; categorias com poucos processos podem produzir intervalos amplos.
Pseudo-R² de McFadden não é proporção de variância explicada. A análise inferencial
não produz uma nova AUC de teste: o desempenho preditivo anterior pertence ao
classificador regularizado, com sua própria divisão treino/teste.

## Publicação

Esta pasta contém apenas agregados selecionados. Microdados, identificadores de
processos/lotes/itens, modelos binários, relatórios locais para edição do TCC e
ZIPs de trabalho ficam fora do versionamento. A reprodução integral depende dos
dados privados e dos artefatos originais; consulte o [README principal](../../README.md).
