#!/usr/bin/env python3
"""Regressao logistica inferencial complementar; ver relatorio das limitacoes.

python codigo/03_regressao_logistica_inferencial.py
Nao modifica os modelos preditivos nem o documento do TCC.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
import statsmodels
import statsmodels.api as sm
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
START = time.time()


def log(s):
    print(f'[{time.time()-START:.1f}s] {s}', flush=True)


def csv(df, path):
    df.to_csv(path, sep=';', index=False, encoding='utf-8-sig')


def save_json(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def legacy_module():
    spec = importlib.util.spec_from_file_location('integracao', ROOT/'codigo/01_integrar_csvs_gerar_base.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # O codigo legado silencia avisos globalmente; nesta analise eles sao preservados.
    warnings.resetwarnings()
    return mod


def prepare(args, out):
    old = legacy_module()
    tracker = old.ProgressTracker(out/'preparacao')
    source = Path(args.input_dir)
    log('Lendo apenas colunas necessarias dos arquivos originais')
    proc = old.read_csv_auto(source/'processos (2).csv', tracker, usecols=[
        'idProcess', 'UF', 'Modalidade', 'TipoDisputa', 'TipoEncerramento',
        'Status', 'MEExclusivo', 'RegExcl', 'LocalExc'])
    proc = old.prepare_processos(proc)
    proc_dup = int(proc.duplicated('idProcess').sum())
    proc = proc.drop_duplicates('idProcess')
    lotes = old.read_csv_auto(source/'lotes.csv', tracker, usecols=[
        'fkProcess', 'idBatch', 'Status', 'TipoLance', 'Titulo', 'ExclMe', 'Regional', 'ExclLocal'])
    lotes = old.prepare_lotes(lotes)
    itens = old.read_csv_auto(source/'itens.csv', tracker, usecols=[
        'fkBatch', 'idBatchItem', 'Descricao', 'VlRef', 'InfoReq', 'ArqReq'])
    itens = old.prepare_itens(itens)
    if itens.idBatchItem.duplicated().any():
        raise ValueError('Chave de item duplicada; revisar antes de estimar.')
    unused_lotes = int((~lotes.idBatch.isin(itens.fkBatch)).sum())
    lotes = lotes.loc[lotes.idBatch.isin(itens.fkBatch)].copy()
    if lotes.idBatch.duplicated().any():
        raise ValueError('Chave de lote vinculada a item duplicada; revisar antes de estimar.')
    log('Relacionando itens, lotes e processos com validacao de cardinalidade')
    lp = lotes.merge(proc, left_on='fkProcess', right_on='idProcess', how='left', validate='many_to_one')
    base = itens.merge(lp, left_on='fkBatch', right_on='idBatch', how='left', validate='many_to_one')
    del itens, lotes, proc, lp
    gc.collect()
    base = old.add_tipo_lote(base)
    base = old.add_target(base)
    audit = {
        'itens_base_completa': len(base),
        'processos_identificados_base_completa': int(base.idProcess.nunique()),
        'processos_duplicados_removidos_regra_legada': proc_dup,
        'linhas_lotes_sem_itens_ignoradas': unused_lotes,
        'itens_elegiveis': int(base.insucesso.notna().sum()),
        'insucessos_elegiveis': int(base.insucesso.eq(1).sum()),
        'status_elegiveis': base.loc[base.insucesso.notna(), 'StatusLoteNorm'].value_counts().to_dict(),
    }
    # Falta de vinculo impede controlar caracteristicas do processo: exclusao explicita.
    eligible = base.insucesso.notna()
    linked = base.idProcess.notna() & ~base.idProcess.astype(str).isin(['', 'nan', 'None'])
    audit['excluidos_sem_processo'] = int((eligible & ~linked).sum())
    audit['insucessos_excluidos_sem_processo'] = int(base.loc[eligible & ~linked, 'insucesso'].sum())
    base = base.loc[eligible & linked].copy().reset_index(drop=True)
    base['valor_cotado_log'] = np.log1p(base.VlRef.clip(lower=0))
    audit['valores_cotados_negativos_tratados_como_zero'] = int(base.VlRef.lt(0).sum())
    for c in ['MEExclusivo','ExclMe','RegExcl','Regional','LocalExc','ExclLocal','InfoReq','ArqReq']:
        base[c] = pd.to_numeric(base[c], errors='coerce').fillna(0).astype('int8')
    me = base.MEExclusivo.eq(1) | base.ExclMe.eq(1)
    regional = base.RegExcl.eq(1) | base.Regional.eq(1) | base.LocalExc.eq(1) | base.ExclLocal.eq(1)
    base['grupo_exclusividade_regionalidade'] = np.select(
        [me & regional, me, regional], ['ME_EPP_E_REGIONAL_LOCAL','SOMENTE_ME_EPP','SOMENTE_REGIONAL_LOCAL'],
        default='SEM_CRITERIO')
    model_dir = Path(args.model_dir)
    vect = joblib.load(model_dir/'tfidf_vectorizer_familias_textuais.joblib')
    km = joblib.load(model_dir/'kmeans_familias_textuais.joblib')
    labels = np.empty(len(base), dtype='int16')
    for st in range(0, len(base), 25000):
        en = min(st+25000, len(base))
        labels[st:en] = km.predict(vect.transform(base.Descricao.iloc[st:en].fillna('').astype(str)))
        if st % 100000 == 0:
            log(f'Familias textuais originais: {en}/{len(base)}')
    base['familia_textual_ml'] = [f'FAMILIA_{x:02d}' for x in labels]
    terms = np.array(vect.get_feature_names_out())
    csv(pd.DataFrame([{'familia_textual_ml':f'FAMILIA_{i:02d}',
                      'termos_centrais':' / '.join(terms[np.argsort(center)[-5:]][::-1])}
                     for i, center in enumerate(km.cluster_centers_)]), out/'familias_textuais.csv')
    keep = ['idProcess','idBatch','idBatchItem','insucesso','UF','Regiao','Modalidade',
            'TipoDisputa','TipoEncerramento','TipoLance','tipo_lote_calc',
            'grupo_exclusividade_regionalidade','familia_textual_ml','valor_cotado_log',
            'descricao_tam_caracteres','descricao_qtd_palavras','qtd_itens_lote','InfoReq','ArqReq']
    base = base[keep].copy()
    audit.update(itens_modelo=len(base), insucessos_modelo=int(base.insucesso.sum()),
                 processos_modelo=int(base.idProcess.nunique()), lotes_modelo=int(base.idBatch.nunique()))
    save_json(audit, out/'auditoria_base.json')
    log(f'Base de inferencia: {len(base)} itens, {int(base.insucesso.sum())} insucessos')
    base.to_csv(out/'base_inferencia_privada.csv.gz', sep=';', index=False, encoding='utf-8-sig')
    return base


def design(base, territory, out):
    numeric = {
        'valor_cotado_log': ('log(1 + valor cotado)', 1.),
        'descricao_tam_caracteres': ('100 caracteres adicionais', 100.),
        'descricao_qtd_palavras': ('10 palavras adicionais', 10.),
        'qtd_itens_lote': ('10 itens adicionais no lote', 10.),
        'InfoReq': ('exigencia de informacao: 1 versus 0', 1.),
        'ArqReq': ('exigencia de arquivo: 1 versus 0', 1.),
    }
    cats = [territory, 'Modalidade','TipoDisputa','TipoEncerramento','TipoLance',
            'tipo_lote_calc','grupo_exclusividade_regionalidade','familia_textual_ml']
    preferred = {'Regiao':'Sudeste','UF':'SP','Modalidade':'PREGÃO ELETRÔNICO',
                 'TipoDisputa':'MENOR LANCE','TipoEncerramento':'ABERTO','TipoLance':'UNITÁRIO',
                 'tipo_lote_calc':'Lote unitário','grupo_exclusividade_regionalidade':'SEM_CRITERIO'}
    columns = [np.ones(len(base))]
    info = [{'termo':'const','variavel':'Intercepto','categoria':'','referencia':'',
             'unidade':'intercepto; numericas centradas','escala_numerica':1.,'centro':0.,'imputados':0}]
    for c, (unit, scale) in numeric.items():
        values = pd.to_numeric(base[c], errors='coerce').replace([np.inf,-np.inf], np.nan)
        missing = int(values.isna().sum())
        median = float(values.median())
        if not np.isfinite(median):
            raise ValueError(f'Variavel inteiramente ausente: {c}')
        values = values.fillna(median).to_numpy(dtype=float)
        center = float(values.mean())
        columns.append((values-center)/scale)
        info.append(dict(termo=c,variavel=c,categoria='',referencia='',unidade=unit,
                         escala_numerica=scale,centro=center,imputados=missing,mediana_imputacao=median))
    counts = []
    for c in cats:
        values = base[c].fillna('NAO_INFORMADO').astype(str).str.strip().replace('', 'NAO_INFORMADO')
        if c == 'familia_textual_ml':
            frequencies = values.value_counts()
            rare = frequencies[frequencies < 100].index
            mapping = pd.DataFrame({'familia_original':frequencies.index,
                                    'n_itens':frequencies.values})
            mapping['familia_modelo'] = mapping.familia_original.where(
                ~mapping.familia_original.isin(rare), 'FAMILIAS_RARAS_MENOS_100')
            csv(mapping,out/'agrupamento_familias_raras.csv')
            values = values.where(~values.isin(rare),'FAMILIAS_RARAS_MENOS_100')
        tab = pd.DataFrame({'categoria':values,'y':base.insucesso,'processo':base.idProcess}).groupby('categoria').agg(
            itens=('y','size'),insucessos=('y','sum'),processos=('processo','nunique')).reset_index()
        tab['sucessos'] = tab.itens-tab.insucessos
        tab.insert(0,'variavel',c)
        counts.append(tab)
        # Falha explicita: nao elimina grupos por desfecho nem produz p-valores invalidos.
        if ((tab.insucessos == 0) | (tab.sucessos == 0)).any():
            csv(pd.concat(counts), out/f'contagens_{territory}.csv')
            raise ValueError(f'Categoria com separacao em {c}; ver contagens e revisar especificacao.')
        levels = sorted(values.unique())
        ref = preferred.get(c, values.value_counts().index[0])
        if ref not in levels:
            raise ValueError(f'Referencia inexistente: {c}={ref}')
        for level in levels:
            if level == ref:
                continue
            columns.append(values.eq(level).to_numpy(dtype=float))
            info.append(dict(termo=f'{c}[{level}]',variavel=c,categoria=level,referencia=ref,
                             unidade='categoria versus referencia',escala_numerica=1.,centro=0.,imputados=0))
    csv(pd.concat(counts,ignore_index=True), out/f'contagens_{territory}.csv')
    X = np.column_stack(columns)
    del columns
    metadata = pd.DataFrame(info)
    # Detecta constantes/aliases antes do ajuste, sem depender do desfecho.
    gram = X.T @ X
    norms = np.sqrt(np.diag(gram))
    active = np.flatnonzero(norms > 1e-12)
    corr = gram[np.ix_(active,active)]/np.outer(norms[active],norms[active])
    _, r, piv = scipy.linalg.qr(corr, pivoting=True)
    rank = int(np.sum(np.abs(np.diag(r)) > 1e-10))
    kept = sorted(active[piv[:rank]])
    removed = sorted(set(range(X.shape[1]))-set(kept))
    if removed:
        csv(metadata.iloc[removed],out/f'colunas_redundantes_{territory}.csv')
        raise ValueError('Matriz redundante; revisar explicitamente as colunas identificadas.')
    condition = float(np.linalg.cond(corr))
    csv(metadata, out/f'dicionario_coeficientes_{territory}.csv')
    return X, metadata, condition


def fit(base, territory, out):
    log(f'Montando matriz do modelo por {territory}')
    X, metadata, condition = design(base, territory, out)
    y = base.insucesso.to_numpy(dtype=float)
    groups, group_names = pd.factorize(base.idProcess, sort=True)
    log(f'Logit sem penalizacao: n={len(y)}, p={X.shape[1]}, clusters={len(group_names)}')
    iteration = [0]
    def callback(params):
        iteration[0] += 1
        log(f'{territory}: iteracao {iteration[0]}')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = sm.Logit(y, X, check_rank=False)
        result = model.fit(method='newton', maxiter=100, tol=1e-8, disp=False, callback=callback,
                           cov_type='cluster', cov_kwds={'groups':groups,'use_correction':True},use_t=False)
    converged = bool(result.mle_retvals['converged'])
    if not converged or not np.isfinite(result.bse).all():
        raise RuntimeError('Ajuste nao convergiu ou erros-padrao invalidos.')
    ci = result.conf_int()
    table = metadata.copy()
    table['beta_log_odds'] = result.params
    table['erro_padrao_cluster'] = result.bse
    table['z'] = result.tvalues
    table['p_valor'] = result.pvalues
    table['beta_ic95_inferior'], table['beta_ic95_superior'] = ci[:,0],ci[:,1]
    table['odds_ratio'] = np.exp(result.params)
    table['or_ic95_inferior'], table['or_ic95_superior'] = np.exp(ci[:,0]), np.exp(ci[:,1])
    table['significativo_5pct'] = result.pvalues < .05
    csv(table, out/f'coeficientes_{territory}.csv')
    cov = np.asarray(result.cov_params())
    csv(pd.DataFrame(cov, columns=metadata.termo).assign(termo=metadata.termo),out/f'covariancia_cluster_{territory}.csv')
    joint = []
    for variable in metadata.variavel.unique():
        if variable == 'Intercepto':
            continue
        ix = np.flatnonzero(metadata.variavel.eq(variable))
        b, v = result.params[ix], cov[np.ix_(ix,ix)]
        df = int(np.linalg.matrix_rank(v))
        if df != len(ix):
            raise RuntimeError(f'Covariancia singular no teste conjunto: {variable}')
        w = float(b @ np.linalg.solve(v,b))
        joint.append(dict(variavel=variable,qui_quadrado_wald=w,gl=df,p_valor=float(stats.chi2.sf(w,df))))
    csv(pd.DataFrame(joint), out/f'testes_conjuntos_{territory}.csv')
    p = result.predict()
    w = p*(1-p)
    # Diagnosticos em blocos para limitar memoria.
    info = np.zeros((X.shape[1],X.shape[1]))
    for start in range(0,len(y),25000):
        xx = X[start:start+25000]
        info += xx.T @ (w[start:start+25000,None]*xx)
    eig = np.linalg.eigvalsh(info)
    metrics = dict(modelo=f'Logit_{territory}',n=len(y),eventos=int(y.sum()),
                   prevalencia=float(y.mean()),processos=len(group_names),parametros=X.shape[1],
                   convergiu=converged,iteracoes=int(result.mle_retvals['iterations']),
                   log_verossimilhanca=float(result.llf),aic=float(result.aic),bic=float(result.bic),
                   pseudo_r2_mcfadden=float(result.prsquared),
                   max_score_por_observacao=float(np.max(np.abs(model.score(result.params)))/len(y)),
                   menor_autovalor_informacao=float(eig.min()),condicao_matriz_correlacao=condition,
                   probabilidade_min=float(p.min()),probabilidade_max=float(p.max()),
                   maior_beta_absoluto=float(np.abs(result.params).max()),
                   covariancia='cluster por idProcess, correcao finita, testes z/Wald assintoticos',
                   avisos=[str(x.message) for x in caught])
    save_json(metrics, out/f'diagnostico_{territory}.json')
    (out/f'summary_{territory}.txt').write_text(result.summary(xname=metadata.termo.tolist()).as_text(),encoding='utf-8')
    log(f'Modelo {territory} concluido: convergiu={converged}; LL={result.llf:.3f}')
    del model, result, X
    gc.collect()
    return table


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',default=str(ROOT/'entrada'))
    parser.add_argument('--model-dir',default=str(ROOT/'modelos'))
    parser.add_argument('--output-dir',default=str(ROOT/'resultados_inferencia'))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--reuse-base',action='store_true')
    parser.add_argument('--territories',nargs='+',choices=['Regiao','UF'],default=['Regiao','UF'])
    args=parser.parse_args()
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        cache=out/'base_inferencia_privada.csv.gz'
        if args.reuse_base:
            base=pd.read_csv(cache,sep=';',encoding='utf-8-sig',dtype={'idProcess':str,'idBatch':str,'idBatchItem':str})
        else:
            base=prepare(args,out)
        if args.prepare_only:
            return
        for territory in args.territories:
            fit(base,territory,out)
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  scipy=scipy.__version__,statsmodels=statsmodels.__version__,sklearn=sklearn.__version__)
    versions['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save_json(versions,out/'versoes.json')
    log('Execucao concluida')


if __name__ == '__main__':
    main()
