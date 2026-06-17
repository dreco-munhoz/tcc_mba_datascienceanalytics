#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algoritmo complementar do TCC - comparativos, estatística e machine learning.
Entrada recomendada: base_analitica_itens_completa.csv.gz gerada pelo algoritmo de integração.
Também funciona como etapa 02 do pipeline: depois de integrar os CSVs, rode este arquivo.
"""
from __future__ import annotations
import argparse, csv, json, time, warnings, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
try:
    import joblib
except Exception:
    joblib=None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt=None

class Progress:
    def __init__(self, out: Path):
        self.out=out; out.mkdir(parents=True, exist_ok=True); self.start=time.time()
        self.status=out/'status_processamento.json'; self.logcsv=out/'log_processamento.csv'; self.log=out/'execucao_algoritmo.log'
        with self.logcsv.open('w', newline='', encoding='utf-8-sig') as f: csv.writer(f, delimiter=';').writerow(['timestamp','etapa','status','mensagem','linhas','elapsed_s'])
    def update(self, etapa, status='andamento', msg='', linhas=None):
        elapsed=round(time.time()-self.start,2); ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        payload={'timestamp':ts,'etapa_atual':etapa,'status':status,'mensagem':msg,'linhas':None if linhas is None else int(linhas),'tempo_decorrido_segundos':elapsed,'tempo_decorrido_minutos':round(elapsed/60,2)}
        self.status.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        with self.logcsv.open('a', newline='', encoding='utf-8-sig') as f: csv.writer(f, delimiter=';').writerow([ts,etapa,status,msg,payload['linhas'],elapsed])
        line=f'[{ts}] {status.upper()} | {etapa} | {msg} | linhas={payload["linhas"]} | {elapsed}s'
        with self.log.open('a', encoding='utf-8') as f: f.write(line+'\n')
        print(line, flush=True)

def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True); df.to_csv(path, sep=';', index=False, encoding='utf-8-sig')

def read_base(path: Path, p: Progress) -> pd.DataFrame:
    p.update('leitura_base','andamento',f'Lendo {path.name}')
    usecols=['idProcess','UF','Regiao','Cidade','Modalidade','TipoDisputa','TipoEncerramento','StatusProcesso','Edital','DtPublicacao','DataDisputa','fkProcess','idBatch','StatusLote','TipoLance','TituloLote','qtd_itens_lote','tipo_lote_calc','idBatchItem','Descricao','Unidade','VlRef','ValorFinal','dif_pct_valor_final_estimado','PropostasLote','MEExclusivo','RegExcl','LocalExc','ExclMe','Regional','ExclLocal','InfoReq','ArqReq','descricao_tam_caracteres','descricao_qtd_palavras','faixa_valor_estimado','status_item_analise','insucesso','status_final_modelagem','baixa_competitividade_descritiva','participacao_unica_descritiva']
    df=pd.read_csv(path, sep=';', encoding='utf-8-sig', usecols=lambda c: c in usecols, low_memory=False)
    p.update('leitura_base','ok','Base carregada',len(df))
    return df

def prep(df: pd.DataFrame, p: Progress) -> pd.DataFrame:
    p.update('preparacao','andamento','Padronizando variáveis',len(df))
    ren={'UF':'uf','Regiao':'regiao','Cidade':'cidade','Modalidade':'modalidade','TipoDisputa':'tipo_disputa','TipoEncerramento':'tipo_encerramento','StatusProcesso':'status_processo','StatusLote':'status_lote','TipoLance':'tipo_lance','TituloLote':'titulo_lote','Descricao':'descricao_item','Unidade':'unidade','VlRef':'valor_cotado','ValorFinal':'valor_homologado','PropostasLote':'numero_propostas'}
    df=df.rename(columns=ren)
    for c in ['valor_cotado','valor_homologado','numero_propostas','dif_pct_valor_final_estimado','qtd_itens_lote','descricao_tam_caracteres','descricao_qtd_palavras']:
        if c in df: df[c]=pd.to_numeric(df[c], errors='coerce')
    for c in ['MEExclusivo','RegExcl','LocalExc','ExclMe','Regional','ExclLocal','InfoReq','ArqReq','baixa_competitividade_descritiva','participacao_unica_descritiva']:
        if c in df: df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0).astype('int8')
        else: df[c]=np.int8(0)
    df['insucesso_principal']=pd.to_numeric(df.get('insucesso',0), errors='coerce').fillna(0).astype('int8')
    df['sucesso_principal']=(df.get('status_item_analise','').astype(str).str.upper().eq('SUCESSO')).astype('int8')
    df['elegivel_modelagem']=((df['insucesso_principal']==1)|(df['sucesso_principal']==1)).astype('int8')
    df['numero_propostas']=df['numero_propostas'].fillna(0)
    df['sem_proposta']=(df['numero_propostas']<=0).astype('int8')
    df['participacao_unica']=(df['numero_propostas']==1).astype('int8')
    df['baixa_competitividade']=((df['numero_propostas']>0)&(df['numero_propostas']<=2)).astype('int8')
    df['insucesso_ampliado']=((df['insucesso_principal']==1)|(df['baixa_competitividade']==1)).astype('int8')
    df['valor_cotado_log']=np.log1p(df['valor_cotado'].clip(lower=0)).astype('float32')
    df['valor_homologado_log']=np.log1p(df['valor_homologado'].clip(lower=0)).astype('float32')
    df['diferenca_cotado_homologado']=df['valor_cotado']-df['valor_homologado']
    df['relacao_homologado_cotado']=np.where((df['valor_cotado']>0)&df['valor_homologado'].notna(),df['valor_homologado']/df['valor_cotado'],np.nan).astype('float32')
    df['desconto_pct']=np.where((df['valor_cotado']>0)&df['valor_homologado'].notna(),(df['valor_cotado']-df['valor_homologado'])/df['valor_cotado'],np.nan).astype('float32')
    df['tem_valor_homologado']=df['valor_homologado'].notna().astype('int8')
    df['criterio_me_epp']=((df['MEExclusivo']==1)|(df['ExclMe']==1)).astype('int8')
    df['criterio_regional']=((df['RegExcl']==1)|(df['Regional']==1)).astype('int8')
    df['criterio_local']=((df['LocalExc']==1)|(df['ExclLocal']==1)).astype('int8')
    df['criterio_regional_ou_local']=((df['criterio_regional']==1)|(df['criterio_local']==1)).astype('int8')
    df['grupo_exclusividade_regionalidade']=np.select([(df['criterio_me_epp']==1)&(df['criterio_regional_ou_local']==1),(df['criterio_me_epp']==1),(df['criterio_regional_ou_local']==1)],['ME_EPP_E_REGIONAL_LOCAL','SOMENTE_ME_EPP','SOMENTE_REGIONAL_LOCAL'],default='SEM_CRITERIO')
    df['descricao_item']=df['descricao_item'].fillna('').astype(str)
    if 'descricao_hash' not in df: df['descricao_hash']=pd.util.hash_pandas_object(df['descricao_item'], index=False).astype('uint64')
    desc=df.groupby('descricao_hash', dropna=False).agg(qtd_desc=('idBatchItem','count'), mediana_cotado_desc=('valor_cotado','median'), mediana_homologado_desc=('valor_homologado','median')).reset_index()
    desc.loc[desc['qtd_desc']<5,['mediana_cotado_desc','mediana_homologado_desc']]=np.nan
    df=df.merge(desc,on='descricao_hash',how='left')
    df['desvio_cotado_mediana_desc']=np.where((df['mediana_cotado_desc']>0)&df['valor_cotado'].notna(),(df['valor_cotado']-df['mediana_cotado_desc'])/df['mediana_cotado_desc'],np.nan).astype('float32')
    df['cotado_abaixo_mediana_desc_15pct']=np.where(df['desvio_cotado_mediana_desc'].notna(),(df['desvio_cotado_mediana_desc']<=-0.15).astype('int8'),np.nan)
    df['cotado_acima_mediana_desc_15pct']=np.where(df['desvio_cotado_mediana_desc'].notna(),(df['desvio_cotado_mediana_desc']>=0.15).astype('int8'),np.nan)
    for c in ['uf','regiao','modalidade','tipo_disputa','tipo_encerramento','status_processo','status_lote','tipo_lance','tipo_lote_calc','faixa_valor_estimado','status_item_analise','grupo_exclusividade_regionalidade']:
        if c in df: df[c]=df[c].astype('category')
    p.update('preparacao','ok','Variáveis preparadas',len(df))
    return df

def add_clusters(df, args, p):
    if args.skip_text_clusters:
        df['familia_textual_ml']='NAO_CALCULADO'; df['familia_textual_rotulo']='NAO_CALCULADO'; return df,None,None
    p.update('familias_textuais','andamento','TF-IDF + MiniBatchKMeans',len(df))
    texts=df['descricao_item'].fillna('').astype(str)
    valid=np.flatnonzero((texts.str.len()>0).to_numpy())
    rng=np.random.default_rng(args.random_state)
    sample=rng.choice(valid, size=min(args.max_text_fit_rows,len(valid)), replace=False)
    vect=TfidfVectorizer(max_features=3000,min_df=10,max_df=.95,ngram_range=(1,2),strip_accents='unicode',lowercase=True)
    X=vect.fit_transform(texts.iloc[sample])
    k=min(args.n_text_clusters,max(2,X.shape[0]//800))
    km=MiniBatchKMeans(n_clusters=k, random_state=args.random_state, batch_size=4096, n_init='auto')
    km.fit(X)
    labels=np.full(len(df),-1,dtype='int16')
    for st in range(0,len(df),100000):
        en=min(st+100000,len(df)); labels[st:en]=km.predict(vect.transform(texts.iloc[st:en])); p.update('familias_textuais','andamento',f'aplicado até {en:,}',en)
    df['familia_textual_ml']=pd.Series(labels,index=df.index).map(lambda x:f'FAMILIA_{int(x):02d}' if x>=0 else 'SEM_TEXTO')
    try:
        terms=np.array(vect.get_feature_names_out()); centers=km.cluster_centers_; lab={}
        for i in range(k): lab[f'FAMILIA_{i:02d}']=' / '.join(terms[np.argsort(centers[i])[-5:]][::-1])
        df['familia_textual_rotulo']=df['familia_textual_ml'].map(lab).fillna(df['familia_textual_ml'])
    except Exception: df['familia_textual_rotulo']=df['familia_textual_ml']
    p.update('familias_textuais','ok','Famílias calculadas',len(df)); return df,vect,km

def group_summary(df, cols, min_n=1):
    g=df.groupby(cols, dropna=False, observed=True).agg(itens=('idBatchItem','count'),processos=('idProcess',pd.Series.nunique),insucesso_qtd=('insucesso_principal','sum'),baixa_comp_qtd=('baixa_competitividade','sum'),insucesso_ampliado_qtd=('insucesso_ampliado','sum'),propostas_media=('numero_propostas','mean'),propostas_mediana=('numero_propostas','median'),valor_cotado_medio=('valor_cotado','mean'),valor_cotado_mediano=('valor_cotado','median'),valor_homologado_medio=('valor_homologado','mean'),valor_homologado_mediano=('valor_homologado','median'),desconto_pct_medio=('desconto_pct','mean'),desconto_pct_mediano=('desconto_pct','median'),me_epp_qtd=('criterio_me_epp','sum'),regional_qtd=('criterio_regional','sum'),local_qtd=('criterio_local','sum')).reset_index()
    g=g[g['itens']>=min_n].copy()
    for q,out in [('insucesso_qtd','taxa_insucesso_pct'),('baixa_comp_qtd','taxa_baixa_comp_pct'),('insucesso_ampliado_qtd','taxa_insucesso_ampliado_pct'),('me_epp_qtd','pct_me_epp'),('regional_qtd','pct_regional'),('local_qtd','pct_local')]: g[out]=np.where(g['itens']>0,g[q]/g['itens']*100,np.nan)
    return g.sort_values(['insucesso_qtd','taxa_insucesso_pct','itens'],ascending=[False,False,False])

def generate_tables(df,args,p):
    p.update('estatisticas','andamento','Gerando tabelas',len(df)); tabs={}
    eleg=int(df['elegivel_modelagem'].sum())
    tabs['indicadores_gerais']=pd.DataFrame([{'itens_total_base':len(df),'processos_total':df['idProcess'].nunique(dropna=True),'itens_elegiveis_sucesso_ou_insucesso':eleg,'itens_insucesso_principal':int(df['insucesso_principal'].sum()),'itens_sucesso_principal':int(df['sucesso_principal'].sum()),'taxa_insucesso_principal_pct':float(df['insucesso_principal'].sum()/eleg*100) if eleg else np.nan,'itens_baixa_competitividade':int(df['baixa_competitividade'].sum()),'taxa_baixa_competititividade_pct':float(df['baixa_competitividade'].mean()*100),'itens_insucesso_ampliado':int(df['insucesso_ampliado'].sum()),'valor_cotado_total':float(df['valor_cotado'].sum(skipna=True)),'valor_homologado_total':float(df['valor_homologado'].sum(skipna=True)),'propostas_media':float(df['numero_propostas'].mean()),'propostas_mediana':float(df['numero_propostas'].median()),'gerado_em':datetime.now().strftime('%Y-%m-%d %H:%M:%S')}])
    specs={'resultado_final':['status_lote'],'fase_final_processo':['status_processo'],'faixa_propostas':['faixa_propostas'],'faixa_valor_estimado':['faixa_valor_estimado'],'uf':['uf'],'regiao':['regiao'],'cidade':['uf','cidade'],'modalidade':['modalidade'],'tipo_disputa':['tipo_disputa'],'tipo_lance':['tipo_lance'],'tipo_lote':['tipo_lote_calc'],'exclusividade_regionalidade':['grupo_exclusividade_regionalidade'],'familia_textual':['familia_textual_ml','familia_textual_rotulo']}
    df['faixa_propostas']=pd.cut(df['numero_propostas'],bins=[-0.1,0,1,2,5,10,20,np.inf],labels=['0','1','2','3_a_5','6_a_10','11_a_20','mais_de_20'])
    for name,cols in specs.items():
        tabs[f'tabela_{name}']=group_summary(df,cols,20 if name in ['cidade','familia_textual'] else 1).head(700); p.update('estatisticas','andamento',name,len(tabs[f'tabela_{name}']))
    b=df.copy(); b['desc_curta']=b['descricao_item'].str.slice(0,260)
    top=b.groupby('descricao_hash',dropna=False).agg(descricao_exemplo=('desc_curta','first'),unidade=('unidade','first'),itens=('idBatchItem','count'),processos=('idProcess',pd.Series.nunique),insucesso_qtd=('insucesso_principal','sum'),baixa_comp_qtd=('baixa_competitividade','sum'),propostas_media=('numero_propostas','mean'),valor_cotado_mediano=('valor_cotado','median'),valor_homologado_mediano=('valor_homologado','median'),desconto_pct_mediano=('desconto_pct','median'),me_epp_qtd=('criterio_me_epp','sum'),regional_qtd=('criterio_regional','sum'),local_qtd=('criterio_local','sum')).reset_index()
    top=top[top['itens']>=args.min_desc_group].copy(); top['taxa_insucesso_pct']=top['insucesso_qtd']/top['itens']*100; top['taxa_baixa_comp_pct']=top['baixa_comp_qtd']/top['itens']*100
    tabs['top_itens_descricao_insucesso']=top.sort_values(['insucesso_qtd','taxa_insucesso_pct','itens'],ascending=[False,False,False]).head(args.top_n)
    # termos
    m=df[df['elegivel_modelagem']==1][['descricao_item','insucesso_principal']]
    fail=m[m['insucesso_principal']==1]; succ=m[m['insucesso_principal']==0]
    if len(fail)>50 and len(succ)>50:
        if len(m)>140000: m=pd.concat([fail.sample(min(len(fail),70000),random_state=args.random_state),succ.sample(min(len(succ),70000),random_state=args.random_state)])
        cv=CountVectorizer(max_features=2500,min_df=10,ngram_range=(1,2),strip_accents='unicode',lowercase=True); X=cv.fit_transform(m['descricao_item'].astype(str)); y=m['insucesso_principal'].to_numpy(); terms=np.array(cv.get_feature_names_out()); fc=np.asarray(X[y==1].sum(axis=0)).ravel()+1; sc=np.asarray(X[y==0].sum(axis=0)).ravel()+1; odds=(fc/fc.sum())/(sc/sc.sum())
        tabs['top_termos_associados_insucesso']=pd.DataFrame({'termo':terms,'freq_insucesso':fc.astype(int)-1,'freq_sucesso':sc.astype(int)-1,'razao_associacao_insucesso':odds}).sort_values('razao_associacao_insucesso',ascending=False).head(args.top_n)
    else: tabs['top_termos_associados_insucesso']=pd.DataFrame()
    p.update('estatisticas','ok','Tabelas geradas',len(tabs)); return tabs

def model_sample(df,args):
    m=df[df['elegivel_modelagem']==1].copy()
    if args.max_model_rows and len(m)>args.max_model_rows:
        fail=m[m['insucesso_principal']==1]; succ=m[m['insucesso_principal']==0]; nf=min(len(fail),int(args.max_model_rows*.30)); ns=min(len(succ),args.max_model_rows-nf); m=pd.concat([fail.sample(nf,random_state=args.random_state),succ.sample(ns,random_state=args.random_state)]).sample(frac=1,random_state=args.random_state)
    return m.reset_index(drop=True)

def train_one(m,args,p,name,post=False):
    p.update(f'modelo_{name}','andamento','Treinando',len(m))
    nums=['valor_cotado_log','descricao_tam_caracteres','descricao_qtd_palavras','MEExclusivo','RegExcl','LocalExc','ExclMe','Regional','ExclLocal','criterio_me_epp','criterio_regional','criterio_local','criterio_regional_ou_local','qtd_itens_lote','InfoReq','ArqReq']
    if post: nums+=['numero_propostas','sem_proposta','participacao_unica','baixa_competitividade','valor_homologado_log','tem_valor_homologado','relacao_homologado_cotado','desconto_pct','desvio_cotado_mediana_desc','cotado_abaixo_mediana_desc_15pct','cotado_acima_mediana_desc_15pct']
    nums=[c for c in nums if c in m]
    cats=[c for c in ['uf','regiao','modalidade','tipo_disputa','tipo_encerramento','tipo_lance','tipo_lote_calc','grupo_exclusividade_regionalidade','familia_textual_ml'] if c in m]
    X=m[nums+cats+['descricao_item']].copy(); y=m['insucesso_principal'].astype(int)
    for c in cats: X[c]=X[c].astype(str).fillna('Não informado')
    X['descricao_item']=X['descricao_item'].fillna('').astype(str)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=args.random_state,stratify=y)
    pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler(with_mean=False))]),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=20))]),cats),('txt',TfidfVectorizer(max_features=1800,min_df=5,max_df=.95,ngram_range=(1,2),strip_accents='unicode',lowercase=True),'descricao_item')],sparse_threshold=.3)
    pipe=Pipeline([('preprocessor',pre),('clf',LogisticRegression(max_iter=220,class_weight='balanced',solver='saga',n_jobs=-1,random_state=args.random_state))])
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    try: auc=float(roc_auc_score(yte,pipe.predict_proba(Xte)[:,1]))
    except Exception: auc=np.nan
    met={'model_name':name,'include_post_event_features':post,'linhas_modelo':len(m),'linhas_treino':len(Xtr),'linhas_teste':len(Xte),'prevalencia_insucesso_teste_pct':float(yte.mean()*100),'accuracy':float(accuracy_score(yte,pred)),'precision':float(precision_score(yte,pred,zero_division=0)),'recall':float(recall_score(yte,pred,zero_division=0)),'f1':float(f1_score(yte,pred,zero_division=0)),'auc':auc,'numeric_features':', '.join(nums),'categorical_features':', '.join(cats)}
    cm=pd.DataFrame(confusion_matrix(yte,pred),index=['real_sucesso','real_insucesso'],columns=['pred_sucesso','pred_insucesso']).reset_index().rename(columns={'index':'classe_real'})
    try:
        names=pipe.named_steps['preprocessor'].get_feature_names_out(); co=pipe.named_steps['clf'].coef_.ravel(); coef=pd.DataFrame({'feature':names,'coeficiente':co}); coef['odds_ratio_aprox']=np.exp(np.clip(coef['coeficiente'],-20,20)); coef['abs_coeficiente']=coef['coeficiente'].abs(); coef=coef.sort_values('abs_coeficiente',ascending=False).head(300)
    except Exception as e: coef=pd.DataFrame({'erro':[str(e)]})
    p.update(f'modelo_{name}','ok',f'AUC={auc:.4f} F1={met["f1"]:.4f}',len(m))
    return {'metrics':met,'cm':cm,'coef':coef,'pipe':pipe}

def run_ml(df,args,p):
    if args.skip_ml: return {}
    m=model_sample(df,args); p.update('ml','andamento','Amostra de modelagem',len(m))
    return {'sample':m,'pre_disputa':train_one(m,args,p,'pre_disputa',False),'diagnostico_final':train_one(m,args,p,'diagnostico_final',True)}

def save_outputs(df,tabs,ml,args,p,vect=None,km=None):
    out=Path(args.output_dir); [mkdir(out/x) for x in ['tabelas','modelos','bases','graficos','texto','codigo']]
    for n,t in tabs.items(): write_csv(t,out/'tabelas'/f'{n}.csv')
    metrics=[]
    for name in ['pre_disputa','diagnostico_final']:
        r=ml.get(name,{}) if ml else {}
        if 'metrics' in r: metrics.append(r['metrics']); write_csv(pd.DataFrame([r['metrics']]),out/'modelos'/f'metricas_{name}.csv')
        if 'cm' in r: write_csv(r['cm'],out/'modelos'/f'matriz_confusao_{name}.csv')
        if 'coef' in r: write_csv(r['coef'],out/'modelos'/f'coeficientes_importancia_{name}.csv')
        if joblib and 'pipe' in r: joblib.dump(r['pipe'],out/'modelos'/f'pipeline_{name}.joblib',compress=3)
    if metrics: write_csv(pd.DataFrame(metrics),out/'modelos'/'metricas_modelos_resumo.csv')
    if joblib and vect is not None: joblib.dump(vect,out/'modelos'/'tfidf_vectorizer_familias_textuais.joblib',compress=3); joblib.dump(km,out/'modelos'/'kmeans_familias_textuais.joblib',compress=3)
    keep=['idProcess','uf','regiao','cidade','modalidade','tipo_disputa','tipo_encerramento','status_processo','Edital','DtPublicacao','DataDisputa','idBatch','status_lote','tipo_lance','tipo_lote_calc','idBatchItem','descricao_item','unidade','valor_cotado','valor_homologado','diferenca_cotado_homologado','relacao_homologado_cotado','desconto_pct','numero_propostas','sem_proposta','participacao_unica','baixa_competitividade','insucesso_principal','insucesso_ampliado','MEExclusivo','ExclMe','criterio_me_epp','RegExcl','Regional','criterio_regional','LocalExc','ExclLocal','criterio_local','grupo_exclusividade_regionalidade','descricao_tam_caracteres','descricao_qtd_palavras','qtd_itens_lote','familia_textual_ml','familia_textual_rotulo','mediana_cotado_desc','mediana_homologado_desc','desvio_cotado_mediana_desc','cotado_abaixo_mediana_desc_15pct','cotado_acima_mediana_desc_15pct']
    keep=[c for c in keep if c in df]
    df[keep].to_csv(out/'bases'/'base_analitica_comparativos_completa.csv.gz',sep=';',index=False,encoding='utf-8-sig',compression='gzip')
    df.loc[df['elegivel_modelagem']==1, keep].to_csv(out/'bases'/'base_modelagem_comparativos.csv.gz',sep=';',index=False,encoding='utf-8-sig',compression='gzip')
    if plt:
        def bar(tab,label,val,title,file):
            if tab is None or tab.empty: return
            d=tab.head(15).copy(); d[label]=d[label].astype(str).str.slice(0,50); plt.figure(figsize=(10,max(4,len(d)*.35))); plt.barh(d[label][::-1],d[val][::-1]); plt.title(title); plt.tight_layout(); plt.savefig(out/'graficos'/file,dpi=150); plt.close()
        bar(tabs.get('tabela_uf'),'uf','taxa_insucesso_pct','Taxa de insucesso por UF','taxa_insucesso_uf.png')
        bar(tabs.get('tabela_exclusividade_regionalidade'),'grupo_exclusividade_regionalidade','taxa_insucesso_pct','Taxa de insucesso por critérios','taxa_insucesso_criterios.png')
    ind=tabs['indicadores_gerais'].iloc[0].to_dict()
    texto=f"""Texto-base dos resultados preliminares\n\nA base analítica foi organizada em nível de item e incluiu número de propostas, valor cotado, valor homologado, resultado final, critérios de exclusividade ME/EPP, regionalidade, localidade, modalidade, UF, município, tipo de lance e descrição dos itens.\n\nForam identificados {int(ind.get('itens_total_base',0)):,} itens, dos quais {int(ind.get('itens_elegiveis_sucesso_ou_insucesso',0)):,} apresentaram desfecho elegível para a modelagem principal. O insucesso principal, definido como item deserto ou fracassado, ocorreu em {int(ind.get('itens_insucesso_principal',0)):,} itens, com taxa de {float(ind.get('taxa_insucesso_principal_pct',0)):.2f}% entre os itens elegíveis.\n\nA baixa competitividade foi mantida como indicador complementar, mensurada por itens com até duas propostas. O valor cotado e o valor homologado foram comparados por meio da diferença absoluta, relação homologado/cotado e percentual de desconto. Também foram geradas tabelas por critérios de regionalidade, localidade e exclusividade ME/EPP, permitindo observar se esses recortes apresentam taxas distintas de insucesso.\n\nForam aplicadas técnicas de aprendizado de máquina em duas abordagens: modelo pré-disputa, sem variáveis posteriores ao certame, e modelo diagnóstico final, com número de propostas, valor homologado e métricas de preço. Os resultados devem ser interpretados como associações estatísticas e operacionais, não como causalidade direta.\n"""
    (out/'texto'/'texto_base_resultados_preliminares.txt').write_text(texto,encoding='utf-8')
    shutil.copy2(Path(__file__).resolve(),out/'codigo'/'algoritmo_tcc_comparativos_ml.py')
    (out/'requirements.txt').write_text('pandas\nnumpy\nscikit-learn\nmatplotlib\njoblib\n',encoding='utf-8')
    (out/'README.md').write_text(f"""# Pacote TCC - Comparativos e Machine Learning\n\n## Execução\n\n```bash\npip install -r requirements.txt\npython codigo/algoritmo_tcc_comparativos_ml.py --base bases/base_analitica_itens_completa.csv.gz --output-dir saida_tcc_comparativos\n```\n\n## Acompanhar processamento\n\n```bash\ntail -f saida_tcc_comparativos/execucao_algoritmo.log\n```\n\nO algoritmo gera atualização contínua em `status_processamento.json`, `log_processamento.csv` e `execucao_algoritmo.log`.\n\n## Saídas principais\n- tabelas/indicadores_gerais.csv\n- tabelas/tabela_faixa_propostas.csv\n- tabelas/tabela_exclusividade_regionalidade.csv\n- tabelas/top_itens_descricao_insucesso.csv\n- modelos/metricas_modelos_resumo.csv\n- modelos/coeficientes_importancia_pre_disputa.csv\n- modelos/coeficientes_importancia_diagnostico_final.csv\n- bases/base_analitica_comparativos_completa.csv.gz\n- texto/texto_base_resultados_preliminares.txt\n\n## Tempo estimado\nNa base atual, a leitura e geração de estatísticas costuma levar poucos minutos. A clusterização textual e o treino dos modelos variam conforme o computador e a quantidade de linhas usadas em `--max-model-rows`.\n""", encoding='utf-8')
    p.update('salvar','ok','Arquivos salvos',len(df))

def parse():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default='/mnt/data/saida_tcc_resultados/bases/base_analitica_itens_completa.csv.gz'); ap.add_argument('--output-dir',default='/mnt/data/saida_tcc_comparativos'); ap.add_argument('--max-model-rows',type=int,default=90000); ap.add_argument('--max-text-fit-rows',type=int,default=50000); ap.add_argument('--n-text-clusters',type=int,default=22); ap.add_argument('--top-n',type=int,default=80); ap.add_argument('--min-desc-group',type=int,default=5); ap.add_argument('--random-state',type=int,default=42); ap.add_argument('--skip-ml',action='store_true'); ap.add_argument('--skip-text-clusters',action='store_true'); return ap.parse_args()
def main():
    args=parse(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); p=Progress(out); p.update('inicio','andamento','iniciando')
    df=read_base(Path(args.base),p); df=prep(df,p); df,vect,km=add_clusters(df,args,p); tabs=generate_tables(df,args,p); ml=run_ml(df,args,p); save_outputs(df,tabs,ml,args,p,vect,km); p.update('fim','ok','concluído',len(df)); return 0
if __name__=='__main__': raise SystemExit(main())
