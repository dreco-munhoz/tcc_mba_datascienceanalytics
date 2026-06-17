#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algoritmo complementar do TCC - comparativos, estatística e machine learning.

Correções incluídas nesta versão:
1. RESULTADO FINAL passa a ser tratado como sucesso, junto com HOMOLOGADO e ADJUDICADO.
2. A quantidade de propostas NÃO é usada no modelo pré-disputa, pois DESERTO implica ausência de propostas.
3. A quantidade de propostas fica apenas como diagnóstico/descrição e no modelo diagnóstico final.
4. Foi adicionada comparação de preço: valor cotado do item versus mediana de valor homologado
   de itens semelhantes com sucesso, por descrição normalizada e por família textual.
5. Foram adicionadas tabelas próprias para deserto, fracassado, baixa competitividade,
   referência de preço, critérios ME/EPP, regionalidade/localidade e resultado final.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


STATUS_SUCESSO = {"HOMOLOGADO", "ADJUDICADO", "RESULTADO FINAL"}
STATUS_INSUCESSO = {"DESERTO", "FRACASSADO"}


class Progress:
    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.start = time.time()
        self.status = out / "status_processamento.json"
        self.logcsv = out / "log_processamento.csv"
        self.log = out / "execucao_algoritmo.log"
        with self.logcsv.open("w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f, delimiter=";").writerow(
                ["timestamp", "etapa", "status", "mensagem", "linhas", "elapsed_s"]
            )

    def update(self, etapa: str, status: str = "andamento", msg: str = "", linhas: Optional[int] = None):
        elapsed = round(time.time() - self.start, 2)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "timestamp": ts,
            "etapa_atual": etapa,
            "status": status,
            "mensagem": msg,
            "linhas": None if linhas is None else int(linhas),
            "tempo_decorrido_segundos": elapsed,
            "tempo_decorrido_minutos": round(elapsed / 60, 2),
        }
        self.status.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.logcsv.open("a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f, delimiter=";").writerow([ts, etapa, status, msg, payload["linhas"], elapsed])
        line = f"[{ts}] {status.upper()} | {etapa} | {msg} | linhas={payload['linhas']} | {elapsed}s"
        with self.log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    txt = str(value).strip().upper()
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def classify_price_deviation(x) -> str:
    if pd.isna(x):
        return "SEM_REFERENCIA"
    if x < -15:
        return "COTADO_ABAIXO_DA_REFERENCIA"
    if x > 15:
        return "COTADO_ACIMA_DA_REFERENCIA"
    return "COTADO_PROXIMO_DA_REFERENCIA"


def read_base(path: Path, p: Progress) -> pd.DataFrame:
    p.update("leitura_base", "andamento", f"Lendo {path.name}")
    usecols = [
        "idProcess", "UF", "Regiao", "Cidade", "Modalidade", "TipoDisputa", "TipoEncerramento",
        "StatusProcesso", "Edital", "DtPublicacao", "DataDisputa", "fkProcess", "idBatch", "StatusLote",
        "TipoLance", "TituloLote", "qtd_itens_lote", "tipo_lote_calc", "idBatchItem", "Descricao",
        "Unidade", "VlRef", "ValorFinal", "dif_pct_valor_final_estimado", "PropostasLote",
        "MEExclusivo", "RegExcl", "LocalExc", "ExclMe", "Regional", "ExclLocal", "InfoReq", "ArqReq",
        "descricao_tam_caracteres", "descricao_qtd_palavras", "faixa_valor_estimado", "status_item_analise",
        "insucesso", "status_final_modelagem", "baixa_competitividade_descritiva", "participacao_unica_descritiva",
    ]
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=lambda c: c in usecols, low_memory=False)
    p.update("leitura_base", "ok", "Base carregada", len(df))
    return df


def prep(df: pd.DataFrame, p: Progress) -> pd.DataFrame:
    p.update("preparacao", "andamento", "Padronizando variáveis e corrigindo alvo", len(df))
    ren = {
        "UF": "uf",
        "Regiao": "regiao",
        "Cidade": "cidade",
        "Modalidade": "modalidade",
        "TipoDisputa": "tipo_disputa",
        "TipoEncerramento": "tipo_encerramento",
        "StatusProcesso": "status_processo",
        "StatusLote": "status_lote",
        "TipoLance": "tipo_lance",
        "TituloLote": "titulo_lote",
        "Descricao": "descricao_item",
        "Unidade": "unidade",
        "VlRef": "valor_cotado",
        "ValorFinal": "valor_homologado",
        "PropostasLote": "numero_propostas",
    }
    df = df.rename(columns=ren)

    for col in ["status_lote", "status_processo", "modalidade", "tipo_disputa", "tipo_encerramento", "tipo_lance"]:
        if col not in df.columns:
            df[col] = ""

    df["resultado_final_normalizado"] = df["status_lote"].map(normalize_text)

    numeric_cols = [
        "valor_cotado", "valor_homologado", "numero_propostas", "dif_pct_valor_final_estimado",
        "qtd_itens_lote", "descricao_tam_caracteres", "descricao_qtd_palavras",
    ]
    df = to_num(df, numeric_cols)

    flag_cols = [
        "MEExclusivo", "RegExcl", "LocalExc", "ExclMe", "Regional", "ExclLocal", "InfoReq", "ArqReq",
        "baixa_competitividade_descritiva", "participacao_unica_descritiva",
    ]
    for c in flag_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")
        else:
            df[c] = np.int8(0)

    # Correção principal: RESULTADO FINAL passa a ser sucesso.
    df["alvo_deserto"] = df["resultado_final_normalizado"].eq("DESERTO").astype("int8")
    df["alvo_fracassado"] = df["resultado_final_normalizado"].eq("FRACASSADO").astype("int8")
    df["alvo_resultado_final_sucesso"] = df["resultado_final_normalizado"].eq("RESULTADO FINAL").astype("int8")
    df["insucesso_principal"] = df["resultado_final_normalizado"].isin(STATUS_INSUCESSO).astype("int8")
    df["sucesso_principal"] = df["resultado_final_normalizado"].isin(STATUS_SUCESSO).astype("int8")
    df["elegivel_modelagem"] = ((df["insucesso_principal"] == 1) | (df["sucesso_principal"] == 1)).astype("int8")

    df["numero_propostas"] = df["numero_propostas"].fillna(0)
    df["sem_proposta"] = (df["numero_propostas"] <= 0).astype("int8")
    df["participacao_unica"] = (df["numero_propostas"] == 1).astype("int8")
    df["baixa_competitividade"] = ((df["numero_propostas"] > 0) & (df["numero_propostas"] <= 2)).astype("int8")
    df["alvo_fracassado_com_proposta"] = ((df["alvo_fracassado"] == 1) & (df["numero_propostas"] > 0)).astype("int8")
    df["insucesso_ampliado"] = ((df["insucesso_principal"] == 1) | (df["baixa_competitividade"] == 1)).astype("int8")

    df["valor_cotado_log"] = np.log1p(df["valor_cotado"].clip(lower=0)).astype("float32")
    df["valor_homologado_log"] = np.log1p(df["valor_homologado"].clip(lower=0)).astype("float32")
    df["diferenca_cotado_homologado"] = df["valor_cotado"] - df["valor_homologado"]
    df["relacao_homologado_cotado"] = np.where(
        (df["valor_cotado"] > 0) & df["valor_homologado"].notna(),
        df["valor_homologado"] / df["valor_cotado"],
        np.nan,
    ).astype("float32")
    df["desconto_pct"] = np.where(
        (df["valor_cotado"] > 0) & df["valor_homologado"].notna(),
        (df["valor_cotado"] - df["valor_homologado"]) / df["valor_cotado"] * 100,
        np.nan,
    ).astype("float32")
    df["tem_valor_homologado"] = df["valor_homologado"].notna().astype("int8")

    df["criterio_me_epp"] = ((df["MEExclusivo"] == 1) | (df["ExclMe"] == 1)).astype("int8")
    df["criterio_regional"] = ((df["RegExcl"] == 1) | (df["Regional"] == 1)).astype("int8")
    df["criterio_local"] = ((df["LocalExc"] == 1) | (df["ExclLocal"] == 1)).astype("int8")
    df["criterio_regional_ou_local"] = ((df["criterio_regional"] == 1) | (df["criterio_local"] == 1)).astype("int8")
    df["grupo_exclusividade_regionalidade"] = np.select(
        [
            (df["criterio_me_epp"] == 1) & (df["criterio_regional_ou_local"] == 1),
            (df["criterio_me_epp"] == 1),
            (df["criterio_regional_ou_local"] == 1),
        ],
        ["ME_EPP_E_REGIONAL_LOCAL", "SOMENTE_ME_EPP", "SOMENTE_REGIONAL_LOCAL"],
        default="SEM_CRITERIO",
    )

    df["descricao_item"] = df.get("descricao_item", "").fillna("").astype(str)
    df["descricao_norm"] = df["descricao_item"].map(normalize_text)
    df["descricao_hash"] = pd.util.hash_pandas_object(df["descricao_norm"], index=False).astype("uint64")

    # Referência de preço por descrição: usa apenas itens bem-sucedidos com valor homologado válido.
    ref_success = df[
        (df["sucesso_principal"] == 1)
        & df["valor_homologado"].notna()
        & (df["valor_homologado"] > 0)
    ].copy()

    desc_count = df.groupby("descricao_hash", dropna=False).agg(qtd_desc=("idBatchItem", "count")).reset_index()
    if len(ref_success) > 0:
        desc_ref = (
            ref_success.groupby("descricao_hash", dropna=False)
            .agg(
                qtd_ref_homologados_desc=("idBatchItem", "count"),
                mediana_homologado_ref_desc=("valor_homologado", "median"),
                q1_homologado_ref_desc=("valor_homologado", lambda s: s.quantile(0.25)),
                q3_homologado_ref_desc=("valor_homologado", lambda s: s.quantile(0.75)),
                mediana_cotado_ref_desc=("valor_cotado", "median"),
            )
            .reset_index()
        )
    else:
        desc_ref = pd.DataFrame({"descricao_hash": []})

    df = df.merge(desc_count, on="descricao_hash", how="left")
    df = df.merge(desc_ref, on="descricao_hash", how="left")
    mask_desc_ref_fraca = df["qtd_ref_homologados_desc"].fillna(0) < 5
    for c in ["mediana_homologado_ref_desc", "q1_homologado_ref_desc", "q3_homologado_ref_desc", "mediana_cotado_ref_desc"]:
        if c in df.columns:
            df.loc[mask_desc_ref_fraca, c] = np.nan

    df["desvio_cotado_vs_ref_homologado_desc_pct"] = np.where(
        (df["mediana_homologado_ref_desc"] > 0) & df["valor_cotado"].notna(),
        (df["valor_cotado"] - df["mediana_homologado_ref_desc"]) / df["mediana_homologado_ref_desc"] * 100,
        np.nan,
    ).astype("float32")
    df["classificacao_preco_ref_desc"] = df["desvio_cotado_vs_ref_homologado_desc_pct"].apply(classify_price_deviation)

    # Aliases mantidos por compatibilidade com saídas anteriores.
    df["mediana_homologado_desc"] = df["mediana_homologado_ref_desc"]
    df["desvio_cotado_mediana_desc"] = df["desvio_cotado_vs_ref_homologado_desc_pct"] / 100
    df["cotado_abaixo_mediana_desc_15pct"] = np.where(
        df["desvio_cotado_vs_ref_homologado_desc_pct"].notna(),
        (df["desvio_cotado_vs_ref_homologado_desc_pct"] <= -15).astype("int8"),
        np.nan,
    )
    df["cotado_acima_mediana_desc_15pct"] = np.where(
        df["desvio_cotado_vs_ref_homologado_desc_pct"].notna(),
        (df["desvio_cotado_vs_ref_homologado_desc_pct"] >= 15).astype("int8"),
        np.nan,
    )

    for c in [
        "uf", "regiao", "modalidade", "tipo_disputa", "tipo_encerramento", "status_processo", "status_lote",
        "tipo_lance", "tipo_lote_calc", "faixa_valor_estimado", "status_item_analise",
        "grupo_exclusividade_regionalidade", "resultado_final_normalizado", "classificacao_preco_ref_desc",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    p.update("preparacao", "ok", "Variáveis preparadas e RESULTADO FINAL incluído como sucesso", len(df))
    return df


def add_clusters(df: pd.DataFrame, args, p: Progress) -> Tuple[pd.DataFrame, Optional[TfidfVectorizer], Optional[MiniBatchKMeans]]:
    if args.skip_text_clusters:
        df["familia_textual_ml"] = "NAO_CALCULADO"
        df["familia_textual_rotulo"] = "NAO_CALCULADO"
        return df, None, None

    p.update("familias_textuais", "andamento", "TF-IDF + MiniBatchKMeans", len(df))
    texts = df["descricao_item"].fillna("").astype(str)
    valid = np.flatnonzero((texts.str.len() > 0).to_numpy())
    if len(valid) < 20:
        df["familia_textual_ml"] = "SEM_TEXTO"
        df["familia_textual_rotulo"] = "SEM_TEXTO"
        return df, None, None

    rng = np.random.default_rng(args.random_state)
    sample = rng.choice(valid, size=min(args.max_text_fit_rows, len(valid)), replace=False)
    vect = TfidfVectorizer(
        max_features=3000,
        min_df=10,
        max_df=0.95,
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
    )
    X = vect.fit_transform(texts.iloc[sample])
    k = min(args.n_text_clusters, max(2, X.shape[0] // 800))
    km = MiniBatchKMeans(n_clusters=k, random_state=args.random_state, batch_size=4096, n_init="auto")
    km.fit(X)

    labels = np.full(len(df), -1, dtype="int16")
    for st in range(0, len(df), 100000):
        en = min(st + 100000, len(df))
        labels[st:en] = km.predict(vect.transform(texts.iloc[st:en]))
        p.update("familias_textuais", "andamento", f"aplicado até {en:,}", en)

    df["familia_textual_ml"] = pd.Series(labels, index=df.index).map(lambda x: f"FAMILIA_{int(x):02d}" if x >= 0 else "SEM_TEXTO")
    try:
        terms = np.array(vect.get_feature_names_out())
        centers = km.cluster_centers_
        lab = {}
        for i in range(k):
            lab[f"FAMILIA_{i:02d}"] = " / ".join(terms[np.argsort(centers[i])[-5:]][::-1])
        df["familia_textual_rotulo"] = df["familia_textual_ml"].map(lab).fillna(df["familia_textual_ml"])
    except Exception:
        df["familia_textual_rotulo"] = df["familia_textual_ml"]

    p.update("familias_textuais", "ok", "Famílias calculadas", len(df))
    return df, vect, km


def add_family_price_reference(df: pd.DataFrame, p: Progress, min_ref: int = 20) -> pd.DataFrame:
    p.update("referencia_preco_familia", "andamento", "Calculando referência por família textual", len(df))
    ref_success = df[
        (df["sucesso_principal"] == 1)
        & df["valor_homologado"].notna()
        & (df["valor_homologado"] > 0)
        & df["familia_textual_ml"].notna()
    ].copy()

    if len(ref_success) == 0:
        df["qtd_ref_homologados_familia"] = np.nan
        df["mediana_homologado_ref_familia"] = np.nan
        df["desvio_cotado_vs_ref_homologado_familia_pct"] = np.nan
        df["classificacao_preco_ref_familia"] = "SEM_REFERENCIA"
        df["mediana_homologado_ref_preferida"] = df["mediana_homologado_ref_desc"]
        df["nivel_referencia_preco"] = np.where(df["mediana_homologado_ref_desc"].notna(), "DESCRICAO", "SEM_REFERENCIA")
        df["desvio_cotado_vs_ref_preferida_pct"] = df["desvio_cotado_vs_ref_homologado_desc_pct"]
        df["classificacao_preco_ref"] = df["classificacao_preco_ref_desc"]
        return df

    fam_ref = (
        ref_success.groupby("familia_textual_ml", dropna=False)
        .agg(
            qtd_ref_homologados_familia=("idBatchItem", "count"),
            mediana_homologado_ref_familia=("valor_homologado", "median"),
            q1_homologado_ref_familia=("valor_homologado", lambda s: s.quantile(0.25)),
            q3_homologado_ref_familia=("valor_homologado", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    df = df.merge(fam_ref, on="familia_textual_ml", how="left")
    weak = df["qtd_ref_homologados_familia"].fillna(0) < min_ref
    for c in ["mediana_homologado_ref_familia", "q1_homologado_ref_familia", "q3_homologado_ref_familia"]:
        df.loc[weak, c] = np.nan

    df["desvio_cotado_vs_ref_homologado_familia_pct"] = np.where(
        (df["mediana_homologado_ref_familia"] > 0) & df["valor_cotado"].notna(),
        (df["valor_cotado"] - df["mediana_homologado_ref_familia"]) / df["mediana_homologado_ref_familia"] * 100,
        np.nan,
    ).astype("float32")
    df["classificacao_preco_ref_familia"] = df["desvio_cotado_vs_ref_homologado_familia_pct"].apply(classify_price_deviation)

    df["mediana_homologado_ref_preferida"] = df["mediana_homologado_ref_desc"].where(
        df["mediana_homologado_ref_desc"].notna(), df["mediana_homologado_ref_familia"]
    )
    df["nivel_referencia_preco"] = np.select(
        [df["mediana_homologado_ref_desc"].notna(), df["mediana_homologado_ref_familia"].notna()],
        ["DESCRICAO", "FAMILIA_TEXTUAL"],
        default="SEM_REFERENCIA",
    )
    df["desvio_cotado_vs_ref_preferida_pct"] = np.where(
        (df["mediana_homologado_ref_preferida"] > 0) & df["valor_cotado"].notna(),
        (df["valor_cotado"] - df["mediana_homologado_ref_preferida"]) / df["mediana_homologado_ref_preferida"] * 100,
        np.nan,
    ).astype("float32")
    df["classificacao_preco_ref"] = df["desvio_cotado_vs_ref_preferida_pct"].apply(classify_price_deviation)
    for c in ["classificacao_preco_ref_familia", "classificacao_preco_ref", "nivel_referencia_preco"]:
        df[c] = df[c].astype("category")
    p.update("referencia_preco_familia", "ok", "Referência de preço calculada", len(df))
    return df


def group_summary(df: pd.DataFrame, cols: List[str], min_n: int = 1) -> pd.DataFrame:
    g = (
        df.groupby(cols, dropna=False, observed=True)
        .agg(
            itens=("idBatchItem", "count"),
            processos=("idProcess", pd.Series.nunique),
            elegiveis_modelagem=("elegivel_modelagem", "sum"),
            sucesso_qtd=("sucesso_principal", "sum"),
            insucesso_qtd=("insucesso_principal", "sum"),
            deserto_qtd=("alvo_deserto", "sum"),
            fracassado_qtd=("alvo_fracassado", "sum"),
            resultado_final_sucesso_qtd=("alvo_resultado_final_sucesso", "sum"),
            baixa_comp_qtd=("baixa_competitividade", "sum"),
            insucesso_ampliado_qtd=("insucesso_ampliado", "sum"),
            propostas_media=("numero_propostas", "mean"),
            propostas_mediana=("numero_propostas", "median"),
            valor_cotado_medio=("valor_cotado", "mean"),
            valor_cotado_mediano=("valor_cotado", "median"),
            valor_homologado_medio=("valor_homologado", "mean"),
            valor_homologado_mediano=("valor_homologado", "median"),
            desconto_pct_medio=("desconto_pct", "mean"),
            desconto_pct_mediano=("desconto_pct", "median"),
            mediana_ref_preco=("mediana_homologado_ref_preferida", "median"),
            desvio_cotado_ref_mediano_pct=("desvio_cotado_vs_ref_preferida_pct", "median"),
            me_epp_qtd=("criterio_me_epp", "sum"),
            regional_qtd=("criterio_regional", "sum"),
            local_qtd=("criterio_local", "sum"),
        )
        .reset_index()
    )
    g = g[g["itens"] >= min_n].copy()
    denom = g["elegiveis_modelagem"].replace(0, np.nan)
    g["taxa_insucesso_pct"] = g["insucesso_qtd"] / denom * 100
    g["taxa_sucesso_pct"] = g["sucesso_qtd"] / denom * 100
    g["taxa_deserto_pct"] = g["deserto_qtd"] / denom * 100
    g["taxa_fracassado_pct"] = g["fracassado_qtd"] / denom * 100
    g["taxa_baixa_comp_pct"] = g["baixa_comp_qtd"] / g["itens"].replace(0, np.nan) * 100
    g["taxa_insucesso_ampliado_pct"] = g["insucesso_ampliado_qtd"] / g["itens"].replace(0, np.nan) * 100
    g["pct_me_epp"] = g["me_epp_qtd"] / g["itens"].replace(0, np.nan) * 100
    g["pct_regional"] = g["regional_qtd"] / g["itens"].replace(0, np.nan) * 100
    g["pct_local"] = g["local_qtd"] / g["itens"].replace(0, np.nan) * 100
    return g.sort_values(["insucesso_qtd", "taxa_insucesso_pct", "itens"], ascending=[False, False, False])


def generate_tables(df: pd.DataFrame, args, p: Progress) -> Dict[str, pd.DataFrame]:
    p.update("estatisticas", "andamento", "Gerando tabelas", len(df))
    tabs: Dict[str, pd.DataFrame] = {}
    eleg = int(df["elegivel_modelagem"].sum())
    ins = int(df["insucesso_principal"].sum())
    suc = int(df["sucesso_principal"].sum())
    tabs["indicadores_gerais"] = pd.DataFrame(
        [
            {
                "itens_total_base": len(df),
                "processos_total": df["idProcess"].nunique(dropna=True) if "idProcess" in df else np.nan,
                "itens_elegiveis_sucesso_ou_insucesso": eleg,
                "itens_insucesso_principal": ins,
                "itens_sucesso_principal_homologado_adjudicado_resultado_final": suc,
                "itens_desertos": int(df["alvo_deserto"].sum()),
                "itens_fracassados": int(df["alvo_fracassado"].sum()),
                "itens_resultado_final_sucesso": int(df["alvo_resultado_final_sucesso"].sum()),
                "taxa_insucesso_principal_pct": float(ins / eleg * 100) if eleg else np.nan,
                "itens_baixa_competitividade": int(df["baixa_competitividade"].sum()),
                "taxa_baixa_competititividade_pct": float(df["baixa_competitividade"].mean() * 100),
                "itens_insucesso_ampliado": int(df["insucesso_ampliado"].sum()),
                "valor_cotado_total": float(df["valor_cotado"].sum(skipna=True)),
                "valor_homologado_total": float(df["valor_homologado"].sum(skipna=True)),
                "propostas_media": float(df["numero_propostas"].mean()),
                "propostas_mediana": float(df["numero_propostas"].median()),
                "itens_com_referencia_preco": int(df["mediana_homologado_ref_preferida"].notna().sum()),
                "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
    )

    df["faixa_propostas"] = pd.cut(
        df["numero_propostas"],
        bins=[-0.1, 0, 1, 2, 5, 10, 20, np.inf],
        labels=["0", "1", "2", "3_a_5", "6_a_10", "11_a_20", "mais_de_20"],
    )

    specs = {
        "resultado_final": ["resultado_final_normalizado"],
        "fase_final_processo": ["status_processo"],
        "faixa_propostas": ["faixa_propostas"],
        "faixa_valor_estimado": ["faixa_valor_estimado"],
        "uf": ["uf"],
        "regiao": ["regiao"],
        "cidade": ["uf", "cidade"],
        "modalidade": ["modalidade"],
        "tipo_disputa": ["tipo_disputa"],
        "tipo_lance": ["tipo_lance"],
        "tipo_lote": ["tipo_lote_calc"],
        "exclusividade_regionalidade": ["grupo_exclusividade_regionalidade"],
        "familia_textual": ["familia_textual_ml", "familia_textual_rotulo"],
        "preco_referencia": ["classificacao_preco_ref"],
        "preco_referencia_resultado": ["resultado_final_normalizado", "classificacao_preco_ref"],
        "preco_referencia_criterios": ["grupo_exclusividade_regionalidade", "classificacao_preco_ref"],
        "nivel_referencia_preco": ["nivel_referencia_preco"],
    }
    for name, cols in specs.items():
        min_n = 20 if name in ["cidade", "familia_textual"] else 1
        tabs[f"tabela_{name}"] = group_summary(df, cols, min_n).head(1000)
        p.update("estatisticas", "andamento", name, len(tabs[f"tabela_{name}"]))

    b = df.copy()
    b["desc_curta"] = b["descricao_item"].str.slice(0, 280)
    top = (
        b.groupby("descricao_hash", dropna=False)
        .agg(
            descricao_exemplo=("desc_curta", "first"),
            unidade=("unidade", "first"),
            familia_textual=("familia_textual_rotulo", "first"),
            itens=("idBatchItem", "count"),
            processos=("idProcess", pd.Series.nunique),
            elegiveis_modelagem=("elegivel_modelagem", "sum"),
            insucesso_qtd=("insucesso_principal", "sum"),
            deserto_qtd=("alvo_deserto", "sum"),
            fracassado_qtd=("alvo_fracassado", "sum"),
            baixa_comp_qtd=("baixa_competitividade", "sum"),
            propostas_media=("numero_propostas", "mean"),
            valor_cotado_mediano=("valor_cotado", "median"),
            valor_homologado_mediano=("valor_homologado", "median"),
            mediana_ref_preco=("mediana_homologado_ref_preferida", "median"),
            desvio_cotado_ref_mediano_pct=("desvio_cotado_vs_ref_preferida_pct", "median"),
            desconto_pct_mediano=("desconto_pct", "median"),
            me_epp_qtd=("criterio_me_epp", "sum"),
            regional_qtd=("criterio_regional", "sum"),
            local_qtd=("criterio_local", "sum"),
        )
        .reset_index()
    )
    top = top[top["itens"] >= args.min_desc_group].copy()
    top["taxa_insucesso_pct"] = top["insucesso_qtd"] / top["elegiveis_modelagem"].replace(0, np.nan) * 100
    top["taxa_deserto_pct"] = top["deserto_qtd"] / top["elegiveis_modelagem"].replace(0, np.nan) * 100
    top["taxa_fracassado_pct"] = top["fracassado_qtd"] / top["elegiveis_modelagem"].replace(0, np.nan) * 100
    top["taxa_baixa_comp_pct"] = top["baixa_comp_qtd"] / top["itens"].replace(0, np.nan) * 100
    tabs["top_itens_descricao_insucesso"] = top.sort_values(
        ["insucesso_qtd", "taxa_insucesso_pct", "itens"], ascending=[False, False, False]
    ).head(args.top_n)

    # Termos associados: apenas elegíveis, e interpretados de forma exploratória.
    m = df[df["elegivel_modelagem"] == 1][["descricao_item", "insucesso_principal"]]
    fail = m[m["insucesso_principal"] == 1]
    succ = m[m["insucesso_principal"] == 0]
    if len(fail) > 50 and len(succ) > 50:
        if len(m) > 140000:
            m = pd.concat(
                [
                    fail.sample(min(len(fail), 70000), random_state=args.random_state),
                    succ.sample(min(len(succ), 70000), random_state=args.random_state),
                ]
            )
        cv = CountVectorizer(
            max_features=2500,
            min_df=10,
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
        )
        X = cv.fit_transform(m["descricao_item"].astype(str))
        y = m["insucesso_principal"].to_numpy()
        terms = np.array(cv.get_feature_names_out())
        fc = np.asarray(X[y == 1].sum(axis=0)).ravel() + 1
        sc = np.asarray(X[y == 0].sum(axis=0)).ravel() + 1
        odds = (fc / fc.sum()) / (sc / sc.sum())
        tabs["top_termos_associados_insucesso"] = (
            pd.DataFrame(
                {
                    "termo": terms,
                    "freq_insucesso": fc.astype(int) - 1,
                    "freq_sucesso": sc.astype(int) - 1,
                    "razao_associacao_insucesso": odds,
                }
            )
            .sort_values("razao_associacao_insucesso", ascending=False)
            .head(args.top_n)
        )
    else:
        tabs["top_termos_associados_insucesso"] = pd.DataFrame()

    p.update("estatisticas", "ok", "Tabelas geradas", len(tabs))
    return tabs


def model_sample(df: pd.DataFrame, args) -> pd.DataFrame:
    m = df[df["elegivel_modelagem"] == 1].copy()
    # max_model_rows <= 0 significa toda a base.
    if args.max_model_rows and args.max_model_rows > 0 and len(m) > args.max_model_rows:
        fail = m[m["insucesso_principal"] == 1]
        succ = m[m["insucesso_principal"] == 0]
        nf = min(len(fail), max(2, int(args.max_model_rows * 0.30)))
        ns = min(len(succ), max(2, args.max_model_rows - nf))
        m = pd.concat(
            [
                fail.sample(nf, random_state=args.random_state),
                succ.sample(ns, random_state=args.random_state),
            ]
        ).sample(frac=1, random_state=args.random_state)
    return m.reset_index(drop=True)


def train_one(m: pd.DataFrame, args, p: Progress, name: str, post: bool = False) -> Dict[str, object]:
    p.update(f"modelo_{name}", "andamento", "Treinando", len(m))
    nums = [
        "valor_cotado_log", "descricao_tam_caracteres", "descricao_qtd_palavras",
        "MEExclusivo", "RegExcl", "LocalExc", "ExclMe", "Regional", "ExclLocal",
        "criterio_me_epp", "criterio_regional", "criterio_local", "criterio_regional_ou_local",
        "qtd_itens_lote", "InfoReq", "ArqReq",
    ]
    if post:
        nums += [
            "numero_propostas", "sem_proposta", "participacao_unica", "baixa_competitividade",
            "valor_homologado_log", "tem_valor_homologado", "relacao_homologado_cotado", "desconto_pct",
            "desvio_cotado_vs_ref_preferida_pct", "desvio_cotado_vs_ref_homologado_desc_pct",
            "desvio_cotado_vs_ref_homologado_familia_pct", "cotado_abaixo_mediana_desc_15pct",
            "cotado_acima_mediana_desc_15pct",
        ]
    nums = [c for c in nums if c in m]
    cats = [
        c
        for c in [
            "uf", "regiao", "modalidade", "tipo_disputa", "tipo_encerramento", "tipo_lance",
            "tipo_lote_calc", "grupo_exclusividade_regionalidade", "familia_textual_ml",
        ]
        if c in m
    ]
    if post:
        cats += [c for c in ["classificacao_preco_ref", "nivel_referencia_preco"] if c in m]

    X = m[nums + cats + ["descricao_item"]].copy()
    y = m["insucesso_principal"].astype(int)
    if y.nunique() < 2 or len(y) < 100:
        return {"metrics": {"model_name": name, "erro": "base insuficiente"}}

    for c in cats:
        X[c] = X[c].astype(str).fillna("Não informado")
    X["descricao_item"] = X["descricao_item"].fillna("").astype(str)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=args.random_state, stratify=y)
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))]), nums),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cats),
            (
                "txt",
                TfidfVectorizer(
                    max_features=1800,
                    min_df=5,
                    max_df=0.95,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    lowercase=True,
                ),
                "descricao_item",
            ),
        ],
        sparse_threshold=0.3,
    )
    pipe = Pipeline(
        [
            ("preprocessor", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    solver="saga",
                    n_jobs=-1,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    try:
        auc = float(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
    except Exception:
        auc = np.nan
    met = {
        "model_name": name,
        "include_post_event_features": post,
        "observacao_metodologica": "modelo diagnostico; usa variaveis posteriores ao certame" if post else "modelo pre-disputa; nao usa propostas nem valor homologado",
        "linhas_modelo": len(m),
        "linhas_treino": len(Xtr),
        "linhas_teste": len(Xte),
        "prevalencia_insucesso_teste_pct": float(yte.mean() * 100),
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "auc": auc,
        "numeric_features": ", ".join(nums),
        "categorical_features": ", ".join(cats),
    }
    cm = pd.DataFrame(
        confusion_matrix(yte, pred),
        index=["real_sucesso", "real_insucesso"],
        columns=["pred_sucesso", "pred_insucesso"],
    ).reset_index().rename(columns={"index": "classe_real"})
    try:
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
        co = pipe.named_steps["clf"].coef_.ravel()
        coef = pd.DataFrame({"feature": names, "coeficiente": co})
        coef["odds_ratio_aprox"] = np.exp(np.clip(coef["coeficiente"], -20, 20))
        coef["abs_coeficiente"] = coef["coeficiente"].abs()
        coef = coef.sort_values("abs_coeficiente", ascending=False).head(300)
    except Exception as e:
        coef = pd.DataFrame({"erro": [str(e)]})
    p.update(f"modelo_{name}", "ok", f"AUC={auc:.4f} F1={met['f1']:.4f}", len(m))
    return {"metrics": met, "cm": cm, "coef": coef, "pipe": pipe}


def run_ml(df: pd.DataFrame, args, p: Progress) -> Dict[str, object]:
    if args.skip_ml:
        return {}
    m = model_sample(df, args)
    p.update("ml", "andamento", "Amostra de modelagem", len(m))
    return {
        "sample": m,
        "pre_disputa": train_one(m, args, p, "pre_disputa", False),
        "diagnostico_final": train_one(m, args, p, "diagnostico_final", True),
    }


def save_outputs(df: pd.DataFrame, tabs: Dict[str, pd.DataFrame], ml: Dict[str, object], args, p: Progress, vect=None, km=None) -> None:
    out = Path(args.output_dir)
    for sub in ["tabelas", "modelos", "bases", "graficos", "texto", "codigo"]:
        mkdir(out / sub)

    for n, t in tabs.items():
        write_csv(t, out / "tabelas" / f"{n}.csv")

    metrics = []
    for name in ["pre_disputa", "diagnostico_final"]:
        r = ml.get(name, {}) if ml else {}
        if "metrics" in r:
            metrics.append(r["metrics"])
            write_csv(pd.DataFrame([r["metrics"]]), out / "modelos" / f"metricas_{name}.csv")
        if "cm" in r:
            write_csv(r["cm"], out / "modelos" / f"matriz_confusao_{name}.csv")
        if "coef" in r:
            write_csv(r["coef"], out / "modelos" / f"coeficientes_importancia_{name}.csv")
        if joblib and "pipe" in r:
            joblib.dump(r["pipe"], out / "modelos" / f"pipeline_{name}.joblib", compress=3)
    if metrics:
        write_csv(pd.DataFrame(metrics), out / "modelos" / "metricas_modelos_resumo.csv")
    if joblib and vect is not None:
        joblib.dump(vect, out / "modelos" / "tfidf_vectorizer_familias_textuais.joblib", compress=3)
        joblib.dump(km, out / "modelos" / "kmeans_familias_textuais.joblib", compress=3)

    keep = [
        "idProcess", "uf", "regiao", "cidade", "modalidade", "tipo_disputa", "tipo_encerramento",
        "status_processo", "Edital", "DtPublicacao", "DataDisputa", "idBatch", "status_lote",
        "resultado_final_normalizado", "tipo_lance", "tipo_lote_calc", "idBatchItem", "descricao_item",
        "unidade", "valor_cotado", "valor_homologado", "diferenca_cotado_homologado",
        "relacao_homologado_cotado", "desconto_pct", "numero_propostas", "sem_proposta",
        "participacao_unica", "baixa_competitividade", "alvo_deserto", "alvo_fracassado",
        "alvo_fracassado_com_proposta", "alvo_resultado_final_sucesso", "insucesso_principal",
        "sucesso_principal", "elegivel_modelagem", "insucesso_ampliado", "MEExclusivo", "ExclMe",
        "criterio_me_epp", "RegExcl", "Regional", "criterio_regional", "LocalExc", "ExclLocal",
        "criterio_local", "grupo_exclusividade_regionalidade", "descricao_tam_caracteres",
        "descricao_qtd_palavras", "qtd_itens_lote", "familia_textual_ml", "familia_textual_rotulo",
        "qtd_ref_homologados_desc", "mediana_homologado_ref_desc", "qtd_ref_homologados_familia",
        "mediana_homologado_ref_familia", "mediana_homologado_ref_preferida", "nivel_referencia_preco",
        "desvio_cotado_vs_ref_homologado_desc_pct", "desvio_cotado_vs_ref_homologado_familia_pct",
        "desvio_cotado_vs_ref_preferida_pct", "classificacao_preco_ref_desc",
        "classificacao_preco_ref_familia", "classificacao_preco_ref",
    ]
    keep = [c for c in keep if c in df]
    df[keep].to_csv(
        out / "bases" / "base_analitica_comparativos_completa.csv.gz",
        sep=";",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    df.loc[df["elegivel_modelagem"] == 1, keep].to_csv(
        out / "bases" / "base_modelagem_comparativos.csv.gz",
        sep=";",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )

    if plt:
        def bar(tab: Optional[pd.DataFrame], label: str, val: str, title: str, file: str) -> None:
            if tab is None or tab.empty or label not in tab or val not in tab:
                return
            d = tab.head(15).copy()
            d[label] = d[label].astype(str).str.slice(0, 50)
            plt.figure(figsize=(10, max(4, len(d) * 0.35)))
            plt.barh(d[label][::-1], d[val][::-1])
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out / "graficos" / file, dpi=150)
            plt.close()

        bar(tabs.get("tabela_uf"), "uf", "taxa_insucesso_pct", "Taxa de insucesso por UF", "taxa_insucesso_uf.png")
        bar(tabs.get("tabela_exclusividade_regionalidade"), "grupo_exclusividade_regionalidade", "taxa_insucesso_pct", "Taxa de insucesso por critérios", "taxa_insucesso_criterios.png")
        bar(tabs.get("tabela_preco_referencia"), "classificacao_preco_ref", "taxa_insucesso_pct", "Taxa de insucesso por referência de preço", "taxa_insucesso_preco_ref.png")

    ind = tabs["indicadores_gerais"].iloc[0].to_dict()
    texto = f"""Texto-base dos resultados preliminares

A base analítica foi organizada em nível de item e incluiu número de propostas, valor cotado, valor homologado, resultado final, critérios de exclusividade ME/EPP, regionalidade, localidade, modalidade, UF, município, tipo de lance, descrição dos itens e referência interna de preço por itens semelhantes.

Foram identificados {int(ind.get('itens_total_base', 0)):,} itens, dos quais {int(ind.get('itens_elegiveis_sucesso_ou_insucesso', 0)):,} apresentaram desfecho elegível para a modelagem principal. O insucesso principal, definido como item deserto ou fracassado, ocorreu em {int(ind.get('itens_insucesso_principal', 0)):,} itens, com taxa de {float(ind.get('taxa_insucesso_principal_pct', 0)):.2f}% entre os itens elegíveis.

Nesta versão, RESULTADO FINAL foi classificado como sucesso, junto com HOMOLOGADO e ADJUDICADO. A baixa competitividade foi mantida como indicador complementar, mensurada por itens com uma ou duas propostas. A quantidade de propostas foi tratada como variável descritiva e diagnóstica, não como explicação causal do deserto, pois a condição de deserto já implica ausência de propostas.

A análise de preço passou a comparar o valor cotado do item com a mediana de valor homologado de itens semelhantes bem-sucedidos. Essa referência foi calculada prioritariamente por descrição normalizada e, quando não houve quantidade mínima de homologações equivalentes, por família textual gerada por TF-IDF e agrupamento. Foram geradas classificações para itens com valor cotado abaixo, próximo ou acima da referência interna.

Foram aplicadas técnicas de aprendizado de máquina em duas abordagens: modelo pré-disputa, sem variáveis posteriores ao certame, e modelo diagnóstico final, com número de propostas, valor homologado e métricas de preço. Os resultados devem ser interpretados como associações estatísticas e operacionais, não como causalidade direta.
"""
    (out / "texto" / "texto_base_resultados_preliminares.txt").write_text(texto, encoding="utf-8")

    shutil.copy2(Path(__file__).resolve(), out / "codigo" / "02_comparativos_estatistica_ml.py")
    (out / "requirements.txt").write_text("pandas\nnumpy\nscikit-learn\nmatplotlib\njoblib\n", encoding="utf-8")
    (out / "README.md").write_text(
        """# Pacote TCC - Comparativos e Machine Learning

## Execução

```bash
pip install -r requirements.txt
python codigo/02_comparativos_estatistica_ml.py --base saida_integracao/bases/base_analitica_itens_completa.csv.gz --output-dir saida_tcc_comparativos --max-model-rows 0 --max-text-fit-rows 40000 --n-text-clusters 20
```

## Acompanhar processamento no PowerShell

```powershell
Get-Content saida_tcc_comparativos\\execucao_algoritmo.log -Wait
```

## Correções desta versão
- RESULTADO FINAL é tratado como sucesso junto com HOMOLOGADO e ADJUDICADO.
- Quantidade de propostas não entra no modelo pré-disputa.
- Deserto e fracassado foram separados em variáveis auxiliares.
- Valor cotado é comparado com a mediana homologada de itens semelhantes.
- O modelo diagnóstico final pode usar propostas, valor homologado e desvios de preço apenas para interpretação operacional.

## Saídas principais
- tabelas/indicadores_gerais.csv
- tabelas/tabela_resultado_final.csv
- tabelas/tabela_preco_referencia_resultado.csv
- tabelas/tabela_preco_referencia_criterios.csv
- tabelas/top_itens_descricao_insucesso.csv
- modelos/metricas_modelos_resumo.csv
- bases/base_analitica_comparativos_completa.csv.gz
- texto/texto_base_resultados_preliminares.txt
""",
        encoding="utf-8",
    )
    p.update("salvar", "ok", "Arquivos salvos", len(df))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/mnt/data/saida_tcc_resultados/bases/base_analitica_itens_completa.csv.gz")
    ap.add_argument("--output-dir", default="/mnt/data/saida_tcc_comparativos")
    ap.add_argument("--max-model-rows", type=int, default=90000, help="0 ou negativo usa toda a base elegível")
    ap.add_argument("--max-text-fit-rows", type=int, default=50000)
    ap.add_argument("--n-text-clusters", type=int, default=22)
    ap.add_argument("--top-n", type=int, default=80)
    ap.add_argument("--min-desc-group", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--skip-ml", action="store_true")
    ap.add_argument("--skip-text-clusters", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = Progress(out)
    p.update("inicio", "andamento", "iniciando")
    df = read_base(Path(args.base), p)
    df = prep(df, p)
    df, vect, km = add_clusters(df, args, p)
    df = add_family_price_reference(df, p)
    tabs = generate_tables(df, args, p)
    ml = run_ml(df, args, p)
    save_outputs(df, tabs, ml, args, p, vect, km)
    p.update("fim", "ok", "concluído", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
