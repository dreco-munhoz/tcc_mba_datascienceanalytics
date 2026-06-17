#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algoritmo completo para gerar base analítica, tabelas de resultados preliminares
e modelo de Machine Learning para o TCC sobre itens desertos e fracassados em
licitações eletrônicas.

Entrada esperada:
- processos (2).csv  -> preferencial, pois contém idProcess
- processos.csv      -> usado como fallback, se não houver processos (2).csv
- lotes.csv
- itens.csv
- resultado-item.csv
- propostas-lote.csv

Saída:
- pasta saida_tcc_resultados/
- base analítica completa
- tabelas descritivas
- métricas do modelo
- gráficos
- log/status de processamento atualizado durante a execução

Autor: gerado para apoio ao TCC de André Francisco Munhoz
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Configurações gerais
# =============================================================================

DEFAULT_OUTPUT_DIR = "saida_tcc_resultados"
DEFAULT_CHUNKSIZE = 150_000
DEFAULT_MODEL_SAMPLE = 50_000
RANDOM_STATE = 42

STATUS_SUCESSO = {"HOMOLOGADO", "ADJUDICADO", "RESULTADO FINAL"}
STATUS_INSUCESSO = {"DESERTO", "FRACASSADO"}
STATUS_EXCLUIR_MODELO = {
    "REVOGADO", "ANULADO", "CANCELADO", "SUSPENSO", "EM RETIFICAÇÃO",
    "RECEPÇÃO DE PROPOSTAS", "ANÁLISE DE PROPOSTAS", "HABILITAÇÃO",
    "DISPUTA", "JULGAMENTO", "PUBLICADO", "EM FINALIZAÇÃO",
    "NAN", "", "NONE"
}

UF_REGIAO = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste", "PE": "Nordeste",
    "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}


# =============================================================================
# Utilidades de progresso e log
# =============================================================================

class ProgressTracker:
    def __init__(self, out_dir: Path, total_steps: int = 12) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps
        self.step = 0
        self.start_time = time.time()
        self.status_path = self.out_dir / "status_processamento.json"
        self.log_path = self.out_dir / "log_processamento.csv"
        if not self.log_path.exists():
            self.log_path.write_text("timestamp;step;percent;mensagem;linhas;arquivo\n", encoding="utf-8-sig")

    def update(self, mensagem: str, linhas: Optional[int] = None, arquivo: Optional[str] = None, step_increment: bool = False) -> None:
        if step_increment:
            self.step += 1
        percent = min(100.0, round((self.step / max(1, self.total_steps)) * 100, 2))
        elapsed = round(time.time() - self.start_time, 2)
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": self.step,
            "total_steps": self.total_steps,
            "percentual_estimado": percent,
            "mensagem": mensagem,
            "linhas_processadas": int(linhas) if linhas is not None and not pd.isna(linhas) else None,
            "arquivo": arquivo,
            "tempo_decorrido_segundos": elapsed,
        }
        self.status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.log_path.open("a", encoding="utf-8-sig") as f:
            f.write(f"{payload['timestamp']};{self.step};{percent};{mensagem};{payload['linhas_processadas'] or ''};{arquivo or ''}\n")
        print(f"[{payload['timestamp']}] {percent:>6.2f}% | {mensagem}" + (f" | linhas={linhas}" if linhas is not None else ""), flush=True)


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "execucao_algoritmo.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# =============================================================================
# Funções de leitura, normalização e escrita
# =============================================================================

def normalize_colname(c: Any) -> str:
    return str(c).strip().replace("\ufeff", "")


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    txt = str(value).strip().upper()
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def safe_numeric(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype(str)
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False)
         .str.replace("R$", "", regex=False)
         .str.strip(),
        errors="coerce",
    )


def safe_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def read_csv_auto(path: Path, tracker: ProgressTracker, chunksize: int = DEFAULT_CHUNKSIZE, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Lê CSV com tentativa automática de encoding.
    Para arquivos grandes, lê em chunks e atualiza status durante a leitura.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    encodings = ["utf-8-sig", "latin1", "cp1252"]
    last_error = None

    for enc in encodings:
        try:
            rows = 0
            chunks = []
            tracker.update(f"Lendo arquivo {path.name} com encoding {enc}", arquivo=str(path))
            reader = pd.read_csv(
                path,
                sep=";",
                encoding=enc,
                low_memory=False,
                chunksize=chunksize,
                usecols=usecols,
                on_bad_lines="skip",
            )
            for i, chunk in enumerate(reader, start=1):
                chunk.columns = [normalize_colname(c) for c in chunk.columns]
                chunk = chunk.loc[:, [c for c in chunk.columns if not str(c).startswith("Unnamed")]]
                rows += len(chunk)
                chunks.append(chunk)
                if i == 1 or i % 3 == 0:
                    tracker.update(f"Lendo {path.name}: chunk {i}", linhas=rows, arquivo=str(path))
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            tracker.update(f"Arquivo lido: {path.name}", linhas=len(df), arquivo=str(path))
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            # Tenta próximo encoding apenas em erros claramente relacionados a leitura.
            continue

    raise RuntimeError(f"Não foi possível ler {path}. Último erro: {last_error}")


def save_csv_chunked(df: pd.DataFrame, path: Path, tracker: ProgressTracker, chunksize: int = DEFAULT_CHUNKSIZE, compression: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(df)
    first = True
    mode = "wt"
    kwargs: Dict[str, Any] = {}
    if compression == "gzip" or str(path).endswith(".gz"):
        kwargs["compression"] = "gzip"

    for start in range(0, total, chunksize):
        end = min(start + chunksize, total)
        df.iloc[start:end].to_csv(
            path,
            sep=";",
            index=False,
            encoding="utf-8-sig",
            mode=mode,
            header=first,
            **kwargs,
        )
        first = False
        mode = "at"
        tracker.update(f"Salvando {path.name}: {end}/{total}", linhas=end, arquivo=str(path))


def compact_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


# =============================================================================
# Construção da base analítica
# =============================================================================

def choose_processos_file(input_dir: Path, explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else input_dir / explicit

    prefer = input_dir / "processos (2).csv"
    fallback = input_dir / "processos.csv"
    if prefer.exists():
        return prefer
    return fallback


def prepare_processos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    # Se idProcess não existir, cria campo vazio para evitar quebra.
    if "idProcess" not in df.columns:
        df["idProcess"] = np.nan

    keep = [
        "idProcess", "UF", "Cidade", "Fornecedor", "CNPJ", "Modalidade", "TipoDisputa",
        "TipoEncerramento", "Status", "Edital", "MEExclusivo", "VlRefVisivel",
        "DtPublicacao", "InicioRecProp", "FimRecProp", "DataDisputa", "Observacao",
        "Obj", "RegExcl", "LocalExc"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    rename = {
        "Status": "StatusProcesso",
        "Obj": "ObjetoProcesso",
        "UF": "UF",
        "Cidade": "Cidade",
    }
    df = df.rename(columns=rename)

    for col in ["idProcess", "UF", "Cidade", "Fornecedor", "CNPJ", "Modalidade", "TipoDisputa", "TipoEncerramento", "StatusProcesso", "Edital"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ["MEExclusivo", "VlRefVisivel", "RegExcl", "LocalExc"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col]).fillna(0).astype("int8")

    for col in ["DtPublicacao", "InicioRecProp", "FimRecProp", "DataDisputa"]:
        if col in df.columns:
            df[col] = safe_datetime(df[col])

    if "DtPublicacao" in df.columns:
        df["ano_publicacao"] = df["DtPublicacao"].dt.year
        df["mes_publicacao"] = df["DtPublicacao"].dt.month
        df["ano_mes_publicacao"] = df["DtPublicacao"].dt.strftime("%Y-%m")
    else:
        df["ano_publicacao"] = np.nan
        df["mes_publicacao"] = np.nan
        df["ano_mes_publicacao"] = np.nan

    if "DataDisputa" in df.columns and "DtPublicacao" in df.columns:
        df["dias_publicacao_disputa"] = (df["DataDisputa"] - df["DtPublicacao"]).dt.total_seconds() / 86400
    else:
        df["dias_publicacao_disputa"] = np.nan

    df["UF"] = df.get("UF", "").astype(str).str.upper().str.strip()
    df["Regiao"] = df["UF"].map(UF_REGIAO).fillna("Não identificado")
    df["StatusProcessoNorm"] = df.get("StatusProcesso", "").map(normalize_text)
    return compact_memory(df)


def prepare_lotes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    keep = [
        "fkProcess", "idBatch", "NumLote", "Status", "TipoLance", "Titulo",
        "Quantidade", "MargemLances", "ExclMe", "Regional", "ExclLocal"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df = df.rename(columns={"Status": "StatusLote", "Titulo": "TituloLote"})
    for col in ["fkProcess", "idBatch", "StatusLote", "TipoLance", "TituloLote"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ["NumLote", "Quantidade", "MargemLances", "ExclMe", "Regional", "ExclLocal"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    df["StatusLoteNorm"] = df.get("StatusLote", "").map(normalize_text)
    return compact_memory(df)


def prepare_itens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    keep = ["fkBatch", "NumLote", "idBatchItem", "NumItem", "Descricao", "Unidade", "VlRef", "InfoReq", "ArqReq"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    for col in ["fkBatch", "idBatchItem", "Descricao", "Unidade"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ["NumLote", "NumItem", "VlRef", "InfoReq", "ArqReq"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    df["Descricao"] = df.get("Descricao", "").fillna("").astype(str)
    df["descricao_tam_caracteres"] = df["Descricao"].str.len()
    df["descricao_qtd_palavras"] = df["Descricao"].str.split().str.len().fillna(0)
    df["descricao_tem_marca"] = df["Descricao"].map(lambda x: int(bool(re.search(r"\b(MARCA|MODELO|REFER[ÊE]NCIA|SIMILAR)\b", normalize_text(x)))))
    df["log_vlref"] = np.log1p(df["VlRef"].clip(lower=0)) if "VlRef" in df.columns else np.nan
    return compact_memory(df)


def prepare_resultado(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    keep = ["idBatchItem", "NumeroItem", "ValorFinal"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df["idBatchItem"] = df["idBatchItem"].astype(str).str.strip()
    if "ValorFinal" in df.columns:
        df["ValorFinal"] = safe_numeric(df["ValorFinal"])
    if "NumeroItem" in df.columns:
        df["NumeroItem"] = safe_numeric(df["NumeroItem"])

    # Em caso de duplicidade no arquivo de resultado, mantém a menor chave e agrega por idBatchItem.
    agg = {"ValorFinal": "first"}
    if "NumeroItem" in df.columns:
        agg["NumeroItem"] = "first"
    df = df.groupby("idBatchItem", as_index=False).agg(agg)
    return compact_memory(df)


def prepare_propostas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    keep = ["idBatch", "PropostasLote"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df["idBatch"] = df["idBatch"].astype(str).str.strip()
    df["PropostasLote"] = safe_numeric(df["PropostasLote"])
    df = df.groupby("idBatch", as_index=False).agg({"PropostasLote": "max"})
    return compact_memory(df)


def add_tipo_lote(base: pd.DataFrame) -> pd.DataFrame:
    counts = base.groupby("idBatch", dropna=False)["idBatchItem"].nunique().rename("qtd_itens_lote").reset_index()
    base = base.merge(counts, on="idBatch", how="left")

    titulo = base.get("TituloLote", "").fillna("").astype(str).map(normalize_text)
    desc = base.get("Descricao", "").fillna("").astype(str).map(normalize_text)
    is_kit = titulo.str.contains(r"\bKIT\b|CONJUNTO|CESTA|LOTE GLOBAL", regex=True, na=False) | desc.str.contains(r"\bKIT\b|CONJUNTO", regex=True, na=False)

    base["tipo_lote_calc"] = np.where(
        base["qtd_itens_lote"].fillna(0) <= 1,
        "Lote unitário",
        np.where(is_kit, "Lote global/kit", "Lote global")
    )
    return base


def add_target(base: pd.DataFrame) -> pd.DataFrame:
    status = base["StatusLoteNorm"].fillna("").astype(str)
    base["status_item_analise"] = np.select(
        [
            status.isin(STATUS_INSUCESSO),
            status.isin(STATUS_SUCESSO),
            status.isin(STATUS_EXCLUIR_MODELO),
        ],
        [
            "Insucesso",
            "Sucesso",
            "Fora da modelagem",
        ],
        default="Outro"
    )

    base["insucesso"] = np.where(status.isin(STATUS_INSUCESSO), 1,
                         np.where(status.isin(STATUS_SUCESSO), 0, np.nan))
    base["status_final_modelagem"] = np.where(base["insucesso"].notna(), "Incluído", "Excluído")
    return base


def add_price_features(base: pd.DataFrame) -> pd.DataFrame:
    if "VlRef" in base.columns and "ValorFinal" in base.columns:
        base["dif_abs_valor_final_estimado"] = base["ValorFinal"] - base["VlRef"]
        base["dif_pct_valor_final_estimado"] = np.where(
            base["VlRef"].fillna(0) > 0,
            (base["ValorFinal"] - base["VlRef"]) / base["VlRef"],
            np.nan
        )
    else:
        base["dif_abs_valor_final_estimado"] = np.nan
        base["dif_pct_valor_final_estimado"] = np.nan

    if "VlRef" in base.columns:
        base["faixa_valor_estimado"] = pd.cut(
            base["VlRef"].fillna(-1),
            bins=[-np.inf, 0, 100, 1000, 10000, 100000, 1_000_000, np.inf],
            labels=["Sem valor/zero", "Até R$100", "R$100 a R$1 mil", "R$1 mil a R$10 mil", "R$10 mil a R$100 mil", "R$100 mil a R$1 milhão", "Acima de R$1 milhão"],
        ).astype(str)
    else:
        base["faixa_valor_estimado"] = "Não informado"
    return base


def build_base_analitica(
    processos: pd.DataFrame,
    lotes: pd.DataFrame,
    itens: pd.DataFrame,
    resultado: pd.DataFrame,
    propostas: pd.DataFrame,
    tracker: ProgressTracker
) -> pd.DataFrame:
    tracker.update("Preparando processos", linhas=len(processos), step_increment=True)
    processos = prepare_processos(processos)

    tracker.update("Preparando lotes", linhas=len(lotes), step_increment=True)
    lotes = prepare_lotes(lotes)

    tracker.update("Preparando itens", linhas=len(itens), step_increment=True)
    itens = prepare_itens(itens)

    tracker.update("Preparando resultados por item", linhas=len(resultado), step_increment=True)
    resultado = prepare_resultado(resultado)

    tracker.update("Preparando propostas por lote", linhas=len(propostas), step_increment=True)
    propostas = prepare_propostas(propostas)

    tracker.update("Relacionando lotes com processos", linhas=len(lotes), step_increment=True)
    if "idProcess" in processos.columns and processos["idProcess"].notna().any():
        processos_small = processos.drop_duplicates(subset=["idProcess"])
        lotes_proc = lotes.merge(processos_small, left_on="fkProcess", right_on="idProcess", how="left", suffixes=("", "_proc"))
    else:
        lotes_proc = lotes.copy()
        logging.warning("processos.csv não possui idProcess. A base ficará sem relação confiável com UF, cidade e modalidade.")

    tracker.update("Relacionando itens com lotes/processos", linhas=len(itens), step_increment=True)
    base = itens.merge(lotes_proc, left_on="fkBatch", right_on="idBatch", how="left", suffixes=("", "_lote"))

    tracker.update("Relacionando valor final e quantidade de propostas", linhas=len(base), step_increment=True)
    base = base.merge(resultado, on="idBatchItem", how="left", suffixes=("", "_resultado"))
    base = base.merge(propostas, on="idBatch", how="left")

    tracker.update("Criando variáveis analíticas", linhas=len(base), step_increment=True)
    base = add_tipo_lote(base)
    base = add_target(base)
    base = add_price_features(base)

    # Variáveis auxiliares binárias.
    for col in ["MEExclusivo", "RegExcl", "LocalExc", "ExclMe", "Regional", "ExclLocal", "InfoReq", "ArqReq", "VlRefVisivel"]:
        if col in base.columns:
            base[col] = safe_numeric(base[col]).fillna(0)

    # Taxa de propostas por lote para análise descritiva, não para predição principal.
    base["baixa_competitividade_descritiva"] = np.where(base["PropostasLote"].fillna(0) <= 2, 1, 0)
    base["participacao_unica_descritiva"] = np.where(base["PropostasLote"].fillna(0) == 1, 1, 0)

    base = compact_memory(base)
    tracker.update("Base analítica construída", linhas=len(base))
    return base


# =============================================================================
# Tabelas de resultados preliminares
# =============================================================================

def pct(x: pd.Series) -> pd.Series:
    total = x.sum()
    return x / total if total else x * np.nan


def tabela_freq(base: pd.DataFrame, col: str, out_path: Path, tracker: ProgressTracker, incluir_insucesso: bool = True) -> pd.DataFrame:
    if col not in base.columns:
        return pd.DataFrame()

    df = base.copy()
    df[col] = df[col].fillna("Não informado").astype(str)
    if incluir_insucesso and "insucesso" in df.columns:
        tab = (
            df.groupby(col, dropna=False)
              .agg(
                  qtd_itens=("idBatchItem", "count"),
                  qtd_insucesso=("insucesso", lambda s: int((s == 1).sum())),
                  qtd_sucesso=("insucesso", lambda s: int((s == 0).sum())),
                  qtd_fora_modelagem=("insucesso", lambda s: int(s.isna().sum())),
                  valor_estimado_total=("VlRef", "sum"),
                  valor_final_total=("ValorFinal", "sum"),
                  media_propostas_lote=("PropostasLote", "mean"),
              )
              .reset_index()
        )
        tab["taxa_insucesso_%"] = np.where((tab["qtd_insucesso"] + tab["qtd_sucesso"]) > 0, tab["qtd_insucesso"] / (tab["qtd_insucesso"] + tab["qtd_sucesso"]) * 100, np.nan)
    else:
        tab = df.groupby(col, dropna=False).agg(qtd_itens=("idBatchItem", "count")).reset_index()

    tab["participacao_%"] = tab["qtd_itens"] / max(1, tab["qtd_itens"].sum()) * 100
    tab = tab.sort_values("qtd_itens", ascending=False)
    tab.to_csv(out_path, sep=";", index=False, encoding="utf-8-sig")
    tracker.update(f"Tabela gerada: {out_path.name}", linhas=len(tab), arquivo=str(out_path))
    return tab


def generate_descriptive_outputs(base: pd.DataFrame, out_dir: Path, tracker: ProgressTracker) -> Dict[str, Any]:
    tabelas_dir = out_dir / "tabelas"
    tabelas_dir.mkdir(parents=True, exist_ok=True)

    tracker.update("Gerando tabelas descritivas", linhas=len(base), step_increment=True)

    total_itens = len(base)
    total_lotes = base["idBatch"].nunique(dropna=True) if "idBatch" in base.columns else np.nan
    total_processos = base["idProcess"].nunique(dropna=True) if "idProcess" in base.columns else np.nan
    qtd_modelagem = int(base["insucesso"].notna().sum())
    qtd_insucesso = int((base["insucesso"] == 1).sum())
    qtd_sucesso = int((base["insucesso"] == 0).sum())

    dt_min = base["DtPublicacao"].min() if "DtPublicacao" in base.columns else pd.NaT
    dt_max = base["DtPublicacao"].max() if "DtPublicacao" in base.columns else pd.NaT

    resumo = pd.DataFrame([{
        "total_itens": total_itens,
        "total_lotes": total_lotes,
        "total_processos": total_processos,
        "itens_incluidos_modelagem": qtd_modelagem,
        "itens_insucesso_deserto_fracassado": qtd_insucesso,
        "itens_sucesso_homologado_adjudicado_resultado_final": qtd_sucesso,
        "taxa_insucesso_modelagem_%": (qtd_insucesso / qtd_modelagem * 100) if qtd_modelagem else np.nan,
        "valor_estimado_total": base["VlRef"].sum(skipna=True) if "VlRef" in base.columns else np.nan,
        "valor_final_total": base["ValorFinal"].sum(skipna=True) if "ValorFinal" in base.columns else np.nan,
        "data_publicacao_min": str(dt_min),
        "data_publicacao_max": str(dt_max),
    }])
    resumo_path = tabelas_dir / "resumo_geral.csv"
    resumo.to_csv(resumo_path, sep=";", index=False, encoding="utf-8-sig")
    tracker.update("Resumo geral gerado", linhas=len(resumo), arquivo=str(resumo_path))

    tables = {
        "resumo_geral": resumo,
        "status_lote": tabela_freq(base, "StatusLoteNorm", tabelas_dir / "tabela_status_lote.csv", tracker),
        "status_processo": tabela_freq(base, "StatusProcessoNorm", tabelas_dir / "tabela_status_processo.csv", tracker),
        "uf": tabela_freq(base, "UF", tabelas_dir / "tabela_uf.csv", tracker),
        "regiao": tabela_freq(base, "Regiao", tabelas_dir / "tabela_regiao.csv", tracker),
        "modalidade": tabela_freq(base, "Modalidade", tabelas_dir / "tabela_modalidade.csv", tracker),
        "tipo_lance": tabela_freq(base, "TipoLance", tabelas_dir / "tabela_tipo_lance.csv", tracker),
        "tipo_lote": tabela_freq(base, "tipo_lote_calc", tabelas_dir / "tabela_tipo_lote.csv", tracker),
        "faixa_valor": tabela_freq(base, "faixa_valor_estimado", tabelas_dir / "tabela_faixa_valor.csv", tracker),
        "mes": tabela_freq(base, "ano_mes_publicacao", tabelas_dir / "tabela_mes.csv", tracker),
    }

    # Tabela com recorte de exclusividade/regionalidade.
    cols_excl = [c for c in ["MEExclusivo", "RegExcl", "LocalExc", "ExclMe", "Regional", "ExclLocal", "InfoReq", "ArqReq", "VlRefVisivel"] if c in base.columns]
    if cols_excl:
        rows = []
        for c in cols_excl:
            b = safe_numeric(base[c]).fillna(0)
            rows.append({
                "variavel": c,
                "qtd_itens_com_flag": int((b > 0).sum()),
                "participacao_%": float((b > 0).mean() * 100),
                "taxa_insucesso_%": float(base.loc[b > 0, "insucesso"].eq(1).sum() / max(1, base.loc[b > 0, "insucesso"].notna().sum()) * 100),
            })
        tab_excl = pd.DataFrame(rows)
        tab_excl.to_csv(tabelas_dir / "tabela_exclusividade_regionalidade.csv", sep=";", index=False, encoding="utf-8-sig")
        tables["exclusividade_regionalidade"] = tab_excl
        tracker.update("Tabela de exclusividade e regionalidade gerada", linhas=len(tab_excl))

    # Estatísticas numéricas.
    num_cols = [c for c in ["VlRef", "ValorFinal", "dif_pct_valor_final_estimado", "PropostasLote", "descricao_qtd_palavras", "qtd_itens_lote", "dias_publicacao_disputa"] if c in base.columns]
    if num_cols:
        desc = base[num_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.reset_index().rename(columns={"index": "variavel"})
        desc.to_csv(tabelas_dir / "estatisticas_numericas.csv", sep=";", index=False, encoding="utf-8-sig")
        tables["estatisticas_numericas"] = desc
        tracker.update("Estatísticas numéricas geradas", linhas=len(desc))

    return {"resumo": resumo.to_dict(orient="records")[0], "tables": list(tables.keys())}


# =============================================================================
# Modelo de Machine Learning
# =============================================================================

def train_model(base: pd.DataFrame, out_dir: Path, tracker: ProgressTracker, max_rows: int = DEFAULT_MODEL_SAMPLE) -> Dict[str, Any]:
    tracker.update("Preparando base de modelagem", linhas=len(base), step_increment=True)
    ml_dir = out_dir / "modelo_ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, average_precision_score, classification_report,
            confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
            roc_curve, precision_recall_curve
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        import joblib
        import matplotlib.pyplot as plt
    except Exception as e:
        msg = f"Bibliotecas de ML não disponíveis: {e}"
        logging.exception(msg)
        tracker.update(msg)
        return {"status": "erro", "mensagem": msg}

    # Importante: não usar ValorFinal e PropostasLote como preditores principais,
    # pois são informações posteriores/durante o certame. Elas ficam nas tabelas descritivas.
    candidate_numeric = [
        "VlRef", "log_vlref", "descricao_tam_caracteres", "descricao_qtd_palavras",
        "descricao_tem_marca", "InfoReq", "ArqReq", "Quantidade", "MargemLances",
        "ExclMe", "Regional", "ExclLocal", "MEExclusivo", "RegExcl", "LocalExc",
        "VlRefVisivel", "qtd_itens_lote", "dias_publicacao_disputa", "mes_publicacao"
    ]
    candidate_categorical = [
        "UF", "Regiao", "Modalidade", "TipoDisputa", "TipoEncerramento",
        "TipoLance", "tipo_lote_calc", "faixa_valor_estimado", "ano_mes_publicacao"
    ]

    numeric_features = [c for c in candidate_numeric if c in base.columns]
    categorical_features = [c for c in candidate_categorical if c in base.columns]

    model_df = base.loc[base["insucesso"].notna(), numeric_features + categorical_features + ["insucesso"]].copy()
    model_df["insucesso"] = model_df["insucesso"].astype(int)

    # Remove linhas totalmente vazias nos preditores principais.
    if "VlRef" in model_df.columns:
        model_df = model_df.loc[model_df["VlRef"].notna()].copy()

    class_counts = model_df["insucesso"].value_counts().to_dict()
    tracker.update(f"Base de modelagem filtrada. Classes: {class_counts}", linhas=len(model_df))

    if model_df["insucesso"].nunique() < 2 or len(model_df) < 100:
        msg = "Base insuficiente para treinar modelo com duas classes."
        tracker.update(msg)
        return {"status": "sem_modelo", "mensagem": msg, "class_counts": class_counts}

    # Amostragem estratificada para acelerar sem perder distribuição.
    # Regra: max_rows <= 0 significa usar TODA a base elegível.
    # A versão anterior interpretava 0 como limite de 0 linhas e criava apenas
    # 1 registro por classe, o que quebrava o train_test_split estratificado.
    if max_rows and max_rows > 0 and len(model_df) > max_rows:
        parts = []
        for cls, grp in model_df.groupby("insucesso"):
            n_cls = max(2, int(max_rows * len(grp) / len(model_df)))
            parts.append(grp.sample(n=min(len(grp), n_cls), random_state=RANDOM_STATE))
        model_df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
        tracker.update(f"Amostra estratificada para modelagem criada: máximo {max_rows} linhas", linhas=len(model_df))
    else:
        tracker.update("Modelagem usando toda a base elegível", linhas=len(model_df))

    X = model_df[numeric_features + categorical_features]
    y = model_df["insucesso"]

    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Compatibilidade com versões do sklearn.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="liblinear",
        penalty="l2",
        random_state=RANDOM_STATE,
    )

    clf = Pipeline(steps=[
        ("preprocessador", preprocessor),
        ("modelo", model),
    ])

    tracker.update("Treinando regressão logística regularizada", linhas=len(X_train))
    clf.fit(X_train, y_train)

    tracker.update("Avaliando modelo", linhas=len(X_test))
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "linhas_modelagem": int(len(model_df)),
        "linhas_treino": int(len(X_train)),
        "linhas_teste": int(len(X_test)),
        "qtd_features_numericas": int(len(numeric_features)),
        "qtd_features_categoricas": int(len(categorical_features)),
        "classe_0_sucesso": int((y == 0).sum()),
        "classe_1_insucesso": int((y == 1).sum()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
    }
    pd.DataFrame([metrics]).to_csv(ml_dir / "metricas_modelo.csv", sep=";", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["real_sucesso_0", "real_insucesso_1"], columns=["pred_sucesso_0", "pred_insucesso_1"])
    cm_df.to_csv(ml_dir / "matriz_confusao.csv", sep=";", encoding="utf-8-sig")

    report = classification_report(y_test, y_pred, target_names=["Sucesso", "Insucesso"], zero_division=0)
    (ml_dir / "relatorio_classificacao.txt").write_text(report, encoding="utf-8")

    # Coeficientes/variáveis mais relevantes.
    try:
        feat_names_num = numeric_features
        cat_encoder = clf.named_steps["preprocessador"].named_transformers_["cat"].named_steps["onehot"]
        feat_names_cat = list(cat_encoder.get_feature_names_out(categorical_features))
        feature_names = feat_names_num + feat_names_cat
        coefs = clf.named_steps["modelo"].coef_[0]
        importance = pd.DataFrame({"variavel": feature_names, "coeficiente": coefs})
        importance["abs_coeficiente"] = importance["coeficiente"].abs()
        importance = importance.sort_values("abs_coeficiente", ascending=False)
        importance.head(100).to_csv(ml_dir / "importancia_variaveis_top100.csv", sep=";", index=False, encoding="utf-8-sig")
    except Exception as e:
        logging.exception("Não foi possível gerar importância de variáveis: %s", e)

    # Curvas.
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Taxa de falsos positivos")
        plt.ylabel("Taxa de verdadeiros positivos")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ml_dir / "curva_roc.png", dpi=150)
        plt.close()

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(rec, prec, label=f"AP = {metrics['average_precision']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precisão")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ml_dir / "curva_precision_recall.png", dpi=150)
        plt.close()
    except Exception as e:
        logging.exception("Não foi possível gerar gráficos do modelo: %s", e)

    # Salva modelo e amostra usada.
    try:
        import joblib
        joblib.dump(clf, ml_dir / "modelo_regressao_logistica.joblib")
    except Exception as e:
        logging.exception("Não foi possível salvar modelo joblib: %s", e)

    model_df.head(5000).to_csv(ml_dir / "amostra_base_modelagem.csv", sep=";", index=False, encoding="utf-8-sig")
    tracker.update("Modelo finalizado", linhas=len(model_df), arquivo=str(ml_dir / "metricas_modelo.csv"))
    return {"status": "ok", "metricas": metrics, "features_numericas": numeric_features, "features_categoricas": categorical_features}


# =============================================================================
# Gráficos descritivos
# =============================================================================

def generate_charts(base: pd.DataFrame, out_dir: Path, tracker: ProgressTracker) -> None:
    tracker.update("Gerando gráficos descritivos", linhas=len(base), step_increment=True)
    graficos_dir = out_dir / "graficos"
    graficos_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.exception("Matplotlib indisponível: %s", e)
        return

    def bar_count(col: str, filename: str, top: int = 15) -> None:
        if col not in base.columns:
            return
        tab = base[col].fillna("Não informado").astype(str).value_counts().head(top).sort_values()
        plt.figure(figsize=(9, max(4, len(tab) * 0.35)))
        tab.plot(kind="barh")
        plt.xlabel("Quantidade de itens")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(graficos_dir / filename, dpi=150)
        plt.close()

    bar_count("StatusLoteNorm", "grafico_status_lote.png")
    bar_count("UF", "grafico_uf.png")
    bar_count("Modalidade", "grafico_modalidade.png")
    bar_count("faixa_valor_estimado", "grafico_faixa_valor.png")

    if "ano_mes_publicacao" in base.columns:
        tab = base.groupby("ano_mes_publicacao")["idBatchItem"].count().sort_index()
        plt.figure(figsize=(9, 5))
        tab.plot(kind="bar")
        plt.xlabel("Mês de publicação")
        plt.ylabel("Quantidade de itens")
        plt.tight_layout()
        plt.savefig(graficos_dir / "grafico_itens_por_mes.png", dpi=150)
        plt.close()

    if "insucesso" in base.columns and "ano_mes_publicacao" in base.columns:
        tab = base.loc[base["insucesso"].notna()].groupby("ano_mes_publicacao")["insucesso"].mean().sort_index() * 100
        plt.figure(figsize=(9, 5))
        tab.plot(kind="line", marker="o")
        plt.xlabel("Mês de publicação")
        plt.ylabel("Taxa de insucesso (%)")
        plt.tight_layout()
        plt.savefig(graficos_dir / "grafico_taxa_insucesso_mes.png", dpi=150)
        plt.close()

    tracker.update("Gráficos gerados", arquivo=str(graficos_dir))


# =============================================================================
# Texto automático para resultados preliminares
# =============================================================================

def generate_texto_resultados(out_dir: Path, resumo: Dict[str, Any], model_result: Dict[str, Any], tracker: ProgressTracker) -> None:
    tracker.update("Gerando texto-base para Resultados Preliminares", step_increment=True)
    texto_path = out_dir / "texto_base_resultados_preliminares.txt"

    metricas = model_result.get("metricas", {}) if isinstance(model_result, dict) else {}
    status_modelo = model_result.get("status") if isinstance(model_result, dict) else "não executado"

    texto = f"""
Texto-base para inserir nos Resultados Preliminares

A base analítica consolidada foi construída a partir da integração dos arquivos de processos, lotes, itens, resultados por item e propostas por lote. Após o relacionamento entre os identificadores de processo, lote e item, foram obtidos {resumo.get('total_itens')} registros em nível de item, vinculados a {resumo.get('total_lotes')} lotes e {resumo.get('total_processos')} processos. A janela temporal identificada nos dados compreendeu publicações entre {resumo.get('data_publicacao_min')} e {resumo.get('data_publicacao_max')}.

Para a modelagem, foram mantidos apenas os itens com resultado final compatível com a definição operacional da variável dependente. Foram classificados como insucesso os itens desertos ou fracassados, enquanto itens homologados, adjudicados ou em resultado final foram classificados como sucesso. Itens revogados, anulados, cancelados, suspensos ou ainda em fases abertas foram excluídos da modelagem principal por representarem situações administrativas distintas ou sem desfecho final consolidado. Com essa regra, {resumo.get('itens_incluidos_modelagem')} itens permaneceram elegíveis para a análise, dos quais {resumo.get('itens_insucesso_deserto_fracassado')} foram classificados como insucesso e {resumo.get('itens_sucesso_homologado_adjudicado_resultado_final')} como sucesso.

A etapa exploratória gerou tabelas de frequência por situação do lote, unidade federativa, região, modalidade, tipo de lance, estrutura do lote, faixa de valor estimado e mês de publicação. Também foram calculadas estatísticas descritivas para valores estimados, valores finais, número de propostas por lote, tamanho da descrição textual, quantidade de itens por lote e intervalo entre publicação e disputa.

O modelo de Machine Learning foi implementado como uma regressão logística regularizada para classificação binária. Foram utilizadas apenas variáveis disponíveis antes ou na abertura da disputa, como valor estimado, características textuais simples da descrição, unidade federativa, região, modalidade, tipo de disputa, tipo de lance, estrutura do lote e indicadores de exclusividade ou regionalidade. Variáveis posteriores ao resultado, como valor final homologado e quantidade de propostas, foram mantidas apenas em análises descritivas, evitando vazamento de informação no modelo.

Status da modelagem: {status_modelo}.
Métricas obtidas, quando aplicável:
- Acurácia: {metricas.get('accuracy')}
- Precisão: {metricas.get('precision')}
- Recall: {metricas.get('recall')}
- F1-score: {metricas.get('f1')}
- AUC ROC: {metricas.get('roc_auc')}

Esses resultados devem ser interpretados de forma associativa e preliminar, sem inferência causal direta. A interpretação final dependerá da validação da qualidade dos dados, da consistência dos status dos itens e da avaliação das variáveis mais relevantes no modelo.
""".strip()

    texto_path.write_text(texto, encoding="utf-8")
    tracker.update("Texto-base gerado", arquivo=str(texto_path))


# =============================================================================
# Execução principal
# =============================================================================

def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir)
    tracker = ProgressTracker(out_dir, total_steps=14)

    tracker.update("Iniciando processamento", step_increment=False)

    processos_path = choose_processos_file(input_dir, args.processos)
    files = {
        "processos": processos_path,
        "lotes": input_dir / args.lotes,
        "itens": input_dir / args.itens,
        "resultado": input_dir / args.resultado_item,
        "propostas": input_dir / args.propostas_lote,
    }

    tracker.update("Arquivos definidos: " + json.dumps({k: str(v) for k, v in files.items()}, ensure_ascii=False), step_increment=True)

    processos = read_csv_auto(files["processos"], tracker, chunksize=args.chunksize)
    lotes = read_csv_auto(files["lotes"], tracker, chunksize=args.chunksize)
    itens = read_csv_auto(files["itens"], tracker, chunksize=args.chunksize)
    resultado = read_csv_auto(files["resultado"], tracker, chunksize=args.chunksize)
    propostas = read_csv_auto(files["propostas"], tracker, chunksize=args.chunksize)

    # Salva diagnóstico bruto de linhas.
    diag = pd.DataFrame([
        {"arquivo": k, "caminho": str(v), "linhas": len(obj)}
        for (k, v), obj in zip(files.items(), [processos, lotes, itens, resultado, propostas])
    ])
    diag.to_csv(out_dir / "diagnostico_arquivos_entrada.csv", sep=";", index=False, encoding="utf-8-sig")

    base = build_base_analitica(processos, lotes, itens, resultado, propostas, tracker)

    # Filtro temporal opcional. Por padrão, não filtra, apenas usa o que veio nos arquivos.
    if args.data_inicio or args.data_fim:
        tracker.update("Aplicando filtro temporal informado pelo usuário")
        if "DtPublicacao" in base.columns:
            inicio = pd.to_datetime(args.data_inicio, errors="coerce", utc=True) if args.data_inicio else base["DtPublicacao"].min()
            fim = pd.to_datetime(args.data_fim, errors="coerce", utc=True) if args.data_fim else base["DtPublicacao"].max()
            antes = len(base)
            base = base.loc[(base["DtPublicacao"] >= inicio) & (base["DtPublicacao"] <= fim)].copy()
            tracker.update(f"Filtro temporal aplicado: {antes} -> {len(base)} linhas", linhas=len(base))

    # Salva bases.
    bases_dir = out_dir / "bases"
    bases_dir.mkdir(parents=True, exist_ok=True)

    essential_cols = [
        "idProcess", "UF", "Regiao", "Cidade", "Fornecedor", "CNPJ", "Modalidade", "TipoDisputa",
        "TipoEncerramento", "StatusProcesso", "Edital", "DtPublicacao", "DataDisputa",
        "fkProcess", "idBatch", "NumLote", "StatusLote", "TipoLance", "TituloLote",
        "qtd_itens_lote", "tipo_lote_calc", "idBatchItem", "NumItem", "Descricao", "Unidade",
        "VlRef", "ValorFinal", "dif_pct_valor_final_estimado", "PropostasLote",
        "MEExclusivo", "RegExcl", "LocalExc", "ExclMe", "Regional", "ExclLocal", "InfoReq", "ArqReq",
        "descricao_tam_caracteres", "descricao_qtd_palavras", "descricao_tem_marca",
        "faixa_valor_estimado", "status_item_analise", "insucesso", "status_final_modelagem",
        "baixa_competitividade_descritiva", "participacao_unica_descritiva"
    ]
    essential_cols = [c for c in essential_cols if c in base.columns]

    save_csv_chunked(base[essential_cols], bases_dir / "base_analitica_itens_completa.csv.gz", tracker, chunksize=args.chunksize, compression="gzip")

    base_modelagem = base.loc[base["insucesso"].notna(), essential_cols].copy()
    save_csv_chunked(base_modelagem, bases_dir / "base_modelagem_itens.csv.gz", tracker, chunksize=args.chunksize, compression="gzip")

    sample_n = min(5000, len(base))
    base[essential_cols].sample(n=sample_n, random_state=RANDOM_STATE).to_csv(
        bases_dir / "amostra_base_analitica.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )
    tracker.update("Bases salvas", linhas=len(base), step_increment=True)

    desc_result = generate_descriptive_outputs(base, out_dir, tracker)
    generate_charts(base, out_dir, tracker)
    model_result = train_model(base, out_dir, tracker, max_rows=args.max_model_rows)
    generate_texto_resultados(out_dir, desc_result["resumo"], model_result, tracker)

    # Manifesto final.
    manifest = {
        "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "arquivos": {k: str(v) for k, v in files.items()},
        "resumo": desc_result.get("resumo"),
        "modelo": model_result,
    }
    (out_dir / "manifesto_execucao.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    tracker.step = tracker.total_steps
    tracker.update("Processamento concluído com sucesso", linhas=len(base), step_increment=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline completo para Resultados Preliminares do TCC.")
    parser.add_argument("--input-dir", default=".", help="Pasta onde estão os CSVs de entrada.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Pasta onde serão salvos os resultados.")
    parser.add_argument("--processos", default=None, help="Nome/caminho do arquivo de processos. Se omitido, usa 'processos (2).csv' quando existir.")
    parser.add_argument("--lotes", default="lotes.csv", help="Arquivo de lotes.")
    parser.add_argument("--itens", default="itens.csv", help="Arquivo de itens.")
    parser.add_argument("--resultado-item", default="resultado-item.csv", help="Arquivo de resultado por item.")
    parser.add_argument("--propostas-lote", default="propostas-lote.csv", help="Arquivo de propostas por lote.")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help="Tamanho do chunk para leitura/escrita.")
    parser.add_argument("--max-model-rows", type=int, default=DEFAULT_MODEL_SAMPLE, help="Máximo de linhas na amostra de modelagem.")
    parser.add_argument("--data-inicio", default=None, help="Filtro opcional de data inicial YYYY-MM-DD.")
    parser.add_argument("--data-fim", default=None, help="Filtro opcional de data final YYYY-MM-DD.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
