#!/usr/bin/env python3
"""
臺股短線(1~2週)觀察清單 - 每日掃描腳本
=========================================

這支腳本做的事情很單純,分四個階段:

  1) 全市場快照
     呼叫 TWSE 官方 OpenAPI STOCK_DAY_ALL,一次拿到「今天」所有上市股票的
     收盤價、漲跌、成交量,不需要對每檔股票各打一次 API。
     文件: https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL

  2) 流動性初篩
     用今天的成交量、成交值、股價做粗篩,把幾千檔股票縮小到一個候選池
     (預設幾百檔),避免後面抓歷史資料時打爆 API。

  3) 歷史資料 + 技術指標
     對候選池裡的每一檔股票,用 yfinance 抓近 90 個交易日的日K,
     計算均線、RSI、量能比、相對大盤強弱等指標。

  4) 綜合評分 + 輸出
     把各項指標依權重合成一個 0~100 分的分數,由高到低排序,
     取前 N 檔連同「進場訊號」標籤一起寫進 docs/data.json,
     給前端網頁讀取顯示。

★ 重要聲明
本腳本純粹是「技術面規則篩選」的產物,篩選邏輯是很常見的動能/量能/均線
組合,不是什麼獨門秘技,也不構成投資建議。過去的價量表現不保證未來報酬,
短線交易風險本來就高,任何進出場決定都是你自己的判斷與責任。

用法:
    python scanner.py                 # 正式模式,會打 TWSE + yfinance
    python scanner.py --demo          # demo 模式,純離線產生假資料(方便測試/預覽)
    python scanner.py --top 20        # 取前 20 名(預設 20)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TWSE_STOCK_DAY_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TAIEX_TICKER = "^TWII"

# ---- 可調參數 -----------------------------------------------------------

MIN_TRADE_VOLUME_SHARES = 1_000_000     # 今日成交量下限(股),約 1,000 張
MIN_PRICE = 10.0                         # 排除過度低價股
MAX_PRICE = 2000.0                       # 排除極端高價股(避免權值股洗版)
CANDIDATE_POOL_SIZE = 300                # 進入歷史資料階段的候選檔數上限
HISTORY_DAYS = "90d"                     # 抓多久的歷史K線
TOP_N_DEFAULT = 20

WEIGHTS = {
    "trend": 0.30,       # 站上均線 + 短均在長均之上
    "momentum": 0.25,    # RSI 落在健康動能區間
    "volume": 0.20,      # 今日量能相對 20 日均量的放大程度
    "relative": 0.25,    # 近 5 日報酬 相對 大盤的超額表現
}


# ---- 資料結構 -------------------------------------------------------------

@dataclass
class Candidate:
    code: str
    name: str
    close: float
    change_pct: float
    score: float = 0.0
    tags: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


# ---- Stage 1: 全市場快照 ---------------------------------------------------

def fetch_market_snapshot() -> pd.DataFrame:
    """呼叫 TWSE STOCK_DAY_ALL,回傳今日全上市股票快照的 DataFrame。"""
    resp = requests.get(TWSE_STOCK_DAY_ALL, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        raise RuntimeError("TWSE STOCK_DAY_ALL 回傳空資料,可能是非交易日或 API 異常。")

    df = pd.DataFrame(raw)

    # TWSE 的欄位名稱偶爾會有出入,這裡做一個保守的容錯:
    # 找不到預期欄位時,把原始欄位印出來方便排錯,而不是直接爛掉。
    expected = {"Code", "Name", "ClosingPrice", "TradeVolume", "Change"}
    missing = expected - set(df.columns)
    if missing:
        print(f"[警告] STOCK_DAY_ALL 缺少預期欄位 {missing},實際欄位: {list(df.columns)}",
              file=sys.stderr)
        print("[提示] 請對照官方文件更新欄位對應: "
              "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", file=sys.stderr)

    def to_float(s):
        return pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False).str.replace("X", "", regex=False),
            errors="coerce",
        )

    df["close"] = to_float(df.get("ClosingPrice", pd.Series(dtype=object)))
    df["volume"] = to_float(df.get("TradeVolume", pd.Series(dtype=object)))
    df["change"] = to_float(df.get("Change", pd.Series(dtype=object)))
    df["code"] = df.get("Code", pd.Series(dtype=object)).astype(str).str.strip()
    df["name"] = df.get("Name", pd.Series(dtype=object)).astype(str).str.strip()

    df["change_pct"] = np.where(
        (df["close"] - df["change"]) > 0,
        df["change"] / (df["close"] - df["change"]) * 100,
        np.nan,
    )

    # 只保留 4 碼的普通股代號,排除 ETF(通常 00xxxx)、權證等衍生商品,
    # 這是一個保守的預設篩法,想涵蓋 ETF 的話把這行拿掉即可。
    df = df[df["code"].str.fullmatch(r"\d{4}")]

    return df.dropna(subset=["close", "volume"])


# ---- Stage 2: 流動性初篩 ---------------------------------------------------

def screen_candidates(df: pd.DataFrame) -> pd.DataFrame:
    liquid = df[
        (df["volume"] >= MIN_TRADE_VOLUME_SHARES)
        & (df["close"].between(MIN_PRICE, MAX_PRICE))
    ].copy()

    # 依成交值(約略以 close * volume 估計)排序,取前 N 檔進入下一階段
    liquid["turnover_est"] = liquid["close"] * liquid["volume"]
    liquid = liquid.sort_values("turnover_est", ascending=False).head(CANDIDATE_POOL_SIZE)
    return liquid


# ---- Stage 3: 歷史資料 + 技術指標 -------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def fetch_history(codes: list[str]) -> dict[str, pd.DataFrame]:
    """用 yfinance 批次抓候選池的歷史日K,回傳 {code: DataFrame}。"""
    import yfinance as yf

    tickers = [f"{c}.TW" for c in codes] + [TAIEX_TICKER]
    raw = yf.download(
        tickers, period=HISTORY_DAYS, interval="1d",
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
    )

    out: dict[str, pd.DataFrame] = {}
    for code in codes:
        ticker = f"{code}.TW"
        try:
            sub = raw[ticker].dropna(how="all")
        except (KeyError, IndexError):
            continue
        if len(sub) >= 25:  # 至少要有夠算 20 日均線的資料
            out[code] = sub

    try:
        out[TAIEX_TICKER] = raw[TAIEX_TICKER].dropna(how="all")
    except (KeyError, IndexError):
        pass

    return out


def score_candidate(code: str, name: str, hist: pd.DataFrame, taiex: pd.DataFrame | None) -> Candidate | None:
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    if len(close) < 25:
        return None

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    rsi = compute_rsi(close, 14)
    vol_avg20 = volume.rolling(20).mean()

    last_close = close.iloc[-1]
    prev_close = close.iloc[-2]
    change_pct = (last_close / prev_close - 1) * 100

    tags: list[str] = []

    # -- 趨勢: 站上短均且短均在長均之上 --
    trend_ok = last_close > ma20.iloc[-1] and ma5.iloc[-1] > ma20.iloc[-1]
    trend_score = 100 if trend_ok else max(0, 50 - abs(last_close - ma20.iloc[-1]) / ma20.iloc[-1] * 100)

    # 近 3 日內是否剛發生 5/20 均線黃金交叉
    cross_recent = False
    diff_series = (ma5 - ma20).dropna()
    if len(diff_series) >= 4:
        recent = diff_series.iloc[-4:]
        cross_recent = (recent.iloc[0] < 0) and (recent.iloc[-1] > 0)
    if cross_recent:
        tags.append("5/20MA黃金交叉")
        trend_score = min(100, trend_score + 15)

    # -- 動能: RSI 落在健康偏強區間,且不是嚴重超買 --
    last_rsi = rsi.iloc[-1]
    if pd.isna(last_rsi):
        momentum_score = 50
    elif 50 <= last_rsi <= 70:
        momentum_score = 100
        if rsi.iloc[-2] < 50 <= last_rsi:
            tags.append("RSI轉強")
    elif 70 < last_rsi <= 80:
        momentum_score = 65
    elif last_rsi > 80:
        momentum_score = 30
        tags.append("短線過熱")
    else:
        momentum_score = max(0, last_rsi)  # RSI 偏低,動能偏弱

    # -- 量能: 今日量 / 20日均量 --
    vol_ratio = volume.iloc[-1] / vol_avg20.iloc[-1] if vol_avg20.iloc[-1] else np.nan
    if pd.isna(vol_ratio):
        volume_score = 50
    else:
        volume_score = float(np.clip((vol_ratio - 0.8) / (2.5 - 0.8) * 100, 0, 100))
        if vol_ratio >= 1.5:
            tags.append("成交量放大")

    # -- 相對強弱: 近 5 日報酬 vs 大盤近 5 日報酬 --
    relative_score = 50.0
    if taiex is not None and len(taiex["Close"].dropna()) >= 6 and len(close) >= 6:
        stock_5d = close.iloc[-1] / close.iloc[-6] - 1
        index_5d = taiex["Close"].dropna().iloc[-1] / taiex["Close"].dropna().iloc[-6] - 1
        excess = stock_5d - index_5d
        relative_score = float(np.clip(50 + excess * 1000, 0, 100))
        if excess > 0:
            tags.append("強於大盤")

    total = (
        trend_score * WEIGHTS["trend"]
        + momentum_score * WEIGHTS["momentum"]
        + volume_score * WEIGHTS["volume"]
        + relative_score * WEIGHTS["relative"]
    )

    return Candidate(
        code=code,
        name=name,
        close=round(float(last_close), 2),
        change_pct=round(float(change_pct), 2),
        score=round(float(total), 1),
        tags=tags,
        detail={
            "rsi14": None if pd.isna(last_rsi) else round(float(last_rsi), 1),
            "vol_ratio_20d": None if pd.isna(vol_ratio) else round(float(vol_ratio), 2),
            "above_ma20": bool(last_close > ma20.iloc[-1]),
        },
    )


# ---- Stage 4: 組裝輸出 -----------------------------------------------------

def build_output(candidates: list[Candidate], top_n: int, taiex_change_pct: float | None) -> dict:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
    tz = timezone(timedelta(hours=8))  # 台北時間
    return {
        "updated_at": datetime.now(tz).isoformat(),
        "taiex_change_pct": taiex_change_pct,
        "methodology": {
            "horizon": "短線 1~2 週波段參考,非當沖、非長期持有",
            "weights": WEIGHTS,
            "note": "本清單為技術面規則篩選結果,不構成投資建議,過去表現不代表未來績效。",
        },
        "candidates": [
            {
                "code": c.code,
                "name": c.name,
                "close": c.close,
                "change_pct": c.change_pct,
                "score": c.score,
                "tags": c.tags,
                "detail": c.detail,
            }
            for c in ranked
        ],
    }


# ---- Demo 模式(離線,方便測試 / 預覽前端) -----------------------------------

def build_demo_output(top_n: int) -> dict:
    rng = np.random.default_rng(42)
    sample_names = [
        ("2330", "台積電"), ("2317", "鴻海"), ("2454", "聯發科"), ("3231", "緯創"),
        ("2382", "廣達"), ("2603", "長榮"), ("3008", "大立光"), ("2379", "瑞昱"),
        ("6669", "緯穎"), ("3661", "世芯-KY"), ("2308", "台達電"), ("2412", "中華電"),
        ("1301", "台塑"), ("2891", "中信金"), ("2881", "富邦金"), ("3037", "欣興"),
        ("2345", "智邦"), ("6446", "藥華藥"), ("2327", "國巨"), ("5347", "世界"),
    ]
    tag_pool = ["5/20MA黃金交叉", "成交量放大", "強於大盤", "RSI轉強", "短線過熱"]
    candidates = []
    for code, name in sample_names[:top_n]:
        score = round(float(rng.uniform(55, 92)), 1)
        n_tags = rng.integers(1, 4)
        tags = list(rng.choice(tag_pool, size=n_tags, replace=False))
        candidates.append({
            "code": code, "name": name,
            "close": round(float(rng.uniform(30, 900)), 1),
            "change_pct": round(float(rng.uniform(-3, 8)), 2),
            "score": score, "tags": tags,
            "detail": {
                "rsi14": round(float(rng.uniform(45, 78)), 1),
                "vol_ratio_20d": round(float(rng.uniform(0.9, 2.6)), 2),
                "above_ma20": bool(rng.random() > 0.2),
            },
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)

    tz = timezone(timedelta(hours=8))
    return {
        "updated_at": datetime.now(tz).isoformat(),
        "taiex_change_pct": round(float(rng.uniform(-1.5, 1.5)), 2),
        "demo": True,
        "methodology": {
            "horizon": "短線 1~2 週波段參考,非當沖、非長期持有",
            "weights": WEIGHTS,
            "note": "這是 demo 假資料,僅供預覽畫面用,不是真實掃描結果。",
        },
        "candidates": candidates,
    }


# ---- main -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="離線 demo 模式,產生假資料")
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT, help="輸出前 N 名")
    parser.add_argument("--out", type=str, default="docs/data.json", help="輸出路徑")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("[demo] 產生離線假資料中...")
        output = build_demo_output(args.top)
    else:
        print("[1/4] 抓取 TWSE 全市場快照...")
        snapshot = fetch_market_snapshot()
        print(f"      取得 {len(snapshot)} 檔股票的今日快照")

        print("[2/4] 流動性初篩...")
        pool = screen_candidates(snapshot)
        print(f"      候選池: {len(pool)} 檔")

        print("[3/4] 抓取歷史K線並計算指標(需要幾分鐘,請耐心等候)...")
        hist_map = fetch_history(pool["code"].tolist())
        taiex_hist = hist_map.get(TAIEX_TICKER)
        taiex_change_pct = None
        if taiex_hist is not None and len(taiex_hist["Close"].dropna()) >= 2:
            c = taiex_hist["Close"].dropna()
            taiex_change_pct = round(float((c.iloc[-1] / c.iloc[-2] - 1) * 100), 2)

        candidates = []
        name_map = dict(zip(pool["code"], pool["name"]))
        for code, hist in hist_map.items():
            if code == TAIEX_TICKER:
                continue
            cand = score_candidate(code, name_map.get(code, code), hist, taiex_hist)
            if cand:
                candidates.append(cand)
        print(f"      成功計算指標: {len(candidates)} 檔")

        print("[4/4] 排序並輸出...")
        output = build_output(candidates, args.top, taiex_change_pct)

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成,已寫入 {out_path}")


if __name__ == "__main__":
    main()
