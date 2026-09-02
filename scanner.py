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
     MAX_PRICE 只排除極端高價股(報價/流動性),不是用來判斷「有沒有漲多」。

  3) 歷史資料 + 技術指標
     對候選池用 yfinance 抓近 90 個交易日的日K,計算均線、Wilder RSI(14)、
     ATR 延伸、量能比、相對大盤強弱。網頁上的現價/漲跌優先用證交所快照。

  4) 結構分 / 擁擠分 + 輸出
     動能延續本身是 1–2 週持有期的合法因子;這裡要區分的是
     「動能剛啟動」(結構) vs 「動能已經走很遠」(擁擠)。
     各因子在候選池內做百分位,結構高、擁擠中等者排前面。
     接近漲停/今日大漲另外降權,並標成風險條件而不是加分。

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
import math
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
MAX_PRICE = 2000.0                       # 排除極端高價股(流動性/報價習慣),不是防追高
CANDIDATE_POOL_SIZE = 300                # 進入歷史資料階段的候選檔數上限
HISTORY_DAYS = "90d"                     # 抓多久的歷史K線
TOP_N_DEFAULT = 20
SNAPSHOTS_KEEP_DAYS = 30

# 結構分:趨勢剛對齊、RSI 健康、價格尚未過度延伸
STRUCTURE_WEIGHTS = {
    "trend": 0.40,
    "rsi": 0.35,
    "not_extended": 0.25,
}
# 擁擠分:量已放大、相對大盤已超額、當日波動大、價格延伸
CROWDING_WEIGHTS = {
    "volume": 0.25,
    "relative": 0.25,
    "day_move": 0.25,
    "extension": 0.25,
}
CROWDING_PENALTY = 0.50                  # 觀察分 = 結構 − 擁擠 × 此係數 − 額外降權
NEAR_LIMIT_PCT = 9.5                     # 接近漲停(一般股 ±10%)
BIG_UP_PCT = 7.0                         # 今日大漲
EXT_ATR_THRESHOLD = 2.5                  # ATR 延伸過遠
NEAR_HIGH_PCT = 0.01                     # 距 20 日高點 1% 內視為延伸
NEAR_LIMIT_PENALTY = 25.0
BIG_UP_PENALTY = 12.0
DEAD_TAPE_VOL = 0.85                 # 低於此視為沒人參與,不是「剛啟動」
STRONG_VOL_RATIO = 1.00              # 強力推薦至少要有均量
STRONG_EXCESS_5D = 0.0               # 近 5 日至少不輸大盤(百分點)
SCALE_EXCESS_5D = -1.0               # 分批進場允許小幅落後
CROWDING_STRONG_MIN = 22.0           # 擁擠過低 = 冷清,不是啟動
CROWDING_STRONG_MAX = 50.0           # 擁擠過高 = 已經走一截
DEAD_TAPE_PENALTY = 10.0             # 觀察分對冷清量能扣分,避免沒人買排第一
STOP_ATR = 1.5                       # 停損距離(×ATR),常見波動停損
TP1_R = 1.5                          # 第一停利 = 1.5 倍風險
TP2_R = 2.5                          # 第二停利 = 2.5 倍風險
ENTRY_ATR_PULLBACK = 0.5             # 進場下緣最多往下等 0.5×ATR
HORIZON = "5–10 個交易日"

ACTION_LABELS = {
    "strong_buy": "強力推薦",
    "scale_in": "分批進場",
    "watch": "觀望",
    "trim": "部分賣出",
    "exit": "出清",
}
ACTION_CAPS = {
    "strong_buy": 5,
    "scale_in": 8,
    "watch": 8,
    "trim": 8,
    "exit": 8,
}

# ---- 資料結構 -------------------------------------------------------------

@dataclass
class FeatureRow:
    code: str
    name: str
    close: float
    change_pct: float
    trend_raw: float
    rsi_health_raw: float
    not_extended_raw: float
    vol_raw: float
    relative_raw: float
    day_move_raw: float
    extension_raw: float
    rsi14: float | None
    vol_ratio: float | None
    above_ma20: bool
    atr_extension: float | None
    dist_from_high: float | None
    excess_5d: float | None
    atr_abs: float | None = None
    ma5: float | None = None
    ma20: float | None = None
    swing_low_10: float | None = None
    swing_high_20: float | None = None
    tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    code: str
    name: str
    close: float
    change_pct: float
    structure: float = 0.0
    crowding: float = 0.0
    score: float = 0.0
    tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    action: str = "watch"
    action_label: str = "觀望"
    reason: str = ""
    plan: dict | None = None


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

def compute_rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI,與券商 / TradingView 常見 RSI(14) 對得上。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))
    no_loss = avg_loss.fillna(0).eq(0)
    rsi = rsi.mask(no_loss & avg_gain.gt(0), 100.0)
    rsi = rsi.mask(no_loss & avg_gain.eq(0), 50.0)
    return rsi


def compute_atr_wilder(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    high = hist["High"]
    low = hist["Low"]
    close = hist["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rsi_health_raw(rsi: float) -> float:
    """RSI 健康度:在 50–70 之間呈山形,峰值約 58,不是整段平台 100 分。"""
    if pd.isna(rsi) or rsi < 40 or rsi > 80:
        return 0.0
    if rsi <= 58:
        return (rsi - 40) / 18.0
    return max(0.0, (75 - rsi) / 17.0)


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
        if len(sub) >= 40:  # Wilder RSI/ATR 需要比 20 日均線更多的暖身
            out[code] = sub

    try:
        out[TAIEX_TICKER] = raw[TAIEX_TICKER].dropna(how="all")
    except (KeyError, IndexError):
        pass

    return out


def extract_features(
    code: str,
    name: str,
    hist: pd.DataFrame,
    taiex: pd.DataFrame | None,
    twse_close: float | None,
    twse_change_pct: float | None,
) -> FeatureRow | None:
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    if len(close) < 40:
        return None

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    rsi = compute_rsi_wilder(close, 14)
    atr = compute_atr_wilder(hist, 14)
    vol_avg20 = volume.rolling(20).mean()
    high20 = hist["High"].rolling(20).max()
    low10 = hist["Low"].rolling(10).min()

    last_close_yf = float(close.iloc[-1])
    prev_close_yf = float(close.iloc[-2])
    yf_change_pct = (last_close_yf / prev_close_yf - 1) * 100

    display_close = float(twse_close) if twse_close is not None and not pd.isna(twse_close) else last_close_yf
    display_change = (
        float(twse_change_pct)
        if twse_change_pct is not None and not pd.isna(twse_change_pct)
        else yf_change_pct
    )

    last_ma5 = float(ma5.iloc[-1])
    last_ma20 = float(ma20.iloc[-1])
    last_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
    last_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) and atr.iloc[-1] else None
    last_high20 = float(high20.iloc[-1]) if not pd.isna(high20.iloc[-1]) and high20.iloc[-1] else None
    last_low10 = float(low10.iloc[-1]) if not pd.isna(low10.iloc[-1]) and low10.iloc[-1] else None

    vol_ratio = None
    if vol_avg20.iloc[-1] and not pd.isna(vol_avg20.iloc[-1]):
        vol_ratio = float(volume.iloc[-1] / vol_avg20.iloc[-1])

    atr_extension = None
    if last_atr:
        atr_extension = (last_close_yf - last_ma20) / last_atr

    dist_from_high = None
    if last_high20:
        dist_from_high = (last_high20 - last_close_yf) / last_high20

    excess_5d = None
    if taiex is not None and len(taiex["Close"].dropna()) >= 6 and len(close) >= 6:
        stock_5d = close.iloc[-1] / close.iloc[-6] - 1
        index_5d = taiex["Close"].dropna().iloc[-1] / taiex["Close"].dropna().iloc[-6] - 1
        excess_5d = float(stock_5d - index_5d)

    tags: list[str] = []
    risk_tags: list[str] = []

    diff_series = (ma5 - ma20).dropna()
    if len(diff_series) >= 4:
        recent = diff_series.iloc[-4:]
        if (recent.iloc[0] < 0) and (recent.iloc[-1] > 0):
            tags.append("5/20MA黃金交叉")

    if last_rsi is not None and len(rsi.dropna()) >= 2:
        prev_rsi = float(rsi.dropna().iloc[-2])
        if prev_rsi < 50 <= last_rsi:
            tags.append("RSI轉強")
        if last_rsi > 80:
            risk_tags.append("短線過熱")

    if vol_ratio is not None and vol_ratio >= 1.5:
        tags.append("成交量放大")

    if excess_5d is not None and excess_5d > 0:
        tags.append("強於大盤")

    if display_change >= NEAR_LIMIT_PCT:
        risk_tags.append("接近漲停")
    elif display_change >= BIG_UP_PCT:
        risk_tags.append("今日大漲")

    if display_change <= -5 and vol_ratio is not None and vol_ratio >= 1.8:
        risk_tags.append("放量長陰")

    if (atr_extension is not None and atr_extension >= EXT_ATR_THRESHOLD) or (
        dist_from_high is not None and dist_from_high <= NEAR_HIGH_PCT
    ):
        risk_tags.append("延伸過遠")

    trend_raw = (last_ma5 - last_ma20) / last_ma20 if last_ma20 else 0.0
    rsi_raw = rsi_health_raw(last_rsi if last_rsi is not None else np.nan)
    ext_for_crowd = atr_extension if atr_extension is not None else 0.0
    not_extended_raw = -ext_for_crowd

    return FeatureRow(
        code=code,
        name=name,
        close=round(display_close, 2),
        change_pct=round(float(display_change), 2),
        trend_raw=float(trend_raw),
        rsi_health_raw=float(rsi_raw),
        not_extended_raw=float(not_extended_raw),
        vol_raw=float(vol_ratio) if vol_ratio is not None else np.nan,
        relative_raw=float(excess_5d) if excess_5d is not None else np.nan,
        day_move_raw=float(abs(display_change)),
        extension_raw=float(ext_for_crowd),
        rsi14=None if last_rsi is None else round(last_rsi, 1),
        vol_ratio=None if vol_ratio is None else round(vol_ratio, 2),
        above_ma20=bool(last_close_yf > last_ma20),
        atr_extension=None if atr_extension is None else round(atr_extension, 2),
        dist_from_high=None if dist_from_high is None else round(dist_from_high * 100, 2),
        excess_5d=None if excess_5d is None else round(excess_5d * 100, 2),
        atr_abs=None if last_atr is None else round(last_atr, 4),
        ma5=round(last_ma5, 4),
        ma20=round(last_ma20, 4),
        swing_low_10=None if last_low10 is None else round(last_low10, 4),
        swing_high_20=None if last_high20 is None else round(last_high20, 4),
        tags=tags,
        risk_tags=risk_tags,
    )


def _pct_rank(values: list[float]) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    return s.rank(method="average", pct=True, na_option="keep").fillna(0.5).to_numpy() * 100.0


def extra_penalty(change_pct: float) -> float:
    if change_pct >= NEAR_LIMIT_PCT:
        return NEAR_LIMIT_PENALTY
    if change_pct >= BIG_UP_PCT:
        return BIG_UP_PENALTY
    return 0.0


def rank_features(rows: list[FeatureRow]) -> list[Candidate]:
    if not rows:
        return []

    trend_p = _pct_rank([r.trend_raw for r in rows])
    rsi_p = _pct_rank([r.rsi_health_raw for r in rows])
    n_ext_p = _pct_rank([r.not_extended_raw for r in rows])
    vol_p = _pct_rank([r.vol_raw for r in rows])
    rel_p = _pct_rank([r.relative_raw for r in rows])
    day_p = _pct_rank([r.day_move_raw for r in rows])
    ext_p = _pct_rank([r.extension_raw for r in rows])

    out: list[Candidate] = []
    for i, r in enumerate(rows):
        structure = (
            trend_p[i] * STRUCTURE_WEIGHTS["trend"]
            + rsi_p[i] * STRUCTURE_WEIGHTS["rsi"]
            + n_ext_p[i] * STRUCTURE_WEIGHTS["not_extended"]
        )
        crowding = (
            vol_p[i] * CROWDING_WEIGHTS["volume"]
            + rel_p[i] * CROWDING_WEIGHTS["relative"]
            + day_p[i] * CROWDING_WEIGHTS["day_move"]
            + ext_p[i] * CROWDING_WEIGHTS["extension"]
        )
        score = structure - CROWDING_PENALTY * crowding - extra_penalty(r.change_pct)
        if r.vol_raw is not None and not pd.isna(r.vol_raw) and r.vol_raw < DEAD_TAPE_VOL:
            score -= DEAD_TAPE_PENALTY
        if r.relative_raw is not None and not pd.isna(r.relative_raw) and r.relative_raw < 0:
            score -= 6.0
        out.append(
            Candidate(
                code=r.code,
                name=r.name,
                close=r.close,
                change_pct=r.change_pct,
                structure=round(float(structure), 1),
                crowding=round(float(crowding), 1),
                score=round(float(score), 1),
                tags=r.tags,
                risk_tags=r.risk_tags,
                detail={
                    "rsi14": r.rsi14,
                    "vol_ratio_20d": r.vol_ratio,
                    "above_ma20": r.above_ma20,
                    "atr_extension": r.atr_extension,
                    "dist_from_high_20d_pct": r.dist_from_high,
                    "excess_5d_pct": r.excess_5d,
                    "atr": r.atr_abs,
                    "ma5": None if r.ma5 is None else round(r.ma5, 2),
                    "ma20": None if r.ma20 is None else round(r.ma20, 2),
                    "swing_low_10": None if r.swing_low_10 is None else round(r.swing_low_10, 2),
                    "swing_high_20": None if r.swing_high_20 is None else round(r.swing_high_20, 2),
                    "factors": {
                        "trend": round(float(trend_p[i]), 0),
                        "rsi": round(float(rsi_p[i]), 0),
                        "not_extended": round(float(n_ext_p[i]), 0),
                        "volume": round(float(vol_p[i]), 0),
                        "relative": round(float(rel_p[i]), 0),
                        "day_move": round(float(day_p[i]), 0),
                    },
                },
            )
        )
    return out


def classify_action(c: Candidate) -> tuple[str, str]:
    """短線規則分級。出場是「若已持有」的假設,不是叫沒持股的人去放空。"""
    d = c.detail or {}
    rsi = d.get("rsi14")
    atr = d.get("atr_extension")
    above = d.get("above_ma20")
    vol = d.get("vol_ratio_20d")
    excess = d.get("excess_5d_pct")
    risks = set(c.risk_tags or [])

    if "接近漲停" in risks:
        return "exit", "接近漲停,短線空間被用完的機率高"
    if "放量長陰" in risks:
        return "exit", "放量長陰,偏高潮後回吐"
    if rsi is not None and rsi > 80:
        return "exit", f"RSI {rsi} 過熱"
    if atr is not None and atr >= EXT_ATR_THRESHOLD:
        return "exit", f"價格已延伸 {atr:.1f} 倍 ATR"
    if above is False and c.crowding >= 55:
        return "exit", "跌破均線且擁擠仍高,結構轉弱"

    if "今日大漲" in risks:
        return "trim", "今日大漲,若已持有考慮先減碼"
    if "延伸過遠" in risks:
        return "trim", "靠近高點或延伸過遠,建議部分停利"
    if "短線過熱" in risks or (rsi is not None and rsi > 70):
        return "trim", f"RSI {rsi} 偏熱,適合減碼而非加碼"
    if c.crowding >= 70:
        return "trim", "擁擠分偏高,追價風險大"

    dead_volume = vol is not None and vol < DEAD_TAPE_VOL
    lagging = excess is not None and excess < SCALE_EXCESS_5D
    if dead_volume or lagging:
        bits = []
        if dead_volume:
            bits.append(f"量比 {vol}")
        if lagging:
            bits.append(f"近5日相對大盤 {excess}%")
        return "watch", "、".join(bits) + ",比較像沒人買或相對弱,不是剛啟動"

    rsi_ok = rsi is not None and 50 <= rsi <= 65
    quiet_day = abs(c.change_pct) < 5
    not_extended = atr is None or atr < 1.5
    vol_strong = vol is None or vol >= STRONG_VOL_RATIO
    not_lagging = excess is None or excess >= STRONG_EXCESS_5D
    crowding_moderate = CROWDING_STRONG_MIN <= c.crowding <= CROWDING_STRONG_MAX
    if (
        above
        and c.structure >= 68
        and crowding_moderate
        and rsi_ok
        and not risks
        and not_extended
        and quiet_day
        and vol_strong
        and not_lagging
    ):
        return "strong_buy", "結構高、量能有跟上、沒明顯輸大盤,比較像動能剛啟動"

    rsi_scale = rsi is None or (48 <= rsi <= 70)
    if (
        above
        and c.structure >= 55
        and c.crowding <= 58
        and rsi_scale
        and c.change_pct < BIG_UP_PCT
        and (atr is None or atr < 2.2)
    ):
        return "scale_in", "結構尚可且不是冷清量能,適合分批而不是一次買滿"

    return "watch", "條件好壞參半,先看不急著動"


def assign_actions(candidates: list[Candidate]) -> None:
    for c in candidates:
        key, reason = classify_action(c)
        c.action = key
        c.action_label = ACTION_LABELS[key]
        c.reason = reason
        if key in ("strong_buy", "scale_in"):
            c.plan = build_trade_plan(c)
        else:
            c.plan = None


def twse_tick(price: float) -> float:
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.10
    if price < 500:
        return 0.50
    if price < 1000:
        return 1.0
    return 5.0


def round_tick(price: float, mode: str = "nearest") -> float:
    if price <= 0:
        return 0.0
    tick = twse_tick(price)
    n = price / tick
    if mode == "down":
        n = math.floor(n + 1e-9)
    elif mode == "up":
        n = math.ceil(n - 1e-9)
    else:
        n = round(n)
    return round(n * tick, 4)


def build_trade_plan(c: Candidate) -> dict | None:
    """公開可覆核的短線計畫:ATR 停損 + 波段低點 + 固定 R 倍數停利。"""
    d = c.detail or {}
    atr = d.get("atr")
    ma5 = d.get("ma5")
    ma20 = d.get("ma20")
    low10 = d.get("swing_low_10")
    high20 = d.get("swing_high_20")
    close = float(c.close)
    if not atr or atr <= 0 or close <= 0:
        return None

    if ma5 is not None and 0 < ma5 < close:
        entry_low = max(float(ma5), close - ENTRY_ATR_PULLBACK * atr)
    else:
        entry_low = close - 0.3 * atr
    entry_high = close
    if entry_low >= entry_high:
        entry_low = close - 0.3 * atr
    entry_low = round_tick(entry_low, "down")
    entry_high = round_tick(entry_high, "nearest")
    if entry_low > entry_high:
        entry_low = entry_high
    entry_mid = (entry_low + entry_high) / 2

    stop_atr = entry_low - STOP_ATR * atr
    stop = stop_atr
    if low10:
        swing = float(low10) - twse_tick(float(low10))
        if swing < entry_low:
            # 短線用較近的停損:10日低點若離入場不超過 2.5×ATR,才採用
            if 0 < (entry_low - swing) <= 2.5 * atr:
                stop = max(stop_atr, swing)
    stop = round_tick(stop, "down")
    if stop >= entry_low:
        stop = round_tick(entry_low - STOP_ATR * atr, "down")
    if stop >= entry_low or entry_mid <= stop:
        return None

    risk = entry_mid - stop
    tp1 = round_tick(entry_mid + TP1_R * risk, "up")
    tp2 = round_tick(entry_mid + TP2_R * risk, "up")
    return {
        "horizon": HORIZON,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "resistance_20d": None if not high20 else round_tick(float(high20), "nearest"),
        "invalid_below": None if not ma20 else round_tick(float(ma20), "down"),
        "atr": round(float(atr), 4),
        "risk_pct": round(risk / entry_mid * 100, 2),
        "tp1_pct": round((tp1 - entry_mid) / entry_mid * 100, 2),
        "tp2_pct": round((tp2 - entry_mid) / entry_mid * 100, 2),
        "rr1": TP1_R,
        "rr2": TP2_R,
        "method": "進場等回檔至5日均線附近;停損預設1.5×Wilder ATR,若10日低點更近且仍在2.5×ATR內則改用結構停損;停利1.5R/2.5R。跳動單位依證交所規定。",
    }


def _action_sort_key(c: Candidate) -> tuple:
    if c.action in ("trim", "exit"):
        return (-c.crowding, -abs(c.change_pct), -c.score)
    return (-c.score, c.crowding)


def build_lazy_pack(candidates: list[Candidate]) -> dict:
    buckets: dict[str, list[Candidate]] = {k: [] for k in ACTION_LABELS}
    for c in candidates:
        buckets[c.action].append(c)

    pack = {}
    counts = {}
    for key, label in ACTION_LABELS.items():
        items = sorted(buckets[key], key=_action_sort_key)
        counts[key] = len(items)
        pack[key] = {
            "label": label,
            "count": len(items),
            "items": [candidate_payload(c) for c in items[: ACTION_CAPS[key]]],
        }
    pack["counts"] = counts
    entry_rows = []
    for key in ("strong_buy", "scale_in"):
        for c in sorted(buckets[key], key=_action_sort_key):
            if c.plan:
                entry_rows.append(candidate_payload(c))
            if len(entry_rows) >= 12:
                break
        if len(entry_rows) >= 12:
            break
    pack["trade_table"] = entry_rows
    pack["trade_method"] = (
        "沒有法人籌碼或某分析師內部模型。進場區間=現價回檔至5日均線附近;"
        "停損預設1.5×Wilder ATR,10日低點只在距離夠近時才用;停利=1.5R/2.5R。"
        "參考持有5–10個交易日,收盤跌破20日均線視為結構失效。風險自負。"
    )
    pack["note"] = (
        "這是同一套技術規則的分級建議,不是保證獲利的進出場點。"
        "強力推薦需要量能跟上且近5日不輸大盤;沒人買的冷清改觀望。"
        "出場假設你已經持有,未持有請忽略出場清單。"
        "任何買賣都是你自己的判斷與責任。"
    )
    return pack


# ---- Stage 4: 組裝輸出 -----------------------------------------------------

def taipei_today() -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).date().isoformat()


def snapshots_path_for(out_path: Path) -> Path:
    return out_path.parent / "snapshots.json"


def load_snapshots(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    days = data.get("days", [])
    return days if isinstance(days, list) else []


def save_snapshots(path: Path, days: list[dict]) -> None:
    path.write_text(
        json.dumps({"days": days[-SNAPSHOTS_KEEP_DAYS:]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _avg_return(prev_candidates: list[dict], today_map: dict[str, float]) -> tuple[float | None, int]:
    rets = []
    for item in prev_candidates:
        code = str(item.get("code", ""))
        prev_close = item.get("close")
        today_close = today_map.get(code)
        if prev_close and today_close and prev_close > 0:
            rets.append(today_close / prev_close - 1)
    if not rets:
        return None, 0
    return round(float(np.mean(rets) * 100), 2), len(rets)


def build_review(
    history_days: list[dict],
    today_df: pd.DataFrame,
    today_date: str,
    taiex_change_pct: float | None,
) -> dict | None:
    usable = [d for d in history_days if d.get("date") and d["date"] != today_date]
    if not usable:
        return None

    today_map = dict(zip(today_df["code"].astype(str), today_df["close"].astype(float)))
    prev = usable[-1]
    avg_1d, n_1d = _avg_return(prev.get("candidates") or [], today_map)
    review = {
        "prev_date": prev["date"],
        "n": n_1d,
        "avg_return_1d": avg_1d,
        "taiex_return_1d": taiex_change_pct,
        "excess_1d": None
        if avg_1d is None or taiex_change_pct is None
        else round(avg_1d - taiex_change_pct, 2),
        "avg_return_5d": None,
        "n_5d": 0,
        "from_date_5d": None,
        "note": "清單報酬用證交所收盤價對帳;大盤日報酬來自 Yahoo Finance ^TWII。",
    }

    if len(usable) >= 5:
        d5 = usable[-5]
        avg_5d, n_5d = _avg_return(d5.get("candidates") or [], today_map)
        review["from_date_5d"] = d5["date"]
        review["avg_return_5d"] = avg_5d
        review["n_5d"] = n_5d

    return review


def candidate_payload(c: Candidate) -> dict:
    return {
        "code": c.code,
        "name": c.name,
        "close": c.close,
        "change_pct": c.change_pct,
        "structure": c.structure,
        "crowding": c.crowding,
        "score": c.score,
        "action": c.action,
        "action_label": c.action_label,
        "reason": c.reason,
        "plan": c.plan,
        "tags": c.tags,
        "risk_tags": c.risk_tags,
        "detail": c.detail,
    }


def build_output(
    candidates: list[Candidate],
    top_n: int,
    taiex_change_pct: float | None,
    review: dict | None,
    as_of: str,
) -> dict:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
    tz = timezone(timedelta(hours=8))
    return {
        "updated_at": datetime.now(tz).isoformat(),
        "as_of": as_of,
        "taiex_change_pct": taiex_change_pct,
        "price_source": "現價與漲跌幅來自證交所 STOCK_DAY_ALL 當日快照",
        "indicator_source": "均線 / RSI(14, Wilder) / ATR / 量比 / 相對強弱來自 Yahoo Finance 歷史K線",
        "methodology": {
            "horizon": "短線 1~2 週波段參考,非當沖、非長期持有",
            "rsi": "RSI(14, Wilder)",
            "ranking": "觀察分 = 結構百分位 − 擁擠百分位 × 0.5 − 接近漲停/今日大漲降權",
            "actions": "強力推薦要結構對、量能跟上、近5日不輸大盤;冷清量能改觀望。出場假設已持有",
            "trade_plan": "進場等回檔至MA5;停損1.5×ATR(近10日低點僅在距離短時採用);停利1.5R/2.5R。不是法人大數據或特定分析師模型。",
            "weights": {
                "structure": STRUCTURE_WEIGHTS,
                "crowding": CROWDING_WEIGHTS,
                "crowding_penalty": CROWDING_PENALTY,
            },
            "note": "本清單為技術面規則篩選結果,不構成投資建議,過去表現不代表未來績效。動能延續是持有期因子;結構分偏「剛啟動」,擁擠分偏「已經走遠」。",
        },
        "review": review,
        "lazy_pack": build_lazy_pack(candidates),
        "candidates": [candidate_payload(c) for c in ranked],
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
    archetypes = [
        {"structure": 78, "crowding": 28, "rsi": 57, "chg": 0.8, "atr": 0.4, "above": True, "risks": [], "vol": 1.3, "excess": 1.2},
        {"structure": 72, "crowding": 36, "rsi": 54, "chg": -0.5, "atr": 0.6, "above": True, "risks": [], "vol": 1.1, "excess": 0.4},
        {"structure": 64, "crowding": 48, "rsi": 61, "chg": 1.2, "atr": 1.1, "above": True, "risks": [], "vol": 1.0, "excess": 0.2},
        {"structure": 58, "crowding": 52, "rsi": 66, "chg": 2.1, "atr": 1.4, "above": True, "risks": [], "vol": 0.95, "excess": -0.3},
        {"structure": 74, "crowding": 26, "rsi": 57, "chg": -0.9, "atr": 0.2, "above": True, "risks": [], "vol": 0.48, "excess": -4.3},
        {"structure": 42, "crowding": 60, "rsi": 46, "chg": -2.4, "atr": -0.3, "above": False, "risks": [], "vol": 1.2, "excess": -1.5},
        {"structure": 55, "crowding": 74, "rsi": 72, "chg": 4.2, "atr": 1.8, "above": True, "risks": ["延伸過遠"], "vol": 2.1, "excess": 3.0},
        {"structure": 48, "crowding": 82, "rsi": 76, "chg": 7.4, "atr": 2.1, "above": True, "risks": ["今日大漲"], "vol": 2.4, "excess": 4.1},
        {"structure": 40, "crowding": 88, "rsi": 83, "chg": 9.7, "atr": 2.8, "above": True, "risks": ["接近漲停"], "vol": 2.8, "excess": 6.0},
        {"structure": 38, "crowding": 79, "rsi": 41, "chg": -6.2, "atr": -0.8, "above": False, "risks": ["放量長陰"], "vol": 2.2, "excess": -5.0},
    ]
    cands: list[Candidate] = []
    for i, (code, name) in enumerate(sample_names):
        spec = archetypes[i] if i < len(archetypes) else {
            "structure": round(float(rng.uniform(40, 75)), 1),
            "crowding": round(float(rng.uniform(25, 70)), 1),
            "rsi": round(float(rng.uniform(45, 70)), 1),
            "chg": round(float(rng.uniform(-2, 3)), 2),
            "atr": round(float(rng.uniform(-0.2, 1.6)), 2),
            "above": True,
            "risks": [],
        }
        structure = float(spec["structure"])
        crowding = float(spec["crowding"])
        px = round(float(rng.uniform(30, 200)), 1)
        atr_abs = round(px * 0.025, 2)
        cands.append(
            Candidate(
                code=code,
                name=name,
                close=px,
                change_pct=float(spec["chg"]),
                structure=structure,
                crowding=crowding,
                score=round(structure - CROWDING_PENALTY * crowding, 1),
                tags=["強於大盤"] if crowding > 50 else [],
                risk_tags=list(spec["risks"]),
                detail={
                    "rsi14": spec["rsi"],
                    "vol_ratio_20d": float(spec.get("vol", round(float(rng.uniform(0.9, 1.8)), 2))),
                    "above_ma20": bool(spec["above"]),
                    "atr_extension": spec["atr"],
                    "dist_from_high_20d_pct": round(float(rng.uniform(0.3, 7.0)), 2),
                    "excess_5d_pct": float(spec.get("excess", round(float(rng.uniform(-0.5, 2.0)), 2))),
                    "atr": atr_abs,
                    "ma5": round(px * 0.985, 2),
                    "ma20": round(px * 0.97, 2),
                    "swing_low_10": round(px * 0.94, 2),
                    "swing_high_20": round(px * 1.05, 2),
                    "factors": {
                        "trend": 70, "rsi": 70, "not_extended": 60,
                        "volume": 40, "relative": 40, "day_move": 40,
                    },
                },
            )
        )
    assign_actions(cands)
    as_of = taipei_today()
    output = build_output(
        cands,
        top_n,
        round(float(rng.uniform(-1.5, 1.5)), 2),
        {
            "prev_date": as_of,
            "n": top_n,
            "avg_return_1d": round(float(rng.uniform(-1.2, 1.8)), 2),
            "taiex_return_1d": round(float(rng.uniform(-1.5, 1.5)), 2),
            "excess_1d": round(float(rng.uniform(-0.8, 1.2)), 2),
            "avg_return_5d": None,
            "n_5d": 0,
            "from_date_5d": None,
            "note": "demo 預覽用的假回顧數字。",
        },
        as_of,
    )
    output["demo"] = True
    output["methodology"]["note"] = "這是 demo 假資料,僅供預覽畫面用,不是真實掃描結果。"
    return output


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

        twse_close_map = dict(zip(pool["code"].astype(str), pool["close"]))
        twse_chg_map = dict(zip(pool["code"].astype(str), pool["change_pct"]))
        name_map = dict(zip(pool["code"].astype(str), pool["name"]))

        rows: list[FeatureRow] = []
        for code, hist in hist_map.items():
            if code == TAIEX_TICKER:
                continue
            feat = extract_features(
                code,
                name_map.get(code, code),
                hist,
                taiex_hist,
                twse_close_map.get(code),
                twse_chg_map.get(code),
            )
            if feat:
                rows.append(feat)
        print(f"      成功計算指標: {len(rows)} 檔")

        print("[4/4] 排序並輸出...")
        candidates = rank_features(rows)
        assign_actions(candidates)
        as_of = taipei_today()
        snap_path = snapshots_path_for(out_path)
        history = load_snapshots(snap_path)
        review = build_review(history, snapshot, as_of, taiex_change_pct)
        output = build_output(candidates, args.top, taiex_change_pct, review, as_of)

        today_entry = {
            "date": as_of,
            "taiex_change_pct": taiex_change_pct,
            "candidates": [
                {"code": c.code, "name": c.name, "close": c.close, "score": c.score}
                for c in sorted(candidates, key=lambda x: x.score, reverse=True)[: args.top]
            ],
        }
        history = [d for d in history if d.get("date") != as_of]
        history.append(today_entry)
        save_snapshots(snap_path, history)

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成,已寫入 {out_path}")


if __name__ == "__main__":
    main()
