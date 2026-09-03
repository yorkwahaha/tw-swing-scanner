#!/usr/bin/env python3
"""
臺股短線(1~2週)觀察清單 - 每日掃描腳本
=========================================

這支腳本做的事情很單純,分四個階段:

  1) 全市場快照
     優先用證交所官網盤後 STOCK_DAY_ALL(當日),失敗才退回 OpenAPI。
     一次拿到該交易日所有上市股票與 ETF 的收盤價、漲跌、成交量/成交金額。
     官網: https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL
     OpenAPI: https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL

  2) 流動性初篩
     用今天的成交量、成交值、股價做粗篩,並剔除處置股與變更交易(全額交割等)。
     MAX_PRICE 只排除極端高價股(報價/流動性),不是用來判斷「有沒有漲多」。

  3) 歷史資料 + 技術指標
     對候選池用 yfinance 抓近 1 年日K,計算均線、Wilder RSI(14)、
     ATR 延伸、量能比、相對大盤強弱。若 Yahoo 缺當日 K 棒,用證交所 OHLC 補上。
     網頁上的現價/漲跌優先用證交所快照。另抓三大法人(T86)當日與近三日投信。

  4) 結構分 / 擁擠分 + 輸出
     動能延續本身是 1–2 週持有期的合法因子;這裡要區分的是
     「動能剛啟動」(結構) vs 「動能已經走很遠」(擁擠)。
     各因子在候選池內做百分位,結構高、擁擠中等者排前面。
     接近漲停/今日大漲/暴跌另外降權,並標成風險條件而不是加分。

★ 重要聲明
本腳本純粹是「技術面規則 + 公開籌碼」篩選的產物,篩選邏輯是很常見的
動能/量能/均線/法人買賣超組合,不是什麼獨門秘技,也不構成投資建議。
過去的價量表現不保證未來報酬,短線交易風險本來就高,任何進出場決定
都是你自己的判斷與責任。

用法:
    python scanner.py                 # 正式模式,會打 TWSE + yfinance
    python scanner.py --demo          # demo 模式,純離線產生假資料(方便測試/預覽)
    python scanner.py --top 20        # 取前 20 名(預設 20)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TWSE_DAY_ALL_OFFICIAL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL"
TWSE_DAY_ALL_OPENAPI = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TWSE_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
TWSE_PUNISH = "https://openapi.twse.com.tw/v1/announcement/punish"
TWSE_ALTERED = "https://openapi.twse.com.tw/v1/exchangeReport/TWT85U"
TAIEX_TICKER = "^TWII"
HTTP_HEADERS = {
    "User-Agent": "tw-swing-scanner/1.1 (github.com/yorkwahaha/tw-swing-scanner)",
    "Accept": "application/json,text/csv,*/*",
}

# ---- 可調參數 -----------------------------------------------------------

MIN_TRADE_VOLUME_SHARES = 1_000_000     # 今日成交量下限(股),約 1,000 張
MIN_ETF_TRADE_VALUE = 80_000_000        # ETF 成交金額下限(元);量少但成交值夠仍可進
MIN_PRICE = 10.0                         # 普通股排除過度低價股
MIN_ETF_PRICE = 5.0                      # ETF 允許較低價
MAX_PRICE = 2000.0                       # 排除極端高價股(流動性/報價習慣),不是防追高
CANDIDATE_POOL_SIZE = 300                # 進入歷史資料階段的候選檔數上限
HISTORY_DAYS = "1y"                      # Wilder RSI/ATR 需要足夠暖身才接近券商數值
TOP_N_DEFAULT = 20
SNAPSHOTS_KEEP_DAYS = 30
EXIT_LOOKBACK_DAYS = 10

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
BIG_DOWN_PCT = -7.0                      # 今日大跌
NEAR_LIMIT_DOWN_PCT = -9.5               # 接近跌停
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
MIN_STOP_ATR = 1.0                   # 結構停損也至少要有 1×ATR,避免摩擦成本吃掉 R
MIN_STOP_PCT = 2.5                   # 或至少 2.5%,不夠繳手續費+交易稅就不做
TP1_R = 1.5                          # 第一停利 = 1.5 倍風險
TP2_R = 2.5                          # 第二停利 = 2.5 倍風險
ENTRY_ATR_PULLBACK = 0.5             # 進場下緣最多往下等 0.5×ATR
HORIZON = "5–10 個交易日"
TRUST_SCORE_BONUS = 3.0              # 當日投信買超加分
TRUST_STREAK_BONUS = 6.0             # 投信連 3 日買超加分
BOTH_BUY_BONUS = 4.0                 # 外資+投信同步買超
BOTH_SELL_PENALTY = 6.0              # 外資投信同步賣超扣分

ETF_CODE_RE = re.compile(r"^00[0-9]{2,4}[A-Z]?$")
STOCK_CODE_RE = re.compile(r"^[1-9]\d{3}$")
LEVERAGE_ETF_RE = re.compile(r"^00\d{3}[LR]$")

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
    kind: str
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
    trust_net: int | None = None
    foreign_net: int | None = None
    trust_streak: int = 0
    tags: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    code: str
    name: str
    close: float
    change_pct: float
    kind: str = "stock"
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


# ---- 小工具 ----------------------------------------------------------------

def taipei_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def taipei_today() -> str:
    return taipei_now().date().isoformat()


def roc_to_iso(raw: str) -> str:
    s = re.sub(r"[^\d]", "", str(raw or ""))
    if len(s) == 7:
        return f"{int(s[:3]) + 1911:04d}-{s[3:5]}-{s[5:7]}"
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    raise ValueError(f"無法解析日期: {raw}")


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("X", "", regex=False).str.replace("+", "", regex=False),
        errors="coerce",
    )


def parse_int(raw) -> int:
    try:
        return int(str(raw).replace(",", "").replace("+", "").strip() or 0)
    except ValueError:
        return 0


def security_kind(code: str) -> str | None:
    code = str(code).strip().upper()
    if ETF_CODE_RE.fullmatch(code):
        return "etf"
    if STOCK_CODE_RE.fullmatch(code):
        return "stock"
    return None


def is_leveraged_or_inverse(code: str, name: str) -> bool:
    code = str(code).strip().upper()
    name = str(name)
    if any(k in name for k in ("槓桿", "反向", "正向二倍", "負向")):
        return True
    return bool(LEVERAGE_ETF_RE.fullmatch(code))


def http_get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", 30)
    kwargs.setdefault("headers", HTTP_HEADERS)
    return requests.get(url, **kwargs)


def normalize_ohlcv_index(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    df.index = _as_naive_dates(df.index)
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Close"])


def overlay_last_bar(hist: pd.DataFrame, as_of: str, bar: dict | None) -> pd.DataFrame:
    """Yahoo 常缺當日收盤;用證交所 OHLC 覆寫/補上同一交易日。"""
    df = normalize_ohlcv_index(hist)
    if not bar or bar.get("close") is None or pd.isna(bar.get("close")):
        return df
    ts = pd.Timestamp(as_of)
    new = {c: (df.loc[ts, c] if ts in df.index else np.nan) for c in df.columns}
    mapping = {
        "Open": bar.get("open"),
        "High": bar.get("high"),
        "Low": bar.get("low"),
        "Close": bar.get("close"),
        "Adj Close": bar.get("close"),
        "Volume": bar.get("volume"),
    }
    for col, val in mapping.items():
        if col in df.columns and val is not None and not pd.isna(val):
            new[col] = val
    df.loc[ts] = pd.Series(new)
    return df.sort_index()


def _as_naive_dates(idx) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Taipei").tz_localize(None)
    return idx.normalize()


def excess_return(stock_close: pd.Series, index_close: pd.Series, days: int = 5) -> float | None:
    a = stock_close.copy()
    b = index_close.copy()
    a.index = _as_naive_dates(a.index)
    b.index = _as_naive_dates(b.index)
    joined = pd.concat([a.rename("s"), b.rename("i")], axis=1, join="inner").dropna()
    if len(joined) < days + 1:
        return None
    stock_n = joined["s"].iloc[-1] / joined["s"].iloc[-(days + 1)] - 1
    index_n = joined["i"].iloc[-1] / joined["i"].iloc[-(days + 1)] - 1
    return float(stock_n - index_n)


# ---- Stage 1: 全市場快照 ---------------------------------------------------

def _snapshot_from_official() -> pd.DataFrame:
    resp = http_get(TWSE_DAY_ALL_OFFICIAL, params={"response": "open_data"})
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str)
    rename = {
        "日期": "Date",
        "證券代號": "Code",
        "證券名稱": "Name",
        "成交股數": "TradeVolume",
        "成交金額": "TradeValue",
        "開盤價": "OpeningPrice",
        "最高價": "HighestPrice",
        "最低價": "LowestPrice",
        "收盤價": "ClosingPrice",
        "漲跌價差": "Change",
        "成交筆數": "Transaction",
    }
    df = df.rename(columns=rename)
    if df.empty:
        raise RuntimeError("證交所官網 STOCK_DAY_ALL 回傳空資料。")
    return df


def _snapshot_from_openapi() -> pd.DataFrame:
    resp = http_get(TWSE_DAY_ALL_OPENAPI)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise RuntimeError("TWSE OpenAPI STOCK_DAY_ALL 回傳空資料。")
    return pd.DataFrame(raw)


def _normalize_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"Code", "Name", "ClosingPrice", "TradeVolume", "Change"}
    missing = expected - set(df.columns)
    if missing:
        print(f"[警告] STOCK_DAY_ALL 缺少預期欄位 {missing},實際欄位: {list(df.columns)}",
              file=sys.stderr)

    out = pd.DataFrame({
        "code": df.get("Code", pd.Series(dtype=object)).astype(str).str.strip().str.upper(),
        "name": df.get("Name", pd.Series(dtype=object)).astype(str).str.strip(),
        "close": to_float_series(df.get("ClosingPrice", pd.Series(dtype=object))),
        "volume": to_float_series(df.get("TradeVolume", pd.Series(dtype=object))),
        "trade_value": to_float_series(df.get("TradeValue", pd.Series(dtype=object))),
        "change": to_float_series(df.get("Change", pd.Series(dtype=object))),
        "open": to_float_series(df.get("OpeningPrice", pd.Series(dtype=object))),
        "high": to_float_series(df.get("HighestPrice", pd.Series(dtype=object))),
        "low": to_float_series(df.get("LowestPrice", pd.Series(dtype=object))),
        "raw_date": df.get("Date", pd.Series(dtype=object)).astype(str),
    })
    out["as_of"] = out["raw_date"].map(lambda x: roc_to_iso(x) if re.search(r"\d", str(x)) else None)
    out["change_pct"] = np.where(
        (out["close"] - out["change"]) > 0,
        out["change"] / (out["close"] - out["change"]) * 100,
        np.nan,
    )
    out["kind"] = out["code"].map(security_kind)
    out = out[out["kind"].notna()].copy()
    out = out[~out.apply(lambda r: is_leveraged_or_inverse(r["code"], r["name"]), axis=1)]
    if out["trade_value"].isna().all():
        out["trade_value"] = out["close"] * out["volume"]
    return out.dropna(subset=["close", "volume"])


def fetch_market_snapshot() -> pd.DataFrame:
    """優先官網當日盤後,OpenAPI 常慢一天,兩者都抓後取日期較新的那份。"""
    frames: list[tuple[str, int, pd.DataFrame]] = []
    for loader, prefer in ((_snapshot_from_official, 1), (_snapshot_from_openapi, 0)):
        label = "官網" if prefer else "OpenAPI"
        try:
            df = _normalize_snapshot(loader())
            as_of = str(df["as_of"].dropna().iloc[0])
            frames.append((as_of, prefer, df))
            print(f"      {label}快照交易日 {as_of}")
        except Exception as exc:
            print(f"[警告] {label} STOCK_DAY_ALL 失敗: {exc}", file=sys.stderr)
    if not frames:
        raise RuntimeError("無法取得證交所全市場快照,可能是非交易日或 API 異常。")
    frames.sort()
    as_of, _, chosen = frames[-1]
    print(f"      採用交易日 {as_of},共 {len(chosen)} 檔(股票+ETF)")
    return chosen


def fetch_restricted_codes() -> set[str]:
    """處置中、變更交易(全額交割/分盤)的代號。注意股不剔除。"""
    blocked: set[str] = set()
    try:
        raw = http_get(TWSE_PUNISH).json()
        for row in raw or []:
            code = str(row.get("Code", "")).strip().upper()
            if code:
                blocked.add(code)
    except Exception as exc:
        print(f"[警告] 處置股名單讀取失敗: {exc}", file=sys.stderr)
    try:
        raw = http_get(TWSE_ALTERED).json()
        for row in raw or []:
            flag = str(row.get("PeriodicCallAuctionTrading", "")).strip()
            if flag:
                code = str(row.get("Code", "")).strip().upper()
                if code:
                    blocked.add(code)
    except Exception as exc:
        print(f"[警告] 變更交易名單讀取失敗: {exc}", file=sys.stderr)
    if blocked:
        print(f"      剔除處置/變更交易 {len(blocked)} 檔: {', '.join(sorted(blocked))}")
    return blocked


def fetch_taiex_official() -> tuple[float | None, float | None]:
    """回傳 (收盤指數, 漲跌百分比)。"""
    try:
        payload = http_get(TWSE_MI_INDEX, params={"response": "json", "type": "IND"}).json()
        tables = payload.get("tables") or []
        for table in tables:
            fields = table.get("fields") or []
            if "指數" not in fields or "漲跌百分比(%)" not in fields:
                continue
            i_name = fields.index("指數")
            i_close = fields.index("收盤指數") if "收盤指數" in fields else None
            i_pct = fields.index("漲跌百分比(%)")
            for row in table.get("data") or []:
                if str(row[i_name]).strip() == "發行量加權股價指數":
                    close = float(str(row[i_close]).replace(",", "")) if i_close is not None else None
                    pct = float(str(row[i_pct]).replace(",", "").replace("+", ""))
                    return close, pct
    except Exception as exc:
        print(f"[警告] 大盤指數讀取失敗: {exc}", file=sys.stderr)
    return None, None


def fetch_t86_map(as_of: str, need_days: int = 3) -> dict[str, dict]:
    """近幾個交易日的外資/投信買賣超(股)。失敗就當沒有籌碼,不中斷掃描。"""
    out: dict[str, dict] = {}
    days: list[tuple[str, list]] = []
    cursor = datetime.fromisoformat(as_of).date()
    for _ in range(12):
        date_s = cursor.strftime("%Y%m%d")
        try:
            payload = http_get(
                TWSE_T86,
                params={"date": date_s, "selectType": "ALL", "response": "json"},
            ).json()
        except Exception:
            payload = {}
        if payload.get("stat") == "OK" and payload.get("data"):
            days.append((roc_to_iso(payload.get("date") or date_s), payload["data"]))
            if len(days) >= need_days:
                break
        cursor -= timedelta(days=1)

    if not days:
        print("[警告] T86 三大法人讀取失敗,這次掃描不含籌碼面。", file=sys.stderr)
        return out

    print(f"      法人資料交易日: {', '.join(d for d, _ in days)}")
    streak_maps: list[dict[str, dict]] = []
    for _, rows in days:
        m = {}
        for row in rows:
            code = str(row[0]).strip().upper()
            foreign = parse_int(row[4]) + parse_int(row[7])
            trust = parse_int(row[10])
            m[code] = {"foreign": foreign, "trust": trust}
        streak_maps.append(m)

    all_codes = set().union(*[m.keys() for m in streak_maps])
    latest = streak_maps[0]
    for code in all_codes:
        trust_series = [m.get(code, {}).get("trust", 0) for m in streak_maps]
        streak = 0
        for n in trust_series:
            if n > 0:
                streak += 1
            else:
                break
        out[code] = {
            "foreign_net": latest.get(code, {}).get("foreign"),
            "trust_net": latest.get(code, {}).get("trust"),
            "trust_streak": streak,
        }
    return out


# ---- Stage 2: 流動性初篩 ---------------------------------------------------

def screen_candidates(df: pd.DataFrame, blocked: set[str]) -> pd.DataFrame:
    work = df[~df["code"].isin(blocked)].copy()
    etf = work["kind"].eq("etf")
    price_ok = np.where(
        etf,
        work["close"].between(MIN_ETF_PRICE, MAX_PRICE),
        work["close"].between(MIN_PRICE, MAX_PRICE),
    )
    volume_ok = work["volume"] >= MIN_TRADE_VOLUME_SHARES
    etf_value_ok = etf & (work["trade_value"] >= MIN_ETF_TRADE_VALUE)
    liquid = work[price_ok & (volume_ok | etf_value_ok)].copy()
    liquid = liquid.sort_values("trade_value", ascending=False).head(CANDIDATE_POOL_SIZE)
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
    """RSI 健康度:40–80 山形,峰值 58;58→80 線性降到 0,沒有 75–80 死區。"""
    if pd.isna(rsi) or rsi < 40 or rsi > 80:
        return 0.0
    if rsi <= 58:
        return (rsi - 40) / 18.0
    return (80 - rsi) / 22.0


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
        sub = normalize_ohlcv_index(sub)
        if len(sub) >= 60:
            out[code] = sub

    try:
        out[TAIEX_TICKER] = normalize_ohlcv_index(raw[TAIEX_TICKER].dropna(how="all"))
    except (KeyError, IndexError):
        pass

    return out


def extract_features(
    code: str,
    name: str,
    kind: str,
    hist: pd.DataFrame,
    taiex: pd.DataFrame | None,
    twse_bar: dict | None,
    twse_change_pct: float | None,
    as_of: str,
    inst: dict | None,
) -> FeatureRow | None:
    hist = overlay_last_bar(hist, as_of, twse_bar)
    close = hist["Close"].dropna()
    volume = hist["Volume"].reindex(close.index)
    if len(close) < 40:
        return None

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    rsi = compute_rsi_wilder(close, 14)
    atr = compute_atr_wilder(hist.reindex(close.index), 14)
    vol_avg20 = volume.rolling(20).mean()
    high20 = hist["High"].reindex(close.index).rolling(20).max()
    low10 = hist["Low"].reindex(close.index).rolling(10).min()

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
    yf_change_pct = (last_close / prev_close - 1) * 100 if prev_close else 0.0

    display_close = last_close
    if twse_bar and twse_bar.get("close") is not None and not pd.isna(twse_bar["close"]):
        display_close = float(twse_bar["close"])
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
    if len(volume.dropna()) and vol_avg20.iloc[-1] and not pd.isna(vol_avg20.iloc[-1]):
        last_vol = volume.iloc[-1]
        if last_vol is not None and not pd.isna(last_vol):
            vol_ratio = float(last_vol / vol_avg20.iloc[-1])

    atr_extension = None
    if last_atr:
        atr_extension = (last_close - last_ma20) / last_atr

    dist_from_high = None
    if last_high20:
        dist_from_high = (last_high20 - last_close) / last_high20

    excess_5d = None
    if taiex is not None and len(taiex["Close"].dropna()) >= 6:
        excess_5d = excess_return(close, taiex["Close"].dropna(), 5)

    tags: list[str] = []
    risk_tags: list[str] = []
    tags.append("ETF" if kind == "etf" else "個股")

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

    trust_net = None if not inst else inst.get("trust_net")
    foreign_net = None if not inst else inst.get("foreign_net")
    trust_streak = 0 if not inst else int(inst.get("trust_streak") or 0)
    if trust_streak >= 3:
        tags.append("投信連3日買超")
    elif trust_net is not None and trust_net > 0:
        tags.append("投信買超")
    if foreign_net is not None and foreign_net > 0 and trust_net is not None and trust_net > 0:
        tags.append("外資投信同步買超")
    elif foreign_net is not None and foreign_net > 0:
        tags.append("外資買超")

    if display_change >= NEAR_LIMIT_PCT:
        risk_tags.append("接近漲停")
    elif display_change >= BIG_UP_PCT:
        risk_tags.append("今日大漲")
    if display_change <= NEAR_LIMIT_DOWN_PCT:
        risk_tags.append("接近跌停")
    elif display_change <= BIG_DOWN_PCT:
        risk_tags.append("今日大跌")

    if display_change <= -5 and vol_ratio is not None and vol_ratio >= 1.8:
        risk_tags.append("放量長陰")

    if (atr_extension is not None and atr_extension >= EXT_ATR_THRESHOLD) or (
        dist_from_high is not None and dist_from_high <= NEAR_HIGH_PCT
    ):
        risk_tags.append("延伸過遠")

    trend_raw = (last_ma5 - last_ma20) / last_ma20 if last_ma20 else 0.0
    rsi_raw = rsi_health_raw(last_rsi if last_rsi is not None else np.nan)
    # 只懲罰「往上延伸」;跌深不會被當成結構加分
    ext_up = max(0.0, atr_extension) if atr_extension is not None else 0.0
    not_extended_raw = -ext_up

    return FeatureRow(
        code=code,
        name=name,
        close=round(display_close, 2),
        change_pct=round(float(display_change), 2),
        kind=kind,
        trend_raw=float(trend_raw),
        rsi_health_raw=float(rsi_raw),
        not_extended_raw=float(not_extended_raw),
        vol_raw=float(vol_ratio) if vol_ratio is not None else np.nan,
        relative_raw=float(excess_5d) if excess_5d is not None else np.nan,
        day_move_raw=float(abs(display_change)),
        extension_raw=float(ext_up),
        rsi14=None if last_rsi is None else round(last_rsi, 1),
        vol_ratio=None if vol_ratio is None else round(vol_ratio, 2),
        above_ma20=bool(last_close > last_ma20),
        atr_extension=None if atr_extension is None else round(atr_extension, 2),
        dist_from_high=None if dist_from_high is None else round(dist_from_high * 100, 2),
        excess_5d=None if excess_5d is None else round(excess_5d * 100, 2),
        atr_abs=None if last_atr is None else round(last_atr, 4),
        ma5=round(last_ma5, 4),
        ma20=round(last_ma20, 4),
        swing_low_10=None if last_low10 is None else round(last_low10, 4),
        swing_high_20=None if last_high20 is None else round(last_high20, 4),
        trust_net=trust_net,
        foreign_net=foreign_net,
        trust_streak=trust_streak,
        tags=tags,
        risk_tags=risk_tags,
    )


def extra_penalty(change_pct: float) -> float:
    if change_pct >= NEAR_LIMIT_PCT or change_pct <= NEAR_LIMIT_DOWN_PCT:
        return NEAR_LIMIT_PENALTY
    if change_pct >= BIG_UP_PCT or change_pct <= BIG_DOWN_PCT:
        return BIG_UP_PENALTY
    return 0.0


def chip_score_adj(r: FeatureRow) -> float:
    adj = 0.0
    if r.trust_streak >= 3:
        adj += TRUST_STREAK_BONUS
    elif r.trust_net is not None and r.trust_net > 0:
        adj += TRUST_SCORE_BONUS
    if r.foreign_net is not None and r.trust_net is not None:
        if r.foreign_net > 0 and r.trust_net > 0:
            adj += BOTH_BUY_BONUS
        elif r.foreign_net < 0 and r.trust_net < 0:
            adj -= BOTH_SELL_PENALTY
    return adj


def rank_features(rows: list[FeatureRow]) -> list[Candidate]:
    if not rows:
        return []

    factor_df = pd.DataFrame({
        "trend": [r.trend_raw for r in rows],
        "rsi": [r.rsi_health_raw for r in rows],
        "not_extended": [r.not_extended_raw for r in rows],
        "volume": [r.vol_raw for r in rows],
        "relative": [r.relative_raw for r in rows],
        "day_move": [r.day_move_raw for r in rows],
        "extension": [r.extension_raw for r in rows],
    })
    # 缺值給中性 50 百分位,只影響排序,不能當成「符合強勢條件」
    ranks = factor_df.rank(method="average", pct=True, na_option="keep") * 100.0
    ranks = ranks.fillna(50.0)

    out: list[Candidate] = []
    for i, r in enumerate(rows):
        structure = (
            ranks.at[i, "trend"] * STRUCTURE_WEIGHTS["trend"]
            + ranks.at[i, "rsi"] * STRUCTURE_WEIGHTS["rsi"]
            + ranks.at[i, "not_extended"] * STRUCTURE_WEIGHTS["not_extended"]
        )
        crowding = (
            ranks.at[i, "volume"] * CROWDING_WEIGHTS["volume"]
            + ranks.at[i, "relative"] * CROWDING_WEIGHTS["relative"]
            + ranks.at[i, "day_move"] * CROWDING_WEIGHTS["day_move"]
            + ranks.at[i, "extension"] * CROWDING_WEIGHTS["extension"]
        )
        score = structure - CROWDING_PENALTY * crowding - extra_penalty(r.change_pct) + chip_score_adj(r)
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
                kind=r.kind,
                structure=round(float(structure), 1),
                crowding=round(float(crowding), 1),
                score=round(float(score), 1),
                tags=r.tags,
                risk_tags=r.risk_tags,
                detail={
                    "kind": r.kind,
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
                    "trust_net": r.trust_net,
                    "foreign_net": r.foreign_net,
                    "trust_streak": r.trust_streak,
                    "factors": {
                        "trend": round(float(ranks.at[i, "trend"]), 0),
                        "rsi": round(float(ranks.at[i, "rsi"]), 0),
                        "not_extended": round(float(ranks.at[i, "not_extended"]), 0),
                        "volume": round(float(ranks.at[i, "volume"]), 0),
                        "relative": round(float(ranks.at[i, "relative"]), 0),
                        "day_move": round(float(ranks.at[i, "day_move"]), 0),
                    },
                },
            )
        )
    return out


def classify_action(c: Candidate, regime: str) -> tuple[str, str]:
    """短線規則分級。出場只應套在日前推薦標的上,這裡先給今日技術狀態。"""
    d = c.detail or {}
    rsi = d.get("rsi14")
    atr = d.get("atr_extension")
    above = d.get("above_ma20")
    vol = d.get("vol_ratio_20d")
    excess = d.get("excess_5d_pct")
    trust_net = d.get("trust_net")
    foreign_net = d.get("foreign_net")
    risks = set(c.risk_tags or [])

    if "接近漲停" in risks:
        return "exit", "接近漲停,短線空間被用完的機率高"
    if "接近跌停" in risks:
        return "exit", "接近跌停,短線先出場觀望"
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
    if "今日大跌" in risks:
        return "trim", "今日大跌,若已持有先減碼控風險"
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

    both_sold = (
        trust_net is not None and foreign_net is not None
        and trust_net < 0 and foreign_net < 0
    )
    rsi_ok = rsi is not None and 50 <= rsi <= 65
    quiet_day = abs(c.change_pct) < 5
    not_extended = atr is not None and atr < 1.5
    vol_strong = vol is not None and vol >= STRONG_VOL_RATIO
    not_lagging = excess is not None and excess >= STRONG_EXCESS_5D
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
        and not both_sold
    ):
        if regime == "risk_off":
            return "scale_in", "結構不錯但大盤在20MA之下,改分批、不一次買滿"
        return "strong_buy", "結構高、量能有跟上、沒明顯輸大盤,比較像動能剛啟動"

    rsi_scale = rsi is not None and 48 <= rsi <= 70
    if (
        above
        and c.structure >= 55
        and c.crowding <= 58
        and rsi_scale
        and c.change_pct < BIG_UP_PCT
        and (atr is not None and atr < 2.2)
        and not both_sold
    ):
        return "scale_in", "結構尚可且不是冷清量能,適合分批而不是一次買滿"

    return "watch", "條件好壞參半,先看不急著動"


def assign_actions(candidates: list[Candidate], regime: str = "risk_on") -> None:
    for c in candidates:
        key, reason = classify_action(c, regime)
        c.action = key
        c.action_label = ACTION_LABELS[key]
        c.reason = reason
        if key in ("strong_buy", "scale_in"):
            c.plan = build_trade_plan(c)
        else:
            c.plan = None


def twse_tick(price: float, is_etf: bool = False) -> float:
    if is_etf:
        return 0.01 if price < 50 else 0.05
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


def round_tick(price: float, mode: str = "nearest", is_etf: bool = False) -> float:
    if price <= 0:
        return 0.0
    tick = twse_tick(price, is_etf=is_etf)
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
    is_etf = c.kind == "etf"
    if not atr or atr <= 0 or close <= 0:
        return None

    if ma5 is not None and 0 < ma5 < close:
        entry_low = max(float(ma5), close - ENTRY_ATR_PULLBACK * atr)
    else:
        entry_low = close - 0.3 * atr
    entry_high = close
    if entry_low >= entry_high:
        entry_low = close - 0.3 * atr
    entry_low = round_tick(entry_low, "down", is_etf=is_etf)
    entry_high = round_tick(entry_high, "nearest", is_etf=is_etf)
    if entry_low > entry_high:
        entry_low = entry_high
    entry_mid = (entry_low + entry_high) / 2

    min_risk = max(MIN_STOP_ATR * atr, MIN_STOP_PCT / 100.0 * entry_mid)
    stop_atr = entry_low - STOP_ATR * atr
    stop = stop_atr
    if low10:
        swing = float(low10) - twse_tick(float(low10), is_etf=is_etf)
        if swing < entry_low:
            dist = entry_low - swing
            # 10日低點只有「夠近但風險仍不小於最低門檻」才採用,避免 1–2 檔假停損
            if min_risk <= dist <= 2.5 * atr:
                stop = max(stop_atr, swing)
    if entry_mid - stop < min_risk:
        stop = entry_mid - min_risk
    stop = round_tick(stop, "down", is_etf=is_etf)
    if stop >= entry_low:
        stop = round_tick(entry_low - min_risk, "down", is_etf=is_etf)
    if stop >= entry_low or entry_mid <= stop:
        return None

    risk = entry_mid - stop
    tp1 = round_tick(entry_mid + TP1_R * risk, "up", is_etf=is_etf)
    tp2 = round_tick(entry_mid + TP2_R * risk, "up", is_etf=is_etf)
    return {
        "horizon": HORIZON,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "resistance_20d": None if not high20 else round_tick(float(high20), "nearest", is_etf=is_etf),
        "invalid_below": None if not ma20 else round_tick(float(ma20), "down", is_etf=is_etf),
        "atr": round(float(atr), 4),
        "risk_pct": round(risk / entry_mid * 100, 2),
        "tp1_pct": round((tp1 - entry_mid) / entry_mid * 100, 2),
        "tp2_pct": round((tp2 - entry_mid) / entry_mid * 100, 2),
        "rr1": TP1_R,
        "rr2": TP2_R,
        "method": (
            "進場等回檔至5日均線附近;停損預設1.5×Wilder ATR,"
            "若10日低點更近且風險仍≥max(1×ATR, 2.5%)才改用結構停損;"
            "停利1.5R/2.5R。ETF 與普通股使用不同跳動單位。"
        ),
    }


def market_regime(taiex_hist: pd.DataFrame | None, official_close: float | None, as_of: str) -> dict:
    if taiex_hist is not None and official_close:
        taiex_hist = overlay_last_bar(
            taiex_hist,
            as_of,
            {"open": official_close, "high": official_close, "low": official_close, "close": official_close},
        )
    close = None if taiex_hist is None else taiex_hist["Close"].dropna()
    if close is None or len(close) < 20:
        return {
            "regime": "unknown",
            "label": "大盤資料不足",
            "above_ma20": None,
            "ma20": None,
            "close": official_close,
            "note": "沒有足夠的加權指數均線,市場燈號先不判斷。",
        }
    ma20 = float(close.rolling(20).mean().iloc[-1])
    last = float(close.iloc[-1])
    above = last > ma20
    if above:
        return {
            "regime": "risk_on",
            "label": "偏多可做",
            "above_ma20": True,
            "ma20": round(ma20, 2),
            "close": round(last, 2),
            "note": "加權指數站上20日均線,短線環境允許找剛啟動的標的,仍須自行控風險。",
        }
    return {
        "regime": "risk_off",
        "label": "偏空防守",
        "above_ma20": False,
        "ma20": round(ma20, 2),
        "close": round(last, 2),
        "note": "加權指數在20日均線之下,系統不發強力推薦,建議多看少做或維持空倉。",
    }


def _action_sort_key(c: Candidate) -> tuple:
    if c.action in ("trim", "exit"):
        return (-c.crowding, -abs(c.change_pct), -c.score)
    return (-c.score, c.crowding)


def collect_prior_recs(history_days: list[dict], today_date: str) -> dict[str, dict]:
    usable = [d for d in history_days if d.get("date") and d["date"] != today_date][-EXIT_LOOKBACK_DAYS:]
    recs: dict[str, dict] = {}
    for day in usable:
        for item in day.get("candidates") or []:
            code = str(item.get("code", "")).strip().upper()
            if not code:
                continue
            action = item.get("action")
            tracked = action in ("strong_buy", "scale_in") or action is None
            if not tracked:
                continue
            if code not in recs:
                recs[code] = {**item, "rec_date": day["date"]}
    return recs


def build_exit_from_history(
    candidates: list[Candidate],
    history_days: list[dict],
    today_date: str,
    today_df: pd.DataFrame,
    blocked: set[str] | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    """只追蹤日前推薦,不要把全市場大跌股塞進出場欄。"""
    prior = collect_prior_recs(history_days, today_date)
    if not prior:
        return [], []
    by_code = {c.code: c for c in candidates}
    today_px = dict(zip(today_df["code"].astype(str), today_df["close"].astype(float)))
    today_chg = dict(zip(today_df["code"].astype(str), today_df["change_pct"].astype(float)))
    today_name = dict(zip(today_df["code"].astype(str), today_df["name"].astype(str)))
    blocked = blocked or set()
    trims: list[Candidate] = []
    exits: list[Candidate] = []

    for code, rec in prior.items():
        rec_date = rec.get("rec_date")
        prefix = f"日前推薦({rec_date}) "
        px = today_px.get(code)

        if code in blocked:
            item = Candidate(
                code=code,
                name=str(rec.get("name") or today_name.get(code, code)),
                close=float(px) if px is not None else float(rec.get("close") or 0),
                change_pct=float(today_chg.get(code) or 0.0),
                kind=str(rec.get("kind") or ("etf" if security_kind(code) == "etf" else "stock")),
                action="exit",
                action_label=ACTION_LABELS["exit"],
                reason=prefix + "已進入處置或變更交易,流動性受限建議出場",
                tags=["日前推薦"],
            )
            exits.append(item)
            continue

        if px is None:
            continue
        live = by_code.get(code)

        if live and live.action in ("trim", "exit"):
            item = Candidate(
                code=live.code,
                name=live.name,
                close=live.close,
                change_pct=live.change_pct,
                kind=live.kind,
                structure=live.structure,
                crowding=live.crowding,
                score=live.score,
                tags=live.tags,
                risk_tags=live.risk_tags,
                detail=live.detail,
                action=live.action,
                action_label=live.action_label,
                reason=prefix + live.reason,
                plan=live.plan,
            )
            (exits if live.action == "exit" else trims).append(item)
            continue

        stop, tp1, tp2, ma20 = rec.get("stop"), rec.get("tp1"), rec.get("tp2"), rec.get("ma20")
        reason = None
        action = None
        if stop and px <= float(stop):
            action, reason = "exit", f"跌破停損 {stop}"
        elif ma20 and px < float(ma20):
            action, reason = "exit", f"收盤跌破20日均線 {ma20}"
        elif tp2 and px >= float(tp2):
            action, reason = "exit", f"已達停利2 {tp2}"
        elif tp1 and px >= float(tp1):
            action, reason = "trim", f"已達停利1 {tp1}"
        if not action:
            continue
        item = Candidate(
            code=code,
            name=str(rec.get("name") or today_name.get(code, code)),
            close=float(px),
            change_pct=float(today_chg.get(code) or 0.0),
            kind=str(rec.get("kind") or ("etf" if security_kind(code) == "etf" else "stock")),
            structure=float(live.structure) if live else 0.0,
            crowding=float(live.crowding) if live else 0.0,
            score=float(live.score) if live else 0.0,
            tags=["日前推薦"],
            risk_tags=[],
            detail=(live.detail if live else {}),
            action=action,
            action_label=ACTION_LABELS[action],
            reason=prefix + reason,
            plan=live.plan if live else None,
        )
        (exits if action == "exit" else trims).append(item)
    return trims, exits


def build_lazy_pack(
    candidates: list[Candidate],
    trim_items: list[Candidate] | None = None,
    exit_items: list[Candidate] | None = None,
) -> dict:
    buckets: dict[str, list[Candidate]] = {k: [] for k in ACTION_LABELS}
    for c in candidates:
        if c.action in ("strong_buy", "scale_in", "watch"):
            buckets[c.action].append(c)
    buckets["trim"] = list(trim_items or [])
    buckets["exit"] = list(exit_items or [])

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
        "進場區間=現價回檔至5日均線附近;停損預設1.5×Wilder ATR,"
        "且不少於 max(1×ATR, 2.5%);停利=1.5R/2.5R。"
        "參考持有5–10個交易日,收盤跌破20日均線視為結構失效。"
        "出場欄只追蹤過去10個交易日本系統推薦過的標的。風險自負。"
    )
    pack["note"] = (
        "這是同一套技術規則加上公開法人買賣超的分級建議,不是保證獲利的進出場點。"
        "強力推薦需要量能跟上、近5日不輸大盤,且缺資料時不會當成強勢。"
        "大盤跌破20MA時不發強力推薦。出場只列出日前推薦標的的停利/停損狀態。"
        "任何買賣都是你自己的判斷與責任。"
    )
    return pack


# ---- Stage 4: 組裝輸出 -----------------------------------------------------

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
        "note": "清單報酬與大盤日報酬都用證交所收盤價對帳;as_of 是快照交易日而非執行當日。",
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
        "kind": c.kind,
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


def snapshot_entry(c: Candidate) -> dict:
    plan = c.plan or {}
    return {
        "code": c.code,
        "name": c.name,
        "kind": c.kind,
        "close": c.close,
        "score": c.score,
        "action": c.action,
        "stop": plan.get("stop"),
        "tp1": plan.get("tp1"),
        "tp2": plan.get("tp2"),
        "ma20": (c.detail or {}).get("ma20"),
    }


def build_output(
    candidates: list[Candidate],
    top_n: int,
    taiex_change_pct: float | None,
    review: dict | None,
    as_of: str,
    market: dict | None,
    lazy_pack: dict | None = None,
) -> dict:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
    return {
        "updated_at": taipei_now().isoformat(),
        "as_of": as_of,
        "taiex_change_pct": taiex_change_pct,
        "market": market,
        "price_source": "現價與漲跌幅來自證交所盤後 STOCK_DAY_ALL(官網優先,OpenAPI 備援)",
        "indicator_source": "均線 / RSI(14, Wilder) / ATR / 量比 / 相對強弱來自 Yahoo Finance;缺當日K則用證交所 OHLC 補上。法人來自 T86。",
        "methodology": {
            "horizon": "短線 1~2 週波段參考,非當沖、非長期持有",
            "rsi": "RSI(14, Wilder)",
            "ranking": "觀察分 = 結構百分位 − 擁擠百分位 × 0.5 − 漲跌停/大漲大跌降權 + 法人加減分",
            "actions": "強力推薦要結構對、量能跟上、近5日不輸大盤;缺資料不當成強勢。大盤在20MA下不發強力推薦。出場只追蹤日前推薦。",
            "trade_plan": "進場等回檔至MA5;停損1.5×ATR且不少於 max(1×ATR, 2.5%);停利1.5R/2.5R。ETF 用 ETF 檔位。",
            "weights": {
                "structure": STRUCTURE_WEIGHTS,
                "crowding": CROWDING_WEIGHTS,
                "crowding_penalty": CROWDING_PENALTY,
            },
            "note": "本清單為技術面規則加上公開三大法人資料的篩選結果,不構成投資建議,過去表現不代表未來績效。動能延續是持有期因子;結構分偏「剛啟動」,擁擠分偏「已經走遠」。",
        },
        "review": review,
        "lazy_pack": lazy_pack or build_lazy_pack(candidates),
        "candidates": [candidate_payload(c) for c in ranked],
    }


# ---- Demo 模式(離線,方便測試 / 預覽前端) -----------------------------------

def build_demo_output(top_n: int) -> dict:
    rng = np.random.default_rng(42)
    sample_names = [
        ("2330", "台積電", "stock"), ("00878", "國泰永續高股息", "etf"),
        ("2454", "聯發科", "stock"), ("0050", "元大台灣50", "etf"),
        ("2382", "廣達", "stock"), ("00919", "群益台灣精選高息", "etf"),
        ("3008", "大立光", "stock"), ("2379", "瑞昱", "stock"),
        ("6669", "緯穎", "stock"), ("3661", "世芯-KY", "stock"),
        ("2308", "台達電", "stock"), ("2412", "中華電", "stock"),
        ("1301", "台塑", "stock"), ("2891", "中信金", "stock"),
        ("2881", "富邦金", "stock"), ("3037", "欣興", "stock"),
        ("2345", "智邦", "stock"), ("6446", "藥華藥", "stock"),
        ("2327", "國巨", "stock"), ("5347", "世界", "stock"),
    ]
    archetypes = [
        {"structure": 78, "crowding": 28, "rsi": 57, "chg": 0.8, "atr": 0.4, "above": True, "risks": [], "vol": 1.3, "excess": 1.2, "trust": 800000, "foreign": 200000, "streak": 3},
        {"structure": 72, "crowding": 36, "rsi": 54, "chg": -0.5, "atr": 0.6, "above": True, "risks": [], "vol": 1.1, "excess": 0.4, "trust": 120000, "foreign": 50000, "streak": 1},
        {"structure": 64, "crowding": 48, "rsi": 61, "chg": 1.2, "atr": 1.1, "above": True, "risks": [], "vol": 1.0, "excess": 0.2, "trust": 0, "foreign": 10000, "streak": 0},
        {"structure": 58, "crowding": 52, "rsi": 66, "chg": 2.1, "atr": 1.4, "above": True, "risks": [], "vol": 0.95, "excess": -0.3, "trust": -20000, "foreign": 8000, "streak": 0},
        {"structure": 74, "crowding": 26, "rsi": 57, "chg": -0.9, "atr": 0.2, "above": True, "risks": [], "vol": 0.48, "excess": -4.3, "trust": 0, "foreign": -90000, "streak": 0},
        {"structure": 42, "crowding": 60, "rsi": 46, "chg": -2.4, "atr": -0.3, "above": False, "risks": [], "vol": 1.2, "excess": -1.5, "trust": -50000, "foreign": -80000, "streak": 0},
        {"structure": 55, "crowding": 74, "rsi": 72, "chg": 4.2, "atr": 1.8, "above": True, "risks": ["延伸過遠"], "vol": 2.1, "excess": 3.0, "trust": 30000, "foreign": 40000, "streak": 1},
        {"structure": 48, "crowding": 82, "rsi": 76, "chg": 7.4, "atr": 2.1, "above": True, "risks": ["今日大漲"], "vol": 2.4, "excess": 4.1, "trust": 10000, "foreign": 20000, "streak": 0},
        {"structure": 40, "crowding": 88, "rsi": 83, "chg": 9.7, "atr": 2.8, "above": True, "risks": ["接近漲停"], "vol": 2.8, "excess": 6.0, "trust": 0, "foreign": 0, "streak": 0},
        {"structure": 38, "crowding": 79, "rsi": 41, "chg": -6.2, "atr": -0.8, "above": False, "risks": ["放量長陰"], "vol": 2.2, "excess": -5.0, "trust": -90000, "foreign": -120000, "streak": 0},
    ]
    cands: list[Candidate] = []
    for i, (code, name, kind) in enumerate(sample_names):
        spec = archetypes[i] if i < len(archetypes) else {
            "structure": round(float(rng.uniform(40, 75)), 1),
            "crowding": round(float(rng.uniform(25, 70)), 1),
            "rsi": round(float(rng.uniform(45, 70)), 1),
            "chg": round(float(rng.uniform(-2, 3)), 2),
            "atr": round(float(rng.uniform(-0.2, 1.6)), 2),
            "above": True,
            "risks": [],
            "vol": round(float(rng.uniform(0.9, 1.8)), 2),
            "excess": round(float(rng.uniform(-0.5, 2.0)), 2),
            "trust": 0,
            "foreign": 0,
            "streak": 0,
        }
        structure = float(spec["structure"])
        crowding = float(spec["crowding"])
        px = round(float(rng.uniform(15, 40 if kind == "etf" else 200)), 2)
        atr_abs = round(px * 0.025, 2)
        tags = ["ETF" if kind == "etf" else "個股"]
        if spec.get("streak", 0) >= 3:
            tags.append("投信連3日買超")
        elif spec.get("trust", 0) > 0:
            tags.append("投信買超")
        cands.append(
            Candidate(
                code=code,
                name=name,
                close=px,
                change_pct=float(spec["chg"]),
                kind=kind,
                structure=structure,
                crowding=crowding,
                score=round(structure - CROWDING_PENALTY * crowding, 1),
                tags=tags,
                risk_tags=list(spec["risks"]),
                detail={
                    "kind": kind,
                    "rsi14": spec["rsi"],
                    "vol_ratio_20d": float(spec.get("vol", 1.0)),
                    "above_ma20": bool(spec["above"]),
                    "atr_extension": spec["atr"],
                    "dist_from_high_20d_pct": round(float(rng.uniform(0.3, 7.0)), 2),
                    "excess_5d_pct": float(spec.get("excess", 0.0)),
                    "atr": atr_abs,
                    "ma5": round(px * 0.985, 2),
                    "ma20": round(px * 0.97, 2),
                    "swing_low_10": round(px * 0.94, 2),
                    "swing_high_20": round(px * 1.05, 2),
                    "trust_net": spec.get("trust"),
                    "foreign_net": spec.get("foreign"),
                    "trust_streak": spec.get("streak", 0),
                    "factors": {
                        "trend": 70, "rsi": 70, "not_extended": 60,
                        "volume": 40, "relative": 40, "day_move": 40,
                    },
                },
            )
        )
    market = {
        "regime": "risk_on",
        "label": "偏多可做",
        "above_ma20": True,
        "ma20": 45000.0,
        "close": 45800.0,
        "note": "demo 預覽用的假市場燈號。",
    }
    assign_actions(cands, market["regime"])
    demo_trim = [c for c in cands if c.action == "trim"][:2]
    demo_exit = [c for c in cands if c.action == "exit"][:2]
    for c in demo_trim + demo_exit:
        c.reason = "日前推薦(demo) " + c.reason
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
        market,
        build_lazy_pack(cands, demo_trim, demo_exit),
    )
    output["demo"] = True
    output["methodology"]["note"] = "這是 demo 假資料,僅供預覽畫面用,不是真實掃描結果。"
    return output


def tracked_candidates(candidates: list[Candidate], top_n: int) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
    by_code = {c.code: c for c in ranked}
    for c in candidates:
        if c.action in ("strong_buy", "scale_in"):
            by_code[c.code] = c
    return list(by_code.values())


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
        as_of = str(snapshot["as_of"].dropna().iloc[0])
        blocked = fetch_restricted_codes()
        official_taiex_close, official_taiex_chg = fetch_taiex_official()
        inst_map = fetch_t86_map(as_of)

        print("[2/4] 流動性初篩...")
        pool = screen_candidates(snapshot, blocked)
        print(f"      候選池: {len(pool)} 檔(個股 {int((pool.kind=='stock').sum())} / ETF {int((pool.kind=='etf').sum())})")

        print("[3/4] 抓取歷史K線並計算指標(需要幾分鐘,請耐心等候)...")
        hist_map = fetch_history(pool["code"].tolist())
        taiex_hist = hist_map.get(TAIEX_TICKER)
        market = market_regime(taiex_hist, official_taiex_close, as_of)
        taiex_change_pct = official_taiex_chg
        if taiex_change_pct is None and taiex_hist is not None and len(taiex_hist["Close"].dropna()) >= 2:
            c = overlay_last_bar(
                taiex_hist, as_of,
                {"close": official_taiex_close} if official_taiex_close else None,
            )["Close"].dropna()
            taiex_change_pct = round(float((c.iloc[-1] / c.iloc[-2] - 1) * 100), 2)
        if isinstance(taiex_change_pct, (int, float)):
            taiex_change_pct = round(float(taiex_change_pct), 2)

        bar_map = {
            str(r.code): {
                "open": r.open, "high": r.high, "low": r.low,
                "close": r.close, "volume": r.volume,
            }
            for r in pool.itertuples()
        }
        twse_chg_map = dict(zip(pool["code"].astype(str), pool["change_pct"]))
        name_map = dict(zip(pool["code"].astype(str), pool["name"]))
        kind_map = dict(zip(pool["code"].astype(str), pool["kind"]))

        rows: list[FeatureRow] = []
        for code, hist in hist_map.items():
            if code == TAIEX_TICKER:
                continue
            feat = extract_features(
                code,
                name_map.get(code, code),
                kind_map.get(code, security_kind(code) or "stock"),
                hist,
                taiex_hist,
                bar_map.get(code),
                twse_chg_map.get(code),
                as_of,
                inst_map.get(code),
            )
            if feat:
                rows.append(feat)
        print(f"      成功計算指標: {len(rows)} 檔")

        print("[4/4] 排序並輸出...")
        candidates = rank_features(rows)
        assign_actions(candidates, market.get("regime") or "unknown")
        snap_path = snapshots_path_for(out_path)
        history = load_snapshots(snap_path)
        review = build_review(history, snapshot, as_of, taiex_change_pct)
        trim_items, exit_items = build_exit_from_history(
            candidates, history, as_of, snapshot, blocked,
        )
        lazy_pack = build_lazy_pack(candidates, trim_items, exit_items)
        output = build_output(
            candidates, args.top, taiex_change_pct, review, as_of, market, lazy_pack,
        )

        today_entry = {
            "date": as_of,
            "taiex_change_pct": taiex_change_pct,
            "market_regime": market.get("regime"),
            "candidates": [snapshot_entry(c) for c in tracked_candidates(candidates, args.top)],
        }
        history = [d for d in history if d.get("date") != as_of]
        history.append(today_entry)
        save_snapshots(snap_path, history)

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成,已寫入 {out_path}")


if __name__ == "__main__":
    main()
