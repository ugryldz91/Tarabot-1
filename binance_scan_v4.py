import asyncio
import aiohttp
import os
import csv
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# ------------------- Ayarlar -------------------
DEFAULT_BASES = [
    "https://data-api.binance.vision",
    "https://api-gcp.binance.com",
    "https://data.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

SESSION_TIMEOUT = aiohttp.ClientTimeout(total=70)
UA_HEADERS = {"User-Agent": "UgurBinanceScan/1.0 (+github actions)"}

EXCLUDE_KEYWORDS = (
    "UP","DOWN","BULL","BEAR","2L","2S","3L","3S","4L","4S","5L","5S","PERP"
)

# ------------------- Pine Script RMA (Wilder EMA) -------------------
def rma(series: pd.Series, period: int) -> pd.Series:
    result = pd.Series(index=series.index, dtype=float)
    for i, val in enumerate(series):
        if i == 0:
            result.iloc[i] = val
        else:
            result.iloc[i] = (val + (period - 1) * result.iloc[i-1]) / period
    return result

def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0)
    return rsi

def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return lower, ma, upper

# ------------------- HTTP helpers -------------------
async def fetch_json(session: aiohttp.ClientSession, url: str, params: Dict[str, Any] = None) -> Any:
    base_delay = 0.8
    for attempt in range(6):
        try:
            async with session.get(url, params=params, headers=UA_HEADERS) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = base_delay * (2 ** attempt) + random.uniform(0,0.6)
                    if retry_after:
                        try: sleep_s = float(retry_after)
                        except: pass
                    print(f"[warn] 429 rate limit, sleeping {sleep_s:.1f}s")
                    await asyncio.sleep(sleep_s)
                    continue
                txt = await resp.text()
                print(f"[warn] GET {url} -> {resp.status}, body={txt[:200]}")
        except Exception as e:
            print(f"[warn] error GET {url}: {e}")
        await asyncio.sleep(base_delay * (1.7 ** attempt) + random.uniform(0,0.6))
    raise RuntimeError(f"GET {url} failed after retries.")

async def try_bases(path: str, params: Dict[str, Any] = None, bases: Optional[List[str]] = None) -> Any:
    bases = bases or DEFAULT_BASES
    async with aiohttp.ClientSession(timeout=SESSION_TIMEOUT) as session:
        last_err = None
        for base in bases:
            url = f"{base}{path}"
            try:
                return await fetch_json(session, url, params=params)
            except Exception as e:
                last_err = e
                print(f"[info] fallback -> {base} failed: {e}")
        if last_err:
            raise last_err
        raise RuntimeError("No base could be used.")

# ------------------- Symbols -------------------
async def get_spot_usdt_symbols(bases=None) -> List[str]:
    try:
        data = await try_bases("/api/v3/exchangeInfo", params={"permissions": "SPOT"}, bases=bases)
        out = []
        for s in data.get("symbols", []):
            if s.get("status") == "TRADING" and s.get("isSpotTradingAllowed", False) and s.get("quoteAsset") == "USDT":
                sym = s.get("symbol","")
                if any(sym.endswith(k) for k in EXCLUDE_KEYWORDS): continue
                out.append(sym)
        return sorted(set(out))
    except:
        data = await try_bases("/api/v3/ticker/price", bases=bases)
        return sorted({i["symbol"] for i in data if i["symbol"].endswith("USDT") and not any(i["symbol"].endswith(k) for k in EXCLUDE_KEYWORDS)})

# ------------------- Klines -------------------
async def get_klines(symbol: str, limit: int = 80, bases=None) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": "1d", "limit": limit}
    raw = await try_bases("/api/v3/klines", params=params, bases=bases)
    cols = ["openTime","open","high","low","close","volume","closeTime","quoteAssetVolume",
            "numberOfTrades","takerBuyBase","takerBuyQuote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume","quoteAssetVolume","takerBuyBase","takerBuyQuote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["openTime"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df["closeTime"] = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    return df

# ------------------- Tarama Koşulları -------------------
def check_conditions(df: pd.DataFrame) -> Tuple[bool, float]:
    if df is None or len(df) < 25: return False, float("nan")
    close, volume = df["close"], df["volume"]
    rsi = wilder_rsi(close, 14)
    bb_lower, _, _ = bollinger_bands(close, 20, 2.0)
    i = len(df)-1
    last_close, last_rsi, last_bb_lower = close.iloc[i], rsi.iloc[i], bb_lower.iloc[i]
    if len(volume) < 4 or pd.isna(volume.iloc[i]): return False, float(last_rsi)
    prev3 = volume.iloc[i-3:i]
    if prev3.isna().any(): return False, float(last_rsi)
    vol_ok = volume.iloc[i] > 1.8 * prev3.mean()
    cond_rsi = (not pd.isna(last_rsi)) and (last_rsi < 30.0)
    cond_bb  = (not pd.isna(last_bb_lower)) and (last_close < last_bb_lower)
    return bool(cond_rsi and cond_bb and vol_ok), float(last_rsi)

# ------------------- Tarama -------------------
async def scan_all(bases=None) -> Tuple[List[Tuple[str,float]], int]:
    symbols = await get_spot_usdt_symbols(bases=bases)
    results: List[Tuple[str,float]] = []
    sem = asyncio.Semaphore(4)
    async def worker(sym: str):
        async with sem:
            try:
                df = await get_klines(sym, 80, bases=bases)
                ok, rsi_val = check_conditions(df)
                if ok: results.append((sym, round(rsi_val,2)))
            except Exception as e:
                print(f"[warn] {sym} klines hata: {e}")
    await asyncio.gather(*[asyncio.create_task(worker(s)) for s in symbols])
    results.sort(key=lambda x: x[1])
    return results, len(symbols)

# ------------------- BTC/ETH RSI -------------------
async def get_rsi_for_symbols(symbols: List[str], bases=None) -> Dict[str,float]:
    results = {}
    sem = asyncio.Semaphore(4)
    async def worker(sym):
        async with sem:
            try:
                df = await get_klines(sym, 80, bases=bases)
                if df is not None and len(df) >= 14:
                    results[sym] = round(float(wilder_rsi(df["close"],14).iloc[-1]),2)
            except Exception as e:
                print(f"[warn] {sym} RSI alınamadı: {e}")
    await asyncio.gather(*[asyncio.create_task(worker(s)) for s in symbols])
    return results

# ------------------- Telegram -------------------
async def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return print("Telegram env eksik; mesaj gönderilmeyecek.")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}
    async with aiohttp.ClientSession(timeout=SESSION_TIMEOUT, headers=UA_HEADERS) as s:
        try: async with s.post(url, json=payload) as resp: _=await resp.text()
        except Exception as e: print(f"[warn] telegram send error: {e}")

def format_message(pairs: List[Tuple[str,float]], scanned: int, btc_eth_rsi: Dict[str,float]) -> str:
    lines = []
    if pairs:
        lines.append(f"Kriterlere uyan coinler (RSI) — Taranan toplam coin: {scanned}")
        for sym, r in pairs: lines.append(f"- {sym}: RSI={r}")
    else:
        lines.append(f"Bugün kriterlere uygun coin bulunamadı.\nTaranan toplam coin: {scanned}")
    for s in ["BTCUSDT","ETHUSDT"]:
        if s in btc_eth_rsi: lines.append(f"{s} RSI={btc_eth_rsi[s]}")
        else: lines.append(f"{s} RSI alınamadı")
    return "\n".join(lines)

def write_csv(pairs: List[Tuple[str,float]], out_dir=".") -> str:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    path = os.path.join(out_dir, f"scan_results_{ts}.csv")
    with open(path,"w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol","rsi"])
        for sym, rsi_val in pairs:
            w.writerow([sym, rsi_val])
    return path

# ------------------- Main -------------------
async def main():
    bases_env = os.getenv("BINANCE_BASES")
    bases = [b.strip() for b in bases_env.split(",")] if bases_env else None
    try:
        pairs, scanned = await scan_all(bases=bases)
        btc_eth_rsi = await get_rsi_for_symbols(["BTCUSDT","ETHUSDT"], bases=bases)
    except Exception as e:
        msg = f"Binance API erişilemedi: {e}"
        print(msg)
        await send_telegram(msg)
        return
    msg = format_message(pairs, scanned, btc_eth_rsi)
    print(msg)
    write_csv(pairs)
    await send_telegram(msg)

if __name__ == "__main__":
    asyncio.run(main())
