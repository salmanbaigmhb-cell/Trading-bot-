# Trading-bot-
Universal Market Analysis python bot

"""
Binance Spot Market Analyser V1
Timeframes: 15m and 1h
Analysis only - does not place orders.
"""

import requests
import pandas as pd
from datetime import datetime, timezone

BASE_URL = "https://api.binance.com/api/v3/klines"

# Settings
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["15m", "1h"]
CANDLE_LIMIT = 200


def get_candles(symbol, interval):
    """Download public Binance Spot candles."""

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": CANDLE_LIMIT
    }

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            raise ValueError("No candle data returned")

        columns = [
            "time", "open", "high", "low", "close",
            "volume", "close_time", "quote_volume",
            "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]

        df = pd.DataFrame(data, columns=columns)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        df["time"] = pd.to_datetime(
            df["time"], unit="ms", utc=True
        )

        # Exclude the currently forming candle.
        df = df.iloc[:-1].copy()

        return df.reset_index(drop=True)

    except (requests.RequestException, ValueError) as error:
        print(f"Data error for {symbol} {interval}: {error}")
        return None


def calculate_indicators(df):
    """Calculate EMA, RSI and ATR."""

    df = df.copy()

    df["ema20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))

    # Handle periods with no losses.
    df.loc[avg_loss == 0, "rsi"] = 100
    df.loc[(avg_gain == 0) & (avg_loss == 0), "rsi"] = 50

    previous_close = df["close"].shift(1)

    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1/14, adjust=False
    ).mean()

    df["volume_avg"] = df["volume"].rolling(20).mean()

    return df


def detect_pattern(df):
    """Detect simple two-candle engulfing patterns."""

    if len(df) < 2:
        return "Insufficient candles"

    previous = df.iloc[-2]
    current = df.iloc[-1]

    bullish = (
        previous["close"] < previous["open"]
        and current["close"] > current["open"]
        and current["open"] <= previous["close"]
        and current["close"] >= previous["open"]
    )

    bearish = (
        previous["close"] > previous["open"]
        and current["close"] < current["open"]
        and current["open"] >= previous["close"]
        and current["close"] <= previous["open"]
    )

    if bullish:
        return "Bullish Engulfing"

    if bearish:
        return "Bearish Engulfing"

    return "No Engulfing Pattern"


def analyse_timeframe(symbol, interval):
    """Analyse one symbol and timeframe."""

    df = get_candles(symbol, interval)

    if df is None or len(df) < 60:
        print(f"Not enough data: {symbol} {interval}")
        return None

    df = calculate_indicators(df)
    current = df.iloc[-1]

    if pd.isna(current["rsi"]) or pd.isna(current["atr"]):
        print("Indicators not ready")
        return None

    close = current["close"]
    ema20 = current["ema20"]
    ema50 = current["ema50"]
    rsi = current["rsi"]
    atr = current["atr"]

    # Trend
    if close > ema20 > ema50:
        trend = "BULLISH"
    elif close < ema20 < ema50:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS / MIXED"

    # Volume
    if pd.notna(current["volume_avg"]) and current["volume_avg"] > 0:
        volume_ratio = current["volume"] / current["volume_avg"]
        volume_status = (
            "HIGH" if volume_ratio > 1.5 else
            "NORMAL" if volume_ratio >= 0.7 else
            "LOW"
        )
    else:
        volume_ratio = 0
        volume_status = "UNKNOWN"

    pattern = detect_pattern(df)

    # Simple score, not a probability of success.
    score = 0

    if trend == "BULLISH":
        score += 2

    if 45 <= rsi <= 65:
        score += 1

    if pattern == "Bullish Engulfing":
        score += 1

    if volume_status == "HIGH":
        score += 1

    if trend == "BEARISH":
        score -= 2

    if rsi > 75 or rsi < 25:
        score -= 1

    if pattern == "Bearish Engulfing":
        score -= 1

    # Preliminary educational setup.
    if score >= 4 and trend == "BULLISH":
        signal = "BUY SETUP"
    elif score <= -2:
        signal = "AVOID / BEARISH"
    else:
        signal = "WAIT"

    # ATR-based levels for a hypothetical long setup.
    stop_loss = close - (1.5 * atr)
    target = close + (3.0 * atr)

    # Recent support and resistance estimates.
    support = df["low"].tail(20).min()
    resistance = df["high"].tail(20).max()

    return {
        "symbol": symbol,
        "timeframe": interval,
        "time": current["time"],
        "price": close,
        "trend": trend,
        "rsi": rsi,
        "ema20": ema20,
        "ema50": ema50,
        "pattern": pattern,
        "volume": volume_status,
        "volume_ratio": volume_ratio,
        "support": support,
        "resistance": resistance,
        "atr": atr,
        "stop_loss": stop_loss,
        "target": target,
        "score": score,
        "signal": signal
    }


def print_report(result):
    """Print readable analysis."""

    if result is None:
        return

    print("\n" + "=" * 55)
    print(f"PAIR       : {result['symbol']}")
    print(f"TIMEFRAME  : {result['timeframe']}")
    print(f"CANDLE UTC : {result['time']}")
    print("-" * 55)

    print(f"Price      : {result['price']:.6f}")
    print(f"Trend      : {result['trend']}")
    print(f"RSI (14)   : {result['rsi']:.2f}")
    print(f"EMA 20     : {result['ema20']:.6f}")
    print(f"EMA 50     : {result['ema50']:.6f}")
    print(f"Pattern    : {result['pattern']}")
    print(f"Volume     : {result['volume']}")
    print(f"Volume x   : {result['volume_ratio']:.2f}")
    print("-" * 55)

    print(f"Support    : {result['support']:.6f}")
    print(f"Resistance : {result['resistance']:.6f}")
    print(f"ATR        : {result['atr']:.6f}")
    print(f"Stop Loss  : {result['stop_loss']:.6f}")
    print(f"Target     : {result['target']:.6f}")
    print("-" * 55)

    print(f"Score      : {result['score']} (rule-based, not a probability)")
    print(f"SIGNAL     : {result['signal']}")
    print("=" * 55)


def main():
    print("\nBINANCE SPOT MARKET ANALYSER V1")
    print("Analysis only | No orders placed")
    print("Time:", datetime.now(timezone.utc))

    for symbol in SYMBOLS:
        results = {}

        for timeframe in TIMEFRAMES:
            result = analyse_timeframe(symbol, timeframe)
            results[timeframe] = result
            print_report(result)

        # Compare both timeframes.
        short = results.get("15m")
        long = results.get("1h")

        print(f"\nMULTI-TIMEFRAME CHECK: {symbol}")

        if short and long:
            print(f"15m: {short['signal']}")
            print(f"1h : {long['signal']}")

            if (
                short["signal"] == "BUY SETUP"
                and long["trend"] == "BULLISH"
            ):
                print("CONFIRMATION: Both timeframes support bullish conditions.")
            else:
                print("CONFIRMATION: Conditions are not fully aligned.")
        else:
            print("CONFIRMATION: Insufficient data.")

    print("\nAnalysis complete.")
    print("Remember: Signals are educational, not guaranteed predictions.")


if __name__ == "__main__":
    main()
