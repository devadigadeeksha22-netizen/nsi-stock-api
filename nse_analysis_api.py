"""
NSE Stock Analysis API  v2.0
================================================================================
FastAPI backend for the "Financial Data Analytics & Visualisation" project.
100% free: only uses yfinance (no API keys, no paid services).

EXISTING ENDPOINTS (unchanged behaviour)
  GET /                      health check
  GET /analyze/{symbol}      technical + fundamental analysis
  GET /chart/{symbol}        candlestick + RSI/MACD/MA series
  GET /watchlist?symbols=    multi-stock snapshot

NEW ENDPOINTS (v2.0)
  GET /signal/{symbol}       composite Buy/Neutral/Sell score + support/resistance + ATR stop-loss
  GET /risk/{symbol}         volatility, Sharpe, Sortino, max drawdown, VaR, beta vs Nifty 50
  GET /compare?symbols=      normalised performance + correlation of 2-5 stocks
  GET /backtest/{symbol}     moving-average crossover strategy vs buy-and-hold
  GET /portfolio?symbols=    portfolio analytics + Monte Carlo efficient frontier
  GET /market-overview       indices, USD/INR, top gainers & losers
  GET /seasonality/{symbol}  month-by-month average returns (heatmap data)

RUN LOCALLY:  uvicorn nse_analysis_api:app --reload --port 8000
API DOCS   :  http://127.0.0.1:8000/docs   (auto-generated, lets you test every endpoint)

EDUCATIONAL USE ONLY - nothing here is investment advice.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime

app = FastAPI(title="NSE Stock Analysis API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRADING_DAYS = 252

# ── SMALL HELPERS ──────────────────────────────────────────────────────────

# Simple in-memory cache (5 minutes). Makes the app faster and keeps us
# under Yahoo Finance's rate limits on Render's free tier.
_CACHE = {}
CACHE_TTL = 300


def get_history(ticker, period="2y"):
    key = (ticker, period)
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < CACHE_TTL:
        return _CACHE[key][1].copy()
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if hist is not None and not hist.empty:
        _CACHE[key] = (now, hist)
    return hist.copy()


def norm_ticker(symbol):
    s = symbol.upper().strip()
    if s.startswith("^") or s.endswith(".NS") or s.endswith(".BO") or s.endswith("=X"):
        return s
    return f"{s}.NS"


def clean_series(s):
    """Remove timezone so series from different tickers can be aligned by date."""
    s = s.copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s


def period_for_years(years):
    if years <= 1:
        return "1y"
    if years <= 2:
        return "2y"
    if years <= 5:
        return "5y"
    return "10y"


def get_close(ticker, period="2y"):
    hist = get_history(ticker, period)
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol.")
    return clean_series(hist["Close"])


def safe_round(val, decimals=2):
    if val is None:
        return None
    try:
        if pd.isna(val) or np.isinf(val):
            return None
        return round(float(val), decimals)
    except Exception:
        return None


def max_drawdown_pct(equity):
    return safe_round(((equity / equity.cummax()) - 1).min() * 100)


def downsample(df, max_points=250):
    step = max(1, len(df) // max_points)
    out = df.iloc[::step]
    if out.index[-1] != df.index[-1]:
        out = pd.concat([out, df.iloc[[-1]]])
    return out


def parse_symbols(symbols, min_n, max_n):
    lst = [s.strip() for s in symbols.split(",") if s.strip()]
    if len(lst) < min_n:
        raise HTTPException(status_code=400, detail=f"Provide at least {min_n} symbol(s), comma-separated.")
    return [norm_ticker(s) for s in lst][:max_n]


# ── INDICATOR FUNCTIONS (original) ─────────────────────────────────────────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_moving_averages(prices, short_window=50, long_window=200):
    return prices.rolling(short_window).mean(), prices.rolling(long_window).mean()


def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100


def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth


def calculate_atr(high, low, close, period=14):
    """NEW: Average True Range - measures typical daily price range (used for stop-loss)."""
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


# ── FUNDAMENTALS (original) ────────────────────────────────────────────────

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth (handles irregularly reported quarters)."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Calculated from income statement + balance sheet (avoids the blocked .info endpoint)."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)
    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None
            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    s = balance_sheet.loc[equity_label].dropna()
                    if len(s) > 0:
                        equity = s.iloc[0]
                        break
            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    s = balance_sheet.loc[debt_label].dropna()
                    if len(s) > 0:
                        total_debt = s.iloc[0]
                        break
            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)
    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass

    return fundamentals


def get_technical_snapshot(hist):
    close = hist["Close"]; high = hist["High"]; low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


# ══════════════════════════════════════════════════════════════════════════
#  EXISTING ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "NSE Stock Analysis API v2.0 is running",
        "endpoints": [
            "/analyze/{symbol}", "/chart/{symbol}", "/watchlist?symbols=",
            "/signal/{symbol}", "/risk/{symbol}", "/compare?symbols=",
            "/backtest/{symbol}", "/portfolio?symbols=", "/market-overview",
            "/seasonality/{symbol}",
        ],
    }


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    ticker = norm_ticker(symbol)
    try:
        stock = yf.Ticker(ticker)
        hist = get_history(ticker, "2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_fundamentals(stock)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "fundamentals": fundamentals,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """OHLC + RSI, MACD, MA50, MA200 series for charting."""
    ticker = norm_ticker(symbol)
    months = min(max(months, 1), 24)
    try:
        hist = get_history(ticker, "2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        recent = hist.tail(months * 21)
        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]),
                "macd_line": safe_round(macd_line.loc[date]),
                "macd_signal": safe_round(signal_line.loc[date]),
                "ma_50": safe_round(ma50.loc[date]),
                "ma_200": safe_round(ma200.loc[date]),
            })
        return {"symbol": ticker, "months": months, "data": rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """Multi-stock snapshot. Example: /watchlist?symbols=RELIANCE,TCS,INFY (max 10)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = norm_ticker(symbol)
        try:
            hist = get_history(ticker, "2y")
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue
            technical, corporate_action_flag = get_technical_snapshot(hist)
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})
    return {"count": len(results), "stocks": results}


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 1: TRADE SIGNAL + SUPPORT/RESISTANCE + ATR STOP-LOSS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/signal/{symbol}")
def trade_signal(symbol: str):
    """Rule-based composite score (-6 to +6) built from the indicators. Fully explainable."""
    ticker = norm_ticker(symbol)
    try:
        hist = get_history(ticker, "2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        tech, _ = get_technical_snapshot(hist)
        price = tech["current_price"]
        score = 0
        reasons = []

        def add(points, text):
            nonlocal score
            score += points
            reasons.append({"points": points, "reason": text})

        rsi = tech["rsi_14"]
        if rsi is not None:
            if rsi < 30:
                add(1, f"RSI {rsi} is below 30 (oversold - possible bounce)")
            elif rsi > 70:
                add(-1, f"RSI {rsi} is above 70 (overbought - possible pullback)")
            else:
                add(0, f"RSI {rsi} is in the neutral zone (30-70)")

        if tech["macd_histogram"] is not None:
            if tech["macd_histogram"] > 0:
                add(1, "MACD histogram is positive (bullish momentum)")
            else:
                add(-1, "MACD histogram is negative (bearish momentum)")

        if tech["ma_trend"]:
            if tech["ma_trend"] == "bullish":
                add(1, "50-day MA is above 200-day MA (long-term uptrend)")
            else:
                add(-1, "50-day MA is below 200-day MA (long-term downtrend)")

        if tech["ma_50"] is not None and price is not None:
            if price > tech["ma_50"]:
                add(1, "Price is above the 50-day MA")
            else:
                add(-1, "Price is below the 50-day MA")

        pb = tech["bollinger_percent_b"]
        if pb is not None:
            if pb < 0.1:
                add(1, "Price is near the lower Bollinger Band (stretched down)")
            elif pb > 0.9:
                add(-1, "Price is near the upper Bollinger Band (stretched up)")
            else:
                add(0, "Price is inside the Bollinger Bands")

        if tech["roc_12"] is not None:
            if tech["roc_12"] > 0:
                add(1, f"12-day rate of change is positive ({tech['roc_12']}%)")
            else:
                add(-1, f"12-day rate of change is negative ({tech['roc_12']}%)")

        if score >= 4:
            label = "Strong Bullish"
        elif score >= 2:
            label = "Bullish"
        elif score > -2:
            label = "Neutral"
        elif score > -4:
            label = "Bearish"
        else:
            label = "Strong Bearish"

        # Support / resistance / pivots / ATR stop-loss
        last = hist.iloc[-1]
        H, L, C = float(last["High"]), float(last["Low"]), float(last["Close"])
        pivot = (H + L + C) / 3
        atr = calculate_atr(hist["High"], hist["Low"], hist["Close"]).iloc[-1]
        recent60 = hist["Close"].tail(60)
        stop = price - 2 * atr if not pd.isna(atr) else None

        levels = {
            "support_60d": safe_round(recent60.min()),
            "resistance_60d": safe_round(recent60.max()),
            "pivot": safe_round(pivot),
            "r1": safe_round(2 * pivot - L),
            "r2": safe_round(pivot + (H - L)),
            "s1": safe_round(2 * pivot - H),
            "s2": safe_round(pivot - (H - L)),
            "atr_14": safe_round(atr),
            "suggested_stop_loss": safe_round(stop),
            "stop_distance_pct": safe_round((price - stop) / price * 100) if stop else None,
            "illustrative_target_2R": safe_round(price + 2 * (price - stop)) if stop else None,
        }

        return {
            "symbol": ticker,
            "current_price": price,
            "score": score,
            "max_score": 6,
            "signal": label,
            "reasons": reasons,
            "levels": levels,
            "disclaimer": "Educational rule-based score, not investment advice.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing signal for {ticker}: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 2: RISK METRICS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/risk/{symbol}")
def risk_metrics(symbol: str, years: int = 3, risk_free_pct: float = 7.0):
    """Volatility, Sharpe, Sortino, max drawdown, VaR, beta vs Nifty 50."""
    ticker = norm_ticker(symbol)
    years = min(max(years, 1), 10)
    try:
        period = period_for_years(years)
        close = get_close(ticker, period).tail(years * TRADING_DAYS + 1)
        rets = close.pct_change().dropna()
        if len(rets) < 30:
            raise HTTPException(status_code=400, detail="Not enough price history for risk metrics.")

        n = len(rets)
        cagr = ((close.iloc[-1] / close.iloc[0]) ** (TRADING_DAYS / n) - 1) * 100
        vol = rets.std() * np.sqrt(TRADING_DAYS) * 100
        downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS) * 100
        sharpe = (cagr - risk_free_pct) / vol if vol else None
        sortino = (cagr - risk_free_pct) / downside if downside else None
        var95 = np.percentile(rets, 5) * 100
        equity = (1 + rets).cumprod()

        beta = None
        try:
            nifty = get_close("^NSEI", period).pct_change().dropna()
            joined = pd.concat([rets, nifty], axis=1, join="inner").dropna()
            joined.columns = ["stock", "nifty"]
            if len(joined) > 30 and joined["nifty"].var() > 0:
                beta = joined["stock"].cov(joined["nifty"]) / joined["nifty"].var()
        except Exception:
            pass

        if vol < 20:
            level = "Low"
        elif vol < 35:
            level = "Moderate"
        else:
            level = "High"

        dd_series = (equity / equity.cummax() - 1) * 100
        dd = dd_series.reset_index()
        dd.columns = ["date", "drawdown"]
        dd_small = downsample(dd.set_index("date"), 200)

        return {
            "symbol": ticker,
            "years": years,
            "risk_free_rate_pct": risk_free_pct,
            "annualised_return_pct": safe_round(cagr),
            "annualised_volatility_pct": safe_round(vol),
            "sharpe_ratio": safe_round(sharpe),
            "sortino_ratio": safe_round(sortino),
            "max_drawdown_pct": max_drawdown_pct(equity),
            "var_95_daily_pct": safe_round(var95),
            "best_day_pct": safe_round(rets.max() * 100),
            "worst_day_pct": safe_round(rets.min() * 100),
            "beta_vs_nifty50": safe_round(beta),
            "risk_level": level,
            "drawdown_series": [
                {"date": d.strftime("%Y-%m-%d"), "drawdown_pct": safe_round(v)}
                for d, v in dd_small["drawdown"].items()
            ],
            "notes": "VaR 95%: on 95% of days the loss should be smaller than this. Sharpe/Sortino use the given risk-free rate.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing risk for {ticker}: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 3: COMPARE STOCKS
# ══════════════════════════════════════════════════════════════════════════

@app.get("/compare")
def compare_stocks(symbols: str, months: int = 12):
    """Rebased-to-100 performance + correlation. Example: /compare?symbols=TCS,INFY,WIPRO&months=12"""
    tickers = parse_symbols(symbols, 2, 5)
    months = min(max(months, 1), 24)
    try:
        cols = {}
        for t in tickers:
            cols[t.replace(".NS", "")] = get_close(t, "2y")
        df = pd.concat(cols, axis=1, join="inner").dropna().tail(months * 21)
        if len(df) < 5:
            raise HTTPException(status_code=400, detail="Not enough overlapping data to compare.")

        rebased = df / df.iloc[0] * 100
        rets = df.pct_change().dropna()
        corr = rets.corr()

        summary = []
        for c in df.columns:
            summary.append({
                "symbol": c,
                "total_return_pct": safe_round((df[c].iloc[-1] / df[c].iloc[0] - 1) * 100),
                "volatility_pct": safe_round(rets[c].std() * np.sqrt(TRADING_DAYS) * 100),
                "max_drawdown_pct": max_drawdown_pct(df[c] / df[c].iloc[0]),
            })
        summary.sort(key=lambda x: (x["total_return_pct"] is None, -(x["total_return_pct"] or 0)))

        small = downsample(rebased, 200)
        series = [
            {"date": d.strftime("%Y-%m-%d"), **{c: safe_round(small.loc[d, c]) for c in small.columns}}
            for d in small.index
        ]

        return {
            "months": months,
            "symbols": list(df.columns),
            "summary": summary,
            "series": series,
            "correlation": {
                "labels": list(corr.columns),
                "matrix": [[safe_round(v, 2) for v in row] for row in corr.values],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing stocks: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 4: BACKTEST (MA CROSSOVER)
# ══════════════════════════════════════════════════════════════════════════

@app.get("/backtest/{symbol}")
def backtest(symbol: str, short: int = 50, long: int = 200, years: int = 5, cost_pct: float = 0.1):
    """Buy when short MA > long MA, otherwise stay in cash. Compared with buy-and-hold."""
    ticker = norm_ticker(symbol)
    if short >= long:
        raise HTTPException(status_code=400, detail="'short' must be smaller than 'long'.")
    years = min(max(years, 1), 10)
    try:
        close = get_close(ticker, period_for_years(years))
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        start = ma_l.first_valid_index()
        if start is None:
            raise HTTPException(status_code=400, detail="Not enough history for this MA window. Lower 'long' or raise 'years'.")

        pos = (ma_s > ma_l).astype(int)
        held = pos.shift(1).fillna(0)  # trade the day AFTER the signal (no look-ahead)
        ret = close.pct_change().fillna(0)
        trades = held.diff().abs().fillna(0)
        strat_ret = held * ret - trades * cost_pct / 100

        sl = slice(start, None)
        strat_ret, ret, held, trades, pos_s = strat_ret[sl], ret[sl], held[sl], trades[sl], pos[sl]
        strat_eq = (1 + strat_ret).cumprod() * 100
        bh_eq = (1 + ret).cumprod() * 100
        n = len(ret)

        def metrics(eq, r):
            return {
                "total_return_pct": safe_round(eq.iloc[-1] - 100),
                "cagr_pct": safe_round(((eq.iloc[-1] / 100) ** (TRADING_DAYS / n) - 1) * 100),
                "max_drawdown_pct": max_drawdown_pct(eq / 100),
                "volatility_pct": safe_round(r.std() * np.sqrt(TRADING_DAYS) * 100),
            }

        changes = pos_s.diff().fillna(0)
        signals = [
            {"date": d.strftime("%Y-%m-%d"), "action": "BUY" if v > 0 else "SELL", "price": safe_round(close.loc[d])}
            for d, v in changes[changes != 0].items()
        ][-10:]

        curve = pd.concat([strat_eq.rename("strategy"), bh_eq.rename("buy_hold")], axis=1)
        small = downsample(curve, 250)

        return {
            "symbol": ticker,
            "strategy": f"Long when {short}-day MA > {long}-day MA, else cash",
            "tested_from": str(start.date()),
            "tested_to": str(close.index[-1].date()),
            "cost_per_trade_pct": cost_pct,
            "strategy_metrics": metrics(strat_eq, strat_ret),
            "buy_hold_metrics": metrics(bh_eq, ret),
            "number_of_trades": int(trades.sum()),
            "currently_in_position": bool(pos.iloc[-1]),
            "recent_signals": signals,
            "equity_curve": [
                {"date": d.strftime("%Y-%m-%d"), "strategy": safe_round(r["strategy"]), "buy_hold": safe_round(r["buy_hold"])}
                for d, r in small.iterrows()
            ],
            "note": "Past performance does not guarantee future results. Taxes and slippage beyond the cost per trade are ignored.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backtesting {ticker}: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 5: PORTFOLIO ANALYTICS + MONTE CARLO FRONTIER
# ══════════════════════════════════════════════════════════════════════════

@app.get("/portfolio")
def portfolio(symbols: str, weights: str = "", years: int = 3, risk_free_pct: float = 7.0, simulations: int = 3000):
    """Example: /portfolio?symbols=TCS,HDFCBANK,RELIANCE&weights=40,30,30"""
    tickers = parse_symbols(symbols, 2, 8)
    years = min(max(years, 1), 5)
    simulations = min(max(simulations, 500), 5000)

    if weights.strip():
        try:
            w = np.array([float(x) for x in weights.split(",") if x.strip()])
        except ValueError:
            raise HTTPException(status_code=400, detail="Weights must be numbers, comma-separated.")
        if len(w) != len(tickers):
            raise HTTPException(status_code=400, detail="Number of weights must match number of symbols.")
        if w.sum() <= 0:
            raise HTTPException(status_code=400, detail="Weights must sum to more than 0.")
        w = w / w.sum()
    else:
        w = np.ones(len(tickers)) / len(tickers)

    try:
        cols = {t.replace(".NS", ""): get_close(t, period_for_years(years)) for t in tickers}
        df = pd.concat(cols, axis=1, join="inner").dropna().tail(years * TRADING_DAYS + 1)
        rets = df.pct_change().dropna()
        if len(rets) < 60:
            raise HTTPException(status_code=400, detail="Not enough overlapping history for these stocks.")

        mean_r = rets.mean().values * TRADING_DAYS
        cov = rets.cov().values * TRADING_DAYS

        def stats(weights_arr):
            r = float(weights_arr @ mean_r) * 100
            v = float(np.sqrt(weights_arr @ cov @ weights_arr)) * 100
            return r, v, (r - risk_free_pct) / v if v else None

        p_ret, p_vol, p_sharpe = stats(w)
        port_daily = rets.values @ w
        equity = pd.Series((1 + port_daily).cumprod(), index=rets.index)

        # Monte Carlo
        np.random.seed(42)
        k = len(tickers)
        rand_w = np.random.random((simulations, k))
        rand_w /= rand_w.sum(axis=1, keepdims=True)
        sim_ret = rand_w @ mean_r * 100
        sim_vol = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov, rand_w)) * 100
        sim_sharpe = (sim_ret - risk_free_pct) / sim_vol
        i_sharpe = int(np.nanargmax(sim_sharpe))
        i_minvol = int(np.nanargmin(sim_vol))

        def pack(i):
            return {
                "weights_pct": {c: safe_round(rand_w[i, j] * 100, 1) for j, c in enumerate(df.columns)},
                "expected_return_pct": safe_round(sim_ret[i]),
                "volatility_pct": safe_round(sim_vol[i]),
                "sharpe_ratio": safe_round(sim_sharpe[i]),
            }

        sample_idx = np.linspace(0, simulations - 1, min(300, simulations)).astype(int)
        corr = rets.corr()
        small = downsample(equity.to_frame("v"), 200)

        return {
            "symbols": list(df.columns),
            "years": years,
            "your_portfolio": {
                "weights_pct": {c: safe_round(w[j] * 100, 1) for j, c in enumerate(df.columns)},
                "expected_return_pct": safe_round(p_ret),
                "volatility_pct": safe_round(p_vol),
                "sharpe_ratio": safe_round(p_sharpe),
                "max_drawdown_pct": max_drawdown_pct(equity),
            },
            "individual_stocks": [
                {
                    "symbol": c,
                    "expected_return_pct": safe_round(mean_r[j] * 100),
                    "volatility_pct": safe_round(np.sqrt(cov[j, j]) * 100),
                }
                for j, c in enumerate(df.columns)
            ],
            "max_sharpe_portfolio": pack(i_sharpe),
            "min_volatility_portfolio": pack(i_minvol),
            "frontier_points": [
                {"return_pct": safe_round(sim_ret[i]), "volatility_pct": safe_round(sim_vol[i]), "sharpe": safe_round(sim_sharpe[i])}
                for i in sample_idx
            ],
            "growth_of_100": [
                {"date": d.strftime("%Y-%m-%d"), "value": safe_round(v * 100)} for d, v in small["v"].items()
            ],
            "correlation": {
                "labels": list(corr.columns),
                "matrix": [[safe_round(v, 2) for v in row] for row in corr.values],
            },
            "note": "Expected return = historical average, which is NOT a forecast.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building portfolio: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 6: MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

INDICES = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Bank Nifty": "^NSEBANK",
    "India VIX": "^INDIAVIX",
    "USD/INR": "INR=X",
}

LARGE_CAPS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL",
    "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "BAJFINANCE", "WIPRO", "NESTLEIND", "NTPC", "POWERGRID", "ONGC", "HCLTECH", "TATASTEEL",
    "M&M", "JSWSTEEL", "COALINDIA", "TECHM", "CIPLA",
]


@app.get("/market-overview")
def market_overview():
    """Index levels, USD/INR, top gainers/losers among 30 NSE large caps."""
    try:
        idx_rows = []
        for name, tkr in INDICES.items():
            try:
                h = get_history(tkr, "1mo")
                c = h["Close"].dropna()
                if len(c) >= 2:
                    idx_rows.append({
                        "name": name,
                        "symbol": tkr,
                        "last": safe_round(c.iloc[-1]),
                        "change_pct": safe_round((c.iloc[-1] / c.iloc[-2] - 1) * 100),
                        "change_1m_pct": safe_round((c.iloc[-1] / c.iloc[0] - 1) * 100),
                    })
            except Exception:
                idx_rows.append({"name": name, "symbol": tkr, "error": "unavailable"})

        movers = []
        key = ("movers", "5d")
        if key in _CACHE and time.time() - _CACHE[key][0] < CACHE_TTL:
            movers = _CACHE[key][1]
        else:
            tickers = [f"{s}.NS" for s in LARGE_CAPS]
            data = yf.download(tickers, period="5d", auto_adjust=True, progress=False)["Close"]
            data = data.dropna(how="all")
            if len(data) >= 2:
                pct = (data.iloc[-1] / data.iloc[-2] - 1) * 100
                for t in pct.dropna().index:
                    movers.append({
                        "symbol": t.replace(".NS", ""),
                        "price": safe_round(data[t].iloc[-1]),
                        "change_pct": safe_round(pct[t]),
                    })
                _CACHE[key] = (time.time(), movers)

        ordered = sorted(movers, key=lambda x: x["change_pct"] if x["change_pct"] is not None else 0, reverse=True)
        return {
            "timestamp": datetime.now().isoformat(),
            "indices": idx_rows,
            "top_gainers": ordered[:5],
            "top_losers": list(reversed(ordered[-5:])),
            "universe": "30 NSE large-cap stocks",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building market overview: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
#  NEW FEATURE 7: SEASONALITY
# ══════════════════════════════════════════════════════════════════════════

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@app.get("/seasonality/{symbol}")
def seasonality(symbol: str, years: int = 10):
    """Average return for each calendar month + year x month heatmap."""
    ticker = norm_ticker(symbol)
    years = min(max(years, 2), 10)
    try:
        close = get_close(ticker, "10y" if years > 5 else "5y")
        df = close.to_frame("c")
        df["y"] = df.index.year
        df["m"] = df.index.month
        month_end = df.groupby(["y", "m"])["c"].last()
        monthly_ret = (month_end.pct_change() * 100).dropna()
        grid = monthly_ret.unstack("m")
        grid = grid.tail(years + 1)

        stats = []
        for m in range(1, 13):
            if m in grid.columns:
                col = grid[m].dropna()
                stats.append({
                    "month": MONTHS[m - 1],
                    "avg_return_pct": safe_round(col.mean()),
                    "median_return_pct": safe_round(col.median()),
                    "pct_positive": safe_round((col > 0).mean() * 100, 0),
                    "best_pct": safe_round(col.max()),
                    "worst_pct": safe_round(col.min()),
                    "samples": int(len(col)),
                })
            else:
                stats.append({"month": MONTHS[m - 1], "avg_return_pct": None})

        valid = [s for s in stats if s.get("avg_return_pct") is not None]
        heat = [
            {"year": int(y), **{MONTHS[m - 1]: safe_round(grid.loc[y, m]) if m in grid.columns else None for m in range(1, 13)}}
            for y in grid.index
        ]
        return {
            "symbol": ticker,
            "years_covered": years,
            "monthly_stats": stats,
            "best_month": max(valid, key=lambda s: s["avg_return_pct"])["month"] if valid else None,
            "worst_month": min(valid, key=lambda s: s["avg_return_pct"])["month"] if valid else None,
            "heatmap": heat,
            "note": "Seasonality is a historical pattern, not a prediction.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing seasonality for {ticker}: {str(e)}")