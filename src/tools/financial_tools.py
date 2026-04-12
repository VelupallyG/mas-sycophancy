"""Financial data tools backed by a local SQLite database.

Three Concordia Tool subclasses that agents can invoke during deliberation:
  - StockPriceLookup  — historical daily OHLCV data
  - EarningsLookup    — quarterly earnings reports
  - MacroIndicatorLookup — macro indicators (VIX, yields, etc.)

All tools open a fresh read-only connection per execute() call so they are
safe to use from any thread without connection pooling.

Usage:
    tools = create_financial_tools(Path("data/financial_db/market_data.db"))
    # Pass to ToolUseActComponent or InteractiveDocumentWithTools
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from concordia.document import tool as tool_module


class StockPriceLookup(tool_module.Tool):
    """Look up historical stock/commodity/currency prices."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "lookup_stock_price"

    @property
    def description(self) -> str:
        return (
            "Look up historical daily price data (open, high, low, close, volume). "
            "Args: ticker (str, e.g. 'META', 'GOOGL', 'GBPUSD', 'BRN', 'LEH'), "
            "start_date (str, YYYY-MM-DD, optional), end_date (str, YYYY-MM-DD, optional). "
            "Returns last 10 trading days if no dates given."
        )

    def execute(self, **kwargs: Any) -> str:
        ticker = kwargs.get("ticker", "").upper()
        start_date = kwargs.get("start_date", "")
        end_date = kwargs.get("end_date", "")

        if not ticker:
            return "Error: ticker is required."

        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        try:
            if start_date and end_date:
                rows = conn.execute(
                    "SELECT date, open, high, low, close, volume "
                    "FROM stock_prices WHERE ticker=? AND date BETWEEN ? AND ? "
                    "ORDER BY date LIMIT 30",
                    (ticker, start_date, end_date),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT date, open, high, low, close, volume "
                    "FROM stock_prices WHERE ticker=? "
                    "ORDER BY date DESC LIMIT 10",
                    (ticker,),
                ).fetchall()
                rows = rows[::-1]  # chronological order
        finally:
            conn.close()

        if not rows:
            return f"No price data found for {ticker}."

        lines = [f"{ticker} prices:"]
        lines.append("Date       | Open   | High   | Low    | Close  | Vol")
        for date, o, h, l, c, v in rows:
            vol = f"{v:>8,}" if v else "     N/A"
            lines.append(f"{date} | {o:6.2f} | {h:6.2f} | {l:6.2f} | {c:6.2f} | {vol}")
        return "\n".join(lines)


class EarningsLookup(tool_module.Tool):
    """Look up quarterly earnings data."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "lookup_earnings"

    @property
    def description(self) -> str:
        return (
            "Look up quarterly earnings data (revenue, EPS, capex, etc.). "
            "Args: ticker (str, e.g. 'META'), quarter (str, e.g. '2022-Q3', optional). "
            "Returns last 4 quarters if no quarter specified."
        )

    def execute(self, **kwargs: Any) -> str:
        ticker = kwargs.get("ticker", "").upper()
        quarter = kwargs.get("quarter", "")

        if not ticker:
            return "Error: ticker is required."

        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        try:
            if quarter:
                rows = conn.execute(
                    "SELECT quarter, revenue_actual, revenue_est, eps_actual, eps_est, "
                    "operating_income, capex, free_cash_flow "
                    "FROM earnings WHERE ticker=? AND quarter=?",
                    (ticker, quarter),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT quarter, revenue_actual, revenue_est, eps_actual, eps_est, "
                    "operating_income, capex, free_cash_flow "
                    "FROM earnings WHERE ticker=? ORDER BY quarter DESC LIMIT 4",
                    (ticker,),
                ).fetchall()
                rows = rows[::-1]
        finally:
            conn.close()

        if not rows:
            return f"No earnings data found for {ticker}."

        lines = [f"{ticker} earnings ($ millions, EPS in $):"]
        for qtr, rev, rev_e, eps, eps_e, oi, capex, fcf in rows:
            beat_rev = "BEAT" if rev and rev_e and rev > rev_e else "MISS"
            beat_eps = "BEAT" if eps and eps_e and eps > eps_e else "MISS"
            lines.append(
                f"  {qtr}: Rev={rev:,.0f} (est {rev_e:,.0f}, {beat_rev}) | "
                f"EPS={eps:.2f} (est {eps_e:.2f}, {beat_eps}) | "
                f"OpInc={oi:,.0f} | CapEx={capex:,.0f} | FCF={fcf:,.0f}"
            )
        return "\n".join(lines)


class MacroIndicatorLookup(tool_module.Tool):
    """Look up macroeconomic indicators."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "lookup_macro"

    @property
    def description(self) -> str:
        return (
            "Look up macroeconomic indicators. "
            "Args: indicator (str, e.g. 'VIX', 'US_10Y_YIELD', 'FED_FUNDS_RATE', "
            "'BRENT_CRUDE', 'SP500'), "
            "start_date (str, YYYY-MM-DD, optional), end_date (str, YYYY-MM-DD, optional). "
            "Returns last 5 data points if no dates given."
        )

    def execute(self, **kwargs: Any) -> str:
        indicator = kwargs.get("indicator", "").upper()
        start_date = kwargs.get("start_date", "")
        end_date = kwargs.get("end_date", "")

        if not indicator:
            return "Error: indicator is required."

        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        try:
            if start_date and end_date:
                rows = conn.execute(
                    "SELECT date, value FROM macro_indicators "
                    "WHERE indicator=? AND date BETWEEN ? AND ? ORDER BY date LIMIT 20",
                    (indicator, start_date, end_date),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT date, value FROM macro_indicators "
                    "WHERE indicator=? ORDER BY date DESC LIMIT 5",
                    (indicator,),
                ).fetchall()
                rows = rows[::-1]
        finally:
            conn.close()

        if not rows:
            return f"No data found for indicator {indicator}."

        lines = [f"{indicator}:"]
        for date, value in rows:
            lines.append(f"  {date}: {value:.2f}")
        return "\n".join(lines)


def create_financial_tools(db_path: Path) -> list[tool_module.Tool]:
    """Create all financial tool instances backed by the given database."""
    return [
        StockPriceLookup(db_path),
        EarningsLookup(db_path),
        MacroIndicatorLookup(db_path),
    ]
