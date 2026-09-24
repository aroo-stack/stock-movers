#!/usr/bin/env python3
"""stock_movers.py -- "What Actually Moves This Stock?"

Measures how much a stock actually moves around three kinds of known events:

    1. earnings dates      (yfinance Ticker.earnings_dates)
    2. news headlines      (Yahoo Finance RSS feed for the ticker)
    3. macro releases      (FRED public CSVs: FEDFUNDS, CPIAUCSL, PAYEMS)

and compares each event type against a random-day control sample drawn from
the same date range, so the result answers: does event X move this stock
more than an ordinary random day would?

Dependencies: yfinance, pandas, numpy, matplotlib, requests
Run:          python stock_movers.py TICKER            (writes report.html + csv)
              python stock_movers.py --json TICKER     (prints the full report
                                                        as one JSON object,
                                                        nothing written to disk)
"""

from __future__ import annotations

import csv
import datetime as dt
import email.utils
import html
import io
import json
import subprocess
import sys
import re
import time
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
      "Accept-Language": "en-US,en;q=0.9"}
RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FINVIZ_URL = "https://finviz.com/quote.ashx?t={ticker}&p=d"
NY = "America/New_York"


class CurlResponse:
    """Minimal stand-in for requests.Response when curl takes over."""
    def __init__(self, text: str):
        self.status_code = 200
        self.text = text
        self.content = text.encode("utf-8")


def _curl_get(url: str, headers: dict) -> CurlResponse:
    """curl fallback. Tries HTTP/1.1 first (some CDN edges reset HTTP/2
    streams under load, while 1.1 stays stable)."""
    last = None
    for proto in ("--http1.1", "--http2"):
        cmd = ["curl", proto, "-sS", "-L", "--max-time", "25", "--compressed"]
        for k, v in (headers or {}).items():
            cmd += ["-H", f"{k}: {v}"]
        cmd += [url]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return CurlResponse(proc.stdout)
        last = RuntimeError(f"curl ({proto}) failed for {url}: {proc.stderr[:200]}")
    raise last


def get(url: str, retries: int = 3, timeout: int = 15,
        headers: "dict | None" = None) -> requests.Response:
    """GET with retries/backoff.  Uses a browser-ish User-Agent for the news
    sites (they block plain clients) but deliberately leaves the UA alone for
    FRED, whose Akamai edge rejects browser-like UAs by dropping the stream."""
    headers = dict(headers or {})
    if "fred.stlouisfed.org" not in url:
        headers.setdefault("User-Agent", UA["User-Agent"])
    if "finviz" in url:
        headers.setdefault("Referer", "https://finviz.com/")
    headers.setdefault("Accept-Language", "en-US,en;q=0.9")
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    # Curl frequently succeeds where python-requests cannot (e.g. certain
    # CDN/proxy paths time out reads).  Keep the download resilient; the
    # retry failures are intentionally ignored (each fetch is idempotent).
    return _curl_get(url, headers)

# Baseline verdict threshold: an event type is "notably above baseline" when
# its average daily move exceeds the baseline average daily move by more than
# one standard deviation of the baseline moves (measured on ordinary days).
THRESHOLD_SD = 1.0

# FRED series definitions.  NOTE: FRED's CSV gives the calendar month COVERED
# by each observation, not the press-release date.  Release dates are
# therefore estimated from each series' standing public publication schedule
# (BLS releases CPI ~the 10th of the month following the data month, and the
# jobs report on the first Friday of the following month).  FEDFUNDS is a
# monthly average of the effective rate, so its "event date" is just the start
# of the month whose average level changed -- a coarse proxy, used only to
# catch months when Fed policy actually moved.  All of this is disclosed in
# the output and in the events CSV.
MACRO_SERIES = {
    "FEDFUNDS": {
        "label": "Fed funds rate",
        "requires_change": True,   # only count months where the rate moved
    },
    "CPIAUCSL": {
        "label": "CPI (inflation)",
        "requires_change": False,  # a flat CPI print is still a release day
    },
    "PAYEMS": {
        "label": "Jobs report",
        "requires_change": False,
    },
}

# Value change per series as reported in the events CSV / console detail.
MACRO_NORMALIZE = {
    "CPIAUCSL": lambda old, new: (new / old - 1.0) * 100.0,   # % mo/mo
    "PAYEMS": lambda old, new: new - old,                     # thousands mo/mo
    "FEDFUNDS": lambda old, new: new - old,                   # pct points mo/mo
}


# ---------------------------------------------------------------------------
# date helpers
# ---------------------------------------------------------------------------
def first_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    while d.weekday() != 4:  # Friday
        d += dt.timedelta(days=1)
    return d


def business_day_on_or_after(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def macro_release_date(series_id: str, obs_date: dt.date) -> dt.date:
    """Estimate the market-relevant release date for a FRED observation."""
    yr = obs_date.year + 1 if obs_date.month == 12 else obs_date.year
    mo = 1 if obs_date.month == 12 else obs_date.month + 1
    if series_id == "PAYEMS":
        return first_friday(yr, mo)
    if series_id == "CPIAUCSL":
        return business_day_on_or_after(dt.date(yr, mo, 10))
    return obs_date  # FEDFUNDS: month start of the data month (proxy)


def to_us_date(value) -> "dt.date | None":
    """Normalize an arbitrary timestamp to a date in America/New_York."""
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts is pd.NaT:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY)
    else:
        ts = ts.tz_convert(NY)
    return ts.date()


def parse_rfc822(text: str) -> "dt.date | None":
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return to_us_date(parsed)


# ---------------------------------------------------------------------------
# data fetchers
# ---------------------------------------------------------------------------
def fetch_price_history(ticker: str) -> pd.DataFrame:
    """Daily OHLCV history (split/dividend-adjusted, auto_adjust=True), with
    the Stock Splits column retained so the sanity check can confirm whether
    any split occurred inside the window.  Returns NaN if unavailable."""
    t = yf.Ticker(ticker)
    df = t.history(period="2y", auto_adjust=True, actions=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if "Stock Splits" not in df.columns:
        df["Stock Splits"] = 0.0
    return df[["Close", "Stock Splits"]].copy()


def fetch_earnings(ticker: str) -> list[dict]:
    """Earnings dates via yfinance's earnings_dates table.  Falls back to the
    screener scrape (which needs no lxml/html5lib) if the page isn't parsed."""
    t = yf.Ticker(ticker)
    df = None
    try:
        df = t.get_earnings_dates(limit=16)
    except Exception:
        try:
            df = t._get_earnings_dates_using_screener(limit=16)
        except Exception:
            df = None

    events = []
    if df is None or df.empty:
        return events

    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    for _, row in df.iterrows():
        d = to_us_date(row["date"])
        if d is None:
            continue
        # If the source distinguishes event kinds, keep only earnings calls.
        if "Event Type" in row.index and isinstance(row["Event Type"], str):
            if row["Event Type"].lower() not in ("earnings",):
                continue
        eps, surprise = row.get("Reported EPS", None), row.get("Surprise(%)", None)
        if isinstance(surprise, float) and not np.isnan(surprise):
            eps_txt = f"EPS {eps:.2f}" if isinstance(eps, float) and not np.isnan(eps) else "EPS"
            detail = f"{eps_txt} (surprise {surprise:+.1f}%)"
        elif isinstance(eps, float) and not np.isnan(eps):
            detail = f"EPS {eps:.2f}"
        else:
            detail = "earnings announcement"
        events.append({"date": d, "type": "Earnings", "detail": detail})
    return events


def fetch_news(ticker: str, max_items: int = 40) -> list[dict]:
    """Headlines from Yahoo's RSS (primary) plus Finviz's recent-news table
    (supplement).  Yahoo RSS only ever holds the last ~24h of stories, so on
    its own it is almost never measurable against known price history."""
    events = []
    try:
        url = RSS_URL.format(ticker=ticker)
        root = ET.fromstring(get(url).content)
        for item in root.iter():
            if not item.tag.endswith("item"):
                continue
            title, pub_date = None, None
            for child in item:
                tag = child.tag.split("}")[-1]
                if tag == "title":
                    title = (child.text or "").strip()
                elif tag == "pubDate":
                    pub_date = child.text
            if not title:
                continue
            d = parse_rfc822(pub_date) if pub_date else None
            if d is None:
                continue
            events.append({"date": d, "type": "News", "detail": title[:120]})
    except Exception as exc:
        print(f"  [warn] could not fetch Yahoo RSS news: {exc}")

    try:
        events += fetch_news_finviz(ticker)
    except Exception as exc:
        print(f"  [warn] could not fetch Finviz news: {exc}")

    seen, out = set(), []
    for ev in events:
        key = (ev["date"], ev["detail"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out[:max_items]


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def _parse_finviz_time(cell: str, today: dt.date) -> "dt.date | None":
    cell = " ".join(cell.split())
    m = re.search(r"([A-Z][a-z]{2})-(\d{1,2})-(\d{2})", cell)
    if m:
        mon, day, yy = _MONTHS.get(m.group(1)), int(m.group(2)), int(m.group(3))
        if mon is None or day < 1 or day > 31:
            return None
        year = 2000 + yy
        try:
            d = dt.date(year, mon, day)
        except ValueError:
            return None
        return d
    if re.match(r"^Today", cell) or re.match(r"^\d{1,2}:\d{2}\s*[AP]M", cell):
        return today
    return None


def fetch_news_finviz(ticker: str, max_items: int = 80) -> list[dict]:
    """Recent headlines from Finviz's news table (no API key required)."""
    html = get(FINVIZ_URL.format(ticker=ticker.lower())).text
    match = re.search(r'id="news-table".*?</table>', html, re.S)
    if not match:
        return []
    table = match.group(0)
    today = dt.datetime.now(ZoneInfo(NY)).date()
    events = []
    for row in re.findall(r'<tr class="cursor-pointer.*?</tr>', table, re.S):
        time_cell = re.search(r'<td[^>]*width="130"[^>]*>(.*?)</td>', row, re.S)
        title = re.search(r'<a class="tab-link-news"[^>]*>(.*?)</a>', row, re.S)
        if not time_cell or not title:
            continue
        d = _parse_finviz_time(re.sub(r"<.*?>", "", time_cell.group(1)), today)
        text = re.sub(r"<.*?>", "", title.group(1))
        text = re.sub(r"\s+", " ", text).strip()
        if d is None or not text:
            continue
        events.append({"date": d, "type": "News", "detail": text[:120]})
    return events[:max_items]


def fetch_fred(series_id: str) -> pd.DataFrame:
    url = FRED_URL.format(series_id=series_id)
    resp = get(url)
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]
    # CSV header is "observation_date,{SERIES_ID}" -- normalize to "value".
    if len(df.columns) >= 2:
        df = df.rename(columns={df.columns[1]: "value"})
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["observation_date", "value"])
    return df.sort_values("observation_date").reset_index(drop=True)


def build_macro_events(start: dt.date) -> list[dict]:
    """Build macro events from all three FRED series; release dates are
    estimated from each series' public publication schedule (see module doc)."""
    events = []
    for series_id, cfg in MACRO_SERIES.items():
        try:
            df = fetch_fred(series_id)
        except Exception as exc:
            print(f"  [warn] could not fetch FRED series {series_id}: {exc}")
            continue
        if len(df) < 2:
            continue
        prev = None
        for _, row in df.iterrows():
            obs_date = row["observation_date"].date()
            cur = float(row["value"])
            if prev is None or np.isnan(prev) or np.isnan(cur):
                prev = cur
                continue
            change = MACRO_NORMALIZE[series_id](prev, cur)
            prev = cur
            if cfg["requires_change"] and abs(change) < 1e-9:
                continue
            rel_date = macro_release_date(series_id, obs_date)
            if rel_date < start:
                continue
            detail = ""
            if series_id == "CPIAUCSL":
                detail = f"CPI index {cur:.1f} ({change:+.2f}% mo/mo)"
            elif series_id == "PAYEMS":
                detail = f"jobs {cur / 1000.0:.1f}M ({change:+.0f}k mo/mo)"
            else:
                detail = f"rate {cur:.2f}% ({change:+.2f} pct pts mo/mo)"
            events.append({"date": rel_date, "type": cfg["label"], "detail": detail})
    return events


# ---------------------------------------------------------------------------
# return measurement
# ---------------------------------------------------------------------------
def build_returns(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """r1: next trading day close / today close - 1
       r3: close three trading days later / today close - 1"""
    close = close.dropna()
    days = [d.date() for d in close.index]
    r1 = pd.Series((close.shift(-1) / close - 1.0).to_numpy(), index=days)
    r3 = pd.Series((close.shift(-3) / close - 1.0).to_numpy(), index=days)
    return r1, r3


def base_position(event_date: dt.date, trading_dates: list[dt.date]) -> "int | None":
    """Index of the base trading day (last trading day on or before the
    event date), or None when the event sits outside the price window (no
    trading day to anchor the measurement, e.g. future or pre-window)."""
    if not trading_dates or event_date < trading_dates[0]:
        return None
    if event_date > trading_dates[-1]:
        return None
    pos = np.searchsorted(np.array(trading_dates, dtype="datetime64[D]"),
                          np.datetime64(event_date), side="right") - 1
    return int(pos)


def measure_events(events: list[dict], trading_dates: list[dt.date],
                   r1: pd.Series, r3: pd.Series) -> list[dict]:
    out = []
    for ev in events:
        pos = base_position(ev["date"], trading_dates)
        rec = {"date": ev["date"], "type": ev["type"], "detail": ev["detail"],
               "base_date": None, "r1": None, "r3": None}
        if pos is None:
            out.append(rec)
            continue
        base = trading_dates[pos]
        rec["base_date"] = base
        if pos + 1 < len(trading_dates) and base in r1.index:
            v = r1.loc[base]
            rec["r1"] = None if pd.isna(v) else float(v)
        if pos + 3 < len(trading_dates) and base in r3.index:
            v = r3.loc[base]
            rec["r3"] = None if pd.isna(v) else float(v)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# data sanity checks
# ---------------------------------------------------------------------------
def report_data_sanity(ticker: str, close: pd.Series,
                       base_dates: set) -> None:
    """(1) Confirm split/dividend adjustment state, (2) scan for any single
    day >15% and check for the ~-50% signature of a missed split, and
    (4a) print an independent realized-volatility reference point."""
    hist = fetch_price_history(ticker)
    splits = hist["Stock Splits"]
    split_dates = splits[splits.abs() > 1e-9]
    print("      price source  : yfinance daily OHLCV, auto_adjust=True "
          "(split/dividend-adjusted closes)")
    print("      split events  : "
          + (", ".join(f"{d.date()} x{s:.1f} (recorded by Yahoo; auto-adjust applied)"
                       for d, s in split_dates.items()) if len(split_dates)
             else "none recorded by Yahoo in the window (raw == adjusted, "
                  "nothing for auto_adjust to rescale)"))

    chg = (close / close.shift(1) - 1.0)
    big = chg.abs()[chg.abs() > 0.15].dropna()
    print(f"      |moves| >15%  : {len(big)}")
    for d, v in chg[big.index].items():
        i = close.index.get_loc(d)
        prev_px = float(close.iloc[i - 1])
        cur_px = float(close.iloc[i])
        labeled = "EVENT day" if d.date() in base_dates else "no known event"
        print(f"        {d.date()}  {prev_px:9.2f} -> {cur_px:9.2f}  "
              f"{v * 100:+.1f}%   (raw before/after closes; {labeled})")
    if len(big):
        worst = chg.abs().max()
        print(f"      largest single-day |move| = {worst * 100:.1f}%; "
              f"no ~-50% split-drop signature present"
              if worst < 0.5 else
              f"      WARNING: a {worst * 100:.1f}% drop occurred -- this looks "
              f"like an UNadjusted stock split, said series is unusable")

    # independent realized-volatility sanity check
    lr = np.log(close / close.shift(1)).dropna()
    vol_ann = float(lr.std(ddof=1) * np.sqrt(252.0))
    vol_daily = float(lr.std(ddof=1))
    implied = vol_daily * np.sqrt(2.0 / np.pi)   # mean |move| ~ 0.80*std if normal
    print(f"      realized vol  : {vol_ann * 100:.1f}% annualized "
          f"(std of daily log returns x sqrt(252)); daily std "
          f"{vol_daily * 100:.2f}%, implying a typical daily move of "
          f"about {implied * 100:.2f}%")


# ---------------------------------------------------------------------------
# control baseline and verdicts
# ---------------------------------------------------------------------------
def eligible_pool(trading_dates: list[dt.date], events: list[dict],
                  r1: pd.Series, r3: pd.Series) -> tuple[pd.Series, pd.Series, set]:
    """Ordinary trading days.  Around each event we exclude the base trading
    day plus the 2 trading days BEFORE it (pre-event drift / positiong,
    e.g. run-ups into earnings) and the 3 trading days AFTER it (which is the
    3-day return window the analysis measures).  Without the pre-event days,
    days like the +13% move two sessions before an earnings print would leak
    into the "ordinary day" baseline."""
    contaminated = set()
    for ev in events:
        if ev.get("base_date") is None:
            continue
        pos = trading_dates.index(ev["base_date"])
        for k in range(-2, 4):
            if 0 <= pos + k < len(trading_dates):
                contaminated.add(trading_dates[pos + k])
    mask_r1, mask_r3 = [], []
    days = []
    for i, day in enumerate(trading_dates):
        if day in contaminated:
            continue
        if i + 1 >= len(trading_dates) or i + 3 >= len(trading_dates):
            continue
        v1, v3 = r1.iloc[i], r3.iloc[i]
        if pd.isna(v1) or pd.isna(v3):
            continue
        days.append(day)
        mask_r1.append(float(v1))
        mask_r3.append(float(v3))
    idx = pd.Index(days)
    return pd.Series(mask_r1, index=idx), pd.Series(mask_r3, index=idx), contaminated


def draw_random_sample(pool: pd.Series, n: int, seed: int) -> pd.Series:
    if n <= 0 or len(pool) == 0:
        return pool.sample(min(n, len(pool)), random_state=seed)
    k = min(n, len(pool))
    return pool.sample(k, random_state=seed)


def verdict(ev_mean_abs: float, base_mean_abs: float, base_std_abs: float,
            n: int) -> str:
    if n < 3:
        return "too few events"
    th = THRESHOLD_SD * base_std_abs
    diff = ev_mean_abs - base_mean_abs
    if diff > th:
        return "notably above baseline"
    if diff < -th:
        return "below baseline"
    return "in line with baseline"


def pct(x: "float | None") -> str:
    return "   --   " if x is None else f"{x * 100:+.2f}%"


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def write_csv(events: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "base_trading_day", "type", "detail",
                    "next_day_return", "three_day_return"])
        for ev in events:
            w.writerow([
                ev["date"].isoformat(),
                "" if ev.get("base_date") is None else ev["base_date"].isoformat(),
                ev["type"],
                ev["detail"],
                "" if ev.get("r1") is None else f"{ev['r1']:.6f}",
                "" if ev.get("r3") is None else f"{ev['r3']:.6f}",
            ])


TYPE_COLORS = {
    "Earnings": "#1f883d",
    "News": "#d62728",
    "Fed funds rate": "#9467bd",
    "CPI (inflation)": "#ff7f0e",
    "Jobs report": "#8c564b",
}
FALLBACK_COLORS = ["#17becf", "#e377c2", "#bcbd22", "#7f7f7f", "#393b79"]


def build_chart_payload(ticker: str, measured: list[dict],
                        close: pd.Series) -> tuple[list, dict]:
    """Plotly traces + layout for the 2y price line with color-coded,
    per-event-type markers.  Reuses only in-memory data (no re-fetching)."""
    events = [m for m in measured if m.get("base_date") is not None]
    tz = close.index.tz
    dates = [d.date().isoformat() for d in close.index]
    prices = [None if v != v else float(v) for v in close.tolist()]

    traces = [{
        "type": "scatter", "mode": "lines", "name": ticker.upper(),
        "x": dates, "y": prices,
        "line": {"color": "#111", "width": 1.5},
        "hovertemplate": "<b>%{x}</b><br>close %{y:,.4f}<extra></extra>",
    }]

    for idx, t in enumerate(sorted({m["type"] for m in events})):
        grp = sorted((m for m in events if m["type"] == t),
                     key=lambda m: m["base_date"])
        x, y, custom = [], [], []
        for m in grp:
            bd = pd.Timestamp(m["base_date"], tz=tz)
            if bd not in close.index:
                continue
            x.append(m["base_date"].isoformat())
            y.append(float(close.loc[bd]))
            r1 = "" if m.get("r1") is None else f"{m['r1'] * 100:+.2f}%"
            r3 = "" if m.get("r3") is None else f"{m['r3'] * 100:+.2f}%"
            custom.append([r1, r3, m["detail"]])
        color = TYPE_COLORS.get(t, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])
        hover = ("<b>%{x}</b>" + '<span style="color:' + color + '">  ' + t
                 + "</span><br>" + "next-day <b>%{customdata[0]}</b>"
                 + " &nbsp;|&nbsp; " + "3-day <b>%{customdata[1]}</b><br>"
                 + "%{customdata[2]}<extra></extra>")
        traces.append({
            "type": "scatter", "mode": "markers", "name": t,
            "x": x, "y": y,
            "marker": {"color": color, "size": 9,
                       "line": {"color": "#ffffff", "width": 1}},
            "customdata": custom,
            "hovertemplate": hover,
        })

    layout = {
        "title": {"text": f"{ticker.upper()} - events overlaid on "
                          f"2y price ({len(events)} events plotted)"},
        "xaxis": {"title": "date", "rangeslider": {"visible": True},
                  "tickangle": -45},
        "yaxis": {"title": "close price"},
        "legend": {"clickmode": "event", "x": 1.01, "xanchor": "left",
                   "orientation": "v"},
        "hovermode": "closest",
        "margin": {"t": 55, "r": 10, "b": 40, "l": 50},
        "height": 560,
        "template": "plotly_white",
    }
    return traces, layout


def html_type_table(types: list[str], stats: dict, r3: bool) -> str:
    """The same next-day (r3=False) / 3-day (r3=True) comparison table that
    is printed on the console, rendered as compact HTML."""
    suf, title = ("r3", "3-day") if r3 else ("r1", "next-day")
    n_k, avg_k = f"n_{suf}", f"avg_abs_{suf}"
    med_k, sgn_k = f"med_abs_{suf}", f"avg_sgn_{suf}"
    bl_k, ver_k = f"bl_avg_abs_{suf}", f"verdict_{suf}"
    cols = ("Event type", "n", f"avg |{suf}|", f"mdn |{suf}|",
            f"avg {suf}", f"control |{suf}|", "diff", f"verdict ({title})")
    rows = []
    for t in types:
        s = stats[t]
        if s is None or s.get(avg_k) is None:
            rows.append("<tr><td colspan='8' class='dim'>%s: no measurable "
                        "%s return</td></tr>" % (html.escape(t), title))
            continue
        n = s[n_k]
        avg = s[avg_k] * 100.0
        med = s[med_k] * 100.0
        sgn = s[sgn_k] * 100.0
        bl = s[bl_k]
        bl_str = "-" if bl is None else f"{bl * 100:.2f}%"
        diff = avg - bl if bl is not None else None
        diff_str = "-" if diff is None else f"{diff:+.2f}%"
        verdict = s[ver_k]
        vcls = ("verdict-above" if verdict == "notably above baseline"
                else "verdict-below" if verdict == "below baseline" else "")
        rows.append(
            "<tr><td>%s</td><td>%d</td><td>%.2f%%</td><td>%.2f%%</td>"
            "<td>%+.2f%%</td><td>%s</td><td>%s</td>"
            "<td class='%s'>%s</td></tr>"
            % (html.escape(t), n, avg, med, sgn, bl_str, diff_str,
               vcls, verdict))
    head = "".join("<th>%s</th>" % html.escape(c) for c in cols)
    return ("<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>"
            % (head, "".join(rows)))


_REPORT_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,
     sans-serif;margin:0;background:#f7f8fa;color:#1a1a1a}
.wrap{max-width:1120px;margin:0 auto;padding:22px 18px 48px}
h1{font-size:25px;margin:0 0 4px}
.sub{color:#5a6472;font-size:14px;margin:0 0 14px}
.box{background:#fff;border:1px solid #e4e7eb;border-radius:10px;
     padding:16px 20px 18px;margin:16px 0;box-shadow:0 1px 2px rgba(0,0,0,.03)}
.box h2{font-size:17px;margin:0 0 10px;padding-bottom:7px;border-bottom:1px solid #eef0f3}
.box h3{font-size:14px;margin:18px 0 8px;color:#333}
.summary{background:#f2f7f2;border-color:#d4e5d4}
.summary ul{margin:8px 0 12px;padding-left:20px}
.summary li{margin:8px 0;line-height:1.55}
.reminder{color:#667085;font-size:13px;font-style:italic;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:right;border:1px solid #eceef1;white-space:nowrap}
th{background:#f4f5f8;font-weight:600}
td:first-child,th:first-child,td.dim{text-align:left}
tr:nth-child(even) td{background:#fafbfc}
.dim{color:#98a0ab;font-style:italic}
.verdict-above{color:#1f883d;font-weight:600}
.verdict-below{color:#c0392b;font-weight:600}
.hint{color:#667085;font-size:13px}
#controls{margin:4px 0 10px}
button{font:inherit;padding:6px 14px;margin-right:8px;border:1px solid #b9c0ca;
       border-radius:6px;background:#fafafa;cursor:pointer}
button:hover{background:#eceff2}
pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#fafbfc;
    border:1px solid #e6e8ec;border-radius:6px;padding:10px 12px;font-size:12px;
    line-height:1.5;white-space:pre-wrap;overflow-x:auto}
ul.notes{margin:8px 0;padding-left:20px}
ul.notes li{margin:7px 0;line-height:1.5;color:#333}
"""


def write_report_html(ticker: str, trading_dates: list[dt.date],
                      measured: list[dict], close: pd.Series, stats: dict,
                      types: list[str], generated_at: str, summary: dict,
                      verify_blocks: list, data_notes: list[str],
                      path: str) -> None:
    """Write <ticker>_report.html -- ONE self-contained page containing the
    plain-English summary, the full comparison tables, the interactive price
    chart, the manual verification trace, and the data notes.  Only reuses
    data already computed in memory."""
    first, last = trading_dates[0].isoformat(), trading_dates[-1].isoformat()
    traces, layout = build_chart_payload(ticker, measured, close)
    payload = json.dumps({"traces": traces, "layout": layout},
                         ensure_ascii=False).replace("</", "<\\/")

    bullets = "".join("<li>%s</li>" % html.escape(line)
                      for line in summary["bullets"])
    context = html.escape(summary["context"])
    reminder = html.escape(summary["reminder"])

    verify_parts = []
    for label, lines in verify_blocks:
        verify_parts.append("<h3>%s</h3><pre>%s</pre>"
                            % (html.escape(label), html.escape("\n".join(lines))))
    verify_html = "".join(verify_parts) if verify_parts else (
        "<p class='hint'>No events had a measurable reaction to trace this run.</p>")

    notes_html = "".join("<li>%s</li>" % html.escape(n) for n in data_notes)

    table_body = (
        "<h3>Next-day move (r1): event-day close -> next trading-day close</h3>"
        + html_type_table(types, stats, False)
        + "<h3>3-day move (r3): event-day close -> +3 trading days later</h3>"
        + html_type_table(types, stats, True))

    js = """<script>
const PAYLOAD = __PAYLOAD__;
function setAll(v) {
  const gd = document.getElementById('chart');
  const n = gd.data.length;
  const vis = Array(n).fill(true);
  if (!v) { for (let i = 1; i < n; i++) vis[i] = false; }
  Plotly.restyle(gd, {visible: vis});
}
Plotly.newPlot('chart', PAYLOAD.traces, PAYLOAD.layout,
               {displaylogo: false, responsive: true});
</script>""".replace("__PAYLOAD__", payload)

    doc = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{ticker.upper()} - event mover report</title>\n"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        f"<style>{_REPORT_CSS}</style>\n</head>\n<body><div class=\"wrap\">\n"
        f"<h1>{ticker.upper()} &mdash; what actually moves it</h1>\n"
        f"<div class=\"sub\">Price range {first} .. {last} (2y) &middot; "
        f"report generated {html.escape(generated_at)} &middot; "
        f"{len(measured)} events collected</div>\n"
        "<div class=\"box summary\">\n<h2>In plain English</h2>\n"
        f"<ul>{bullets}</ul>\n<p>{context}</p>\n"
        f"<p class=\"reminder\">{reminder}</p>\n</div>\n"
        "<div class=\"box\"><h2>Event-type impact vs random-day baseline "
        f"(threshold: baseline avg + {THRESHOLD_SD:.0f} SD)</h2>\n"
        f"{table_body}\n</div>\n"
        "<div class=\"box\"><h2>Price chart with event markers</h2>\n"
        '<div id="controls">\n'
        '<button onclick="setAll(true)">Show all</button>\n'
        '<button onclick="setAll(false)">Hide event markers</button>\n'
        '<span class="hint">tip: click a legend entry to toggle that event type</span>\n'
        "</div>\n<div id=\"chart\" style=\"width:100%\"></div>\n</div>\n"
        "<div class=\"box\"><h2>Manual verification against raw closes</h2>\n"
        f"{verify_html}\n</div>\n"
        "<div class=\"box\"><h2>Data notes</h2>\n"
        f"<ul class=\"notes\">{notes_html}</ul>\n</div>\n"
        "</div>\n"
        f"{js}\n"
        "</body>\n</html>\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)


def verify_event(ev: dict, close: pd.Series, trading_dates: list[dt.date],
                 r1: pd.Series, r3: pd.Series, kind: str) -> list[str]:
    """Manual trace of the return arithmetic against raw close prices.  Prints
    the trace (console) and returns the same lines for the HTML report."""
    pos = close.index.get_loc(pd.Timestamp(ev["base_date"], tz=close.index.tz))
    base_px = float(close.iloc[pos])
    nxt_px = float(close.iloc[pos + 1])
    r1_manual = nxt_px / base_px - 1.0
    lines = [f"  Verify {kind.upper()} event: {ev['date']}  ({ev['detail']})",
             f"    {ev['base_date']} (base, close)          = {base_px:.4f}",
             f"    next trading day close                 = {nxt_px:.4f}",
             f"    next-day r  = {nxt_px:.4f}/{base_px:.4f} - 1 = {r1_manual:+.4%}"]
    if pos + 3 < len(close):
        thr_px = float(close.iloc[pos + 3])
        r3_manual = thr_px / base_px - 1.0
        lines += [f"    +3 trading days close                  = {thr_px:.4f}",
                  f"    3-day r     = {thr_px:.4f}/{base_px:.4f} - 1 = {r3_manual:+.4%}",
                  f"    (script values: next-day {ev['r1']:+.4%}, 3-day {ev['r3']:+.4%})"]
    else:
        lines += ["    +3 trading days close                  = (beyond history)",
                  "    3-day r     = n/a (no +3 trading days of price data yet)",
                  f"    (script values: next-day {ev['r1']:+.4%})"]
    for line in lines:
        print(line)
    return lines


# ---------------------------------------------------------------------------
# plain-language summary
# ---------------------------------------------------------------------------
FRIENDLY_TYPE = {
    "Earnings": "earnings reports",
    "News": "news headlines",
    "Fed funds rate": "Fed rate decisions",
    "CPI (inflation)": "CPI inflation reports",
    "Jobs report": "jobs reports",
}


def _friendly(t: str) -> str:
    return FRIENDLY_TYPE.get(t, t.lower())


def _join_friendly(names: list[str]) -> str:
    names = [_friendly(t) for t in names]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def plain_language_summary(ticker: str, types: list[str], stats: dict,
                           base_avg_abs: float,
                           base_med_abs: float = 0.0) -> dict:
    """Render the numeric verdicts as a few everyday sentences, derived
    entirely from the stats already computed above (never hardcoded).
    Returns {"bullets", "context", "reminder"} so both the console printer
    and the HTML report use the exact same wording."""
    def has(t):
        return stats[t] is not None

    above = [t for t in types if has(t) and stats[t]["verdict_r1"] == "notably above baseline"]
    below = [t for t in types if has(t) and stats[t]["verdict_r1"] == "below baseline"]
    inline = [t for t in types if has(t) and stats[t]["verdict_r1"] == "in line with baseline"]
    sparse = [t for t in types if not has(t) or stats[t]["verdict_r1"] == "too few events"]

    base_pct = base_avg_abs * 100.0

    bullets = []
    if above or inline or below:
        if above:
            for t in above:
                ev_pct = stats[t]["avg_abs_r1"] * 100.0
                bullets.append(
                    f"{_friendly(t).capitalize()} tend to move {ticker}'s price noticeably more than "
                    f"a typical day — about {ev_pct:.1f}% the day after, versus about "
                    f"{base_pct:.1f}% on an ordinary day.")
        if inline or below:
            if inline:
                bullets.append(
                    f"{_join_friendly(inline)} did not move {ticker}'s price meaningfully differently "
                    f"than an ordinary day — the stock's normal day-to-day ups and downs already "
                    f"cover about as much ground.")
            if below:
                bullets.append(
                    f"{_join_friendly(below).capitalize()} tended to move {ticker}'s price LESS than "
                    f"an average day.")
    else:
        bullets.append(
            f"None of the event types measured showed a real, above-baseline reaction worth "
            f"flagging for {ticker} on this run.")

    for t in sparse:
        if not has(t):
            bullets.append(
                f"There isn't enough {_friendly(t)} data yet to say anything reliable about how "
                f"they move {ticker}'s price.")
        else:
            bullets.append(
                f"There weren't enough {_friendly(t)} yet to say anything reliable about how they "
                f"move {ticker}'s price.")

    context = (f"For context: a typical day for {ticker} moves the price about "
               f"{base_med_abs * 100:.1f}% (median), while the average skip day is "
               f"{base_avg_abs * 100:.1f}% -- the average is pulled up by a few "
               f"extreme days, so the median is the better 'usual day' reference "
               f"when judging these reactions.")

    reminder = ("That's historical pattern-matching over a limited sample, "
                "not a prediction of future moves.")

    return {"bullets": bullets, "context": context, "reminder": reminder}


def print_plain_language(ticker: str, summary: dict) -> None:
    """Print the [6] plain-English section (kept separate from the builder so
    the HTML report can reuse the identical text)."""
    print("\n[6] In plain English")
    for line in summary["bullets"]:
        print(f"  - {line}")
    print(f"\n  {summary['context']}")
    print(f"\n  One reminder: {summary['reminder']}")


def json_report(ticker: str, trading_dates: list[dt.date],
                measured: list[dict], close: pd.Series, stats: dict,
                types: list[str], generated_at: str,
                summary: dict, verify_blocks: list[tuple[str, list[str]]],
                data_notes: list[str], pool_r1: pd.Series,
                base_abs_mu_r1: float, base_abs_sd_r1: float,
                base_abs_med_r1: float, base_abs_mu_r3: float,
                base_abs_med_r3: float) -> dict:
    """Everything the HTML report renders, as one JSON-serializable dict
    (used by the --json CLI mode / web app; nothing is written to disk)."""
    traces, layout = build_chart_payload(ticker, measured, close)
    plotted = [m for m in measured if m["base_date"] is not None]
    return {
        "meta": {
            "ticker": ticker.upper(),
            "first": trading_dates[0].isoformat(),
            "last": trading_dates[-1].isoformat(),
            "generated_at": generated_at,
            "n_trading_days": len(trading_dates),
            "n_events": len(measured),
            "n_plotted": len(plotted),
        },
        "chart": {"traces": traces, "layout": layout},
        "stats": stats,
        "types": types,
        "baseline": {
            "pool_n": int(len(pool_r1)),
            "avg_abs_r1": base_abs_mu_r1,
            "sd_abs_r1": base_abs_sd_r1,
            "med_abs_r1": base_abs_med_r1,
            "avg_abs_r3": base_abs_mu_r3,
            "med_abs_r3": base_abs_med_r3,
        },
        "summary": summary,
        "verification": [{"label": label, "lines": lines}
                         for label, lines in verify_blocks],
        "notes": data_notes,
        "events": [_serialize_event(m) for m in measured],
    }


def _serialize_event(m: dict) -> dict:
    return {
        "date": m["date"].isoformat() if m["date"] else None,
        "base_date": m["base_date"].isoformat() if m["base_date"] else None,
        "type": m["type"],
        "detail": m["detail"],
        "r1": m["r1"],
        "r3": m["r3"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(ticker: str, json_mode: bool = False) -> None:
    real_stdout = sys.stdout
    if json_mode:
        # in --json mode all the normal console chatter goes to a throwaway
        # buffer so stdout carries ONLY the one JSON object at the end
        # (progress is still logged to stderr for the web server to see)
        sys.stdout = sys.stderr
        print(f"\n[0] --json mode for {ticker.upper()}", file=sys.stderr)
    print(f"\n{'=' * 78}")
    print(f"  What Actually Moves {ticker.upper()} -- event impact analyzer")
    print(f"{'=' * 78}")

    # --- price history -----------------------------------------------------
    print(f"\n[1] Pulling 2y daily price history for {ticker.upper()} ...")
    hist = fetch_price_history(ticker)
    if hist.empty:
        print(f"  [error] no price data for {ticker}. aborting.")
        sys.exit(1)
    trading_dates = [d.date() for d in hist.index]
    start = trading_dates[0]
    close = hist["Close"]
    r1, r3 = build_returns(close)
    print(f"      {len(trading_dates)} trading days, {trading_dates[0]} .. {trading_dates[-1]}")

    # --- events ------------------------------------------------------------
    print("[2] Collecting event dates ...")
    earnings = fetch_earnings(ticker)
    print(f"      earnings dates : {len(earnings)}")
    news = fetch_news(ticker)
    print(f"      news headlines : {len(news)}")
    macro = build_macro_events(start)
    print(f"      macro releases : {len(macro)}")
    events = earnings + news + macro
    if not events:
        print(f"  [error] no events found for {ticker}. aborting.")
        sys.exit(1)

    measured = measure_events(events, trading_dates, r1, r3)
    mie = [m for m in measured if m["base_date"] is not None]

    # --- data sanity ------------------------------------------------------
    report_data_sanity(ticker, close, {m["base_date"] for m in mie})

    # --- control baseline --------------------------------------------------
    pool_r1, pool_r3, _ = eligible_pool(trading_dates, measured, r1, r3)
    if len(pool_r1) == 0:
        print("  [error] no eligible control days. aborting.")
        sys.exit(1)
    base_abs_mu_r1, base_abs_sd_r1 = pool_r1.abs().mean(), pool_r1.abs().std(ddof=1)
    base_abs_mu_r3, base_abs_sd_r3 = pool_r3.abs().mean(), pool_r3.abs().std(ddof=1)
    base_abs_med_r1 = float(np.median(pool_r1.abs()))
    base_abs_med_r3 = float(np.median(pool_r3.abs()))
    print(f"\n      control pool (ordinary days): n={len(pool_r1)} "
          f"(event day, 2 days before, and 3 days after each event excluded)")
    print(f"        next-day avg |move|={base_abs_mu_r1 * 100:.2f}%  "
          f"median={base_abs_med_r1 * 100:.2f}%  sd={base_abs_sd_r1 * 100:.2f}%")
    print(f"        3-day    avg |move|={base_abs_mu_r3 * 100:.2f}%  "
          f"median={base_abs_med_r3 * 100:.2f}%")

    # sanity: does the baseline agree with realized volatility?
    lr = np.log(close / close.shift(1)).dropna()
    vol_ann = float(lr.std(ddof=1) * np.sqrt(252.0))
    implied_typical = float(lr.std(ddof=1) * np.sqrt(2.0 / np.pi))
    agree = abs(base_abs_mu_r1 - implied_typical) < 0.3 * implied_typical
    print(f"        realized vol {vol_ann * 100:.1f}% ann. implies a typical "
          f"daily move of ~{implied_typical * 100:.2f}% "
          f"(measured baseline {base_abs_mu_r1 * 100:.2f}% -> "
          f"{'consistent' if agree else 'CHECK: inconsistent!'})")
    realized_note = (f"Realized volatility {vol_ann * 100:.1f}% annualized "
                     f"(std of daily log returns x sqrt(252)) implies a typical "
                     f"daily move of about {implied_typical * 100:.2f}%, which is "
                     f"{'consistent' if agree else 'NOT consistent'} with the "
                     f"measured ordinary-day baseline of "
                     f"{base_abs_mu_r1 * 100:.2f}%.")
    if base_abs_med_r1 < 0.75 * base_abs_mu_r1:
        print(f"        median ({base_abs_med_r1 * 100:.2f}%) is well below the mean "
              f"({base_abs_mu_r1 * 100:.2f}%): a handful of extreme days skew the "
              f"average up, so the median is the more representative 'ordinary day' "
              f"number to keep in mind")

    # --- per-type stats ----------------------------------------------------
    types = sorted({m["type"] for m in mie})
    stats = {}
    for t in types:
        grp = [m for m in mie if m["type"] == t and m["r1"] is not None]
        grp3 = [m for m in mie if m["type"] == t and m["r3"] is not None]
        if not grp:
            stats[t] = None
            continue
        r1v = np.array([m["r1"] for m in grp])
        r3v = np.array([m["r3"] for m in grp3]) if grp3 else np.array([])
        seedy = 1234 + sum(ord(c) for c in t)
        sample_r1 = draw_random_sample(pool_r1.abs(), len(r1v), seedy)
        bl_avg_abs_r1 = sample_r1.mean()
        bl_avg_abs_r3 = None
        if len(r3v):
            sample_r3 = draw_random_sample(pool_r3.abs(), len(r3v), seedy + 1)
            bl_avg_abs_r3 = sample_r3.mean()
        stats[t] = {
            "n_r1": len(r1v),
            "n_r3": len(r3v),
            "avg_abs_r1": float(np.abs(r1v).mean()),
            "med_abs_r1": float(np.median(np.abs(r1v))),
            "avg_sgn_r1": float(r1v.mean()),
            "std_r1": float(r1v.std(ddof=1)) if len(r1v) > 1 else 0.0,
            "avg_abs_r3": float(np.abs(r3v).mean()) if len(r3v) else None,
            "med_abs_r3": float(np.median(np.abs(r3v))) if len(r3v) else None,
            "avg_sgn_r3": float(r3v.mean()) if len(r3v) else None,
            "bl_avg_abs_r1": float(bl_avg_abs_r1),
            "bl_avg_abs_r3": bl_avg_abs_r3,
            "verdict_r1": verdict(float(np.abs(r1v).mean()), float(base_abs_mu_r1),
                                  float(base_abs_sd_r1), len(r1v)),
            "verdict_r3": (verdict(float(np.abs(r3v).mean()), float(base_abs_mu_r3),
                                   float(base_abs_sd_r3), len(r3v))
                           if len(r3v) else "too few events"),
        }

    # --- comparison table --------------------------------------------------
    print(f"\n[3] Event-type impact vs random-day baseline (threshold: baseline avg + "
          f"{THRESHOLD_SD:.0f} SD)")
    hdr = (f"\n  {'event type':<18}{'n':>4}  {'avg |r1|':>9}{'mdn |r1|':>9}"
           f"{'avg r1':>9}  {'control |r1|':>13}{'diff':>8}  verdict (next-day)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t in types:
        s = stats[t]
        if s is None:
            print(f"  {t:<18}{'--':>4}  (no measurable next-day return)")
            continue
        diff = s["avg_abs_r1"] - s["bl_avg_abs_r1"]
        print(f"  {t:<18}{s['n_r1']:>4}  {s['avg_abs_r1'] * 100:>8.2f}%"
              f"{s['med_abs_r1'] * 100:>8.2f}%"
              f"{s['avg_sgn_r1'] * 100:>+8.2f}%  {s['bl_avg_abs_r1'] * 100:>12.2f}%"
              f"{diff * 100:>+8.2f}%  {s['verdict_r1']}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"\n  {'event type':<18}{'n':>4}  {'avg |r3|':>9}{'mdn |r3|':>9}"
          f"{'avg r3':>9}  {'control |r3|':>13}{'diff':>8}  verdict (3-day)")
    print("  " + "-" * (len(hdr) - 2))
    for t in types:
        s = stats[t]
        if s is None or s["avg_abs_r3"] is None:
            print(f"  {t:<18}{'--':>4}  (no measurable 3-day return)")
            continue
        diff = s["avg_abs_r3"] - s["bl_avg_abs_r3"]
        print(f"  {t:<18}{s['n_r3']:>4}  {s['avg_abs_r3'] * 100:>8.2f}%"
              f"{s['med_abs_r3'] * 100:>8.2f}%"
              f"{s['avg_sgn_r3'] * 100:>+8.2f}%  {s['bl_avg_abs_r3'] * 100:>12.2f}%"
              f"{diff * 100:>+8.2f}%  {s['verdict_r3']}")
    print("  " + "-" * (len(hdr) - 2))
    if any((stats[t] and stats[t]["verdict_r1"] == "too few events")
           for t in types):
        print("\n  note: 'too few events' usually means the headlines/stories are from the\n"
              "        last day or two and simply have no price history yet to measure\n"
              "        against.  Re-run this script after a few trading sessions and the\n"
              "        news category will fill in with measurable reactions.")
    print("\n  Legend: r1 = return from event-day close to next trading-day close.\n"
          "          r3 = return over the following 3 trading days. A positive 'avg r'\n"
          "          means the stock tended to RISE after that event type.\n"
          "          'control' = same-size random sample of ordinary days (its |r| metric -\n"
          "          mean, median, sd - is what each event type is measured against).\n"
          "          'mdn |r|' is the median absolute return (the 'middle' move); the median\n"
          "          is less inflated by a few extreme days than the mean.")

    # --- manual verification ------------------------------------------------
    print("\n[4] Manual verification against raw close prices")
    ver = [m for m in mie if m["r1"] is not None]
    ev_earn = next((m for m in ver if m["type"] == "Earnings"), None)
    ev_macro = next((m for m in ver if not (m["type"] == "Earnings" or m["type"] == "News")), None)
    ev_news = next((m for m in ver if m["type"] == "News"), None)
    verify_blocks: list[tuple[str, list[str]]] = []
    if ev_earn:
        verify_blocks.append(("earnings event",
                              verify_event(ev_earn, close, trading_dates, r1, r3, "earnings")))
    if ev_macro:
        label = f"macro {ev_macro['type']} event"
        verify_blocks.append((label, verify_event(ev_macro, close, trading_dates,
                                                  r1, r3, f"macro {ev_macro['type']}")))
    if ev_news and ev_news is not ev_macro:
        verify_blocks.append(("news headline event",
                              verify_event(ev_news, close, trading_dates, r1, r3, "news")))
    if not ev_earn and not ev_macro:
        print("  (no events with a measurable reaction to trace)")

    # --- outputs ------------------------------------------------------------
    summary = plain_language_summary(ticker, types, stats,
                                     base_abs_mu_r1, base_abs_med_r1)
    noticed = [m for m in measured if m["base_date"] is None]
    data_notes = [
        ("Prices: Yahoo Finance daily closes over the 2-year window, "
         "auto_adjust=True (split/dividend-adjusted); any missing day renders "
         "as a gap in the chart line."),
        ("Events: Yahoo earnings dates, Yahoo Finance RSS headlines, and FRED "
         "macro releases (FEDFUNDS, CPIAUCSL, PAYEMS)."),
        (f"Control pool: {len(pool_r1)} ordinary trading days (each event's day, "
         f"the 2 days before it, and the 3 days after it are excluded); next-day "
         f"avg |move| {base_abs_mu_r1 * 100:.2f}% (median {base_abs_med_r1 * 100:.2f}%, "
         f"sd {base_abs_sd_r1 * 100:.2f}%), 3-day avg |move| "
         f"{base_abs_mu_r3 * 100:.2f}% (median {base_abs_med_r3 * 100:.2f}%)."),
        realized_note,
        ("Legend: r1 = return from event-day close to next trading-day close; "
         "r3 = return over the following 3 trading days; 'control' = same-size "
         "random sample of ordinary days; 'mdn |r|' = median absolute return "
         "(the 'middle' move, less inflated by extreme days than the mean)."),
    ]
    if any(stats[t] and stats[t]["verdict_r1"] == "too few events"
           for t in types):
        data_notes.append(
            "'Too few events' usually means the headlines/stories are from the "
            "last day or two and simply have no price history yet to measure "
            "against; re-run after a few trading sessions and that category "
            "fills in.")
    if noticed:
        data_notes.append(
            f"{len(noticed)} event(s) fell outside the measurable price window "
            "(before it, in the future, or without +3 trading days of history "
            "yet, e.g. today's headlines); they are listed with blank returns "
            "in the event list below so nothing is silently dropped.")

    generated_at = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    if json_mode:
        result = json_report(ticker, trading_dates, measured, close, stats,
                             types, generated_at, summary, verify_blocks,
                             data_notes, pool_r1, base_abs_mu_r1,
                             base_abs_sd_r1, base_abs_med_r1, base_abs_mu_r3,
                             base_abs_med_r3)
        sys.stdout = real_stdout
        print(json.dumps(result, ensure_ascii=False))
        return

    report_path = f"{ticker.lower()}_report.html"
    csv_path = f"{ticker.lower()}_events.csv"
    data_notes.append(f"Full per-event log with every return: {csv_path} "
                      "(in the same folder, for spreadsheet use).")

    write_csv(measured, csv_path)
    write_report_html(ticker, trading_dates, measured, close, stats, types,
                      generated_at, summary, verify_blocks, data_notes,
                      report_path)
    print("\n[5] Outputs written:")
    print(f"      event log : {csv_path}  ({len(measured)} events)")
    print(f"      report    : {report_path}  (open in a browser - everything is in here)")
    if noticed:
        print(f"      note: {len(noticed)} event(s) fell outside the measured price window\n"
              f"            (before the window, in the future, or without +3 trading days of\n"
              f"            history yet, e.g. today's headlines); they are in the CSV and\n"
              f"            data-notes section with blank returns.")

    print_plain_language(ticker, summary)


# sector name -> broad-market ETF / fund (any word in a name resolves to it)
SECTOR_ETFS: dict[str, str] = {
    "energy": "XLE", "materials": "XLB", "industrials": "XLI",
    "consumer discretionary": "XLY", "consumer staples": "XLP",
    "health care": "XLV", "financials": "XLF", "technology": "XLK",
    "communication services": "XLC", "utilities": "XLU",
    "real estate": "XLRE", "gold": "GLD", "silver": "SLV",
}
SECTOR_ALIASES: dict[str, str] = {  # common one-word aliases
    "xle": "energy", "xlb": "materials", "xli": "industrials",
    "xly": "consumer discretionary", "xlp": "consumer staples",
    "xlv": "health care", "xlf": "financials", "xlk": "technology",
    "xlc": "communication services", "xlu": "utilities",
    "xlre": "real estate", "gld": "gold", "slv": "silver",
    "tech": "technology", "energy": "energy", "gold": "gold",
    "silver": "silver", "healthcare": "health care",
    "consumer": "consumer discretionary", "comm": "communication services",
    "reits": "real estate",
}

# commodity / metal funds and continuous futures you can analyze
COMMODITIES: list[tuple[str, str, str]] = [
    ("gold", "GLD", "SPDR Gold Trust (ETF)"),
    ("silver", "SLV", "iShares Silver Trust (ETF)"),
    ("platinum", "PPLT", "abrdn Physical Platinum Shares (ETF)"),
    ("copper", "CPER", "U.S. Copper Index Fund (ETF)"),
    ("palladium", "PALL", "abrdn Physical Palladium Shares (ETF)"),
    ("crude oil", "USO", "United States Oil Fund (ETF)"),
    ("natural gas", "UNG", "United States Natural Gas Fund (ETF)"),
    ("gold futures", "GC=F", "COMEX gold, continuous contract"),
    ("silver futures", "SI=F", "COMEX silver, continuous contract"),
    ("copper futures", "HG=F", "COMEX copper, continuous contract"),
    ("crude futures", "CL=F", "NYMEX WTI crude, continuous contract"),
]


def show_catalog() -> None:
    """Non-interactive listing of everything you can analyze."""
    width = 74
    print("=" * width)
    print("  What you can analyze (python stock_movers.py <thing>)")
    print("=" * width)
    print("\n  SECTORS  (11 Select Sector SPDRs + the whole market)")
    for name, sym in SECTOR_ETFS.items():
        if sym.startswith("XL"):
            print(f"    {name:<26} {sym:<6}   python stock_movers.py {name}")
    print(f"    {'whole market':<26} SPY     python stock_movers.py market")
    print("\n  COMMODITIES & METALS")
    for name, sym, desc in COMMODITIES:
        print(f"    {name:<26} {sym:<6}   {desc}")
    print("\n  STOCKS  -- any ticker works:")
    print("    python stock_movers.py MU   AAPL   TSLA   NVDA   ...")
    print("\n  GROUPS  -- analyze several at once:")
    print("    python stock_movers.py technology gold MU")
    print("    python stock_movers.py SECTORS      (all 11 sectors + SPY)")
    print("    python stock_movers.py metals       (all commodity funds)")


def resolve_symbols(args: list[str]) -> list[str]:
    """Turn raw CLI args into a list of ticker symbols.  Accepts tickers by
    themselves, sector names (mapped to their broad-market ETFs), the
    SECTORS/ALL keyword (all 11 Select Sector SPDRs), METALS/COMMODITIES
    (all commodity funds above), and SPY/MARKET."""
    out: list[str] = []
    for raw in args:
        arg = " ".join(raw.lower().split())
        if arg in ("sectors", "all", "every"):
            out = [v for v in SECTOR_ETFS.values() if v.startswith("XL")] + ["SPY"]
            continue
        if arg in ("metals", "commodities", "commodity"):
            out = [sym for _, sym, _ in COMMODITIES]
            continue
        if arg in ("market", "spy", "whole market"):
            out.append("SPY")
            continue
        if arg in SECTOR_ALIASES:
            out.append(SECTOR_ETFS[SECTOR_ALIASES[arg]])
            continue
        if arg in {k for k in SECTOR_ETFS if len(k) > 1}:
            out.append(SECTOR_ETFS[arg])
            continue
        match = [sym for name, sym, _ in COMMODITIES if name == arg]
        if match:
            out.append(match[0])
            continue
        out.append(raw.upper())
    return out


def interactive_menu() -> None:
    """No arguments given: show the catalog, then keep asking what to run."""
    print("\nStock/economics event mover analyzer")
    print("type 'list' to see everything, a sector/commodity name or ticker to\n"
          "analyze it, 'sectors'/'metals' for the full sweep, or 'quit' to exit.\n")
    while True:
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not choice:
            continue
        lowered = " ".join(choice.lower().split())
        if lowered in ("quit", "exit", "q"):
            return
        if lowered in ("list", "menu", "catalog", "help", "-h", "--help", "?"):
            show_catalog()
            continue
        resolved = resolve_symbols([choice])
        print(f"\nRunning: {', '.join(resolved)}")
        for sym in resolved:
            try:
                run(sym)
            except Exception as exc:  # keep the menu alive on a bad ticker
                print(f"  [error] {sym}: {exc}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--json":
        if len(args) < 2:
            print("usage: python stock_movers.py --json TICKER [TICKER ...]\n"
                  "       (prints one JSON report object per line to stdout,\n"
                  "        writes nothing to disk)", file=sys.stderr)
            sys.exit(1)
        for sym in resolve_symbols(args[1:]):
            run(sym, json_mode=True)
        return
    if len(sys.argv) < 2:
        interactive_menu()
        return
    first = " ".join(sys.argv[1:]).lower()
    if first in ("list", "menu", "catalog", "help", "-h", "--help"):
        show_catalog()
        return
    for sym in resolve_symbols(sys.argv[1:]):
        run(sym)


if __name__ == "__main__":
    main()