"""Asset-specific feature assembly:
- Gold: real Treasury yields, DXY, inflation expectations, rate spreads
- Silver: gold-silver ratio, industrial proxy
- Crypto: funding rates, open interest, volume anomalies
All features are point-in-time joined by timestamp (as_of-aware)."""
from __future__ import annotations

import pandas as pd


def macro_features(macro_frame: pd.DataFrame) -> pd.DataFrame:
    """Expects columns: series_id, ts (release), value. Returns pivoted series
    ffill'd — macro releases are sparse by nature."""
    if macro_frame.empty:
        return pd.DataFrame()
    piv = macro_frame.pivot_table(index="ts", columns="series_id", values="value")
    piv = piv.sort_index().ffill()
    out = pd.DataFrame(index=piv.index)
    if "DGS10" in piv and "T10YIE" in piv:
        out["real_yield_10y"] = piv["DGS10"] - piv["T10YIE"]
    if "DGS10" in piv and "DGS2" in piv:
        out["yield_curve_spread"] = piv["DGS10"] - piv["DGS2"]
    if "DTWEXBGS" in piv:
        out["dxy"] = piv["DTWEXBGS"]
    if "CPIAUCSL" in piv:
        out["cpi_yoy"] = piv["CPIAUCSL"].pct_change(12)  # works on monthly series
    if "UNRATE" in piv:
        out["unemployment"] = piv["UNRATE"]
    if "PAYEMS" in piv:
        out["payrolls_change"] = piv["PAYEMS"].diff()
    return out


def metals_features(gold: pd.Series, silver: pd.Series, dxy: pd.Series) -> pd.DataFrame:
    """Gold/silver-specific features aligned on the candle grid."""
    out = pd.DataFrame(index=gold.index)
    out["gold_silver_ratio"] = gold / silver
    out["gold_silver_ratio_z"] = (
        out["gold_silver_ratio"] - out["gold_silver_ratio"].rolling(168).mean()
    ) / out["gold_silver_ratio"].rolling(168).std().replace(0, pd.NA)
    if len(dxy):
        dxy_aligned = dxy.reindex(gold.index, method="ffill")
        out["dxy_change_24"] = dxy_aligned.pct_change(24)
        out["dxy_change_168"] = dxy_aligned.pct_change(168)
    return out


def crypto_features(deriv_frame: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Expects derivative rows: ts, funding_rate, open_interest, asset_symbol.
    Aligned onto the candle index."""
    out = pd.DataFrame(index=candles.index)
    if deriv_frame.empty:
        return out
    fr = deriv_frame.groupby("ts")["funding_rate"].mean().sort_index()
    oi = deriv_frame.groupby("ts")["open_interest"].mean().sort_index()
    if len(fr):
        fr_s = fr.reindex(candles.index, method="ffill")
        out["funding_rate"] = fr_s
        out["funding_rate_cum_24"] = fr_s.rolling(24).sum()
    if len(oi):
        oi_s = oi.reindex(candles.index, method="ffill")
        out["open_interest"] = oi_s
        out["oi_change_24"] = oi_s.pct_change(24)
    return out


def assemble(asset_symbol: str, candles: pd.DataFrame, *,
             macro: pd.DataFrame | None = None,
             other_assets: dict[str, pd.DataFrame] | None = None,
             derivatives: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full point-in-time feature frame for one asset on its candle grid."""
    from features.statistical import statistical_features
    from features.technical import technical_features

    parts = [candles.reset_index(drop=True)]
    parts.append(technical_features(candles))
    parts.append(statistical_features(candles))

    if macro is not None and not macro.empty and "ts" in candles:
        macro_ts = macro.sort_index()
        joined = pd.merge_asof(
            candles[["ts"]], macro_ts,
            left_on="ts", right_index=True, direction="backward",
        ).drop(columns=["ts"])
        joined.index = candles.index
        parts.append(joined)

    if asset_symbol in ("XAUUSD", "XAGUSD") and other_assets:
        gold = other_assets.get("XAUUSD", pd.DataFrame())
        silver = other_assets.get("XAGUSD", pd.DataFrame())
        dxy = other_assets.get("DXY", pd.DataFrame())
        if not gold.empty and not silver.empty:
            g = gold.set_index("ts")["close"]
            s = silver.set_index("ts")["close"]
            aligned = pd.merge_asof(candles[["ts"]], pd.DataFrame({"gold": g, "silver": s}),
                                     left_on="ts", right_index=True, direction="backward")
            aligned.index = candles.index
            d = dxy.set_index("ts")["close"] if not dxy.empty else pd.Series(dtype=float)
            parts.append(metals_features(
                aligned["gold"], aligned["silver"], d.reindex(aligned.index, method="ffill")))

    if asset_symbol in ("BTCUSD", "ETHUSD") and derivatives is not None and not derivatives.empty:
        parts.append(crypto_features(derivatives, candles))

    full = pd.concat(parts, axis=1)
    return full
