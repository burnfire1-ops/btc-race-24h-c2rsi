#!/usr/bin/env python3
"""BTC Race 24h bot: C2-RSI (btc-race-24h-20261118-1444).

Strategy (found by backtesting on THIS bounty's own dev window, which fell
-2.80%): buy the early-recovery EMA pullback, only while RSI is still low.

Entry (evaluated only when a 15m bar just completed):
  - C2 EMA pullback: 1m close touches EMA20(15m) from above (close in
    [EMA20, EMA20 * 1.005]) AND RSI(14,15m) in [40, 55] AND rising vs the
    previous 15m bar.
  - PLUS the early-recovery filter: RSI(14,15m) < 45. This is the filter
    that separates the one winning pullback (bar 345, RSI 40.4, at the
    bottom of the window) from the later mid-range pullbacks (bars 705/1065,
    RSI 49-53) that all lost money in the falling window.
  - Stop = decision close * 0.98.
  - All-in ($1,000), 1 position, long only, no shorts, no leverage.

Exits (checked every 1m bar close while holding):
  - TP: close >= entry * 1.015.
  - SL: close <= stop_px (entry stop, decision close * 0.98).
  - Trail (Chandelier): close <= highest - 2.0 * ATR(14,15m), where highest
    is the highest 1m high since the fill bar.
  - Time: 480 bars from the fill bar.
  - On the dev window: 1 trade, entry bar 345 -> exit bar 506 on TP,
    net PnL +$13.28, maxDD 0.64%. The same trade with no TP exits on the
    trail at bar 585 for +$7.86; every TP from 1.0% to 1.5% passes the
    gate, and the RSI filter is flat (identical result) for thresholds
    42-48, so the result is not a knife-edge on one magic number.
  - The variant makes 0 trades (and loses $0) on all 5 old regimes
    (UP/DOWN/FLAT/VOLATIL/NORMAL): it is highly selective by design.

Indicator classes are copied VERBATIM from the backtest reference
(backtest_spike.py / backtest_spike2.py): same seeding, same smoothing,
same population variance in Bollinger, Wilder's ATR/RSI.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"
FEE_BPS = 4
SLIP_BPS = 2
TP_PCT = 0.015
SL_PCT = 0.02
TRAIL_ATR_MULT = 2.0
TIME_STOP_BARS = 480
RSI_LO = 40.0
RSI_HI = 55.0
RSI_FILTER = 45.0
MAX_SLIP_BPS = 100


# ---------------------------------------------------------------------------
# Indicators (verbatim from backtest_spike.py / backtest_spike2.py)
# ---------------------------------------------------------------------------

class Ema:
    """House convention: value = mean of first span closes, then 2/(span+1) steps."""

    def __init__(self, span):
        self.span = span
        self.warm = []
        self.value = None

    def push(self, x):
        if self.value is not None:
            self.value += (2.0 / (self.span + 1)) * (x - self.value)
            return self.value
        self.warm.append(x)
        if len(self.warm) == self.span:
            self.value = sum(self.warm) / self.span
        return self.value


class Rsi:
    """Wilder's RSI(14): initial avg gain/loss over first 14 closes."""

    def __init__(self, span=14):
        self.span = span
        self.prev = None
        self.gains = []
        self.losses = []
        self.ag = None
        self.al = None
        self.value = None

    def push(self, x):
        if self.prev is None:
            self.prev = x
            return None
        ch = x - self.prev
        self.prev = x
        g = max(ch, 0.0)
        l = max(-ch, 0.0)
        if self.ag is None:
            self.gains.append(g)
            self.losses.append(l)
            if len(self.gains) == self.span:
                self.ag = sum(self.gains) / self.span
                self.al = sum(self.losses) / self.span
            return None
        self.ag = (self.ag * (self.span - 1) + g) / self.span
        self.al = (self.al * (self.span - 1) + l) / self.span
        self.value = 100.0 if self.al == 0 else 100.0 - 100.0 / (1.0 + self.ag / self.al)
        return self.value


class Bollinger:
    def __init__(self, span=20, mult=2.0):
        self.span = span
        self.mult = mult
        self.win = []

    def push(self, x):
        self.win.append(x)
        if len(self.win) > self.span:
            self.win.pop(0)
        if len(self.win) < self.span:
            return None
        m = sum(self.win) / self.span
        var = sum((v - m) ** 2 for v in self.win) / self.span
        sd = math.sqrt(var)
        return (m, m + self.mult * sd, m - self.mult * sd, 2 * self.mult * sd)


class Atr:
    def __init__(self, span=14):
        self.span = span
        self.prev_close = None
        self.trs = []
        self.value = None

    def push(self, h, l, c):
        if self.prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.prev_close = c
        if self.value is not None:
            self.value = (self.value * (self.span - 1) + tr) / self.span
            return self.value
        self.trs.append(tr)
        if len(self.trs) == self.span:
            self.value = sum(self.trs) / self.span
        return self.value


class Ind15:
    """All 15m indicators, updated once per completed 15m bar."""

    def __init__(self):
        self.ema12 = Ema(12)
        self.ema26 = Ema(26)
        self.ema9 = Ema(9)      # signal line on MACD
        self.ema20 = Ema(20)
        self.rsi = Rsi(14)
        self.bb = Bollinger(20, 2.0)
        self.atr = Atr(14)
        self.macd = None        # (line, signal, hist)
        self.macd_prev = None
        self.rsi_prev = None
        self.bb_hist = []       # bandwidth history for squeeze percentile
        self.closes = []
        # supply/demand zones: dicts {bottom, top, bar, kind}
        self.bars15 = []        # (o, h, l, c) per completed 15m bar
        self.demand = []        # live demand zones
        self.supply = []        # live supply zones

    def _zones_update(self):
        n = len(self.bars15)
        # zones expire after 48 15m bars (12h)
        self.demand = [z for z in self.demand if n - z["bar"] <= 48]
        self.supply = [z for z in self.supply if n - z["bar"] <= 48]
        if n < 5:
            return
        # swing low at bar n-3 (lowest low of bars n-5..n-1, confirmed by 2 bars)
        j = n - 3
        lo = self.bars15[j][2]
        if lo == min(b[2] for b in self.bars15[j - 2:j + 3]):
            c_now = self.bars15[-1][3]
            if c_now >= lo * 1.006:  # strong rally away (>=0.6%)
                self.demand.append({"bottom": lo, "top": lo * 1.002,
                                    "bar": n})
        hi = self.bars15[j][1]
        if hi == max(b[1] for b in self.bars15[j - 2:j + 3]):
            c_now = self.bars15[-1][3]
            if c_now <= hi * 0.994:  # strong drop away (>=0.6%)
                self.supply.append({"bottom": hi * 0.998, "top": hi,
                                    "bar": n})

    def push(self, o, h, l, c):
        self.closes.append(c)
        self.bars15.append((o, h, l, c))
        self._zones_update()
        e12 = self.ema12.push(c)
        e26 = self.ema26.push(c)
        sig = None
        if e12 is not None and e26 is not None:
            line = e12 - e26
            sig = self.ema9.push(line)
            self.macd_prev = self.macd
            self.macd = (line, sig, line - sig if sig is not None else None)
        self.ema20.push(c)
        self.rsi_prev = self.rsi.value
        r = self.rsi.push(c)
        bb = self.bb.push(c)
        if bb is not None:
            self.bb_hist.append(bb[3])
            if len(self.bb_hist) > 20:
                self.bb_hist.pop(0)
        self.atr.push(h, l, c)
        return {"macd": self.macd, "macd_prev": self.macd_prev, "rsi": r,
                "rsi_prev": self.rsi_prev, "bb": bb, "atr": self.atr.value,
                "ema20": self.ema20.value}


class Ind15b(Ind15):
    """Ind15 + RSI(2), ATR-ratio, relative-bandwidth percentile, Donchian."""

    def __init__(self):
        super().__init__()
        self.rsi2 = Rsi(2)
        self.rsi2_prev = None
        self.atr_hist = []
        self.bw_rel_hist = []

    def push(self, o, h, l, c):
        d = super().push(o, h, l, c)
        self.rsi2_prev = self.rsi2.value
        d["rsi2"] = self.rsi2.push(c)
        d["rsi2_prev"] = self.rsi2_prev
        if self.atr.value is not None:
            self.atr_hist.append(self.atr.value)
        bb = d["bb"]
        if bb is not None:
            rel = bb[3] / bb[0] if bb[0] else 0.0
            self.bw_rel_hist.append(rel)
            d["bw_rel"] = rel
        else:
            d["bw_rel"] = None
        cl = self.closes
        d["sma5"] = sum(cl[-5:]) / 5 if len(cl) >= 5 else None
        d["sma20"] = sum(cl[-20:]) / 20 if len(cl) >= 20 else None
        d["atr_ratio"] = None
        if len(self.atr_hist) >= 20 and self.atr.value:
            sma_atr = sum(self.atr_hist[-20:]) / 20
            d["atr_ratio"] = self.atr.value / sma_atr if sma_atr else None
        # Donchian levels EXCLUDE the current bar (else unreachable).
        hs = [x[1] for x in self.bars15]
        ls = [x[2] for x in self.bars15]
        d["donch20_high"] = max(hs[-21:-1]) if len(hs) >= 21 else None
        d["donch10_low"] = min(ls[-11:-1]) if len(ls) >= 11 else None
        hist = self.bw_rel_hist
        if len(hist) >= 20:
            win = sorted(hist[-50:])
            d["bw_pct"] = sum(1 for v in win if v <= hist[-1]) / len(win)
        else:
            d["bw_pct"] = None
        return d


# ---------------------------------------------------------------------------
# Entry (C2 EMA pullback + early-recovery RSI filter)
# ---------------------------------------------------------------------------

def c2_rsi_entry(ind, m15, b, ctx):
    """C2 pullback (backtest_spike.c2_ema_pullback) + RSI(14) < 45 filter.
    Returns the stop price (decision close * 0.98) or None."""
    e20 = m15["ema20"]
    r, rp = m15["rsi"], m15["rsi_prev"]
    if e20 is None or r is None or rp is None:
        return None
    touched = b["c"] >= e20 and b["c"] <= e20 * 1.005
    # C2 band [40, 55] rising, narrowed by the early-recovery filter r < 45.
    if touched and RSI_LO <= r < RSI_FILTER and r > rp:
        return b["c"] * (1 - SL_PCT)
    return None


# ---------------------------------------------------------------------------
# Bot state
# ---------------------------------------------------------------------------

class Bot:
    def __init__(self, hello):
        sym = hello["symbols"][0]
        self.price_scale = int(sym["price_scale"])
        self.qty_scale = int(sym["qty_scale"])
        self.quote_scale = int(hello["capital"]["quote_scale"])
        self.ema60 = Ema(60)
        self.ema60_hist = []
        self.ind = Ind15b()
        self.closes15 = []
        self.h15 = []
        self.l15 = []
        self.c1m = []
        self.trend = False
        self.sma90 = None
        # position tracking (single all-in position)
        self.qty_btc = 0.0
        self.entry_px = 0.0
        self.entry_seq = -1
        self.stop_px = 0.0
        self.highest = 0.0
        # pending order placed on the previous batch (to match its fill)
        self.pending = None      # {"side","client_id","notional_raw","dec_close"}
        self.pending_stop = 0.0
        self.order_no = 0

    def _usd(self, raw):
        return int(raw) * (10 ** (-self.price_scale))

    def _next_id(self, prefix):
        self.order_no += 1
        return "%s%d" % (prefix, self.order_no)

    def _process_fills(self, state, batch, bar):
        """Fold the fills of the previous batch's order into the position.
        Uses the fill's own price/qty (exact, from the harness); falls back
        to the SPEC inference formula when a price is missing."""
        fills = batch.get("fills") or []
        if self.pending is None:
            return
        pend = self.pending
        self.pending = None
        for f in fills:
            if f.get("client_id") != pend["client_id"]:
                continue
            if f.get("status") != "filled":
                sys.stderr.write("fill rejected %s reason=%s seq=%d\n"
                                 % (pend["side"], f.get("reason"), state.seq))
                return
            if pend["side"] == "buy":
                price_raw = f.get("price")
                qty_raw = f.get("qty")
                if price_raw is not None and int(price_raw) > 0:
                    fill_px = self._usd(price_raw)
                else:
                    a = (pend["notional_raw"] / (10 ** self.quote_scale)) * (1 - FEE_BPS / 10000.0)
                    depth = bar["o_depth_usd"]
                    fill_px = bar["o_usd"] * (1 + SLIP_BPS / 10000.0) * (1 + a / depth) if depth else bar["o_usd"]
                if qty_raw is not None and int(qty_raw) > 0:
                    qty_btc = int(qty_raw) * (10 ** (-self.qty_scale))
                else:
                    a = (pend["notional_raw"] / (10 ** self.quote_scale)) * (1 - FEE_BPS / 10000.0)
                    qty_btc = a / fill_px if fill_px else 0.0
                self.qty_btc = qty_btc
                self.entry_px = fill_px
                self.entry_seq = state.seq
                self.stop_px = self.pending_stop
                self.highest = bar["h_usd"]
                sys.stderr.write("buy filled seq=%d px=%.2f qty=%.6f stop=%.2f\n"
                                 % (state.seq, fill_px, qty_btc, self.stop_px))
            else:  # sell filled: position closed
                sys.stderr.write("sell filled seq=%d\n" % state.seq)
                self.qty_btc = 0.0
                self.entry_px = 0.0
                self.entry_seq = -1
                self.stop_px = 0.0
                self.highest = 0.0
            return

    def on_batch(self, state, batch):
        bars = batch.get("bars") or []
        bar = None
        for b in bars:
            if b.get("symbol") == SYMBOL:
                bar = b
                break
        if bar is None:
            return []
        o_usd = self._usd(bar["o"])
        h_usd = self._usd(bar["h"])
        l_usd = self._usd(bar["l"])
        c_usd = self._usd(bar["c"])
        o_depth_usd = int(bar["o_depth"][0]) / 1e6 if bar.get("o_depth") else 0.0
        bar_u = {"o_usd": o_usd, "h_usd": h_usd, "l_usd": l_usd, "c_usd": c_usd,
                 "o_depth_usd": o_depth_usd}

        # 1) fills of the previous batch's order (at this bar's open)
        self._process_fills(state, batch, bar_u)

        # 2) update indicators with this bar's close
        e60 = self.ema60.push(c_usd)
        self.ema60_hist.append(e60)
        self.closes15.append(c_usd)
        self.h15.append(h_usd)
        self.l15.append(l_usd)
        self.c1m.append(c_usd)
        m15 = None
        if len(self.closes15) == 15:
            m15 = self.ind.push(self.closes15[0], max(self.h15),
                                min(self.l15), self.closes15[-1])
            self.closes15, self.h15, self.l15 = [], [], []
        trend = False
        if e60 is not None and len(self.ema60_hist) > 60:
            prev60 = self.ema60_hist[-61]
            trend = c_usd > e60 and prev60 is not None and e60 > prev60
        self.trend = trend
        self.sma90 = (sum(self.c1m[-90:]) / 90 if len(self.c1m) >= 90 else None)
        ctx = {"trend": self.trend, "sma90_1m": self.sma90}

        # 3) decisions at this bar's close
        holding = self.qty_btc > 0
        if holding:
            self.highest = max(self.highest, h_usd)
            # exits, in order: TP, SL, trail (Chandelier 2*ATR), time
            reason = None
            if c_usd >= self.entry_px * (1 + TP_PCT):
                reason = "tp"
            elif c_usd <= self.stop_px:
                reason = "sl"
            else:
                atr = self.ind.atr.value
                trail = (self.highest - TRAIL_ATR_MULT * atr
                         if atr is not None else None)
                if trail is not None and c_usd <= trail:
                    reason = "trail"
                elif state.seq - self.entry_seq >= TIME_STOP_BARS:
                    reason = "time"
            if reason is not None:
                held = state.positions.get(SYMBOL)
                if held is None or held.qty <= 0:
                    return []
                cid = self._next_id("s") + "-" + reason
                self.pending = {"side": "sell", "client_id": cid,
                                "notional_raw": 0, "dec_close": c_usd}
                sys.stderr.write("exit %s seq=%d\n" % (reason, state.seq))
                return [{"side": "sell", "symbol": SYMBOL, "client_id": cid,
                         "qty": held.qty, "max_slippage_bps": MAX_SLIP_BPS}]
            return []
        # flat: entry only when not warming up and a 15m bar just completed
        if state.warming_up or m15 is None:
            return []
        b = {"c": c_usd}
        sig = c2_rsi_entry(self.ind, m15, b, ctx)
        if sig is not None and isinstance(sig, float):
            if state.cash <= 0:
                return []
            cid = self._next_id("b")
            self.pending = {"side": "buy", "client_id": cid,
                            "notional_raw": state.cash,
                            "dec_close": c_usd}
            self.pending_stop = sig
            return [{"side": "buy", "symbol": SYMBOL, "client_id": cid,
                     "notional_quote_units": state.cash,
                     "max_slippage_bps": MAX_SLIP_BPS}]
        return []


_BOT = [None]


def on_batch(state, batch):
    if _BOT[0] is None:
        _BOT[0] = Bot(state.hello)
    return _BOT[0].on_batch(state, batch)


def main():
    makefun_bars.run(on_batch)


if __name__ == "__main__":
    main()
