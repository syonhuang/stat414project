import io, os, gc, warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning)


VENUE      = "kalshi"
N_TOP      = 100
FREQ       = "1min"
HORIZON    = 5
NEAR       = 0.05
VOL_WIN    = 15
TEST_FRAC  = 0.30
BID_SIDE   = "BUY"
COLMAP     = {}
SEED       = 0

SOURCES = {
    "kalshi": dict(
        url="https://d389h2c6tmngyk.cloudfront.net/events-2026-10-09T23.parquet",
        auth=("kalshidatabase", "KalshiData2026!")),
}

BLUE, GRAY, ORANGE = "#2f6db5", "#7f8c8d", "#e08a1e"
NAVY, LIGHT = "#10206b", "#8ad3e8"
plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.frameon": False,
})

os.makedirs("figs", exist_ok=True)
cache = f"{VENUE}_events.parquet"
STEP = pd.Timedelta(FREQ)
STEP_MS = int(STEP.total_seconds() * 1000)
rng = np.random.default_rng(SEED)


def show(name):
    for a in plt.gcf().axes:
        a.margins(y=.12)
    plt.tight_layout()
    plt.savefig(f"figs/{VENUE}_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


events = None
if os.path.exists(cache):
    try:
        events = pd.read_parquet(cache)
    except Exception as e:
        print("Cache unreadable, re-downloading:", e)
        os.remove(cache)

if events is None:
    src = SOURCES[VENUE]
    r = requests.get(src["url"], auth=src["auth"], timeout=300)
    r.raise_for_status()
    if not r.content.startswith(b"PAR1"):
        raise ValueError("Download is not a parquet file:\n" + r.text[:300])
    events = pd.read_parquet(io.BytesIO(r.content))
    events.to_parquet(cache + ".tmp")
    os.replace(cache + ".tmp", cache)
    del r; gc.collect()

events = events.rename(columns=COLMAP)
required = {"ts_ms", "asset_id", "side", "price", "size"}
missing_cols = required - set(events.columns)
assert not missing_cols, f"Missing columns {missing_cols}. Set COLMAP to rename them."

side_counts = events.side.value_counts()
assert BID_SIDE in set(side_counts.index), \
    f"BID_SIDE='{BID_SIDE}' is not among the side values {list(side_counts.index)}"
PRICE_DIVISOR = 100.0 if events.price.max() > 1.5 else 1.0


n_rows, n_assets = len(events), events.asset_id.nunique()
t_min = pd.to_datetime(events.ts_ms.min(), unit="ms", utc=True)
t_max = pd.to_datetime(events.ts_ms.max(), unit="ms", utc=True)
all_counts = events.asset_id.value_counts()

minute = events.ts_ms.astype("int64") // 60_000
per_min = minute.value_counts().sort_index()
per_min = per_min.reindex(np.arange(per_min.index.min(), per_min.index.max() + 1), fill_value=0)
per_min.index = pd.to_datetime(per_min.index.values * 60_000, unit="ms", utc=True)
avg_min = per_min.mean()

top_rows = all_counts.head(N_TOP).sum()
print(f"[Overview] window {t_min} -> {t_max}")
print(f"[Overview] {n_rows:,} rows | {n_assets:,} assets | {avg_min:,.0f} rows/min (~{avg_min/60:,.0f}/s)")
print(f"[Overview] mean updates/asset {all_counts.mean():,.0f} | median {all_counts.median():,.0f}")
print(f"[Overview] top {N_TOP} assets: {top_rows:,} rows ({top_rows/n_rows:.1%}); the rest: {1 - top_rows/n_rows:.1%}")


c = np.sort(all_counts.values)
n = len(c)
gini = (2 * np.sum(np.arange(1, n + 1) * c) / (n * c.sum())) - (n + 1) / n
ks = [20, N_TOP]
id_pct = [100 * k / n_assets for k in ks]
upd_pct = [100 * all_counts.head(k).sum() / n_rows for k in ks]
print(f"[Concentration] Gini = {gini:.3f}")
print(f"[Concentration] top 20 = {id_pct[0]:.2f}% of IDs -> {upd_pct[0]:.1f}% of updates | "
      f"top {N_TOP} = {id_pct[1]:.2f}% of IDs -> {upd_pct[1]:.1f}% of updates")
print(f"[Concentration] median asset {int(np.median(c)):,} updates vs busiest {int(c.max()):,} "
      f"({c.max()/np.median(c):,.0f}x)")

fig, ax = plt.subplots(figsize=(6, 4))
x, w = np.arange(len(ks)), .38
b1 = ax.bar(x - w/2, id_pct, w, color=LIGHT, label=f"Share of {n_assets:,} asset IDs")
b2 = ax.bar(x + w/2, upd_pct, w, color=NAVY, label=f"Share of {n_rows:,} updates")
ax.bar_label(b1, fmt="%.2f"); ax.bar_label(b2, fmt="%.1f")
ax.set_xticks(x); ax.set_xticklabels([f"Top {k}" for k in ks])
ax.set_ylabel("%"); ax.set_title("Share of asset IDs vs share of updates (%)"); ax.legend()
show("concentration")


per_5 = per_min.resample("5min").sum()
pk, qt = per_min.idxmax(), per_min.idxmin()
print(f"[Activity] average {avg_min:,.0f}/min | peak {per_min.max():,} at {pk:%H:%M} UTC "
      f"({per_min.max()/avg_min:.2f}x) | quietest {per_min.min():,} at {qt:%H:%M} UTC ({per_min.min()/avg_min:.2f}x)")
print(f"[Activity] 5-min totals {per_5.min()/1e3:.1f}k to {per_5.max()/1e3:.1f}k "
      f"({per_5.min()/(5*avg_min):.2f}x to {per_5.max()/(5*avg_min):.2f}x the per-minute average x5)")

cols6 = [NAVY if v == per_5.max() else LIGHT if v == per_5.min() else BLUE for v in per_5.values]
fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.bar(range(1, len(per_5) + 1), per_5.values / 1e3, color=cols6)
ax.bar_label(bars, fmt="%.2f", fontsize=8)
ax.set_xticks(range(1, len(per_5) + 1)); ax.set_xlabel("5-minute bucket")
ax.set_title("Updates per 5-minute bucket, thousands")
show("activity")


px = np.round(events.price.astype("float64").values / PRICE_DIVISOR, 4)
pb = np.clip(np.floor(px * 10 + 1e-9).astype(int), 0, 9)
p_counts = np.bincount(pb, minlength=10)
q25, q50, q75 = np.percentile(px, [25, 50, 75])
print(f"[Prices] mean {px.mean():.3f} | median {q50:.2f} | 25th-75th pct {q25:.2f}-{q75:.2f}")
print(f"[Prices] below 0.10: {p_counts[0]/1e3:.0f}k rows ({p_counts[0]/n_rows:.1%}) | "
      f"<0.05 or >0.95: {((px < .05) | (px > .95)).mean():.1%} | total {p_counts.sum()/1e3:.1f}k")

cols7 = [NAVY if i in (0, 9) else BLUE for i in range(10)]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(10), p_counts / 1e3, color=cols7)
ax.bar_label(bars, fmt="%.0f", fontsize=8)
ax.set_xticks(range(10)); ax.set_xticklabels([f"{i/10:.1f}" for i in range(10)])
ax.set_xlabel("price bucket (start of 0.1-wide bucket)")
ax.set_title("Updates by price bucket (thousands of rows)")
show("prices")


sz = events["size"].astype("float64").values
edges = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]
sb = np.where(sz == 0, 0, np.where(sz < 1, 1, 2 + np.searchsorted(edges, sz, side="right") - 1))
s_counts = np.bincount(sb, minlength=9)
s_labels = ["0", "<1", "1", "10", "100", "1k", "10k", "100k", "1M"]
print(f"[Sizes] median {np.median(sz):,.0f} | mean {sz.mean():,.0f} | p99 {np.percentile(sz, 99):,.0f} | max {sz.max():,.0f}")
print(f"[Sizes] size-0 rows {s_counts[0]/1e3:.0f}k | 10-99: {s_counts[3]/1e3:.0f}k | 100-999: {s_counts[4]/1e3:.0f}k")

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(9), s_counts / 1e3, color=[ORANGE] + [BLUE] * 8)
ax.bar_label(bars, fmt="%.0f", fontsize=8)
ax.set_xticks(range(9)); ax.set_xticklabels(s_labels)
ax.set_xlabel("size bin (bin start, shares)")
ax.set_title("Updates by size (thousands of rows)")
show("sizes")


comp = pd.crosstab(events.side, np.where(sz == 0, "removal", "set")).reindex(columns=["set", "removal"], fill_value=0)
comp_pct = comp / comp.values.sum() * 100
row_share = comp.sum(axis=1) / comp.values.sum()
zero_by_side = comp["removal"] / comp.sum(axis=1)
print("[Sides] share of rows by side:", {s: f"{v:.1%}" for s, v in row_share.items()})
print("[Sides] zero-size share within side:", {s: f"{v:.1%}" for s, v in zero_by_side.items()},
      f"| overall {(sz == 0).mean():.1%}")

fig, ax = plt.subplots(figsize=(6, 4.2))
bottom = np.zeros(len(comp_pct))
for kind, colr, lbl in [("set", BLUE, "Size > 0 (quote set or changed)"), ("removal", ORANGE, "Size = 0 (level removed)")]:
    vals = comp_pct[kind].values
    ax.bar(range(len(vals)), vals, bottom=bottom, color=colr, label=lbl)
    for i, v in enumerate(vals):
        ax.text(i, bottom[i] + v / 2, f"{v:.1f}", ha="center", va="center", color="white")
    bottom += vals
ax.set_xticks(range(len(comp_pct)))
ax.set_xticklabels([f"{s} ({row_share[s]:.1%} of rows)" for s in comp_pct.index])
ax.set_ylabel("Share of all rows (%)"); ax.set_title("Share of rows by side (%)")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=1)
show("sides")


top = all_counts.head(N_TOP).index
events = events.loc[events.asset_id.isin(top), ["ts_ms", "asset_id", "side", "price", "size"]].copy()
events["asset_id"] = events.asset_id.astype(str).astype("category")
events["side"] = events.side.astype("category")
events["price"] = (events.price.astype("float64") / PRICE_DIVISOR).round(4)
events["size"] = events["size"].astype("float64")
del px, pb, sz, sb, minute
gc.collect()


def book_series(d):
    d = d.sort_values("ts_ms", kind="stable")
    ts = d.ts_ms.values
    is_bid = (d.side == BID_SIDE).values
    pr, sz_ = d.price.values, d["size"].values
    bucket = ts // STEP_MS
    n_ = len(ts)
    bids, asks, rows = {}, {}, []
    nan = np.nan
    for i in range(n_):
        book = bids if is_bid[i] else asks
        p, s = pr[i], sz_[i]
        if s == 0:
            book.pop(p, None)
        else:
            book[p] = s
        if bids and asks:
            bb, ba = max(bids), min(asks)
            if bb < ba:
                if i == n_ - 1 or bucket[i + 1] != bucket[i]:
                    rows.append((ts[i], bb, ba, bids[bb], asks[ba],
                                 sum(bids.values()), sum(asks.values()),
                                 sum(v for q, v in bids.items() if q >= bb - NEAR),
                                 sum(v for q, v in asks.items() if q <= ba + NEAR)))
                else:
                    rows.append((ts[i], bb, ba, bids[bb], asks[ba], nan, nan, nan, nan))
    out = pd.DataFrame(rows, columns=["ts", "bid", "ask", "bid_sz", "ask_sz",
                                      "bid_depth", "ask_depth", "near_bid", "near_ask"])
    out["ts"] = pd.to_datetime(out.ts, unit="ms", utc=True)
    out[["bid", "ask"]] = out[["bid", "ask"]].astype("float64").round(4)
    return out.set_index("ts")


books, coverage = {}, {}
groups = list(events.groupby("asset_id", observed=True))
for i, (a, d) in enumerate(groups):
    b = book_series(d)
    if len(b) > 20:
        books[a] = b
        coverage[a] = len(b) / len(d)
    if (i + 1) % 25 == 0:
        print(f"  rebuilt {i+1}/{len(groups)} books")
cov = pd.Series(coverage)
print(f"[Order book] {len(books)} of {len(groups)} assets have > 20 valid book states")
print(f"[Order book] median share of updates kept as valid states: {cov.median():.1%} (lowest {cov.min():.1%})")


def make_features(b):
    g = b.resample(FREQ).last()
    cnt = b.bid.resample(FREQ).count()
    g = g.ffill()
    g["n_upd"] = cnt.reindex(g.index).fillna(0)
    g = g.dropna()
    g["mid"] = ((g.bid + g.ask) / 2).round(5)
    g["spread"] = (g.ask - g.bid).round(5)
    g["imb"] = (g.bid_sz - g.ask_sz) / (g.bid_sz + g.ask_sz)
    g["depth_imb"] = (g.bid_depth - g.ask_depth) / (g.bid_depth + g.ask_depth)
    g["near_imb"] = (g.near_bid - g.near_ask) / (g.near_bid + g.near_ask)
    g["ret_1"] = g.mid.diff().round(5)
    g["ret_5"] = g.mid.diff(5).round(5)
    g["vol"] = g.ret_1.rolling(VOL_WIN).std()
    g["fwd"] = (g.mid.shift(-HORIZON) - g.mid).round(5)
    g["spread_fwd"] = g.spread.shift(-HORIZON)
    g["up"] = (g.fwd > 0).astype(int)
    g["moved"] = g.fwd != 0
    return g.dropna()


feat = pd.concat([make_features(b).assign(asset=a) for a, b in books.items()])
print(f"[Order book] {feat.asset.nunique()} assets with usable features")
print(f"[Grid] {feat.moved.sum():,} of {len(feat):,} grid rows moved ({feat.moved.mean():.0%})")

del events, groups, books
gc.collect()


X_cols = ["imb", "near_imb", "depth_imb", "ret_1", "ret_5", "spread", "vol", "n_upd"]
t0, t1 = feat.index.min(), feat.index.max()
cutoff = t0 + (1 - TEST_FRAC) * (t1 - t0)
train = feat[(feat.index <= cutoff - HORIZON * STEP) & feat.moved]
test_all = feat[feat.index > cutoff]
test = test_all[test_all.moved]
assert len(train) > 50 and len(test) > 50, f"Too few moved rows (train {len(train)}, test {len(test)})."
print(f"[Split] train (moved) {len(train):,} | test (moved) {len(test):,} | test (all rows) {len(test_all):,}")

fig, ax = plt.subplots(figsize=(5.5, 4))
bars = ax.bar(["Train: moved", "Test: moved", "Test: all rows"],
              [len(train), len(test), len(test_all)], color=[BLUE, NAVY, LIGHT])
ax.bar_label(bars, fmt="{:,.0f}".format)
ax.set_ylabel("Number of 1-minute grid rows"); ax.set_title("Rows by sample")
show("rows_by_sample")


logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[X_cols], train.up)
gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=150, min_samples_leaf=50,
                                     l2_regularization=1.0, random_state=SEED).fit(train[X_cols], train.up)
P = {"Logistic": logit.predict_proba(test[X_cols])[:, 1], "Boosting": gbm.predict_proba(test[X_cols])[:, 1]}
base_p = np.full(len(test), train.up.mean())
base_ll = log_loss(test.up, base_p)
base_acc = max(test.up.mean(), 1 - test.up.mean())


def boot_auc(y, p, n_boot=500):
    y = np.asarray(y); out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


res = {}
for k, p in P.items():
    lo, hi = boot_auc(test.up, p)
    ll = log_loss(test.up, p)
    res[k] = dict(AUC=roc_auc_score(test.up, p), AUC_lo=lo, AUC_hi=hi,
                  LogLoss=ll, LogLoss_gain=base_ll - ll, Accuracy=accuracy_score(test.up, p > .5))
results = pd.DataFrame(res).T
print(f"[Model scores] baseline log loss {base_ll:.4f} | always-majority accuracy {base_acc:.1%}")
print(results.round(4))

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
bars = ax[0].bar(["Logistic", "Boosting"], results.AUC, color=[BLUE, NAVY]); ax[0].bar_label(bars, fmt="%.3f")
ax[0].set_title("AUC")
bars = ax[1].bar(["Logistic", "Boosting"], results.LogLoss_gain, color=[BLUE, NAVY]); ax[1].bar_label(bars, fmt="%.4f")
ax[1].set_title(f"Log loss gain over the {base_ll:.3f} baseline")
show("model_scores")


def simulate(df, p):
    s = df.copy()
    s["side"] = np.where(p > .5, 1, -1)
    s["gross"] = s.side * s.fwd
    s["cost"] = (s.spread + s.spread_fwd) / 2
    s["net"] = s.gross - s.cost
    s["k"] = s.groupby("asset").cumcount()
    return s[s.k % HORIZON == 0].sort_index(kind="mergesort")


ptest = {"Logistic": logit.predict_proba(test_all[X_cols])[:, 1], "Boosting": gbm.predict_proba(test_all[X_cols])[:, 1]}
sims = {k: simulate(test_all, p) for k, p in ptest.items()}
econ_tbl = pd.DataFrame({k: dict(trades=len(s), gross_c=s.gross.mean() * 100,
                                 cost_c=s.cost.mean() * 100, net_c=s.net.mean() * 100)
                         for k, s in sims.items()})
print("[Trading test] cents per trade (1c = 1% of a $1 contract)")
print(econ_tbl.round(2))

cats = ["Gross edge", "Spread cost", "Net result"]
fig, ax = plt.subplots(figsize=(7, 4.2))
x, w = np.arange(3), .38
for j, (k, colr) in enumerate([("Logistic", BLUE), ("Boosting", ORANGE)]):
    vals = [econ_tbl.loc["gross_c", k], -econ_tbl.loc["cost_c", k], econ_tbl.loc["net_c", k]]
    bars = ax.bar(x + (j - .5) * w, vals, w, color=colr, label=k)
    ax.bar_label(bars, fmt="%+.2f", fontsize=8)
ax.axhline(0, color="k", lw=.8)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("Cents per trade (1c = 1% of a $1 contract)")
ax.set_title("Per-trade economics"); ax.legend()
show("per_trade_economics")

print("Done. Figures saved in ./figs with prefix", VENUE + "_")
