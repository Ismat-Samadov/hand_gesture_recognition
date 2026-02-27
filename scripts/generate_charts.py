import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_CSV = Path(__file__).parent.parent / "data" / "data.csv"
CHARTS_DIR = Path(__file__).parent.parent / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
PRIMARY   = "#1B4F72"
ACCENT    = "#2E86C1"
HIGHLIGHT = "#F39C12"
MUTED     = "#AEB6BF"
BG        = "#FAFAFA"

TOP_MAKES = ["BMW", "Toyota", "Kia", "Hyundai", "Mercedes-Benz", "Subaru", "Volvo"]

# ── Load data ────────────────────────────────────────────────────────────────
def load():
    rows = []
    with open(DATA_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["_price"]   = int(r["price_gel"])   if r["price_gel"].isdigit()   else None
            r["_mileage"] = int(r["mileage_km"])  if r["mileage_km"].isdigit()  else None
            r["_make"]    = r["title"].split()[0] if r["title"] else ""
            r["_year"]    = int(r["year"])         if r["year"].isdigit()        else None
            # Normalize fuel type
            eng = r["engine"]
            if "hibrid"  in eng: r["_fuel"] = "Hybrid"
            elif "dizel" in eng: r["_fuel"] = "Diesel"
            elif "benzin" in eng: r["_fuel"] = "Petrol"
            else: r["_fuel"] = "Other"
            rows.append(r)
    return rows

# ── Helpers ──────────────────────────────────────────────────────────────────
def save(fig, name):
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved {name}")

def styled_fig(w=12, h=6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555")
    ax.yaxis.label.set_color("#555555")
    ax.xaxis.label.set_color("#555555")
    ax.title.set_color("#1A1A1A")
    return fig, ax

def fmt_gel(x, _=None):
    return f"GEL {x:,.0f}"

# ── Chart 1 — Market Share by Brand (listing volume) ─────────────────────────
def chart_brand_volume(rows):
    from collections import Counter
    counts = Counter(r["_make"] for r in rows if r["_make"])
    labels, vals = zip(*counts.most_common(10))
    colors = [HIGHLIGHT if l in ("BMW", "Toyota") else ACCENT for l in labels]

    fig, ax = styled_fig(12, 6)
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + 60, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=10, color="#333333")
    ax.set_xlabel("Number of Listings", fontsize=11)
    ax.set_title("Market Supply by Brand — Top 10 Most Listed Brands", fontsize=14, fontweight="bold", pad=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlim(0, max(vals) * 1.12)
    fig.text(0.01, 0.01, f"Total listings: {len(rows):,}", fontsize=9, color=MUTED)
    save(fig, "01_brand_volume.png")

# ── Chart 2 — Average Asking Price by Brand ───────────────────────────────────
def chart_avg_price_brand(rows):
    make_prices = defaultdict(list)
    for r in rows:
        if r["_make"] in TOP_MAKES and r["_price"]:
            make_prices[r["_make"]].append(r["_price"])

    labels = sorted(TOP_MAKES, key=lambda m: -np.mean(make_prices[m]))
    avgs   = [int(np.mean(make_prices[m])) for m in labels]
    colors = [HIGHLIGHT if a == max(avgs) else ACCENT for a in avgs]

    fig, ax = styled_fig(12, 6)
    bars = ax.bar(labels, avgs, color=colors, width=0.55)
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 400,
                fmt_gel(v), ha="center", fontsize=10, color="#333333")
    ax.set_ylabel("Average Price (GEL)", fontsize=11)
    ax.set_title("Average Asking Price by Brand", fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_gel))
    ax.set_ylim(0, max(avgs) * 1.15)
    save(fig, "02_avg_price_brand.png")

# ── Chart 3 — Inventory Volume by Manufacture Year ───────────────────────────
def chart_listings_by_year(rows):
    from collections import Counter
    counts = Counter(r["_year"] for r in rows if r["_year"])
    years  = sorted(counts.keys())
    vals   = [counts[y] for y in years]
    colors = [HIGHLIGHT if v == max(vals) else ACCENT for v in vals]

    fig, ax = styled_fig(14, 6)
    bars = ax.bar([str(y) for y in years], vals, color=colors, width=0.7)
    for bar, v in zip(bars, vals):
        if v > 100:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 30,
                    f"{v:,}", ha="center", fontsize=8, color="#333333")
    ax.set_xlabel("Manufacture Year", fontsize=11)
    ax.set_ylabel("Number of Listings", fontsize=11)
    ax.set_title("Inventory Volume by Manufacture Year", fontsize=14, fontweight="bold", pad=14)
    plt.xticks(rotation=45, ha="right")
    save(fig, "03_listings_by_year.png")

# ── Chart 4 — Price Depreciation Curve (avg price by year) ───────────────────
def chart_price_by_year(rows):
    year_prices = defaultdict(list)
    for r in rows:
        if r["_year"] and r["_price"]:
            year_prices[r["_year"]].append(r["_price"])

    # Only years with meaningful sample size
    years = sorted(y for y, p in year_prices.items() if len(p) >= 50)
    avgs  = [int(np.mean(year_prices[y])) for y in years]
    counts = [len(year_prices[y]) for y in years]

    fig, ax = styled_fig(14, 6)
    ax2 = ax.twinx()
    ax2.set_facecolor(BG)

    bars = ax2.bar([str(y) for y in years], counts, color=MUTED, alpha=0.45, width=0.7, label="# Listings")
    ax.plot([str(y) for y in years], avgs, color=HIGHLIGHT, linewidth=2.5,
            marker="o", markersize=7, zorder=5, label="Avg Price")

    ax.set_ylabel("Average Asking Price (GEL)", fontsize=11, color=HIGHLIGHT)
    ax2.set_ylabel("Number of Listings", fontsize=11, color=MUTED)
    ax.set_title("Price vs. Manufacture Year — Depreciation & Supply Overlap", fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_gel))
    ax.tick_params(axis="y", colors=HIGHLIGHT)
    ax2.tick_params(axis="y", colors=MUTED)
    ax2.spines["top"].set_visible(False)
    plt.xticks(rotation=45, ha="right")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    save(fig, "04_price_by_year.png")

# ── Chart 5 — Inventory by City ───────────────────────────────────────────────
def chart_city_inventory(rows):
    from collections import Counter
    counts = Counter(r["city"] for r in rows if r["city"])
    labels, vals = zip(*counts.most_common(5))
    pct = [v / sum(vals) * 100 for v in vals]
    colors = [HIGHLIGHT if i == 0 else ACCENT for i in range(len(labels))]

    fig, ax = styled_fig(10, 5)
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for bar, v, p in zip(bars, vals, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 50,
                f"{v:,}\n({p:.0f}%)", ha="center", fontsize=10, color="#333333")
    ax.set_ylabel("Number of Listings", fontsize=11)
    ax.set_title("Geographic Inventory Distribution by City", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(0, max(vals) * 1.18)
    save(fig, "05_city_inventory.png")

# ── Chart 6 — Body Type Distribution ─────────────────────────────────────────
def chart_body_type(rows):
    from collections import Counter
    # Translate Azerbaijani body types
    body_map = {
        "sedan":                        "Sedan",
        "xizəkli-tırtıllı avtomobil":  "SUV / Crossover",
        "minivan":                      "Minivan",
        "heçbek":                       "Hatchback",
    }
    counts = Counter(body_map.get(r["body_type"], r["body_type"])
                     for r in rows if r["body_type"])
    labels, vals = zip(*counts.most_common())
    colors = [HIGHLIGHT if i == 0 else (ACCENT if i == 1 else MUTED)
              for i in range(len(labels))]

    fig, ax = styled_fig(10, 5)
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 40,
                f"{v:,}", ha="center", fontsize=11, color="#333333")
    ax.set_ylabel("Number of Listings", fontsize=11)
    ax.set_title("Inventory by Vehicle Body Type", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(0, max(vals) * 1.14)
    save(fig, "06_body_type.png")

# ── Chart 7 — Mileage Bracket Analysis ───────────────────────────────────────
def chart_mileage_brackets(rows):
    brackets = [
        ("0–30k km",   0,      30000),
        ("30–60k km",  30000,  60000),
        ("60–100k km", 60000,  100000),
        ("100–150k km",100000, 150000),
        ("150–200k km",150000, 200000),
        ("200k+ km",   200000, 10**9),
    ]
    counts = []
    avg_prices = []
    for label, lo, hi in brackets:
        subset = [r for r in rows if r["_mileage"] and lo <= r["_mileage"] < hi]
        counts.append(len(subset))
        priced = [r["_price"] for r in subset if r["_price"]]
        avg_prices.append(int(np.mean(priced)) if priced else 0)

    labels = [b[0] for b in brackets]
    colors = [HIGHLIGHT if c == max(counts) else ACCENT for c in counts]

    fig, ax = styled_fig(13, 6)
    ax2 = ax.twinx()
    ax2.set_facecolor(BG)

    bars = ax.bar(labels, counts, color=colors, alpha=0.85, width=0.55, label="# Listings")
    ax2.plot(labels, avg_prices, color=PRIMARY, linewidth=2.5,
             marker="s", markersize=7, zorder=5, label="Avg Price")

    ax.set_ylabel("Number of Listings", fontsize=11)
    ax2.set_ylabel("Average Asking Price (GEL)", fontsize=11, color=PRIMARY)
    ax.set_title("Inventory & Pricing by Mileage Bracket", fontsize=14, fontweight="bold", pad=14)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_gel))
    ax2.tick_params(axis="y", colors=PRIMARY)
    ax2.spines["top"].set_visible(False)

    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 30,
                f"{v:,}", ha="center", fontsize=9, color="#333333")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)
    save(fig, "07_mileage_brackets.png")

# ── Chart 8 — Brand Price Range (min / avg / max) ─────────────────────────────
def chart_brand_price_range(rows):
    make_prices = defaultdict(list)
    for r in rows:
        if r["_make"] in TOP_MAKES and r["_price"]:
            make_prices[r["_make"]].append(r["_price"])

    makes  = sorted(TOP_MAKES, key=lambda m: -np.mean(make_prices[m]))
    mins   = [min(make_prices[m])          for m in makes]
    avgs   = [int(np.mean(make_prices[m])) for m in makes]
    maxs   = [max(make_prices[m])          for m in makes]

    x = np.arange(len(makes))
    w = 0.25

    fig, ax = styled_fig(13, 6)
    ax.bar(x - w, mins,  width=w, label="Min Price",  color=MUTED,      alpha=0.9)
    ax.bar(x,     avgs,  width=w, label="Avg Price",  color=ACCENT,     alpha=0.9)
    ax.bar(x + w, maxs,  width=w, label="Max Price",  color=HIGHLIGHT,  alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(makes, fontsize=11)
    ax.set_ylabel("Price (GEL)", fontsize=11)
    ax.set_title("Price Range per Brand — Min / Average / Max", fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_gel))
    ax.legend(fontsize=10)
    save(fig, "08_brand_price_range.png")

# ── Chart 9 — Fuel Type by Brand (stacked bar) ───────────────────────────────
def chart_fuel_by_brand(rows):
    fuels = ["Petrol", "Diesel", "Hybrid"]
    data  = {f: [] for f in fuels}

    for make in TOP_MAKES:
        subset = [r for r in rows if r["_make"] == make]
        total  = len(subset) or 1
        for f in fuels:
            data[f].append(sum(1 for r in subset if r["_fuel"] == f) / total * 100)

    x      = np.arange(len(TOP_MAKES))
    bottom = np.zeros(len(TOP_MAKES))
    palette = [ACCENT, PRIMARY, HIGHLIGHT]

    fig, ax = styled_fig(13, 6)
    for i, (fuel, color) in enumerate(zip(fuels, palette)):
        vals = np.array(data[fuel])
        ax.bar(x, vals, bottom=bottom, label=fuel, color=color, width=0.55)
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 6:
                ax.text(x[j], b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(TOP_MAKES, fontsize=11)
    ax.set_ylabel("Share of Listings (%)", fontsize=11)
    ax.set_title("Fuel Type Mix by Brand (% of Listings)", fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=10)
    save(fig, "09_fuel_by_brand.png")

# ── Chart 10 — Customs Clearance Status by Brand ─────────────────────────────
def chart_customs_by_brand(rows):
    data_cleared     = []
    data_not_cleared = []

    for make in TOP_MAKES:
        subset = [r for r in rows if r["_make"] == make and r["customs_cleared"]]
        total  = len(subset) or 1
        cleared     = sum(1 for r in subset if r["customs_cleared"] == "yes")
        not_cleared = sum(1 for r in subset if r["customs_cleared"] == "no")
        data_cleared.append(cleared / total * 100)
        data_not_cleared.append(not_cleared / total * 100)

    x = np.arange(len(TOP_MAKES))
    fig, ax = styled_fig(13, 6)
    ax.bar(x,     data_not_cleared, width=0.5, label="Not Cleared", color=HIGHLIGHT, alpha=0.9)
    ax.bar(x, data_cleared,         width=0.5, label="Cleared",     color=ACCENT,   alpha=0.9,
           bottom=data_not_cleared)

    ax.set_xticks(x)
    ax.set_xticklabels(TOP_MAKES, fontsize=11)
    ax.set_ylabel("Share of Listings (%)", fontsize=11)
    ax.set_title("Customs Clearance Status by Brand", fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=10)
    save(fig, "10_customs_by_brand.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    print("Loading data...")
    rows = load()
    print(f"  {len(rows):,} records loaded\n")

    print("Generating charts...")
    chart_brand_volume(rows)
    chart_avg_price_brand(rows)
    chart_listings_by_year(rows)
    chart_price_by_year(rows)
    chart_city_inventory(rows)
    chart_body_type(rows)
    chart_mileage_brackets(rows)
    chart_brand_price_range(rows)
    chart_fuel_by_brand(rows)
    chart_customs_by_brand(rows)

    print(f"\nAll charts saved to: {CHARTS_DIR.resolve()}")
