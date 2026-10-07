"""Marketing funnel analysis for the Olist qualified-leads and closed-deals data.

Run this script from any folder. It reads only the two specified source CSVs
inside the project's data/ folder and writes tables, charts, and a short report
to outputs/. Missing required columns or invalid lead IDs produce clear errors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
QUALIFIED_FILE = DATA_DIR / "olist_marketing_qualified_leads_dataset.csv"
CLOSED_FILE = DATA_DIR / "olist_closed_deals_dataset.csv"

REQUIRED_QUALIFIED_COLUMNS = {"mql_id", "first_contact_date", "origin"}
REQUIRED_CLOSED_COLUMNS = {"mql_id", "won_date"}

# Short descriptions for columns in the official Olist datasets.
COLUMN_DESCRIPTIONS = {
    "mql_id": "Unique identifier for a marketing qualified lead; the common key used to connect both files.",
    "first_contact_date": "Date when the qualified lead first contacted Olist.",
    "landing_page_id": "Identifier for the landing page associated with the lead.",
    "origin": "Recorded marketing origin/source for the qualified lead.",
    "seller_id": "Identifier of the seller associated with the closed deal.",
    "sdr_id": "Identifier of the Sales Development Representative associated with the lead.",
    "sr_id": "Identifier of the Sales Representative associated with the lead.",
    "won_date": "Date and time when the lead became a closed deal.",
    "business_segment": "Business category/segment recorded for the closed deal.",
    "lead_type": "Recorded classification/type of the lead.",
    "lead_behaviour_profile": "Recorded lead behavior/profile category.",
    "has_company": "Whether the lead was recorded as having a company.",
    "has_gtin": "Whether the lead was recorded as having GTIN product identifiers.",
    "average_stock": "Recorded average stock information for the lead.",
    "business_type": "Type of business recorded for the closed deal.",
    "declared_product_catalog_size": "Declared size of the lead's product catalog.",
    "declared_monthly_revenue": "Monthly revenue declared by the lead; this is not necessarily realized Olist revenue.",
}


def require_files() -> None:
    """Stop with a useful message if either required source CSV is missing."""
    missing = [path for path in (QUALIFIED_FILE, CLOSED_FILE) if not path.is_file()]
    if missing:
        missing_list = "\n".join(f" - {path}" for path in missing)
        raise FileNotFoundError(
            "Required Olist dataset file(s) not found. Add the original CSVs under data/:\n"
            f"{missing_list}\n"
            "The analysis does not use marketing_funnel.csv or any churn dataset."
        )


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two specified Olist CSV files without modifying their raw values."""
    qualified = pd.read_csv(QUALIFIED_FILE)
    closed = pd.read_csv(CLOSED_FILE)
    return qualified, closed


def validate_required_columns(
    qualified: pd.DataFrame, closed: pd.DataFrame
) -> None:
    """Check required columns before calculating any funnel results."""
    missing_qualified = REQUIRED_QUALIFIED_COLUMNS.difference(qualified.columns)
    missing_closed = REQUIRED_CLOSED_COLUMNS.difference(closed.columns)
    problems = []
    if missing_qualified:
        problems.append(
            "Qualified-leads file is missing: " + ", ".join(sorted(missing_qualified))
        )
    if missing_closed:
        problems.append(
            "Closed-deals file is missing: " + ", ".join(sorted(missing_closed))
        )
    if problems:
        raise ValueError(
            "Cannot calculate the requested funnel metrics:\n - "
            + "\n - ".join(problems)
        )

    if qualified["mql_id"].isna().any() or closed["mql_id"].isna().any():
        raise ValueError("The mql_id join key contains missing values.")

    if qualified["mql_id"].duplicated().any():
        raise ValueError(
            "The qualified-leads file has duplicate mql_id values; "
            "the lead denominator would be ambiguous."
        )
    if closed["mql_id"].duplicated().any():
        raise ValueError(
            "The closed-deals file has duplicate mql_id values; "
            "the deal count would be ambiguous."
        )


def inspect_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Summarize actual CSV dtypes, missingness, uniqueness, and duplicates."""
    report = pd.DataFrame(
        {
            "dataset": dataset_name,
            "column": df.columns,
            "data_type": [str(dtype) for dtype in df.dtypes],
            "missing_values": df.isna().sum().values,
            "missing_percent": (df.isna().mean() * 100).round(2).values,
            "unique_values": df.nunique(dropna=True).values,
            "duplicate_rows_in_dataset": int(df.duplicated().sum()),
            "description": [
                COLUMN_DESCRIPTIONS.get(
                    column, "No predefined description available for this column."
                )
                for column in df.columns
            ],
        }
    )
    return report


def inspect_and_print(
    qualified: pd.DataFrame, closed: pd.DataFrame
) -> pd.DataFrame:
    """Print basic inspection for both source datasets and save a combined report."""
    reports = []
    for name, frame in (
        (QUALIFIED_FILE.name, qualified),
        (CLOSED_FILE.name, closed),
    ):
        print(f"\n=== DATA INSPECTION: {name} ===")
        print(f"Rows: {len(frame):,}")
        print(f"Columns: {len(frame.columns)}")
        print(f"Column names: {frame.columns.tolist()}")
        print("Data types:")
        print(frame.dtypes.to_string())
        print("Missing values:")
        print(frame.isna().sum().to_string())
        print(f"Duplicate rows: {int(frame.duplicated().sum()):,}")
        print("Column descriptions:")
        for column in frame.columns:
            description = COLUMN_DESCRIPTIONS.get(
                column, "No predefined description available for this column."
            )
            print(f" - {column}: {description}")
        reports.append(inspect_dataset(frame, name))

    combined = pd.concat(reports, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "data_inspection.csv", index=False)
    return combined


def create_joined_leads(
    qualified: pd.DataFrame, closed: pd.DataFrame
) -> pd.DataFrame:
    """Left-join closed-deal status/date onto every qualified lead by mql_id."""
    closed_columns = ["mql_id", "won_date"] + [
        column
        for column in closed.columns
        if column not in {"mql_id", "won_date"}
    ]
    joined = qualified.merge(
        closed[closed_columns],
        on="mql_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    joined["closed_won"] = joined["_merge"].eq("both")
    joined.drop(columns="_merge", inplace=True)
    return joined


def make_funnel_summary(
    qualified: pd.DataFrame, closed: pd.DataFrame
) -> pd.DataFrame:
    """Summarize the two stages supported by the source data."""
    total_qualified = len(qualified)
    total_closed = int(qualified["mql_id"].isin(closed["mql_id"]).sum())
    conversion = total_closed / total_qualified if total_qualified else np.nan
    dropoff = 1 - conversion if pd.notna(conversion) else np.nan
    return pd.DataFrame(
        [
            {
                "funnel_stage": "Marketing Qualified Leads",
                "number_of_leads": total_qualified,
                "conversion_from_previous_stage_pct": 100.0,
                "dropoff_from_previous_stage_pct": 0.0,
            },
            {
                "funnel_stage": "Closed/Won Leads",
                "number_of_leads": total_closed,
                "conversion_from_previous_stage_pct": conversion * 100,
                "dropoff_from_previous_stage_pct": dropoff * 100,
            },
        ]
    )


def make_origin_performance(joined: pd.DataFrame) -> pd.DataFrame:
    """Calculate MQL-to-closed conversion for each marketing origin."""
    leads = joined.copy()
    leads["origin"] = leads["origin"].fillna("Not provided")
    performance = (
        leads.groupby("origin", dropna=False)
        .agg(
            qualified_leads=("mql_id", "nunique"),
            closed_deals=("closed_won", "sum"),
        )
        .reset_index()
    )
    performance["conversion_rate_pct"] = (
        performance["closed_deals"] / performance["qualified_leads"] * 100
    )
    performance["dropoff_rate_pct"] = 100 - performance["conversion_rate_pct"]
    performance["performance_by_conversion"] = "Below overall rate"
    overall_rate = (
        joined["closed_won"].sum() / len(joined) * 100 if len(joined) else np.nan
    )
    performance.loc[
        performance["conversion_rate_pct"].ge(overall_rate),
        "performance_by_conversion",
    ] = "At or above overall rate"
    performance.loc[
        performance["origin"].eq("Not provided"), "performance_by_conversion"
    ] = "Origin not provided"
    performance.loc[
        performance["origin"].eq("unknown"), "performance_by_conversion"
    ] = "Origin unknown"
    return performance.sort_values(
        ["conversion_rate_pct", "qualified_leads"], ascending=[False, False]
    ).reset_index(drop=True)


def make_closed_deal_breakdown(
    closed: pd.DataFrame, column: str
) -> pd.DataFrame | None:
    """Count closed deals by an available descriptive business field."""
    if column not in closed.columns:
        return None
    values = closed[column].fillna("Not provided")
    result = (
        closed.assign(_category=values)
        .groupby("_category", dropna=False)
        .agg(closed_deals=("mql_id", "nunique"))
        .reset_index()
        .rename(columns={"_category": column})
    )
    total = result["closed_deals"].sum()
    result["share_of_closed_deals_pct"] = (
        result["closed_deals"] / total * 100 if total else np.nan
    )
    return result.sort_values(
        ["closed_deals", column], ascending=[False, True]
    ).reset_index(drop=True)


def make_monthly_trend(joined: pd.DataFrame) -> pd.DataFrame | None:
    """Calculate qualified leads and closed leads by first-contact month."""
    first_contact = pd.to_datetime(joined["first_contact_date"], errors="coerce")
    if first_contact.notna().sum() == 0:
        print(
            "Monthly funnel trend unavailable: first_contact_date contains no "
            "values that can be parsed as dates."
        )
        return None
    trend_data = joined.assign(
        month=first_contact.dt.to_period("M").astype("string")
    ).dropna(subset=["month"])
    trend = (
        trend_data.groupby("month")
        .agg(
            qualified_leads=("mql_id", "nunique"),
            closed_deals=("closed_won", "sum"),
        )
        .reset_index()
    )
    trend["conversion_rate_pct"] = (
        trend["closed_deals"] / trend["qualified_leads"] * 100
    )
    return trend


def make_days_to_close(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Calculate close delay for matched deals, excluding invalid/negative intervals."""
    converted = joined.loc[joined["closed_won"]].copy()
    first_contact = pd.to_datetime(converted["first_contact_date"], errors="coerce")
    won_date = pd.to_datetime(converted["won_date"], errors="coerce")
    days = (won_date - first_contact).dt.total_seconds() / 86400
    valid = days.notna() & days.ge(0)
    summary = pd.DataFrame(
        [
            {
                "matched_closed_deals": len(converted),
                "valid_nonnegative_date_intervals": int(valid.sum()),
                "invalid_or_negative_intervals_excluded": int((~valid).sum()),
                "average_days_to_close": days.loc[valid].mean(),
                "median_days_to_close": days.loc[valid].median(),
            }
        ]
    )
    return summary, days


def plot_funnel(funnel: pd.DataFrame) -> None:
    """Save a simple two-stage funnel-style bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C78A8", "#59A14F"]
    bars = ax.barh(
        funnel["funnel_stage"],
        funnel["number_of_leads"],
        color=colors,
        height=0.58,
    )
    ax.invert_yaxis()
    ax.set_title("Olist Marketing Funnel: Qualified Leads to Closed Deals")
    ax.set_xlabel("Number of leads")
    ax.set_ylabel("")
    ax.bar_label(bars, labels=[f"{int(value):,}" for value in funnel["number_of_leads"]], padding=6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "funnel_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_leads_by_origin(origin: pd.DataFrame) -> None:
    """Save qualified lead and closed deal counts by origin."""
    plot_data = origin.sort_values("qualified_leads", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(plot_data))
    height = 0.38
    ax.barh(
        positions - height / 2,
        plot_data["qualified_leads"],
        height,
        label="Qualified leads",
        color="#4C78A8",
    )
    ax.barh(
        positions + height / 2,
        plot_data["closed_deals"],
        height,
        label="Closed deals",
        color="#59A14F",
    )
    ax.set_yticks(positions, plot_data["origin"])
    ax.set_title("Qualified Leads and Closed Deals by Marketing Origin")
    ax.set_xlabel("Number of leads")
    ax.set_ylabel("Marketing origin")
    ax.legend(frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "leads_by_origin.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_conversion_by_origin(origin: pd.DataFrame) -> None:
    """Save a marketing origin conversion-rate chart."""
    plot_data = origin.sort_values("conversion_rate_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        plot_data["origin"],
        plot_data["conversion_rate_pct"],
        color="#F28E2B",
    )
    ax.axvline(
        origin["closed_deals"].sum()
        / origin["qualified_leads"].sum()
        * 100,
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        label="Overall conversion",
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_title("Qualified-Lead to Closed-Deal Conversion by Origin")
    ax.set_xlabel("Conversion rate (%)")
    ax.set_ylabel("Marketing origin")
    ax.legend(frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "conversion_rate_by_origin.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_closed_deal_breakdown(
    breakdown: pd.DataFrame | None, column: str
) -> None:
    """Save a top-10 category chart for an available closed-deal dimension."""
    if breakdown is None or breakdown.empty:
        print(f"Skipping {column} chart: this field is not available.")
        return
    plot_data = breakdown.head(10).sort_values("closed_deals", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        plot_data[column].astype(str),
        plot_data["closed_deals"],
        color="#76B7B2",
    )
    ax.bar_label(bars, padding=4)
    ax.set_title(f"Closed Deals by {column.replace('_', ' ').title()} (Top 10)")
    ax.set_xlabel("Number of closed deals")
    ax.set_ylabel(column.replace("_", " ").title())
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"closed_deals_by_{column}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_monthly_trend(trend: pd.DataFrame | None) -> None:
    """Save the monthly qualified-lead and closed-deal volume chart."""
    if trend is None or trend.empty:
        print("Monthly trend chart not created because there is no valid monthly data.")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        trend["month"],
        trend["qualified_leads"],
        marker="o",
        label="Qualified leads",
        color="#4C78A8",
    )
    ax.plot(
        trend["month"],
        trend["closed_deals"],
        marker="o",
        label="Closed deals",
        color="#59A14F",
    )
    ax.set_title("Monthly Funnel Volumes by First-Contact Month")
    ax.set_xlabel("First-contact month")
    ax.set_ylabel("Number of leads")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_funnel_trend.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_insights(
    qualified: pd.DataFrame,
    closed: pd.DataFrame,
    origin: pd.DataFrame,
    days_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Create a numeric KPI summary and findings/recommendations from the data."""
    total_leads = len(qualified)
    total_closed = len(closed)
    matched_deals = int(qualified["mql_id"].isin(closed["mql_id"]).sum())
    conversion_pct = matched_deals / total_leads * 100 if total_leads else np.nan
    dropoff_pct = 100 - conversion_pct if pd.notna(conversion_pct) else np.nan

    eligible_origins = origin.loc[
        ~origin["origin"].isin(["Not provided", "unknown"])
    ].copy()
    best = eligible_origins.sort_values(
        ["conversion_rate_pct", "qualified_leads"], ascending=[False, False]
    ).iloc[0]
    weakest = eligible_origins.sort_values(
        ["conversion_rate_pct", "qualified_leads"], ascending=[True, False]
    ).iloc[0]
    most_volume = eligible_origins.sort_values(
        ["qualified_leads", "closed_deals"], ascending=[False, False]
    ).iloc[0]
    weakest_origins = eligible_origins.nsmallest(3, "conversion_rate_pct")
    weakest_origin_text = ", ".join(
        f"{row.origin} ({row.conversion_rate_pct:.2f}%)"
        for row in weakest_origins.itertuples()
    )
    closed_sum = int(origin["closed_deals"].sum())
    unmatched_closed = total_closed - matched_deals

    metrics = [
        {
            "metric": "Marketing qualified leads",
            "result": total_leads,
            "business_meaning": "Number of unique qualified leads in the qualified-leads file.",
        },
        {
            "metric": "Closed deals",
            "result": total_closed,
            "business_meaning": "Number of unique records in the closed-deals file.",
        },
        {
            "metric": "Qualified-lead to closed-deal conversion",
            "result": round(conversion_pct, 2),
            "business_meaning": "Share of qualified leads that are present in the closed-deals file (percent).",
        },
        {
            "metric": "Overall funnel drop-off",
            "result": round(dropoff_pct, 2),
            "business_meaning": "Share of qualified leads without a matching closed-deal record (percent).",
        },
        {
            "metric": "Best-performing identified origin",
            "result": str(best["origin"]),
            "business_meaning": (
                f"{best['conversion_rate_pct']:.2f}% conversion from "
                f"{int(best['qualified_leads']):,} qualified leads "
                f"({int(best['closed_deals']):,} closed deals)."
            ),
        },
        {
            "metric": "Lowest-performing identified origin",
            "result": str(weakest["origin"]),
            "business_meaning": (
                f"{weakest['conversion_rate_pct']:.2f}% conversion from "
                f"{int(weakest['qualified_leads']):,} qualified leads "
                f"({int(weakest['closed_deals']):,} closed deals)."
            ),
        },
        {
            "metric": "Highest-volume origin",
            "result": str(most_volume["origin"]),
            "business_meaning": (
                f"{int(most_volume['qualified_leads']):,} qualified leads and "
                f"{int(most_volume['closed_deals']):,} closed deals."
            ),
        },
        {
            "metric": "Matched closed deals",
            "result": matched_deals,
            "business_meaning": "Closed-deal IDs found among the qualified leads.",
        },
        {
            "metric": "Closed deals without a qualified-lead match",
            "result": unmatched_closed,
            "business_meaning": "Closed-deal IDs not found in the qualified-leads file.",
        },
        {
            "metric": "Average days from first contact to won date",
            "result": round(float(days_summary.loc[0, "average_days_to_close"]), 2),
            "business_meaning": (
                "Mean elapsed days for matched deals with valid, nonnegative dates; "
                "see close-time output for excluded intervals."
            ),
        },
    ]
    metrics_df = pd.DataFrame(metrics)

    insights = [
        "# Marketing Funnel Performance Analysis — Findings",
        "",
        "## Data scope",
        (
            f"The analysis includes {total_leads:,} qualified leads and "
            f"{total_closed:,} closed deals from the two Olist source CSVs. "
            f"{matched_deals:,} closed deals match a qualified lead by `mql_id`; "
            f"{unmatched_closed:,} closed-deal records do not match."
        ),
        (
            "The available files support a two-stage funnel (Marketing Qualified "
            "Leads to Closed/Won Leads). They do not provide visitor, opportunity, "
            "or intermediate sales-stage counts, so those stages are not estimated."
        ),
        "",
        "## Key findings",
        (
            f"- Funnel conversion is {conversion_pct:.2f}% "
            f"({total_closed:,} of {total_leads:,}); {dropoff_pct:.2f}% of "
            "qualified leads have no matching closed-deal record."
        ),
        (
            f"- Best conversion among identified marketing origins (excluding "
            f"`unknown` and missing labels): **{best['origin']}** at "
            f"{best['conversion_rate_pct']:.2f}% "
            f"({int(best['closed_deals']):,}/{int(best['qualified_leads']):,})."
        ),
        (
            f"- Lowest conversion among identified marketing origins: **{weakest['origin']}** "
            f"at {weakest['conversion_rate_pct']:.2f}% "
            f"({int(weakest['closed_deals']):,}/{int(weakest['qualified_leads']):,})."
        ),
        f"- Other low-converting identified origins include: {weakest_origin_text}.",
        (
            f"- The largest qualified-lead volume comes from **{most_volume['origin']}** "
            f"({int(most_volume['qualified_leads']):,} leads, "
            f"{int(most_volume['closed_deals']):,} closed deals)."
        ),
        (
            f"- Average time from first contact to won date is "
            f"{days_summary.loc[0, 'average_days_to_close']:.2f} days for "
            f"{int(days_summary.loc[0, 'valid_nonnegative_date_intervals']):,} "
            "valid, nonnegative matched-deal intervals."
        ),
        (
            "- The closed-deals file contains declared monthly revenue, but no "
            "marketing spend; ROI or return-on-ad-spend cannot be calculated. "
            "Declared monthly revenue should not be treated as realized sales."
        ),
        "",
        "## Practical recommendations",
        (
            f"- Investigate the lowest-converting identified origins "
            f"({weakest_origin_text}). Review lead quality, targeting, and "
            "follow-up before changing investment."
        ),
        (
            f"- **{best['origin']}** has the highest conversion among identified "
            "origins. Compare its lead quality and acquisition practices with "
            "organic_search, which brings the largest lead volume, before testing "
            "any budget changes; spend data is not available here."
        ),
        (
            f"- Prioritize follow-up process improvements across the full MQL pool: "
            f"{dropoff_pct:.2f}% have no matching win in this dataset. The source "
            "files do not show intermediate stage or contact-attempt history, so "
            "they cannot identify the exact point or cause of loss."
        ),
        "- Track channel-level spend and realized revenue in future reporting to support budget allocation and ROI comparisons.",
        "- Improve capture of missing origin (and missing closed-deal profile fields) to make channel and segment comparisons more complete.",
        "",
        "## Interpretation notes",
        (
            "Channel rates are descriptive historical rates, not causal effects. "
            "The literal `unknown` category and blank origins are kept separate "
            "from named channels and excluded from best/weakest identified-origin "
            "labels. "
            "Business-segment, lead-type, behavior-profile, and business-type "
            "breakdowns describe closed deals only because those attributes are "
            "not supplied for all qualified leads."
        ),
    ]
    return metrics_df, "\n".join(insights) + "\n"


def main() -> None:
    """Run the complete inspection, funnel analysis, charts, and reporting."""
    sns.set_theme(style="whitegrid", palette="deep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    require_files()
    qualified, closed = load_datasets()
    validate_required_columns(qualified, closed)

    inspection = inspect_and_print(qualified, closed)
    print("\n=== JOIN VALIDATION ===")
    print(f"Qualified leads: {len(qualified):,}")
    print(f"Closed-deal records: {len(closed):,}")
    print(
        "Closed deals matching qualified leads by mql_id: "
        f"{int(qualified['mql_id'].isin(closed['mql_id']).sum()):,}"
    )
    print(
        "Closed deals without a qualified-lead match: "
        f"{int((~closed['mql_id'].isin(qualified['mql_id'])).sum()):,}"
    )

    joined = create_joined_leads(qualified, closed)
    joined.to_csv(OUTPUT_DIR / "qualified_leads_with_closed_status.csv", index=False)

    funnel = make_funnel_summary(qualified, closed)
    funnel.to_csv(OUTPUT_DIR / "funnel_summary.csv", index=False)

    origin = make_origin_performance(joined)
    origin.to_csv(OUTPUT_DIR / "origin_performance.csv", index=False)

    breakdowns: dict[str, pd.DataFrame] = {}
    for column in (
        "business_segment",
        "lead_type",
        "lead_behaviour_profile",
        "business_type",
    ):
        breakdown = make_closed_deal_breakdown(closed, column)
        if breakdown is None:
            print(
                f"Optional category analysis unavailable: closed-deals dataset "
                f"does not contain '{column}'."
            )
            continue
        breakdowns[column] = breakdown
        breakdown.to_csv(OUTPUT_DIR / f"closed_deals_by_{column}.csv", index=False)

    days_summary, day_intervals = make_days_to_close(joined)
    days_summary.to_csv(OUTPUT_DIR / "days_to_close_summary.csv", index=False)
    invalid_count = int((day_intervals.isna() | day_intervals.lt(0)).sum())
    if invalid_count:
        print(
            f"\nDate quality note: {invalid_count} matched deal interval(s) had "
            "invalid dates or a won_date before first_contact_date; excluded "
            "from average and median time-to-close."
        )

    trend = make_monthly_trend(joined)
    if trend is not None:
        trend.to_csv(OUTPUT_DIR / "monthly_funnel_trend.csv", index=False)

    metrics, report = build_insights(qualified, closed, origin, days_summary)
    metrics.to_csv(OUTPUT_DIR / "business_summary.csv", index=False)
    (OUTPUT_DIR / "business_insights_and_recommendations.md").write_text(
        report, encoding="utf-8"
    )

    plot_funnel(funnel)
    plot_leads_by_origin(origin)
    plot_conversion_by_origin(origin)
    plot_monthly_trend(trend)
    for column in ("business_segment", "lead_type"):
        plot_closed_deal_breakdown(breakdowns.get(column), column)

    print("\n=== FUNNEL SUMMARY ===")
    print(funnel.to_string(index=False))
    print("\n=== MARKETING ORIGIN PERFORMANCE ===")
    print(origin.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\n=== TIME TO CLOSE ===")
    print(days_summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\n=== BUSINESS SUMMARY ===")
    print(metrics.to_string(index=False))
    print("\n=== INSIGHTS AND RECOMMENDATIONS ===")
    print(report)
    print(f"Inspection rows saved: {len(inspection):,}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
