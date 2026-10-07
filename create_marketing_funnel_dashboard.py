"""Create a one-page PNG dashboard from the generated Olist analysis tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_PATH = OUTPUT_DIR / "marketing_funnel_dashboard.png"

NAVY = "#14263D"
BLUE = "#3E78A8"
TEAL = "#35A99A"
ORANGE = "#E69F42"
RED = "#D96B62"
MUTED = "#66758A"
TEXT = "#25364A"
PAGE = "#F2F5F9"
WHITE = "#FFFFFF"
GRID = "#E6EBF1"


def load_output(name: str, required_columns: set[str]) -> pd.DataFrame:
    """Load a generated CSV and make missing files/columns explicit."""
    path = OUTPUT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Required generated result is missing: {path}")
    frame = pd.read_csv(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required output column(s): {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError(f"Required generated result contains no rows: {path}")
    return frame


def get_metric(summary: pd.DataFrame, metric_name: str) -> str:
    """Read a single KPI result from the generated business summary."""
    match = summary.loc[summary["metric"].eq(metric_name), "result"]
    if match.empty:
        raise ValueError(f"Metric is missing from business_summary.csv: {metric_name}")
    return str(match.iloc[0])


def add_panel(fig: plt.Figure, rect: tuple[float, float, float, float]) -> None:
    """Draw a white panel behind the chart and text axes."""
    panel_ax = fig.add_axes(rect, zorder=0)
    panel_ax.set_facecolor(WHITE)
    panel_ax.set_xticks([])
    panel_ax.set_yticks([])
    for spine in panel_ax.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(0.7)


def panel_heading(
    fig: plt.Figure, x: float, y: float, title: str, subtitle: str = ""
) -> None:
    """Add a consistent heading inside a dashboard panel."""
    fig.text(
        x,
        y,
        title,
        fontsize=14,
        color=NAVY,
        fontweight="bold",
        va="center",
        zorder=30,
    )
    if subtitle:
        fig.text(
            x, y - 0.029, subtitle, fontsize=9.5, color=MUTED, va="top", zorder=30
        )


def draw_kpi(
    fig: plt.Figure,
    x: float,
    label: str,
    value: str,
    accent: str,
    width: float = 0.176,
) -> None:
    """Draw one KPI card."""
    add_panel(fig, (x, 0.805, width, 0.085))
    fig.add_artist(
        plt.Line2D(
            [x + 0.012, x + 0.012],
            [0.822, 0.873],
            transform=fig.transFigure,
            color=accent,
            linewidth=3,
            solid_capstyle="round",
            zorder=30,
        )
    )
    fig.text(
        x + 0.024,
        0.867,
        label.upper(),
        fontsize=9,
        color=MUTED,
        fontweight="bold",
        va="top",
        zorder=30,
    )
    fig.text(
        x + 0.024,
        0.827,
        value,
        fontsize=21,
        color=NAVY,
        fontweight="bold",
        va="bottom",
        zorder=30,
    )


def draw_origin_chart(
    fig: plt.Figure,
    origins: pd.DataFrame,
    best_rate: float,
    overall_conversion: float,
) -> None:
    """Compare lead volume by origin and annotate closed count and conversion."""
    panel_heading(
        fig,
        0.055,
        0.762,
        "CHANNEL PERFORMANCE",
        "Leads by bar length   |   Right label = wins and conversion   |   Teal = best named origin",
    )
    ax = fig.add_axes([0.112, 0.445, 0.508, 0.265], zorder=2)
    plot_data = origins.sort_values(
        "qualified_leads", ascending=False
    ).reset_index(drop=True)
    display_names = {
        "direct_traffic": "direct traffic",
        "other_publicities": "other publicity",
        "Not provided": "missing origin",
    }
    display_origins = [
        display_names.get(str(origin), str(origin)) for origin in plot_data["origin"]
    ]

    def color_for_origin(row: pd.Series) -> str:
        name = str(row["origin"])
        if name == "paid_search":
            return TEAL
        if name == "other":
            return RED
        if name in {"unknown", "Not provided"}:
            return "#AAB4C0"
        return BLUE

    colors = [color_for_origin(row) for _, row in plot_data.iterrows()]
    positions = np.arange(len(plot_data))
    ax.barh(
        positions,
        plot_data["qualified_leads"],
        color=colors,
        height=0.66,
        zorder=3,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(display_origins, fontsize=10, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 3650)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
    ax.set_xlabel("Qualified leads", fontsize=10, color=MUTED, labelpad=6)
    ax.tick_params(axis="x", labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for y, (_, row) in enumerate(plot_data.iterrows()):
        ax.text(
            2630,
            y,
            f"{int(row['closed_deals']):,}  •  {row['conversion_rate_pct']:.1f}%",
            va="center",
            fontsize=9.5,
            color=TEXT,
            fontweight="bold" if row["origin"] == "paid_search" else "normal",
        )
def draw_funnel(
    fig: plt.Figure,
    qualified_count: int,
    closed_count: int,
    conversion_pct: float,
    dropoff_pct: float,
) -> None:
    """Show the two observed funnel stages and their saved rates."""
    panel_heading(
        fig,
        0.665,
        0.762,
        "QUALIFIED-LEAD FUNNEL",
        "Only the stages present in the source data are shown",
    )
    ax = fig.add_axes([0.685, 0.574, 0.245, 0.112], zorder=2)
    ax.barh(
        [1, 0],
        [100, closed_count / qualified_count * 100],
        color=[BLUE, TEAL],
        height=0.48,
    )
    ax.set_xlim(0, 110)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Qualified", "Closed / won"], fontsize=10, color=TEXT)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of qualified leads (%)", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(98, 1, f"{qualified_count:,}", va="center", ha="right", color=WHITE, fontsize=10, fontweight="bold")
    ax.text(
        closed_count / qualified_count * 100 + 2,
        0,
        f"{closed_count:,}",
        va="center",
        color=TEXT,
        fontsize=10,
        fontweight="bold",
    )

    fig.text(
        0.69,
        0.520,
        f"CONVERTED   {conversion_pct:.2f}%",
        fontsize=12,
        color=TEAL,
        fontweight="bold",
        zorder=30,
    )
    fig.text(
        0.69,
        0.484,
        f"DROPPED OFF   {dropoff_pct:.2f}%",
        fontsize=12,
        color=RED,
        fontweight="bold",
        zorder=30,
    )
    fig.text(
        0.69,
        0.446,
        f"{qualified_count - closed_count:,} qualified leads have no matching closed deal.",
        fontsize=9.5,
        color=MUTED,
        zorder=30,
    )


def draw_business_panel(
    fig: plt.Figure, segments: pd.DataFrame, lead_types: pd.DataFrame
) -> None:
    """Show leading closed-deal segment and lead type as metric callouts."""
    panel_heading(
        fig,
        0.055,
        0.36,
        "BUSINESS PERFORMANCE",
        "Closed-deal counts only; these are not category conversion rates",
    )
    segment = segments.iloc[0]
    lead_type = lead_types.iloc[0]
    ax = fig.add_axes([0.055, 0.065, 0.265, 0.265], zorder=2)
    ax.set_axis_off()
    ax.text(
        0.03,
        0.91,
        "TOP BUSINESS SEGMENT",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.03,
        0.62,
        str(segment["business_segment"]).replace("_", " "),
        transform=ax.transAxes,
        fontsize=13.5,
        color=NAVY,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.03,
        0.35,
        f"{int(segment['closed_deals']):,} wins  |  "
        f"{segment['share_of_closed_deals_pct']:.2f}% share",
        transform=ax.transAxes,
        fontsize=9.5,
        color=TEXT,
        va="top",
    )
    ax.plot([0.51, 0.51], [0.12, 0.92], transform=ax.transAxes, color=GRID, linewidth=1)
    ax.text(
        0.57,
        0.91,
        "TOP LEAD TYPE",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.57,
        0.62,
        str(lead_type["lead_type"]).replace("_", " "),
        transform=ax.transAxes,
        fontsize=13.5,
        color=NAVY,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.57,
        0.35,
        f"{int(lead_type['closed_deals']):,} wins  |  "
        f"{lead_type['share_of_closed_deals_pct']:.2f}% share",
        transform=ax.transAxes,
        fontsize=9.5,
        color=TEXT,
        va="top",
    )


def draw_bullets(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    title: str,
    bullets: list[str],
    color: str,
) -> None:
    """Write short, manually wrapped bullets in a dashboard panel."""
    x, y, width, height = rect
    panel_heading(fig, x + 0.02, y + height - 0.022, title)
    ax = fig.add_axes(
        [x + 0.018, y + 0.02, width - 0.036, height - 0.07], zorder=2
    )
    ax.axis("off")
    y_pos = 0.94
    for bullet in bullets:
        wrapped_bullet = "\n".join(textwrap.wrap(bullet, width=50))
        ax.text(
            0.02,
            y_pos,
            "\u2022",
            transform=ax.transAxes,
            fontsize=11,
            color=color,
            fontweight="bold",
            va="top",
        )
        text = ax.text(
            0.085,
            y_pos,
            wrapped_bullet,
            transform=ax.transAxes,
            fontsize=10.5,
            color=TEXT,
            va="top",
            linespacing=1.15,
            wrap=True,
        )
        # Fixed spacing leaves enough room for every concise bullet at page size.
        y_pos -= 0.19
        del text


def draw_sidebar(fig: plt.Figure) -> None:
    """Add dashboard-style navigation for a familiar portfolio layout."""
    sidebar = fig.add_axes([0, 0.04, 0.145, 0.86], zorder=0)
    sidebar.set_facecolor(WHITE)
    sidebar.set_xticks([])
    sidebar.set_yticks([])
    for spine in sidebar.spines.values():
        spine.set_color("#E2E8F0")
        spine.set_linewidth(0.7)

    fig.text(
        0.025,
        0.862,
        "REPORT SECTIONS",
        fontsize=8.5,
        color=MUTED,
        fontweight="bold",
        va="center",
        zorder=30,
    )
    labels = [
        "Overview",
        "Funnel Analysis",
        "Origin Performance",
        "Business Profile",
        "Insights",
        "Recommendations",
    ]
    section_numbers = ["01", "02", "03", "04", "05", "06"]
    section_colors = ["#2878B5", "#21A38A", "#7758C5", "#D58A25", "#D95D76", "#287FA1"]
    section_backgrounds = [
        "#EAF3FA",
        "#EAF7F3",
        "#F1EDFA",
        "#FBF3E7",
        "#FBECEF",
        "#EAF4F7",
    ]
    for index, label in enumerate(labels):
        y = 0.82 - index * 0.065
        row = fig.add_axes([0.012, y - 0.021, 0.12, 0.043], zorder=1)
        row.set_facecolor(section_backgrounds[index])
        row.set_xticks([])
        row.set_yticks([])
        for spine in row.spines.values():
            spine.set_visible(False)
        row.plot(
            [0.035, 0.035],
            [0.18, 0.82],
            color=section_colors[index],
            linewidth=3,
            solid_capstyle="round",
            transform=row.transAxes,
            clip_on=False,
        )
        fig.text(
            0.021,
            y,
            section_numbers[index],
            fontsize=7.5,
            color=section_colors[index],
            fontweight="bold",
            va="center",
            zorder=30,
        )
        fig.text(
            0.044,
            y,
            label,
            fontsize=8.4,
            color=section_colors[index] if index == 0 else TEXT,
            fontweight="bold" if index == 0 else "normal",
            va="center",
            zorder=30,
        )

    fig.text(
        0.022,
        0.105,
        "DATA SCOPE",
        fontsize=8,
        color=MUTED,
        fontweight="bold",
        va="center",
        zorder=30,
    )
    fig.text(
        0.022,
        0.073,
        "Olist MQLs\nand closed deals",
        fontsize=9.2,
        color=TEXT,
        linespacing=1.4,
        va="top",
        zorder=30,
    )


def draw_compact_funnel(
    fig: plt.Figure, qualified_count: int, closed_count: int, conversion: float, dropoff: float
) -> None:
    """Draw the observed two-stage funnel with counts and exact saved percentages."""
    x, y, width, height = 0.165, 0.485, 0.265, 0.28
    add_panel(fig, (x, y, width, height))
    panel_heading(fig, x + 0.018, y + height - 0.018, "MARKETING FUNNEL")
    fig.text(
        x + 0.018,
        y + height - 0.05,
        "Qualified leads to closed deals",
        fontsize=8.4,
        color=MUTED,
        va="top",
        zorder=30,
    )

    ax = fig.add_axes([x + 0.03, y + 0.09, width - 0.06, 0.13], zorder=2)
    counts = [qualified_count, closed_count]
    max_count = max(counts)
    ax.barh(
        [1, 0],
        counts,
        color=[BLUE, TEAL],
        height=0.56,
        zorder=3,
    )
    ax.set_xlim(0, max_count * 1.18)
    ax.set_yticks([1, 0], ["Qualified", "Closed / won"])
    ax.tick_params(axis="y", labelsize=8.2, colors=TEXT, length=0, pad=5)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for position, count in ((1, qualified_count), (0, closed_count)):
        ax.text(
            count - max_count * 0.025,
            position,
            f"{count:,}",
            ha="right",
            va="center",
            color=WHITE,
            fontsize=9,
            fontweight="bold",
        )

    fig.text(
        x + 0.02,
        y + 0.058,
        f"CONVERSION  {conversion:.2f}%",
        fontsize=9.3,
        color=TEAL,
        fontweight="bold",
        va="center",
        zorder=30,
    )
    fig.text(
        x + 0.02,
        y + 0.027,
        f"DROP-OFF  {dropoff:.2f}%  ({qualified_count - closed_count:,} leads)",
        fontsize=9,
        color=RED,
        fontweight="bold",
        va="center",
        zorder=30,
    )


def draw_origin_performance(
    fig: plt.Figure,
    origins: pd.DataFrame,
    overall_conversion: float,
    best_origin: str,
    best_rate: float,
) -> None:
    """Compare lead volume, closed deals, and conversion across all origins."""
    x, y, width, height = 0.44, 0.485, 0.545, 0.28
    add_panel(fig, (x, y, width, height))
    panel_heading(
        fig,
        x + 0.018,
        y + height - 0.018,
        "ORIGIN PERFORMANCE",
        "Bar = qualified leads   |   Right label = closed deals • conversion",
    )

    data = origins.sort_values("qualified_leads", ascending=False).reset_index(drop=True)
    names = {
        "Not provided": "Missing origin",
        "direct_traffic": "Direct traffic",
        "other_publicities": "Other publicity",
    }
    y_labels = [names.get(str(origin), str(origin)) for origin in data["origin"]]
    color_map = []
    for origin in data["origin"]:
        if origin == best_origin:
            color_map.append(TEAL)
        elif origin == "other":
            color_map.append(RED)
        elif origin in ("unknown", "Not provided"):
            color_map.append("#AAB4C0")
        else:
            color_map.append(BLUE)

    ax = fig.add_axes([x + 0.15, y + 0.065, width - 0.17, 0.15], zorder=2)
    y_positions = np.arange(len(data))
    ax.barh(y_positions, data["qualified_leads"], color=color_map, height=0.64, zorder=3)
    ax.set_yticks(y_positions, y_labels, fontsize=8.8, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, 3550)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xlabel("Qualified leads", fontsize=8, color=MUTED, labelpad=3)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for position, (_, row) in enumerate(data.iterrows()):
        ax.text(
            3430,
            position,
            f"{int(row['closed_deals']):,}  •  {row['conversion_rate_pct']:.1f}%",
            ha="right",
            va="center",
            fontsize=8.4,
            color=TEXT,
            fontweight="bold" if row["origin"] == best_origin else "normal",
        )

    fig.text(
        x + 0.018,
        y + 0.016,
        f"Best named origin: {best_origin} ({best_rate:.2f}%)  |  Overall: {overall_conversion:.2f}%",
        fontsize=8.3,
        color=MUTED,
        va="bottom",
        zorder=30,
    )


def draw_top_categories(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    frame: pd.DataFrame,
    category_column: str,
    title: str,
) -> None:
    """Create a compact top-category horizontal bar chart from closed-deal results."""
    x, y, width, height = rect
    add_panel(fig, rect)
    panel_heading(
        fig,
        x + 0.018,
        y + height - 0.018,
        title,
        "Closed-deal counts only",
    )
    data = frame.head(5).sort_values("closed_deals", ascending=True)
    labels = [str(value).replace("_", " ") for value in data[category_column]]
    ax = fig.add_axes([x + 0.125, y + 0.045, width - 0.15, height - 0.105], zorder=2)
    bars = ax.barh(
        np.arange(len(data)),
        data["closed_deals"],
        color=[BLUE] * max(0, len(data) - 1) + ([TEAL] if len(data) else []),
        height=0.62,
    )
    ax.set_yticks(np.arange(len(data)), labels, fontsize=8.5, color=TEXT)
    ax.set_xlim(0, float(data["closed_deals"].max()) * 1.28)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.bar_label(bars, padding=3, fontsize=8.5, color=TEXT)


def draw_insights(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    bullets: list[str],
) -> None:
    """Show concise factual findings in a matching panel."""
    x, y, width, height = rect
    add_panel(fig, rect)
    panel_heading(fig, x + 0.018, y + height - 0.018, "KEY INSIGHTS")
    y_pos = y + height - 0.05
    for bullet in bullets:
        fig.text(
            x + 0.025,
            y_pos,
            "•",
            fontsize=10,
            color=BLUE,
            fontweight="bold",
            va="top",
            zorder=30,
        )
        fig.text(
            x + 0.047,
            y_pos,
            textwrap.fill(bullet, width=43),
            fontsize=8.8,
            color=TEXT,
            va="top",
            linespacing=1.1,
            zorder=30,
        )
        y_pos -= 0.036


def draw_recommendation_cards(fig: plt.Figure, recommendations: list[str]) -> None:
    """Arrange practical actions as uniform cards along the dashboard footer."""
    x_start, gap, width, y, height = 0.165, 0.012, 0.196, 0.04, 0.115
    for index, recommendation in enumerate(recommendations):
        x = x_start + index * (width + gap)
        add_panel(fig, (x, y, width, height))
        fig.text(
            x + 0.013,
            y + height / 2,
            f"{index + 1:02d}",
            fontsize=10,
            color=TEAL,
            fontweight="bold",
            va="top",
            zorder=30,
        )
        fig.text(
            x + 0.045,
            y + height / 2,
            textwrap.fill(recommendation, width=30),
            fontsize=9.1,
            color=TEXT,
            va="center",
            linespacing=1.2,
            zorder=30,
        )


def main() -> None:
    """Read generated analysis results and render the requested single PNG."""
    funnel = load_output(
        "funnel_summary.csv",
        {
            "funnel_stage",
            "number_of_leads",
            "conversion_from_previous_stage_pct",
            "dropoff_from_previous_stage_pct",
        },
    )
    origins = load_output(
        "origin_performance.csv",
        {"origin", "qualified_leads", "closed_deals", "conversion_rate_pct"},
    )
    summary = load_output("business_summary.csv", {"metric", "result"})
    timing = load_output(
        "days_to_close_summary.csv",
        {"average_days_to_close", "valid_nonnegative_date_intervals"},
    )
    segments = load_output(
        "closed_deals_by_business_segment.csv",
        {"business_segment", "closed_deals", "share_of_closed_deals_pct"},
    )
    lead_types = load_output(
        "closed_deals_by_lead_type.csv",
        {"lead_type", "closed_deals", "share_of_closed_deals_pct"},
    )

    qualified_count = int(float(get_metric(summary, "Marketing qualified leads")))
    closed_count = int(float(get_metric(summary, "Closed deals")))
    conversion_pct = float(
        get_metric(summary, "Qualified-lead to closed-deal conversion")
    )
    dropoff_pct = float(get_metric(summary, "Overall funnel drop-off"))
    average_days = float(timing.loc[0, "average_days_to_close"])

    best = origins.loc[origins["origin"].eq("paid_search")]
    if best.empty:
        raise ValueError("Expected identified best origin is missing from origin table.")
    best_rate = float(best.iloc[0]["conversion_rate_pct"])

    # Keep non-identifiable source labels visible, but do not treat them as channels.
    channel_data = origins.copy()
    channel_data["origin"] = channel_data["origin"].replace(
        {"Not provided": "Missing origin"}
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig = plt.figure(figsize=(20, 13.3), facecolor=PAGE)

    # Header strip
    header = fig.add_axes([0, 0.91, 1, 0.09])
    header.set_facecolor(NAVY)
    header.set_xticks([])
    header.set_yticks([])
    for spine in header.spines.values():
        spine.set_visible(False)
    header.text(
        0.075,
        0.62,
        "Marketing Funnel Performance Analysis",
        color=WHITE,
        fontsize=21,
        fontweight="bold",
        va="center",
    )
    header.text(
        0.075,
        0.22,
        "Future Interns – Data Science & Analytics | Task 3",
        color="#CCD7E4",
        fontsize=10,
        va="center",
    )
    header.plot([0.023, 0.055], [0.72, 0.72], color=TEAL, linewidth=4, solid_capstyle="round")
    header.plot([0.023, 0.049], [0.60, 0.60], color="#8ECFC5", linewidth=4, solid_capstyle="round")
    header.plot([0.023, 0.043], [0.48, 0.48], color="#D4E4F0", linewidth=4, solid_capstyle="round")
    header.plot([0.635, 0.635], [0.18, 0.82], color="#7890A8", linewidth=0.8)
    header_stats = [
        ("QUALIFIED LEADS", f"{qualified_count:,}", 0.665),
        ("CLOSED DEALS", f"{closed_count:,}", 0.775),
        ("CONVERSION", f"{conversion_pct:.2f}%", 0.89),
    ]
    for label, value, x in header_stats:
        fig.text(
            x,
            0.973,
            label,
            color="#B8C8D8",
            fontsize=8,
            ha="left",
            va="center",
            fontweight="bold",
            zorder=30,
        )
        fig.text(
            x,
            0.94,
            value,
            color=WHITE,
            fontsize=14,
            ha="left",
            va="center",
            fontweight="bold",
            zorder=30,
        )

    draw_sidebar(fig)

    card_width = 0.154
    positions = [0.165 + index * (card_width + 0.01) for index in range(5)]
    draw_kpi(fig, positions[0], "Qualified leads", f"{qualified_count:,}", BLUE, card_width)
    draw_kpi(fig, positions[1], "Closed deals", f"{closed_count:,}", TEAL, card_width)
    draw_kpi(fig, positions[2], "Conversion rate", f"{conversion_pct:.2f}%", TEAL, card_width)
    draw_kpi(fig, positions[3], "Funnel drop-off", f"{dropoff_pct:.2f}%", RED, card_width)
    draw_kpi(fig, positions[4], "Avg. days to close", f"{average_days:.2f} d", ORANGE, card_width)

    # Upper analysis row
    draw_compact_funnel(
        fig,
        qualified_count,
        closed_count,
        conversion_pct,
        dropoff_pct,
    )
    named_origins = origins.loc[
        ~origins["origin"].isin(["unknown", "Not provided"])
    ]
    best_origin_row = named_origins.sort_values(
        ["conversion_rate_pct", "qualified_leads"], ascending=[False, False]
    ).iloc[0]
    best_origin = str(best_origin_row["origin"])
    best_rate = float(best_origin_row["conversion_rate_pct"])
    organic_row = origins.loc[origins["origin"].eq("organic_search")]
    organic_insight = (
        f"organic_search: {int(organic_row.iloc[0]['qualified_leads']):,} leads "
        f"and {int(organic_row.iloc[0]['closed_deals']):,} closed deals."
        if not organic_row.empty
        else "organic_search results are not available."
    )
    draw_origin_performance(
        fig, channel_data, conversion_pct, best_origin, best_rate
    )

    # Business breakdowns and insights share a uniform second row.
    draw_top_categories(
        fig,
        (0.165, 0.235, 0.255, 0.225),
        segments,
        "business_segment",
        "TOP BUSINESS SEGMENTS",
    )
    draw_top_categories(
        fig,
        (0.43, 0.235, 0.255, 0.225),
        lead_types,
        "lead_type",
        "CLOSED DEALS BY LEAD TYPE",
    )
    draw_insights(
        fig,
        (0.695, 0.235, 0.29, 0.225),
        [
            "842 of 8,000 qualified leads closed (10.53%).",
            f"{best_origin} led identified origins at {best_rate:.2f}%.",
            organic_insight,
            (
                f"{segments.iloc[0]['business_segment']}: "
                f"{int(segments.iloc[0]['closed_deals'])} wins; "
                f"{lead_types.iloc[0]['lead_type']}: "
                f"{int(lead_types.iloc[0]['closed_deals'])} wins."
            ),
            f"Average close time: {average_days:.2f} days.",
        ],
    )

    # Recommendations in matching cards, inspired by the supplied reference.
    recommendation_panel = (0.165, 0.04, 0.82, 0.17)
    add_panel(fig, recommendation_panel)
    panel_heading(
        fig,
        0.183,
        0.195,
        "PRACTICAL RECOMMENDATIONS",
        "Actions linked to the available results",
    )
    draw_recommendation_cards(
        fig,
        [
            "Review lead quality and follow-up for other, email, and other publicity.",
            "Learn from paid_search and test carefully using conversion and win counts.",
            "Build on organic_search volume and keep monitoring conversion.",
            "Capture lead stages, contact activity, origin, and marketing spend.",
        ],
    )

    fig.text(
        0.165,
        0.022,
        "Source: generated Olist output CSVs  |  Two observed funnel stages; intermediate stages and spend are not available.",
        fontsize=8,
        color=MUTED,
        va="center",
        zorder=30,
    )
    for text_artist in fig.texts:
        text_artist.set_zorder(30)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        DASHBOARD_PATH,
        dpi=300,
        facecolor=PAGE,
        bbox_inches=None,
        metadata={
            "Title": "Marketing Funnel Performance Analysis",
            "Description": "Single-page Olist marketing funnel dashboard based on generated analysis outputs.",
        },
    )
    plt.close(fig)
    print(f"Dashboard saved: {DASHBOARD_PATH}")
    print(f"Image size: {20 * 300} x {13.3 * 300} pixels at 300 DPI")


if __name__ == "__main__":
    main()
