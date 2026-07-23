#!/usr/bin/env python3
"""Reproduce the synthetic PR-review analysis from the accompanying workbook.

This script treats the ``PR Data`` and ``Review Events`` sheets as the raw
tables. It recomputes all durations and classifications instead of relying on
formula cells cached inside Excel.

Usage:
    python analyze_pr_reviews.py synthetic_pr_review_model.xlsx \
        --output analysis_output

Outputs:
    headline_metrics.csv
    weekly_metrics.csv
    metrics_by_pr_size.csv
    metrics_by_ownership_areas.csv
    metrics_by_reviewer_workload.csv
    reviewer_load.csv
    analysis_summary.json
    weekly_queue_trend.png
    review_bottleneck_segments.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PR_SHEET = "PR Data"
REVIEW_SHEET = "Review Events"

PR_REQUIRED_COLUMNS = {
    "pr_id",
    "ready_for_review_at",
    "first_review_at",
    "merged_at",
    "additions",
    "deletions",
    "ownership_areas_touched",
    "open_review_requests_at_assignment",
    "reverted_within_14_days",
    "linked_incident",
    "status",
}

REVIEW_REQUIRED_COLUMNS = {
    "review_id",
    "pr_id",
    "reviewer_id",
    "reviewer_requested_at",
    "review_submitted_at",
    "review_state",
}

SIZE_ORDER = ["Small", "Medium", "Large", "XL"]
WORKLOAD_ORDER = ["0-2", "3-5", "6-8", "9+"]

NAVY = "#14213D"
TEAL = "#0E7490"
ORANGE = "#F59E0B"
LIGHT_GRID = "#D1D5DB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the synthetic PR-review workbook."
    )
    parser.add_argument("workbook", type=Path, help="Path to the .xlsx workbook")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_output"),
        help="Directory for generated tables and charts",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, expected: set[str], sheet_name: str) -> None:
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise ValueError(f"{sheet_name!r} is missing required columns: {missing}")


def load_and_prepare(workbook_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not workbook_path.exists():
        raise FileNotFoundError(workbook_path)

    prs = pd.read_excel(workbook_path, sheet_name=PR_SHEET)
    reviews = pd.read_excel(workbook_path, sheet_name=REVIEW_SHEET)
    require_columns(prs, PR_REQUIRED_COLUMNS, PR_SHEET)
    require_columns(reviews, REVIEW_REQUIRED_COLUMNS, REVIEW_SHEET)

    pr_dates = [
        "ready_for_review_at",
        "first_review_at",
        "merged_at",
        "created_at",
        "closed_at",
    ]
    for column in pr_dates:
        if column in prs:
            prs[column] = pd.to_datetime(prs[column], errors="coerce", utc=True)

    review_dates = ["reviewer_requested_at", "review_submitted_at"]
    for column in review_dates:
        reviews[column] = pd.to_datetime(reviews[column], errors="coerce", utc=True)

    numeric_pr_columns = [
        "additions",
        "deletions",
        "ownership_areas_touched",
        "open_review_requests_at_assignment",
        "reverted_within_14_days",
        "linked_incident",
    ]
    for column in numeric_pr_columns:
        prs[column] = pd.to_numeric(prs[column], errors="coerce")

    prs["review_wait_hours"] = (
        prs["first_review_at"] - prs["ready_for_review_at"]
    ).dt.total_seconds() / 3600
    prs["ready_to_merge_hours"] = (
        prs["merged_at"] - prs["ready_for_review_at"]
    ).dt.total_seconds() / 3600
    prs["total_lines_changed"] = prs["additions"] + prs["deletions"]
    prs["size_bucket"] = pd.cut(
        prs["total_lines_changed"],
        bins=[-np.inf, 150, 500, 1000, np.inf],
        labels=SIZE_ORDER,
        ordered=True,
    )
    prs["wait_over_24h"] = prs["review_wait_hours"] > 24
    prs["cross_domain"] = prs["ownership_areas_touched"] > 1
    prs["merged"] = prs["status"].eq("Merged")
    prs["rollback"] = prs["reverted_within_14_days"].fillna(0).astype(int).eq(1)
    prs["linked_incident_flag"] = prs["linked_incident"].fillna(0).astype(int).eq(1)
    prs["workload_bucket"] = pd.cut(
        prs["open_review_requests_at_assignment"],
        bins=[-np.inf, 2, 5, 8, np.inf],
        labels=WORKLOAD_ORDER,
        ordered=True,
    )

    # Convert to timezone-naive dates only after durations are calculated.
    ready_naive = prs["ready_for_review_at"].dt.tz_convert(None)
    prs["week_start"] = ready_naive.dt.to_period("W-SUN").dt.start_time

    if prs["review_wait_hours"].isna().any():
        count = int(prs["review_wait_hours"].isna().sum())
        raise ValueError(f"{count} PR rows have missing review-wait timestamps")

    return prs, reviews


def headline_metrics(prs: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    reviewer_counts = reviews["reviewer_id"].value_counts()
    top_reviewer_count = max(1, math.ceil(len(reviewer_counts) * 0.20))
    top_20_share = reviewer_counts.iloc[:top_reviewer_count].sum() / reviewer_counts.sum()

    merged = prs.loc[prs["merged"]]
    rows = [
        ("PRs ready for review", len(prs), "count"),
        ("Review events", len(reviews), "count"),
        ("Median first-review wait", prs["review_wait_hours"].median(), "hours"),
        ("Average first-review wait", prs["review_wait_hours"].mean(), "hours"),
        ("Waiting over 24 hours", prs["wait_over_24h"].mean() * 100, "percent"),
        ("Median ready-to-merge", merged["ready_to_merge_hours"].median(), "hours"),
        ("Top 20% reviewer share", top_20_share * 100, "percent"),
        ("14-day rollback rate", merged["rollback"].mean() * 100, "percent"),
        ("Linked-incident rate", merged["linked_incident_flag"].mean() * 100, "percent"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit"])


def weekly_metrics(prs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for week_start, group in prs.groupby("week_start", sort=True):
        merged = group.loc[group["merged"]]
        records.append(
            {
                "week_start": week_start,
                "prs_ready": len(group),
                "average_review_wait_hours": group["review_wait_hours"].mean(),
                "median_review_wait_hours": group["review_wait_hours"].median(),
                "waiting_over_24h_pct": group["wait_over_24h"].mean() * 100,
                "average_ready_to_merge_hours": merged[
                    "ready_to_merge_hours"
                ].mean(),
                "cross_domain_prs_pct": group["cross_domain"].mean() * 100,
            }
        )
    return pd.DataFrame(records)


def segment_metrics(
    prs: pd.DataFrame, column: str, order: list[object] | None = None
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    groups = prs.groupby(column, observed=True, sort=False)
    for value, group in groups:
        merged = group.loc[group["merged"]]
        records.append(
            {
                column: str(value),
                "pr_count": len(group),
                "average_review_wait_hours": group["review_wait_hours"].mean(),
                "median_review_wait_hours": group["review_wait_hours"].median(),
                "waiting_over_24h_pct": group["wait_over_24h"].mean() * 100,
                "average_ready_to_merge_hours": merged[
                    "ready_to_merge_hours"
                ].mean(),
                "rollback_rate_pct": merged["rollback"].mean() * 100,
            }
        )

    result = pd.DataFrame(records)
    if order:
        order_map = {str(value): index for index, value in enumerate(order)}
        result["_sort"] = result[column].map(order_map)
        result = result.sort_values("_sort").drop(columns="_sort")
    return result.reset_index(drop=True)


def reviewer_load_metrics(reviews: pd.DataFrame) -> pd.DataFrame:
    result = (
        reviews.groupby("reviewer_id")
        .agg(
            review_events=("review_id", "count"),
            distinct_prs=("pr_id", "nunique"),
            approvals=("review_state", lambda values: values.eq("approved").sum()),
            changes_requested=(
                "review_state",
                lambda values: values.eq("changes_requested").sum(),
            ),
        )
        .sort_values("review_events", ascending=False)
        .reset_index()
    )
    result["share_of_review_events_pct"] = (
        result["review_events"] / result["review_events"].sum() * 100
    )
    result["cumulative_share_pct"] = result["share_of_review_events_pct"].cumsum()
    result.insert(0, "reviewer_rank", np.arange(1, len(result) + 1))
    result["top_20pct_reviewer"] = result["reviewer_rank"] <= math.ceil(
        len(result) * 0.20
    )
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(LIGHT_GRID)
    axis.spines["bottom"].set_color(LIGHT_GRID)
    axis.grid(axis="y", color=LIGHT_GRID, linewidth=0.8, alpha=0.65)
    axis.set_axisbelow(True)


def plot_weekly_trend(weekly: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    labels = weekly["week_start"].dt.strftime("%b %-d")

    axes[0].plot(labels, weekly["prs_ready"], color=TEAL, linewidth=2.5, marker="o")
    axes[0].set_title("PR throughput rises across the synthetic quarter", color=NAVY)
    axes[0].set_ylabel("PRs ready")
    style_axis(axes[0])

    axes[1].plot(
        labels,
        weekly["average_review_wait_hours"],
        color=ORANGE,
        linewidth=2.5,
        marker="o",
    )
    axes[1].set_title("Human-review wait grows with the queue", color=NAVY)
    axes[1].set_ylabel("Average wait (hours)")
    axes[1].tick_params(axis="x", rotation=40)
    style_axis(axes[1])

    fig.suptitle("Synthetic PR Review Capacity Model", fontsize=15, color=NAVY)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_segments(
    size: pd.DataFrame, workload: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].bar(
        size["size_bucket"], size["average_review_wait_hours"], color=TEAL
    )
    axes[0].set_title("Larger PRs wait longer", color=NAVY)
    axes[0].set_ylabel("Average wait (hours)")
    style_axis(axes[0])

    axes[1].bar(
        workload["workload_bucket"],
        workload["average_review_wait_hours"],
        color=ORANGE,
    )
    axes[1].set_title("Reviewer backlog predicts longer waits", color=NAVY)
    axes[1].set_ylabel("Average wait (hours)")
    style_axis(axes[1])

    fig.suptitle("Where the synthetic review bottleneck appears", fontsize=15, color=NAVY)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def round_numeric_columns(frame: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=["number"]).columns
    result[numeric_columns] = result[numeric_columns].round(decimals)
    return result


def save_analysis(
    output_dir: Path,
    headlines: pd.DataFrame,
    weekly: pd.DataFrame,
    by_size: pd.DataFrame,
    by_areas: pd.DataFrame,
    by_workload: pd.DataFrame,
    reviewer_load: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    round_numeric_columns(headlines).to_csv(
        output_dir / "headline_metrics.csv", index=False
    )
    round_numeric_columns(weekly).to_csv(
        output_dir / "weekly_metrics.csv", index=False, date_format="%Y-%m-%d"
    )
    round_numeric_columns(by_size).to_csv(
        output_dir / "metrics_by_pr_size.csv", index=False
    )
    round_numeric_columns(by_areas).to_csv(
        output_dir / "metrics_by_ownership_areas.csv", index=False
    )
    round_numeric_columns(by_workload).to_csv(
        output_dir / "metrics_by_reviewer_workload.csv", index=False
    )
    round_numeric_columns(reviewer_load).to_csv(
        output_dir / "reviewer_load.csv", index=False
    )

    summary = {
        row.metric: {"value": round(float(row.value), 6), "unit": row.unit}
        for row in headlines.itertuples(index=False)
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "disclosure": (
                    "Synthetic, assumption-driven scenario. Not observed Persona or "
                    "company data."
                ),
                "headline_metrics": summary,
            },
            file,
            indent=2,
        )

    plot_weekly_trend(weekly, output_dir / "weekly_queue_trend.png")
    plot_segments(by_size, by_workload, output_dir / "review_bottleneck_segments.png")


def main() -> None:
    args = parse_args()
    prs, reviews = load_and_prepare(args.workbook)

    headlines = headline_metrics(prs, reviews)
    weekly = weekly_metrics(prs)
    by_size = segment_metrics(prs, "size_bucket", SIZE_ORDER)
    by_areas = segment_metrics(
        prs, "ownership_areas_touched", sorted(prs["ownership_areas_touched"].dropna().unique())
    )
    by_workload = segment_metrics(prs, "workload_bucket", WORKLOAD_ORDER)
    reviewer_load = reviewer_load_metrics(reviews)

    save_analysis(
        args.output,
        headlines,
        weekly,
        by_size,
        by_areas,
        by_workload,
        reviewer_load,
    )

    print("Synthetic PR-review analysis complete.\n")
    print(headlines.to_string(index=False, formatters={"value": "{:,.3f}".format}))
    print(f"\nOutputs written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
