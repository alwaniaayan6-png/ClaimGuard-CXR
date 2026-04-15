"""Publication-quality comparison figure for ClaimGuard-CXR poster.

Generates a 4-row by 4-column comparison table positioning ClaimGuard-CXR
against two prior radiology verification approaches (Conformal Alignment
and VeriFact) along four evaluation criteria: granularity, error control,
evidence conditioning, and cross-dataset transfer.

Style: two-color palette (navy + teal) with a single red accent reserved
for negative indicators. No gradients, drop shadows, or decorative icons.
Designed to look like a figure from a Nature Machine Intelligence or
NeurIPS paper, not a marketing infographic.

Outputs (saved to the current working directory):
    claimguard_comparison.pdf  (vector, for paper figures)
    claimguard_comparison.png  (300 DPI raster, for posters and slides)

Usage:
    python claimguard_comparison_figure.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# ---- Color palette ----
NAVY = "#1e3a5f"
TEAL = "#2a9d8f"
RED = "#c0392b"


def main() -> None:
    """Build and save the comparison figure as PDF and PNG."""
    # Typography: prefer Helvetica/Arial, fall back to DejaVu Sans.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Helvetica",
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ]
    # Embed TrueType fonts in PDF/PS so the figure is editable downstream.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # ---- Figure setup ----
    fig_width = 14.0
    fig_height = 5.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Layout (axis coordinates equal inches because of set_aspect) ----
    margin_x = 0.4
    margin_top = 0.3
    header_height = 1.2
    data_row_height = 0.8
    n_data_rows = 4

    # Column x-boundaries: criteria column, then three approach columns.
    crit_col_width = 2.8
    data_col_width = 3.47
    col_boundaries = [
        margin_x,
        margin_x + crit_col_width,
        margin_x + crit_col_width + data_col_width,
        margin_x + crit_col_width + 2 * data_col_width,
        fig_width - margin_x,
    ]

    # Row y-boundaries (top-down).
    top = fig_height - margin_top
    row_boundaries = [top]
    row_boundaries.append(top - header_height)
    for _ in range(n_data_rows):
        row_boundaries.append(row_boundaries[-1] - data_row_height)

    table_top = row_boundaries[0]
    table_bottom = row_boundaries[-1]

    # ---- ClaimGuard-CXR column subtle fill (border drawn last) ----
    cg_x = col_boundaries[3]
    cg_w = col_boundaries[4] - col_boundaries[3]
    cg_y = table_bottom
    cg_h = table_top - table_bottom

    ax.add_patch(Rectangle(
        (cg_x, cg_y), cg_w, cg_h,
        facecolor=TEAL, alpha=0.08,
        edgecolor="none",
        zorder=0.5,
    ))

    # ---- Thin grid lines ----
    for x in col_boundaries:
        ax.plot(
            [x, x], [table_bottom, table_top],
            color=NAVY, linewidth=0.5, zorder=2,
        )
    for y in row_boundaries:
        ax.plot(
            [col_boundaries[0], col_boundaries[-1]], [y, y],
            color=NAVY, linewidth=0.5, zorder=2,
        )

    # ---- Header row ----
    approach_names = [
        "",
        "Report-level conformal",
        "LLM-as-judge",
        "ClaimGuard-CXR",
    ]
    approach_refs = [
        "",
        "(Conformal Alignment, NeurIPS 2024)",
        "(VeriFact, NEJM AI 2025)",
        "(ours)",
    ]
    granularity_subtitles = [
        "",
        "report",
        "claim",
        "claim + conformal",
    ]

    header_top = row_boundaries[0]
    for col in range(1, 4):
        cx = (col_boundaries[col] + col_boundaries[col + 1]) / 2
        ax.text(
            cx, header_top - 0.32, approach_names[col],
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=NAVY,
        )
        ax.text(
            cx, header_top - 0.62, approach_refs[col],
            ha="center", va="center",
            fontsize=8, color=NAVY,
        )
        ax.text(
            cx, header_top - 0.94, granularity_subtitles[col],
            ha="center", va="center",
            fontsize=9, color=NAVY, fontstyle="italic",
        )

    # ---- Data rows ----
    criteria = [
        "Granularity",
        "Error control",
        "Evidence conditioning",
        "Cross-dataset transfer",
    ]

    # Each cell: (indicator, text)
    #   indicator = "yes" -> teal filled circle
    #   indicator = "no"  -> red open circle
    #   indicator = None  -> text only
    cells = [
        # Row 1: Granularity
        [
            ("no", "report-level"),
            ("yes", "claim-level"),
            ("yes", "claim-level"),
        ],
        # Row 2: Error control
        [
            ("yes", "FDR ≤ α"),
            ("no", "none"),
            ("yes", "FDR ≤ α"),
        ],
        # Row 3: Evidence conditioning
        [
            ("no", "no external evidence"),
            ("no", "self-consistency only"),
            ("yes", "retrieved evidence"),
        ],
        # Row 4: Cross-dataset transfer
        [
            ("no", "not evaluated"),
            ("no", "not evaluated"),
            ("yes", "FDR preserved on OpenI"),
        ],
    ]

    for row_idx in range(n_data_rows):
        y_top = row_boundaries[1 + row_idx]
        y_bottom = row_boundaries[2 + row_idx]
        y_mid = (y_top + y_bottom) / 2

        # Criteria label in leftmost column.
        crit_cx = (col_boundaries[0] + col_boundaries[1]) / 2
        ax.text(
            crit_cx, y_mid, criteria[row_idx],
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=NAVY,
        )

        # Data cells for the three approach columns.
        for col_idx in range(3):
            x_left = col_boundaries[col_idx + 1]
            x_right = col_boundaries[col_idx + 2]
            indicator, text = cells[row_idx][col_idx]

            if indicator == "yes":
                cx = x_left + 0.38
                ax.add_patch(Circle(
                    (cx, y_mid), 0.09,
                    facecolor=TEAL, edgecolor=TEAL,
                    linewidth=1.2, zorder=4,
                ))
                ax.text(
                    cx + 0.22, y_mid, text,
                    ha="left", va="center",
                    fontsize=10, color=NAVY,
                )
            elif indicator == "no":
                cx = x_left + 0.38
                ax.add_patch(Circle(
                    (cx, y_mid), 0.09,
                    facecolor="white", edgecolor=RED,
                    linewidth=1.5, zorder=4,
                ))
                ax.text(
                    cx + 0.22, y_mid, text,
                    ha="left", va="center",
                    fontsize=10, color=NAVY,
                )
            else:
                mid = (x_left + x_right) / 2
                ax.text(
                    mid, y_mid, text,
                    ha="center", va="center",
                    fontsize=10, color=NAVY,
                )

    # ---- ClaimGuard-CXR column border on top of everything ----
    ax.add_patch(Rectangle(
        (cg_x, cg_y), cg_w, cg_h,
        facecolor="none",
        edgecolor=NAVY, linewidth=1.5,
        zorder=10,
    ))

    # ---- Save outputs ----
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(
        "claimguard_comparison.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
    )
    fig.savefig(
        "claimguard_comparison.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
    )
    print("Saved: claimguard_comparison.pdf")
    print("Saved: claimguard_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
