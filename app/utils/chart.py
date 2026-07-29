"""PNG chart rendering with an optional matplotlib/seaborn enhancement.

Plotting libraries are imported only when a chart is requested.  If they are
missing or cannot initialize on the host, a small standard-library renderer
still produces a useful PNG instead of making the web application fail.
"""

from __future__ import annotations

import bisect
import io
import logging
import math
import struct
import zlib
from collections.abc import Sequence

logger = logging.getLogger(__name__)

_COLORS = (
    (37, 99, 235),
    (220, 38, 38),
    (22, 163, 74),
    (245, 158, 11),
    (124, 58, 237),
    (8, 145, 178),
    (219, 39, 119),
    (107, 114, 128),
)

# Compact fallback font.  Matplotlib remains the preferred renderer, but these
# glyphs keep the dependency-free chart self-describing.
_FONT_5X7 = {
    " ": ("00000",) * 7,
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _chart_data(
    summary_rows: Sequence[Sequence[object]],
) -> list[tuple[str, float, float]]:
    raw: list[tuple[str, float, float | None]] = []
    for row in summary_rows[1:]:
        value = _numeric_value(row[1] if len(row) > 1 else None)
        share = _numeric_value(row[2] if len(row) > 2 else None)
        if value is not None and value > 0:
            raw.append((str(row[0] if row else "Unknown"), value, share))

    total = sum(item[1] for item in raw)
    if total <= 0:
        return []
    return [
        (label, value, share if share is not None and share >= 0 else value / total)
        for label, value, share in raw
    ]


def seaborn_pie_summary_to_png(
    summary_rows: Sequence[Sequence[object]],
) -> bytes:
    """Render a polished donut chart using optional plotting dependencies."""

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ImportError:
        sns = None

    data = _chart_data(summary_rows)
    if sns is not None:
        sns.set_theme(style="white", context="talk")
        colors = sns.color_palette("Set2", n_colors=max(len(data), 3))
    else:
        colors = plt.colormaps["Set2"].colors

    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
    try:
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        if not data:
            ax.text(
                0.5,
                0.5,
                "No chart data",
                ha="center",
                va="center",
                fontsize=22,
                weight="bold",
                color="#b91c1c",
            )
            ax.axis("off")
        else:
            labels = [item[0] for item in data]
            values = [item[1] for item in data]
            shares = [item[2] for item in data]
            legend_labels = [
                f"{label}  {share * 100:.1f}% ({value:,.1f} kg CO2e)"
                for label, value, share in zip(labels, values, shares)
            ]
            wedges, _texts, autotexts = ax.pie(
                values,
                labels=None,
                autopct=lambda percent: f"{percent:.1f}%" if percent >= 3 else "",
                startangle=90,
                counterclock=False,
                pctdistance=0.72,
                colors=colors,
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
                textprops={"color": "#111827", "fontsize": 12, "weight": "bold"},
            )
            for text in autotexts:
                text.set_color("#111827")
                text.set_fontsize(11)
                text.set_weight("bold")

            ax.add_artist(plt.Circle((0, 0), 0.45, fc="white", ec="white"))
            ax.legend(
                wedges,
                legend_labels,
                title="Material contribution",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=11,
                title_fontsize=13,
            )
            ax.axis("equal")

        fig.suptitle(
            "GWP Contribution by Material",
            fontsize=24,
            weight="bold",
            color="#111827",
        )
        fig.text(
            0.5,
            0.93,
            "Materials below 5% contribution are grouped into Other",
            ha="center",
            fontsize=13,
            color="#6b7280",
        )
        fig.tight_layout(rect=(0.02, 0.02, 0.84, 0.90))

        output = io.BytesIO()
        fig.savefig(
            output,
            format="png",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        return output.getvalue()
    finally:
        plt.close(fig)


def _put_pixel(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if not (0 <= x < width and 0 <= y < height):
        return
    offset = (y * width + x) * 3
    pixels[offset : offset + 3] = bytes(color)


def _fill_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    rect_width: int,
    rect_height: int,
    color: tuple[int, int, int],
) -> None:
    start_x = max(x, 0)
    end_x = min(x + rect_width, width)
    start_y = max(y, 0)
    end_y = min(y + rect_height, height)
    if start_x >= end_x or start_y >= end_y:
        return
    row_data = bytes(color) * (end_x - start_x)
    for py in range(start_y, end_y):
        offset = (py * width + start_x) * 3
        pixels[offset : offset + len(row_data)] = row_data


def _draw_text(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: object,
    color: tuple[int, int, int] = (31, 41, 55),
    scale: int = 2,
) -> None:
    cursor_x = x
    for character in str(text).upper():
        glyph = _FONT_5X7.get(character, _FONT_5X7[" "])
        for row_index, row in enumerate(glyph):
            for column_index, bit in enumerate(row):
                if bit == "1":
                    _fill_rect(
                        pixels,
                        width,
                        height,
                        cursor_x + column_index * scale,
                        y + row_index * scale,
                        scale,
                        scale,
                        color,
                    )
        cursor_x += 6 * scale


def _png_bytes(
    width: int,
    height: int,
    pixels: bytes | bytearray,
) -> bytes:
    expected_length = width * height * 3
    if len(pixels) != expected_length:
        raise ValueError(
            f"RGB pixel data must contain {expected_length} bytes, got {len(pixels)}"
        )

    scanlines = bytearray()
    row_length = width * 3
    for y in range(height):
        scanlines.append(0)
        start = y * row_length
        scanlines.extend(pixels[start : start + row_length])

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", checksum)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(scanlines), 9))
        + chunk(b"IEND", b"")
    )


def fallback_pie_summary_to_png(
    summary_rows: Sequence[Sequence[object]],
) -> bytes:
    """Render a dependency-free donut chart as an RGB PNG."""

    width, height = 1000, 700
    pixels = bytearray(b"\xff") * (width * height * 3)
    data = _chart_data(summary_rows)

    _draw_text(
        pixels,
        width,
        height,
        40,
        30,
        "GWP CONTRIBUTION",
        (17, 24, 39),
        scale=3,
    )
    _draw_text(
        pixels,
        width,
        height,
        40,
        68,
        "ITEMS BELOW 5% GROUPED AS OTHER",
        (75, 85, 99),
        scale=2,
    )

    if not data:
        _draw_text(
            pixels,
            width,
            height,
            40,
            140,
            "NO CHART DATA",
            (220, 38, 38),
            scale=3,
        )
        return _png_bytes(width, height, pixels)

    total = sum(item[1] for item in data)
    cumulative: list[float] = []
    running = 0.0
    for _label, value, _share in data:
        running += value / total
        cumulative.append(running)
    cumulative[-1] = 1.0

    center_x, center_y = 300, 370
    outer_radius, inner_radius = 220, 96
    outer_squared = outer_radius * outer_radius
    inner_squared = inner_radius * inner_radius

    for y in range(center_y - outer_radius, center_y + outer_radius + 1):
        for x in range(center_x - outer_radius, center_x + outer_radius + 1):
            dx = x - center_x
            dy = y - center_y
            distance_squared = dx * dx + dy * dy
            if distance_squared > outer_squared:
                continue
            if distance_squared < inner_squared:
                _put_pixel(pixels, width, height, x, y, (255, 255, 255))
                continue
            # Start at twelve o'clock and progress clockwise.
            fraction = ((math.atan2(dy, dx) + math.pi / 2) % math.tau) / math.tau
            color_index = min(
                bisect.bisect_left(cumulative, fraction),
                len(data) - 1,
            )
            _put_pixel(
                pixels,
                width,
                height,
                x,
                y,
                _COLORS[color_index % len(_COLORS)],
            )

    legend_x = 570
    row_height = max(26, min(48, 500 // max(len(data), 1)))
    for index, (material, _value, share) in enumerate(data):
        y = 125 + index * row_height
        if y + 20 >= height:
            break
        color = _COLORS[index % len(_COLORS)]
        _fill_rect(pixels, width, height, legend_x, y, 18, 18, color)
        label = f"{material[:42]} {share * 100:.1f}%"
        _draw_text(
            pixels,
            width,
            height,
            legend_x + 28,
            y + 4,
            label,
            (31, 41, 55),
            scale=1,
        )

    return _png_bytes(width, height, pixels)


def pie_summary_to_png(
    summary_rows: Sequence[Sequence[object]],
    *,
    prefer_matplotlib: bool = True,
) -> bytes:
    """Return a PNG contribution chart.

    ``prefer_matplotlib=False`` is useful for constrained deployments and
    deterministic tests.  The public default retains the richer source-app
    chart when optional plotting packages are available.
    """

    if prefer_matplotlib:
        try:
            return seaborn_pie_summary_to_png(summary_rows)
        except Exception as exc:  # plotting is optional; the fallback is deliberate
            logger.warning(
                "matplotlib/seaborn chart unavailable; using fallback PNG: %s",
                exc,
            )
    return fallback_pie_summary_to_png(summary_rows)


__all__ = [
    "fallback_pie_summary_to_png",
    "pie_summary_to_png",
    "seaborn_pie_summary_to_png",
]
