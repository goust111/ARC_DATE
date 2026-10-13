from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
DRAW_COLORS = (1, 2, 3, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _separator_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert rows.size == 1, "expected exactly one solid grey separator row"
    return int(rows[0])


def _parse_bands(top: np.ndarray) -> list[dict[str, object]]:
    bands: list[dict[str, object]] = []
    for r in range(top.shape[0]):
        markers = np.flatnonzero(top[r] == YELLOW)
        if markers.size == 0:
            continue
        assert markers.size == 1, "each rule row must contain exactly one yellow marker"
        marker_c = int(markers[0])
        start_c = marker_c + 1
        assert start_c < top.shape[1], "yellow marker must sit immediately left of a band"
        assert int(top[r, start_c]) != BLACK, "band must start right after its yellow marker"

        seq: list[int] = []
        c = start_c
        while c < top.shape[1] and int(top[r, c]) != BLACK:
            color = int(top[r, c])
            assert color not in (BLACK, YELLOW, GREY), "band cells must use draw colors"
            seq.append(color)
            c += 1

        assert len(seq) >= 2, "each band must encode at least two weaving steps"
        palette = sorted(set(seq))
        assert len(palette) == 2, "each band must alternate exactly two colors"
        assert all(seq[i] != seq[i + 1] for i in range(len(seq) - 1)), "band must alternate colors"
        bands.append({"row": r, "seq": seq, "colors": set(palette)})

    assert bands, "expected at least one rule band"
    return bands


def _find_wefts(bottom: np.ndarray) -> list[dict[str, int]]:
    wefts: list[dict[str, int]] = []
    for r in range(bottom.shape[0]):
        cols = np.flatnonzero(bottom[r] != BLACK)
        if cols.size < 2:
            continue
        if not np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)):
            continue
        colors = np.unique(bottom[r, cols])
        if colors.size != 1:
            continue
        wefts.append(
            {
                "row": int(r),
                "color": int(colors[0]),
                "start": int(cols[0]),
                "end": int(cols[-1]),
            }
        )

    assert len(wefts) >= 2, "expected at least two horizontal wefts"
    spans = {(weft["start"], weft["end"]) for weft in wefts}
    assert len(spans) == 1, "all wefts must share one horizontal weaving span"
    colors = [weft["color"] for weft in wefts]
    assert len(colors) == len(set(colors)), "weft colors must be unique"
    return wefts


def _find_warps(bottom: np.ndarray, wefts: list[dict[str, int]]) -> list[dict[str, int]]:
    span_start = wefts[0]["start"]
    span_end = wefts[0]["end"]
    row_lo = min(weft["row"] for weft in wefts)
    row_hi = max(weft["row"] for weft in wefts)
    weft_rows = {weft["row"] for weft in wefts}

    warps: list[dict[str, int]] = []
    for c in range(span_start, span_end + 1):
        segment = bottom[row_lo : row_hi + 1, c]
        if not np.all(segment != BLACK):
            continue

        non_intersection = [int(bottom[r, c]) for r in range(row_lo, row_hi + 1) if r not in weft_rows]
        assert non_intersection, "warp color must be visible between wefts"
        warp_color = non_intersection[0]
        assert all(color == warp_color for color in non_intersection), "warp must be monochrome"
        warps.append({"col": int(c), "color": int(warp_color)})

    assert len(warps) >= 2, "expected at least two vertical warps"
    warp_colors = [warp["color"] for warp in warps]
    assert len(warp_colors) == len(set(warp_colors)), "warp colors must be unique"
    return warps


def _band_for_weft_color(bands: list[dict[str, object]], weft_color: int) -> dict[str, object]:
    matches = [band for band in bands if weft_color in band["colors"]]
    assert len(matches) == 1, "each weft color must identify one unique rule band"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each weaving intersection by consulting the matching upper alternating band.

    Input structure:
    - The grid has a black background and one full grey row splitting it into an upper rule panel and a lower weaving panel.
    - In the upper panel, each non-empty row contains one yellow marker pixel immediately left of a horizontal two-color alternating band.
    - In the lower panel, single-pixel horizontal wefts and vertical warps cross each other, and every input intersection currently shows the weft color.

    Rule:
    - Keep the separator row, the upper rule panel, and every non-intersection line pixel unchanged.
    - For each weft, find the unique upper band that contains that weft's color among its two alternating colors.
    - Visit that weft's intersections from left to right. If the matching band color at the same step equals the weft color, keep the intersection as the weft; otherwise replace it with the corresponding warp color.

    Key insight:
    - The upper alternating bands are lookup tables for over-under weaving order, and the matching key is the weft color itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    bands = _parse_bands(g[:sep])
    bottom = g[sep + 1 :]
    wefts = _find_wefts(bottom)
    warps = _find_warps(bottom, wefts)

    out = g.copy()
    global_row_offset = sep + 1
    for weft in wefts:
        band = _band_for_weft_color(bands, weft["color"])
        seq = band["seq"]
        assert len(seq) >= len(warps), "rule band must be long enough for all intersections"
        for step, warp in enumerate(warps):
            if int(seq[step]) != weft["color"]:
                out[global_row_offset + weft["row"], warp["col"]] = warp["color"]

    return out


def _alternating_sequence(primary: int, secondary: int, length: int, start_with_primary: bool) -> list[int]:
    return [
        primary if (idx % 2 == 0) == start_with_primary else secondary
        for idx in range(length)
    ]


def _band_rows(count: int) -> tuple[list[int], int]:
    first = random.randint(0, 1)
    rows = [first]
    for _ in range(count - 1):
        rows.append(rows[-1] + random.randint(1, 2))
    top_height = rows[-1] + 1 + random.randint(0, 2)
    return rows, top_height


def _weft_rows(count: int) -> tuple[list[int], int]:
    first = random.randint(1, 3)
    rows = [first]
    for _ in range(count - 1):
        rows.append(rows[-1] + random.randint(2, 4))
    bottom_height = rows[-1] + 1 + random.randint(1, 3)
    return rows, bottom_height


def _warp_columns(count: int) -> list[int]:
    first = random.randint(1, 3)
    cols = [first]
    for _ in range(count - 1):
        cols.append(cols[-1] + random.randint(2, 4))
    return cols


def _expected_from_layout(
    grid: np.ndarray,
    sep: int,
    weft_rows: list[int],
    weft_colors: list[int],
    warp_cols: list[int],
    warp_colors: list[int],
    sequences: list[list[int]],
) -> np.ndarray:
    out = grid.copy()
    offset = sep + 1
    for row, weft_color, seq in zip(weft_rows, weft_colors, sequences):
        for step, (col, warp_color) in enumerate(zip(warp_cols, warp_colors)):
            if seq[step] != weft_color:
                out[offset + row, col] = warp_color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_wefts = random.randint(2, 4)
    max_warps = min(5, len(DRAW_COLORS) - n_wefts)
    n_warps = random.randint(2, max_warps)

    weft_colors = random.sample(DRAW_COLORS, n_wefts)
    remaining = [color for color in DRAW_COLORS if color not in weft_colors]
    warp_colors = random.sample(remaining, n_warps)
    alt_colors = [random.choice(remaining) for _ in range(n_wefts)]

    band_rows, top_height = _band_rows(n_wefts)
    weft_rows, bottom_height = _weft_rows(n_wefts)
    warp_cols = _warp_columns(n_warps)

    band_starts = [random.randint(1, 4) for _ in range(n_wefts)]
    band_ends = [start + n_warps for start in band_starts]

    left_tail = random.randint(0, 2)
    right_tail = random.randint(0, 2)
    weave_start = max(0, warp_cols[0] - left_tail)
    weave_end = warp_cols[-1] + right_tail

    width = max(max(band_ends, default=0), weave_end + 1 + random.randint(1, 3))
    height = top_height + 1 + bottom_height
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    sep = top_height
    grid[sep, :] = GREY

    sequences: list[list[int]] = []
    for row, start, weft_color, alt_color in zip(band_rows, band_starts, weft_colors, alt_colors):
        start_with_weft = bool(random.getrandbits(1))
        seq = _alternating_sequence(weft_color, alt_color, n_warps, start_with_weft)
        sequences.append(seq)
        grid[row, start - 1] = YELLOW
        grid[row, start : start + n_warps] = seq

    bottom_offset = sep + 1
    warp_top = max(0, weft_rows[0] - random.randint(0, 1))
    warp_bottom = min(bottom_height - 1, weft_rows[-1] + random.randint(0, 1))
    for col, color in zip(warp_cols, warp_colors):
        grid[bottom_offset + warp_top : bottom_offset + warp_bottom + 1, col] = color

    for row, color in zip(weft_rows, weft_colors):
        grid[bottom_offset + row, weave_start : weave_end + 1] = color

    expected = _expected_from_layout(grid, sep, weft_rows, weft_colors, warp_cols, warp_colors, sequences)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving input")
