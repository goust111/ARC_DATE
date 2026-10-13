from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_BAND_GAP = 3
MIN_ARROW_GAP = 4
MAX_GENERATE_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _sample_spaced_positions(
    rng: random.Random, low: int, high: int, count: int, min_gap: int
) -> list[int] | None:
    if high < low or count <= 0:
        return None
    candidates = list(range(low, high + 1))
    for _ in range(200):
        picks = sorted(rng.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _detect_vertical_bands(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    bands: dict[int, int] = {}
    for c in range(w):
        col = grid[:, c]
        if np.any(col == BLACK):
            continue
        counts = Counter(int(v) for v in col if int(v) != GREY)
        if not counts:
            continue
        color, _count = counts.most_common(1)[0]
        if color != BLACK:
            bands[int(c)] = int(color)
    return bands


def _detect_horizontal_bands(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    bands: dict[int, int] = {}
    for r in range(h):
        row = grid[r]
        if np.any(row == BLACK):
            continue
        counts = Counter(int(v) for v in row if int(v) != GREY)
        if not counts:
            continue
        color, _count = counts.most_common(1)[0]
        if color != BLACK:
            bands[int(r)] = int(color)
    return bands


def _parse_arrow(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    pts = {(int(r), int(c)) for r, c in coords}
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    row_counts = [sum((rr, cc) in pts for cc in range(c0, c1 + 1)) for rr in range(r0, r1 + 1)]
    col_counts = [sum((rr, cc) in pts for rr in range(r0, r1 + 1)) for cc in range(c0, c1 + 1)]

    if h == 3 and w >= 3 and row_counts == [1, w, 1]:
        shaft_row = r0 + 1
        if col_counts[0] == 3 and all(v == 1 for v in col_counts[1:]):
            shaft_cols = list(range(c1, c0, -1))
            expected = {(shaft_row, c) for c in range(c0 + 1, c1 + 1)}
            expected |= {(r0, c0), (shaft_row, c0), (r1, c0)}
            if expected == pts and len(shaft_cols) >= 2:
                return {
                    "direction": "left",
                    "bbox": (r0, c0, r1, c1),
                    "shaft_cols": shaft_cols,
                    "ray_start": (shaft_row, c0 - 1),
                    "step": (0, -1),
                }
        if col_counts[-1] == 3 and all(v == 1 for v in col_counts[:-1]):
            shaft_cols = list(range(c0, c1))
            expected = {(shaft_row, c) for c in range(c0, c1)}
            expected |= {(r0, c1), (shaft_row, c1), (r1, c1)}
            if expected == pts and len(shaft_cols) >= 2:
                return {
                    "direction": "right",
                    "bbox": (r0, c0, r1, c1),
                    "shaft_cols": shaft_cols,
                    "ray_start": (shaft_row, c1 + 1),
                    "step": (0, 1),
                }

    if w == 3 and h >= 3 and col_counts == [1, h, 1]:
        shaft_col = c0 + 1
        if row_counts[0] == 3 and all(v == 1 for v in row_counts[1:]):
            shaft_rows = list(range(r1, r0, -1))
            expected = {(r, shaft_col) for r in range(r0 + 1, r1 + 1)}
            expected |= {(r0, c0), (r0, shaft_col), (r0, c1)}
            if expected == pts and len(shaft_rows) >= 2:
                return {
                    "direction": "up",
                    "bbox": (r0, c0, r1, c1),
                    "shaft_rows": shaft_rows,
                    "ray_start": (r0 - 1, shaft_col),
                    "step": (-1, 0),
                }
        if row_counts[-1] == 3 and all(v == 1 for v in row_counts[:-1]):
            shaft_rows = list(range(r0, r1))
            expected = {(r, shaft_col) for r in range(r0, r1)}
            expected |= {(r1, c0), (r1, shaft_col), (r1, c1)}
            if expected == pts and len(shaft_rows) >= 2:
                return {
                    "direction": "down",
                    "bbox": (r0, c0, r1, c1),
                    "shaft_rows": shaft_rows,
                    "ray_start": (r1 + 1, shaft_col),
                    "step": (1, 0),
                }

    return None


def _draw_horizontal_arrow(grid: np.ndarray, row: int, direction: str, start: int, end: int) -> None:
    assert direction in {"left", "right"}
    if direction == "right":
        assert end >= start + 1
        grid[row, start : end + 1] = GREY
        grid[row - 1 : row + 2 : 2, end + 1] = GREY
        grid[row, end + 1] = GREY
    else:
        assert end >= start + 1
        grid[row, start : end + 1] = GREY
        grid[row - 1 : row + 2 : 2, start - 1] = GREY
        grid[row, start - 1] = GREY


def _draw_vertical_arrow(grid: np.ndarray, col: int, direction: str, start: int, end: int) -> None:
    assert direction in {"up", "down"}
    if direction == "down":
        assert end >= start + 1
        grid[start : end + 1, col] = GREY
        grid[end + 1, col - 1 : col + 2] = GREY
    else:
        assert end >= start + 1
        grid[start : end + 1, col] = GREY
        grid[start - 1, col - 1 : col + 2] = GREY


def _vertical_loaded_specs(w: int, band_cols: list[int]) -> list[tuple[str, int, int]]:
    left_band = band_cols[0]
    right_band = band_cols[-1]
    specs: list[tuple[str, int, int]] = []

    for i, col in enumerate(band_cols):
        low = 1 if i == 0 else band_cols[i - 1] + 1
        high = min(col, right_band - 1)
        for start in range(low, high + 1):
            specs.append(("right", start, right_band))

    for j, col in enumerate(band_cols):
        low = max(col, left_band + 1) if j == 0 else col
        high = (w - 2) if j == len(band_cols) - 1 else band_cols[j + 1] - 1
        for end in range(low, high + 1):
            specs.append(("left", left_band, end))

    return specs


def _vertical_empty_specs(w: int, band_cols: list[int]) -> list[tuple[str, int, int]]:
    left_band = band_cols[0]
    right_band = band_cols[-1]
    specs: list[tuple[str, int, int]] = []
    max_len = 7

    for start in range(1, left_band - 1):
        for end in range(start + 1, min(left_band - 2, start + max_len - 1) + 1):
            specs.append(("right", start, end))
    for start in range(right_band + 1, w - 1):
        for end in range(start + 1, min(w - 2, start + max_len - 1) + 1):
            specs.append(("right", start, end))
    for start in range(1, left_band):
        for end in range(start + 1, min(left_band - 1, start + max_len - 1) + 1):
            specs.append(("left", start, end))
    for start in range(right_band + 2, w - 1):
        for end in range(start + 1, min(w - 2, start + max_len - 1) + 1):
            specs.append(("left", start, end))

    return specs


def _horizontal_loaded_specs(h: int, band_rows: list[int]) -> list[tuple[str, int, int]]:
    top_band = band_rows[0]
    bottom_band = band_rows[-1]
    specs: list[tuple[str, int, int]] = []

    for i, row in enumerate(band_rows):
        low = 1 if i == 0 else band_rows[i - 1] + 1
        high = min(row, bottom_band - 1)
        for start in range(low, high + 1):
            specs.append(("down", start, bottom_band))

    for j, row in enumerate(band_rows):
        low = max(row, top_band + 1) if j == 0 else row
        high = (h - 2) if j == len(band_rows) - 1 else band_rows[j + 1] - 1
        for end in range(low, high + 1):
            specs.append(("up", top_band, end))

    return specs


def _horizontal_empty_specs(h: int, band_rows: list[int]) -> list[tuple[str, int, int]]:
    top_band = band_rows[0]
    bottom_band = band_rows[-1]
    specs: list[tuple[str, int, int]] = []
    max_len = 7

    for start in range(1, top_band - 1):
        for end in range(start + 1, min(top_band - 2, start + max_len - 1) + 1):
            specs.append(("down", start, end))
    for start in range(bottom_band + 1, h - 1):
        for end in range(start + 1, min(h - 2, start + max_len - 1) + 1):
            specs.append(("down", start, end))
    for start in range(1, top_band):
        for end in range(start + 1, min(top_band - 1, start + max_len - 1) + 1):
            specs.append(("up", start, end))
    for start in range(bottom_band + 2, h - 1):
        for end in range(start + 1, min(h - 2, start + max_len - 1) + 1):
            specs.append(("up", start, end))

    return specs


def _gen_vertical_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(11, 28)
    w = rng.randint(12, 30)
    max_bands = min(4, 1 + (w - 7) // MIN_BAND_GAP)
    max_arrows = min(4, 1 + (h - 3) // MIN_ARROW_GAP)
    if max_bands < 1 or max_arrows < 2:
        return None

    band_count = rng.randint(1, max_bands)
    arrow_count = rng.randint(2, max_arrows)
    band_cols = _sample_spaced_positions(rng, 3, w - 4, band_count, MIN_BAND_GAP)
    arrow_rows = _sample_spaced_positions(rng, 1, h - 2, arrow_count, MIN_ARROW_GAP)
    if band_cols is None or arrow_rows is None:
        return None

    loaded_specs = _vertical_loaded_specs(w, band_cols)
    empty_specs = _vertical_empty_specs(w, band_cols)
    if not loaded_specs or not empty_specs:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    colors = rng.sample(PALETTE, band_count)
    for c, color in zip(band_cols, colors):
        grid[:, c] = int(color)

    num_loaded = rng.randint(1, arrow_count - 1)
    tags = ["loaded"] * num_loaded + ["empty"] * (arrow_count - num_loaded)
    rng.shuffle(tags)
    for row, tag in zip(arrow_rows, tags):
        direction, start, end = rng.choice(loaded_specs if tag == "loaded" else empty_specs)
        _draw_horizontal_arrow(grid, int(row), direction, int(start), int(end))

    return grid


def _gen_horizontal_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 30)
    w = rng.randint(11, 28)
    max_bands = min(4, 1 + (h - 7) // MIN_BAND_GAP)
    max_arrows = min(4, 1 + (w - 3) // MIN_ARROW_GAP)
    if max_bands < 1 or max_arrows < 2:
        return None

    band_count = rng.randint(1, max_bands)
    arrow_count = rng.randint(2, max_arrows)
    band_rows = _sample_spaced_positions(rng, 3, h - 4, band_count, MIN_BAND_GAP)
    arrow_cols = _sample_spaced_positions(rng, 1, w - 2, arrow_count, MIN_ARROW_GAP)
    if band_rows is None or arrow_cols is None:
        return None

    loaded_specs = _horizontal_loaded_specs(h, band_rows)
    empty_specs = _horizontal_empty_specs(h, band_rows)
    if not loaded_specs or not empty_specs:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    colors = rng.sample(PALETTE, band_count)
    for r, color in zip(band_rows, colors):
        grid[r, :] = int(color)

    num_loaded = rng.randint(1, arrow_count - 1)
    tags = ["loaded"] * num_loaded + ["empty"] * (arrow_count - num_loaded)
    rng.shuffle(tags)
    for col, tag in zip(arrow_cols, tags):
        direction, start, end = rng.choice(loaded_specs if tag == "loaded" else empty_specs)
        _draw_vertical_arrow(grid, int(col), direction, int(start), int(end))

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each gray arrow with a cyclic colored ray loaded from the full-row or full-column bands its shaft crosses.

    Input structure:
    - The background is black.
    - Some non-black, non-gray colors form width-1 full rows and/or full columns that act as color bands.
    - Gray connected components are arrows: a horizontal or vertical shaft of length at least 2 with a 3-pixel tip on one end.

    Rule:
    - Determine every arrow direction from its tip shape.
    - For a horizontal arrow, read the vertical-band colors crossed by its shaft from tail to tip; for a vertical arrow, read the horizontal-band colors crossed by its shaft from tail to tip.
    - If that loaded color list is non-empty, start one cell in front of the tip and draw a ray to the grid edge, cycling through the loaded colors in order.
    - Keep the original arrows and the original bands unchanged.

    Key insight:
    - The arrowhead gives the motion direction, while the shaft acts as a scanner that samples crossed color bands into an ordered palette that is then repeated along the outgoing ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    vertical_bands = _detect_vertical_bands(g)
    horizontal_bands = _detect_horizontal_bands(g)

    arrows: list[dict[str, object]] = []
    for comp in _components(g == GREY):
        arrow = _parse_arrow(comp)
        assert arrow is not None, "every gray component must be a valid arrow"
        arrows.append(arrow)
    assert arrows, "expected at least one arrow"

    out = g.copy()
    for arrow in sorted(arrows, key=lambda item: tuple(int(v) for v in item["bbox"])):
        direction = str(arrow["direction"])
        if direction in {"left", "right"}:
            colors = [vertical_bands[c] for c in arrow["shaft_cols"] if c in vertical_bands]
        else:
            colors = [horizontal_bands[r] for r in arrow["shaft_rows"] if r in horizontal_bands]

        if not colors:
            continue

        r, c = map(int, arrow["ray_start"])
        dr, dc = map(int, arrow["step"])
        idx = 0
        while 0 <= r < h and 0 <= c < w:
            if int(g[r, c]) == BLACK:
                out[r, c] = int(colors[idx % len(colors)])
            r += dr
            c += dc
            idx += 1

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_vertical_case(rng) if rng.random() < 0.5 else _gen_horizontal_case(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape or np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid instance")
