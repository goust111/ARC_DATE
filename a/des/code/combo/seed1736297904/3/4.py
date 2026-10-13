from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _band_rows(comp: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in comp]
    return min(rows), max(rows)


def _find_bands(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    bands: list[dict[str, int]] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        touches_left = c0 == 0
        touches_right = c1 == w - 1
        assert touches_left ^ touches_right, "each grey arrow must touch exactly one side edge"
        for rr in range(r0, r1 + 1):
            assert any(r == rr for r, _c in comp), "arrow rows must define one continuous band"
        bands.append(
            {
                "r0": int(r0),
                "r1": int(r1),
                "direction": 1 if touches_left else -1,
                "edge_col": 0 if touches_left else w - 1,
            }
        )

    bands.sort(key=lambda band: (band["r0"], band["r1"]))
    for prev, cur in zip(bands, bands[1:]):
        assert prev["r1"] < cur["r0"], "row bands must not overlap"
    return bands


def _paint_growth(
    out: np.ndarray,
    r0: int,
    r1: int,
    direction: int,
    color: int,
    seed_cols: list[int],
) -> None:
    width = len(seed_cols)
    ordered = sorted(seed_cols) if direction == 1 else sorted(seed_cols, reverse=True)
    for c in ordered:
        left = c if direction == 1 else c - width + 1
        right = c + width - 1 if direction == 1 else c
        left = max(0, int(left))
        right = min(out.shape[1] - 1, int(right))
        if left <= right:
            out[r0 : r1 + 1, left : right + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a full-height rectangle from every seed across its arrow's row band.

    Input structure:
    - Black background.
    - One or more grey horizontal arrows touch the left or right grid edge.
      Each arrow defines a non-overlapping row band and its direction.
    - Colored single-pixel seeds lie in the non-arrow area of those row bands.

    Rule:
    - Keep the arrows unchanged.
    - For each row band, count its seeds: that count is the rectangle width.
    - Every seed grows a solid rectangle covering the full band height and
      extending along the arrow direction from the seed column.
    - Growth clips at the grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    for band in _find_bands(g):
        r0, r1 = band["r0"], band["r1"]
        mask = (np.arange(g.shape[0])[:, None] >= r0) & (np.arange(g.shape[0])[:, None] <= r1)
        seed_mask = mask & (g != BG) & (g != GREY)
        coords = np.argwhere(seed_mask)
        if len(coords) == 0:
            continue
        colors = np.unique(g[seed_mask])
        assert len(colors) == 1, "all seeds in one row band must have the same color"
        color = int(colors[0])
        seed_cols = [int(c) for _r, c in coords]
        _paint_growth(out, r0, r1, band["direction"], color, seed_cols)

    return out


# --------------- generator ---------------

MIN_BANDS = 2
MAX_BANDS = 4
HEIGHT_CHOICES = (3, 4, 5, 6)
WIDTH_RANGE = (14, 26)
TOP_MARGIN_RANGE = (0, 3)
BOTTOM_MARGIN_RANGE = (0, 3)
GAP_RANGE = (1, 3)
SEED_COUNT_RANGE = (1, 4)
SHAFT_LEN_RANGE = (2, 5)
GEN_RETRIES = 300


def _sample_layout() -> tuple[int, list[tuple[int, int]]]:
    for _ in range(200):
        n_bands = random.randint(MIN_BANDS, MAX_BANDS)
        heights = random.sample(HEIGHT_CHOICES, n_bands)
        random.shuffle(heights)
        top = random.randint(*TOP_MARGIN_RANGE)
        bottom = random.randint(*BOTTOM_MARGIN_RANGE)
        gaps = [random.randint(*GAP_RANGE) for _ in range(n_bands - 1)]
        total_h = top + bottom + sum(heights) + sum(gaps)
        if 12 <= total_h <= 30:
            bands: list[tuple[int, int]] = []
            r = top
            for i, height in enumerate(heights):
                bands.append((r, r + height - 1))
                r += height
                if i < len(gaps):
                    r += gaps[i]
            return total_h, bands
    raise RuntimeError("failed to sample row-band layout")


def _sample_seed_columns(width: int, avail_lo: int, avail_hi: int, count: int, direction: int) -> list[int]:
    all_choices: dict[tuple[bool, bool], list[tuple[int, ...]]] = {}
    for cols in itertools.combinations(range(avail_lo, avail_hi + 1), count):
        overlap = any(cols[i + 1] - cols[i] < count for i in range(count - 1))
        clip = cols[-1] + count - 1 >= width if direction == 1 else cols[0] - count + 1 < 0
        all_choices.setdefault((overlap, clip), []).append(cols)

    assert all_choices, "band must leave room for at least one seed placement"
    categories = list(all_choices)
    if count == 1:
        chosen = (False, False)
    else:
        weighted: list[tuple[bool, bool]] = []
        for category in categories:
            overlap, clip = category
            weight = 1 + int(overlap) + int(clip)
            weighted.extend([category] * weight)
        chosen = random.choice(weighted)
    return list(random.choice(all_choices[chosen]))


def _draw_arrow(grid: np.ndarray, r0: int, r1: int, direction: int, shaft_len: int) -> None:
    mid = (r0 + r1) // 2
    if direction == 1:
        grid[r0 : r1 + 1, 0:shaft_len] = GREY
        grid[mid - 1 : mid + 2, shaft_len] = GREY
        grid[mid, shaft_len + 1] = GREY
    else:
        w = grid.shape[1]
        grid[r0 : r1 + 1, w - shaft_len : w] = GREY
        grid[mid - 1 : mid + 2, w - shaft_len - 1] = GREY
        grid[mid, w - shaft_len - 2] = GREY


def _build_scene() -> tuple[np.ndarray, np.ndarray]:
    height, row_bands = _sample_layout()
    width = random.randint(*WIDTH_RANGE)
    colors = list(PALETTE)
    random.shuffle(colors)
    counts = [random.randint(*SEED_COUNT_RANGE) for _ in row_bands]
    if all(count == 1 for count in counts):
        counts[random.randrange(len(counts))] = random.randint(2, SEED_COUNT_RANGE[1])

    bands: list[dict[str, object]] = []
    for (r0, r1), count, color in zip(row_bands, counts, colors):
        max_shaft = min(SHAFT_LEN_RANGE[1], width - count - 2)
        if max_shaft < SHAFT_LEN_RANGE[0]:
            raise RuntimeError("grid width too small for sampled seed count")
        shaft_len = random.randint(SHAFT_LEN_RANGE[0], max_shaft)
        direction = random.choice((1, -1))
        if direction == 1:
            avail_lo = shaft_len + 2
            avail_hi = width - 1
        else:
            avail_lo = 0
            avail_hi = width - shaft_len - 3
        seed_cols = _sample_seed_columns(width, avail_lo, avail_hi, count, direction)
        seeds = [(random.randint(r0, r1), c) for c in seed_cols]
        bands.append(
            {
                "r0": r0,
                "r1": r1,
                "direction": direction,
                "shaft_len": shaft_len,
                "color": int(color),
                "seeds": seeds,
            }
        )

    inp = np.full((height, width), BG, dtype=int)
    for band in bands:
        _draw_arrow(inp, int(band["r0"]), int(band["r1"]), int(band["direction"]), int(band["shaft_len"]))
        for r, c in band["seeds"]:
            assert inp[r, c] == BG
            inp[r, c] = int(band["color"])

    expected = inp.copy()
    for band in bands:
        seed_cols = [int(c) for _r, c in band["seeds"]]
        _paint_growth(
            expected,
            int(band["r0"]),
            int(band["r1"]),
            int(band["direction"]),
            int(band["color"]),
            seed_cols,
        )
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected = _build_scene()
        except Exception:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
