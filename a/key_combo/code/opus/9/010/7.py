from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
TEAL = 8

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_BAR_COUNT_CHOICES = {
    2: (6, 8, 10),
    3: (8, 9),
    4: (10,),
}
_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = _N4 if int(connectivity) == 4 else _N8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _scene_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BLACK

    colors = set(map(int, np.unique(g)))
    assert colors.issubset({BLACK, YELLOW, TEAL})
    assert YELLOW in colors and TEAL in colors

    marker_comps = _components_mask(g == YELLOW, connectivity=8)
    assert 2 <= len(marker_comps) <= 4
    for comp in marker_comps:
        assert comp.shape == (1, 2)

    bar_comps = _components_mask(g == TEAL, connectivity=4)
    assert 6 <= len(bar_comps) <= 10

    bars = []
    for comp in bar_comps:
        r0, c0, r1, c1 = _bbox_coords(comp)
        width = c1 - c0 + 1
        assert r0 == r1
        assert int(comp.shape[0]) == width
        assert width >= 3
        bars.append(
            {
                "coords": comp,
                "row": r0,
                "col": c0,
                "width": width,
                "end_col": c1,
            }
        )
    bars.sort(key=lambda bar: (bar["row"], bar["col"]))

    n = len(marker_comps)
    assert len(bars) >= 2 * n + 1

    markers = [tuple(map(int, comp[0])) for comp in marker_comps]
    return {"grid": g, "markers": markers, "bars": bars, "n": n}


def _selected_indices(bar_count: int, stride: int) -> tuple[int, ...]:
    return tuple(range(0, int(bar_count), int(stride)))


def _paint_stride(grid: np.ndarray, bars: list[dict], stride: int, *, from_top: bool = True) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[out == YELLOW] = BLACK
    if from_top:
        chosen = range(0, len(bars), stride)
    else:
        chosen = range(len(bars) - 1, -1, -stride)
    for idx in chosen:
        coords = bars[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count yellow singleton markers, then recolor every N-th teal bar from the top.

    Input structure:
    - The grid has a black background, several one-cell-thick teal horizontal bars, and scattered yellow singleton markers.
    - Each teal connected component is a separate horizontal bar, and each yellow connected component is exactly one pixel.

    Rule:
    - Count the yellow singleton markers to obtain N.
    - Sort all teal bars by increasing row, breaking ties by increasing left column.
    - Recolor bars numbered 1, 1+N, 1+2N, ... to red and keep the other teal bars unchanged.
    - Remove all yellow markers by turning them into black background.

    Key insight:
    - The yellow pixels are not spatial targets; their count is an abstract stride parameter that controls periodic selection over the ordered teal-bar list.
    """
    info = _scene_info(grid)
    g = info["grid"]
    out = np.array(g, copy=True)
    out[out == YELLOW] = BLACK

    for idx in _selected_indices(len(info["bars"]), info["n"]):
        coords = info["bars"][idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = RED
    return out


def _random_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(total + parts - 1), parts - 1)) if parts > 1 else []
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _sample_rows(height: int, count: int, rng: random.Random) -> list[int]:
    slack = height - (2 * count - 1)
    assert slack >= 0
    extras = _random_partition(slack, count + 1, rng)
    rows = [extras[0]]
    for idx in range(1, count):
        rows.append(rows[-1] + 2 + extras[idx])
    return rows


def _marker_is_valid(r: int, c: int, grid: np.ndarray, markers: list[tuple[int, int]]) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(grid[rr, cc]) == TEAL:
                return False
    for mr, mc in markers:
        if max(abs(r - mr), abs(c - mc)) <= 1:
            return False
    return True


def _distance_to_bar(marker: tuple[int, int], bar: dict) -> int:
    r, c = marker
    row_gap = abs(r - bar["row"])
    if c < bar["col"]:
        col_gap = bar["col"] - c
    elif c > bar["end_col"]:
        col_gap = c - bar["end_col"]
    else:
        col_gap = 0
    return max(row_gap, col_gap)


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        info = _scene_info(grid)
        g = info["grid"]
        bars = info["bars"]
        markers = info["markers"]
        n = info["n"]

        assert len({bar["width"] for bar in bars}) >= 2
        assert len(_selected_indices(len(bars), n)) >= 3
        assert len(_selected_indices(len(bars), n)) < len(bars)

        for marker in markers:
            assert all(_distance_to_bar(marker, bar) >= 2 for bar in bars)
        for idx, marker in enumerate(markers):
            for other in markers[idx + 1 :]:
                assert max(abs(marker[0] - other[0]), abs(marker[1] - other[1])) >= 2

        true_out = main(g)
        assert np.all(true_out[g == YELLOW] == BLACK)
        for idx, bar in enumerate(bars):
            coords = bar["coords"]
            expected = RED if idx in _selected_indices(len(bars), n) else TEAL
            assert np.all(true_out[coords[:, 0], coords[:, 1]] == expected)

        for alt in (2, 3, 4):
            if alt == n:
                continue
            alt_out = _paint_stride(g, bars, alt, from_top=True)
            assert not np.array_equal(true_out, alt_out)

        bottom_out = _paint_stride(g, bars, n, from_top=False)
        assert not np.array_equal(true_out, bottom_out)
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(2, 4)
    bar_count = rng.choice(_BAR_COUNT_CHOICES[n])

    min_h = max(14, 2 * bar_count + 2)
    max_h = min(28, min_h + 4)
    height = rng.randint(min_h, max_h)
    width = rng.randint(18, 26)

    rows = _sample_rows(height, bar_count, rng)
    grid = np.full((height, width), BLACK, dtype=int)

    max_bar_width = min(width - 4, 12)
    if max_bar_width < 5:
        return None

    widths = []
    for _ in range(bar_count):
        bar_width = rng.randint(3, max_bar_width)
        start_col = rng.randint(1, width - bar_width - 2)
        widths.append(bar_width)
        grid[rows[len(widths) - 1], start_col : start_col + bar_width] = TEAL

    if len(set(widths)) < 2:
        return None

    candidates = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(candidates)
    markers: list[tuple[int, int]] = []
    for r, c in candidates:
        if _marker_is_valid(r, c, grid, markers):
            markers.append((r, c))
            grid[r, c] = YELLOW
            if len(markers) == n:
                break
    if len(markers) != n:
        return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
