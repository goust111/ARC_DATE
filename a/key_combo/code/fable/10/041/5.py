from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, get_bg


PairKey = tuple[int, int]


def _pair_key(a: int, b: int) -> PairKey:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _slice_size(s: slice) -> int:
    return int(s.stop - s.start)


def _layout_from_corner(
    corner: str, panel_h: int, panel_w: int, main_h: int, main_w: int
) -> dict[str, object]:
    total_h = int(panel_h + 1 + main_h)
    total_w = int(panel_w + 1 + main_w)
    assert total_h <= 30 and total_w <= 30

    if corner == "tl":
        div_r, div_c = panel_h, panel_w
        panel_rows = slice(0, panel_h)
        panel_cols = slice(0, panel_w)
        main_rows = slice(div_r + 1, total_h)
        main_cols = slice(div_c + 1, total_w)
        divider_cells = [(div_r, c) for c in range(0, div_c + 1)] + [
            (r, div_c) for r in range(0, div_r)
        ]
    elif corner == "tr":
        div_r, div_c = panel_h, main_w
        panel_rows = slice(0, panel_h)
        panel_cols = slice(div_c + 1, total_w)
        main_rows = slice(div_r + 1, total_h)
        main_cols = slice(0, div_c)
        divider_cells = [(div_r, c) for c in range(div_c, total_w)] + [
            (r, div_c) for r in range(0, div_r)
        ]
    elif corner == "bl":
        div_r, div_c = main_h, panel_w
        panel_rows = slice(div_r + 1, total_h)
        panel_cols = slice(0, panel_w)
        main_rows = slice(0, div_r)
        main_cols = slice(div_c + 1, total_w)
        divider_cells = [(div_r, c) for c in range(0, div_c + 1)] + [
            (r, div_c) for r in range(div_r + 1, total_h)
        ]
    elif corner == "br":
        div_r, div_c = main_h, main_w
        panel_rows = slice(div_r + 1, total_h)
        panel_cols = slice(div_c + 1, total_w)
        main_rows = slice(0, div_r)
        main_cols = slice(0, div_c)
        divider_cells = [(div_r, c) for c in range(div_c, total_w)] + [
            (r, div_c) for r in range(div_r + 1, total_h)
        ]
    else:
        raise AssertionError(f"unknown corner: {corner}")

    return {
        "corner": corner,
        "shape": (total_h, total_w),
        "panel_rows": panel_rows,
        "panel_cols": panel_cols,
        "main_rows": main_rows,
        "main_cols": main_cols,
        "divider_cells": divider_cells,
        "div_r": int(div_r),
        "div_c": int(div_c),
    }


def _find_layout(grid: np.ndarray, bg: int) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    candidates: list[dict[str, object]] = []

    for color in colors_present(g, exclude={bg}):
        mask = g == int(color)
        coords = np.argwhere(mask)
        if coords.shape[0] < 3:
            continue

        row_values, row_counts = np.unique(coords[:, 0], return_counts=True)
        col_values, col_counts = np.unique(coords[:, 1], return_counts=True)
        row_candidates = [int(r) for r, cnt in zip(row_values, row_counts) if cnt > 1]
        col_candidates = [int(c) for c, cnt in zip(col_values, col_counts) if cnt > 1]
        if len(row_candidates) != 1 or len(col_candidates) != 1:
            continue

        div_r = row_candidates[0]
        div_c = col_candidates[0]
        if not np.all((coords[:, 0] == div_r) | (coords[:, 1] == div_c)):
            continue

        row_cols = np.flatnonzero(mask[div_r])
        col_rows = np.flatnonzero(mask[:, div_c])
        if row_cols.size < 2 or col_rows.size < 2:
            continue
        if not np.array_equal(row_cols, np.arange(row_cols[0], row_cols[-1] + 1)):
            continue
        if not np.array_equal(col_rows, np.arange(col_rows[0], col_rows[-1] + 1)):
            continue

        layout: dict[str, object] | None = None
        if (
            row_cols[0] == 0
            and row_cols[-1] == div_c
            and col_rows[0] == 0
            and col_rows[-1] == div_r
        ):
            layout = {
                "corner": "tl",
                "panel_rows": slice(0, div_r),
                "panel_cols": slice(0, div_c),
                "main_rows": slice(div_r + 1, height),
                "main_cols": slice(div_c + 1, width),
            }
        elif (
            row_cols[0] == div_c
            and row_cols[-1] == width - 1
            and col_rows[0] == 0
            and col_rows[-1] == div_r
        ):
            layout = {
                "corner": "tr",
                "panel_rows": slice(0, div_r),
                "panel_cols": slice(div_c + 1, width),
                "main_rows": slice(div_r + 1, height),
                "main_cols": slice(0, div_c),
            }
        elif (
            row_cols[0] == 0
            and row_cols[-1] == div_c
            and col_rows[0] == div_r
            and col_rows[-1] == height - 1
        ):
            layout = {
                "corner": "bl",
                "panel_rows": slice(div_r + 1, height),
                "panel_cols": slice(0, div_c),
                "main_rows": slice(0, div_r),
                "main_cols": slice(div_c + 1, width),
            }
        elif (
            row_cols[0] == div_c
            and row_cols[-1] == width - 1
            and col_rows[0] == div_r
            and col_rows[-1] == height - 1
        ):
            layout = {
                "corner": "br",
                "panel_rows": slice(div_r + 1, height),
                "panel_cols": slice(div_c + 1, width),
                "main_rows": slice(0, div_r),
                "main_cols": slice(0, div_c),
            }

        if layout is None:
            continue

        panel_h = _slice_size(layout["panel_rows"])
        panel_w = _slice_size(layout["panel_cols"])
        main_h = _slice_size(layout["main_rows"])
        main_w = _slice_size(layout["main_cols"])
        if panel_h <= 0 or panel_w <= 0 or main_h < 3 or main_w < 3:
            continue

        layout = dict(layout)
        layout["divider_color"] = int(color)
        candidates.append(layout)

    assert len(candidates) == 1, "expected a unique corner divider"
    return candidates[0]


def _legend_lookup(panel: np.ndarray, bg: int) -> tuple[dict[PairKey, int], list[int]]:
    entries: list[tuple[int, int, int]] = []
    for r in range(panel.shape[0]):
        row = [int(v) for v in panel[r] if int(v) != bg]
        if not row:
            continue
        assert len(row) == 3, "each legend row must contain exactly three colors"
        entries.append((row[0], row[1], row[2]))

    assert entries, "legend panel must contain mapping rows"
    lookup: dict[PairKey, int] = {}
    op_colors: set[int] = set()
    for a, b, out in entries:
        key = _pair_key(a, b)
        if key in lookup:
            assert lookup[key] == out, "duplicate legend rows must agree"
        lookup[key] = int(out)
        op_colors.update((int(a), int(b), int(out)))

    color_list = sorted(op_colors)
    for a, b in itertools.combinations_with_replacement(color_list, 2):
        assert _pair_key(a, b) in lookup, "legend must define every unordered color pair"
    return lookup, color_list


def _solve_main_region(region: np.ndarray, lookup: dict[PairKey, int], bg: int) -> np.ndarray:
    h, w = map(int, region.shape)
    out = region.copy()
    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}

    for r in range(h):
        left = int(region[r, 0])
        right = int(region[r, w - 1])
        if left != bg and right != bg and np.all(region[r, 1 : w - 1] == bg):
            row_color = int(lookup[_pair_key(left, right)])
            out[r, 1 : w - 1] = row_color
            row_colors[r] = row_color

    for c in range(w):
        top = int(region[0, c])
        bottom = int(region[h - 1, c])
        if top != bg and bottom != bg and np.all(region[1 : h - 1, c] == bg):
            col_color = int(lookup[_pair_key(top, bottom)])
            out[1 : h - 1, c] = col_color
            col_colors[c] = col_color

    for r, row_color in row_colors.items():
        for c, col_color in col_colors.items():
            out[r, c] = int(lookup[_pair_key(row_color, col_color)])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color every clear edge-to-edge corridor via a legend-table lookup, then recolor corridor crossings by looking up the two corridor colors again.

    Input structure:
    - A black-background main region is separated from one corner legend panel by a colored L-shaped divider.
    - The legend panel contains rows of three colored cells, where the first two colors form an unordered pair and the third cell gives that pair's output color.
    - Around the main region border are colored edge cells, and inside the region are a few colored blocker cells.

    Rule:
    - In the main region, any row whose left and right border cells are foreground and whose interior is entirely black becomes a horizontal corridor colored by the legend lookup of its two endpoint colors.
    - Any column whose top and bottom border cells are foreground and whose interior is entirely black becomes a vertical corridor colored by the legend lookup of its two endpoint colors.
    - If a horizontal and vertical corridor cross, recolor that intersection cell by looking up the unordered pair of the two corridor colors in the same legend.
    - Preserve the divider, the legend panel, all original border cells, and all original blocker cells.

    Key insight:
    - The legend defines a reusable symmetric binary operation: border-endpoint pairs produce corridor colors, and corridor-color pairs produce the final crossing colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    layout = _find_layout(g, bg)
    panel = g[layout["panel_rows"], layout["panel_cols"]]
    lookup, _op_colors = _legend_lookup(panel, bg)

    out = g.copy()
    region = g[layout["main_rows"], layout["main_cols"]]
    out[layout["main_rows"], layout["main_cols"]] = _solve_main_region(region, lookup, bg)
    return out


PANEL_WIDTH = 3
OP_COLOR_COUNT_CHOICES = (3, 4)
CORNERS = ("tl", "tr", "bl", "br")
MAIN_HEIGHT_RANGE = (7, 17)
MAIN_WIDTH_RANGE = (8, 19)
CLEAR_ROW_RANGE = (1, 4)
CLEAR_COL_RANGE = (1, 3)
EXTRA_BLOCKER_CAP = 8
GENERATION_RETRIES = 500


def _sample_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _random_lookup(
    op_colors: list[int], rng: np.random.Generator
) -> dict[PairKey, int] | None:
    pair_keys = [_pair_key(a, b) for a, b in itertools.combinations_with_replacement(op_colors, 2)]
    for _ in range(80):
        lookup = {
            key: int(op_colors[int(rng.integers(0, len(op_colors)))]) for key in pair_keys
        }
        outputs = set(lookup.values())
        if len(outputs) < min(3, len(op_colors)):
            continue
        if not any(key[0] != key[1] and lookup[key] not in set(key) for key in pair_keys):
            continue
        if not any(lookup[_pair_key(c, c)] != c for c in op_colors):
            continue
        return lookup
    return None


def _choose_clear_indices(
    length: int, bounds: tuple[int, int], rng: np.random.Generator
) -> set[int]:
    interior = list(range(1, length - 1))
    max_clear = min(bounds[1], len(interior) - 1)
    min_clear = min(bounds[0], max_clear)
    assert max_clear >= 1
    count = _sample_int(rng, min_clear, max_clear)
    return set(map(int, rng.choice(interior, size=count, replace=False)))


def _paint_border(region: np.ndarray, op_colors: list[int], rng: np.random.Generator) -> None:
    h, w = region.shape
    region[0, :] = rng.choice(op_colors, size=w)
    region[h - 1, :] = rng.choice(op_colors, size=w)
    region[:, 0] = rng.choice(op_colors, size=h)
    region[:, w - 1] = rng.choice(op_colors, size=h)


def _seed_blockers(
    region: np.ndarray,
    blocked_rows: list[int],
    blocked_cols: list[int],
    op_colors: list[int],
    rng: np.random.Generator,
) -> None:
    positions: set[tuple[int, int]] = set()
    for r in blocked_rows:
        positions.add((int(r), int(blocked_cols[int(rng.integers(0, len(blocked_cols)))])))
    covered_cols = {c for _r, c in positions}
    for c in blocked_cols:
        if c not in covered_cols:
            positions.add((int(blocked_rows[int(rng.integers(0, len(blocked_rows)))]), int(c)))

    candidates = [
        (int(r), int(c))
        for r in blocked_rows
        for c in blocked_cols
        if (int(r), int(c)) not in positions
    ]
    extra_limit = min(EXTRA_BLOCKER_CAP, len(candidates))
    if extra_limit > 0:
        extra_count = _sample_int(rng, 0, extra_limit)
        if extra_count > 0:
            chosen = rng.choice(len(candidates), size=extra_count, replace=False)
            for idx in np.atleast_1d(chosen):
                positions.add(candidates[int(idx)])

    for r, c in positions:
        region[r, c] = int(op_colors[int(rng.integers(0, len(op_colors)))])


def _lookup_rows_to_panel(
    op_colors: list[int], lookup: dict[PairKey, int], rng: np.random.Generator
) -> np.ndarray:
    rows = [
        [int(a), int(b), int(lookup[_pair_key(a, b)])]
        for a, b in itertools.combinations_with_replacement(op_colors, 2)
    ]
    rng.shuffle(rows)
    return np.asarray(rows, dtype=int)


def _design_expected_region(
    region: np.ndarray,
    lookup: dict[PairKey, int],
    clear_rows: set[int],
    clear_cols: set[int],
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    out = region.copy()
    h, w = region.shape
    row_colors = {
        int(r): int(lookup[_pair_key(int(region[r, 0]), int(region[r, w - 1]))])
        for r in clear_rows
    }
    col_colors = {
        int(c): int(lookup[_pair_key(int(region[0, c]), int(region[h - 1, c]))])
        for c in clear_cols
    }

    for r, color in row_colors.items():
        out[r, 1 : w - 1] = int(color)
    for c, color in col_colors.items():
        out[1 : h - 1, c] = int(color)
    for r, row_color in row_colors.items():
        for c, col_color in col_colors.items():
            out[r, c] = int(lookup[_pair_key(row_color, col_color)])
    return out, row_colors, col_colors


def _semantic_checks(
    region: np.ndarray,
    expected: np.ndarray,
    lookup: dict[PairKey, int],
    clear_rows: set[int],
    clear_cols: set[int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
) -> bool:
    h, w = region.shape
    blocked_rows = [r for r in range(1, h - 1) if r not in clear_rows]
    blocked_cols = [c for c in range(1, w - 1) if c not in clear_cols]
    if not blocked_rows or not blocked_cols:
        return False

    for r in clear_rows:
        if not np.all(region[r, 1 : w - 1] == Color.BLACK):
            return False
    for c in clear_cols:
        if not np.all(region[1 : h - 1, c] == Color.BLACK):
            return False
    for r in blocked_rows:
        if not np.any(region[r, 1 : w - 1] != Color.BLACK):
            return False
    for c in blocked_cols:
        if not np.any(region[1 : h - 1, c] != Color.BLACK):
            return False

    if not any(
        row_colors[r] not in {int(region[r, 0]), int(region[r, w - 1])} for r in clear_rows
    ):
        return False
    if not any(
        col_colors[c] not in {int(region[0, c]), int(region[h - 1, c])} for c in clear_cols
    ):
        return False
    if not any(
        lookup[_pair_key(row_colors[r], col_colors[c])] not in {row_colors[r], col_colors[c]}
        for r in clear_rows
        for c in clear_cols
    ):
        return False

    filled = int(np.count_nonzero((expected != region) & (region == Color.BLACK)))
    interior_area = int((h - 2) * (w - 2))
    if filled <= 0 or filled > int(np.ceil(interior_area * 0.6)):
        return False

    visible_colors = set(map(int, expected.ravel())) - {Color.BLACK}
    if len(visible_colors) < 4:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    op_count = int(OP_COLOR_COUNT_CHOICES[int(rng.integers(0, len(OP_COLOR_COUNT_CHOICES)))])
    available = [c for c in range(1, 10)]
    picked = rng.choice(available, size=op_count + 1, replace=False)
    op_colors = [int(v) for v in picked[:op_count]]
    divider_color = int(picked[-1])

    lookup = _random_lookup(op_colors, rng)
    if lookup is None:
        return None

    panel_rows = _lookup_rows_to_panel(op_colors, lookup, rng)
    panel_h = int(panel_rows.shape[0])
    panel_w = PANEL_WIDTH

    corner = str(CORNERS[int(rng.integers(0, len(CORNERS)))])
    main_h = _sample_int(rng, *MAIN_HEIGHT_RANGE)
    main_w = _sample_int(rng, *MAIN_WIDTH_RANGE)
    if panel_h + 1 + main_h > 30 or panel_w + 1 + main_w > 30:
        return None

    clear_rows = _choose_clear_indices(main_h, CLEAR_ROW_RANGE, rng)
    clear_cols = _choose_clear_indices(main_w, CLEAR_COL_RANGE, rng)
    blocked_rows = [r for r in range(1, main_h - 1) if r not in clear_rows]
    blocked_cols = [c for c in range(1, main_w - 1) if c not in clear_cols]
    if not blocked_rows or not blocked_cols:
        return None

    layout = _layout_from_corner(corner, panel_h, panel_w, main_h, main_w)
    grid = np.zeros(layout["shape"], dtype=int)
    for r, c in layout["divider_cells"]:
        grid[r, c] = divider_color

    grid[layout["panel_rows"], layout["panel_cols"]] = panel_rows
    region = grid[layout["main_rows"], layout["main_cols"]]
    _paint_border(region, op_colors, rng)
    _seed_blockers(region, blocked_rows, blocked_cols, op_colors, rng)

    expected_region, row_colors, col_colors = _design_expected_region(
        region, lookup, clear_rows, clear_cols
    )
    if not _semantic_checks(
        region, expected_region, lookup, clear_rows, clear_cols, row_colors, col_colors
    ):
        return None

    expected = grid.copy()
    expected[layout["main_rows"], layout["main_cols"]] = expected_region
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor palette input")
