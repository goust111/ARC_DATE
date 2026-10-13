from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, get_bg


GREY = Color.GREY
BLACK = Color.BLACK
PIXEL_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

GEN_RETRIES = 400


def _is_open_gap(grid: np.ndarray, r: int, c: int, bg: int, guide_color: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if c == 0 or c == w - 1:
        return False
    return (
        int(grid[r, c]) == bg
        and int(grid[r, c - 1]) == guide_color
        and int(grid[r, c + 1]) == guide_color
    )


def _simulate(grid: np.ndarray, *, bg: int = BLACK, guide_color: int = GREY) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    pixel_mask = (out != bg) & (out != guide_color)
    pixel_coords = np.argwhere(pixel_mask)
    out[pixel_mask] = bg

    pixels_by_col: dict[int, list[tuple[int, int]]] = {}
    for r, c in pixel_coords:
        pixels_by_col.setdefault(int(c), []).append((int(r), int(g[r, c])))

    for c, items in pixels_by_col.items():
        items.sort(reverse=True)
        for start_r, color in items:
            settled = False
            for r in range(start_r + 1, out.shape[0]):
                if _is_open_gap(out, r, c, bg, guide_color):
                    out[r, c] = guide_color
                    settled = True
                    break
                if int(out[r, c]) != bg:
                    land_r = r - 1
                    if land_r >= 0:
                        out[land_r, c] = color
                    settled = True
                    break
            if not settled:
                continue

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """Let colored pixels fall column by column so they plug grey guide-bar gaps or stack above the first support they reach.

    Input structure:
    - The background is black.
    - One-cell-thick horizontal grey guide bars occupy some row segments, and some guide cells are missing as single-cell black gaps between grey neighbors.
    - Colored single pixels are scattered above the guides, and multiple pixels may share a column.

    Rule:
    - Process each column independently, taking its colored pixels from lower to higher.
    - A falling pixel is consumed if it passes through the first open guide gap in that column, and that gap cell turns grey.
    - Otherwise the pixel stops immediately above the first already-solid cell below it, where solid means a grey guide cell or a previously settled colored pixel.
    - If a pixel has no guide cell, no open gap, and no settled support anywhere below, it disappears; keep the original guide bars plus every gap that got plugged.

    Key insight:
    - Open gaps are consumable supports: the first low pixel closes the hole, which changes the resting height for later pixels in the same column.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    return _simulate(g, bg=bg, guide_color=GREY)


def _pick_rows(h: int, count: int) -> list[int]:
    candidates = list(range(4, h - 2))
    random.shuffle(candidates)
    rows: list[int] = []
    for r in candidates:
        if all(abs(r - prev) >= 3 for prev in rows):
            rows.append(r)
            if len(rows) == count:
                break
    return sorted(rows)


def _interval_without_touch(
    existing: list[tuple[int, int]],
    width: int,
    w: int,
    blocked_cols: set[int] | None = None,
) -> tuple[int, int] | None:
    blocked_cols = blocked_cols or set()
    starts = list(range(0, w - width + 1))
    random.shuffle(starts)
    for c0 in starts:
        c1 = c0 + width - 1
        if any(c0 <= blocked <= c1 for blocked in blocked_cols):
            continue
        if all(c1 + 2 < e0 or e1 + 2 < c0 for e0, e1 in existing):
            return c0, c1
    return None


def _draw_interval(grid: np.ndarray, row: int, c0: int, c1: int, gaps: set[int]) -> None:
    for c in range(c0, c1 + 1):
        if c not in gaps:
            grid[row, c] = GREY


def _feature_map(guide: np.ndarray) -> list[list[tuple[int, str]]]:
    h, w = guide.shape
    features: list[list[tuple[int, str]]] = [[] for _ in range(w)]
    for c in range(w):
        for r in range(h):
            if int(guide[r, c]) == GREY:
                features[c].append((r, "solid"))
            elif _is_open_gap(guide, r, c, BLACK, GREY):
                features[c].append((r, "gap"))
    return features


def _choose_extra_columns(used: set[int], pools: dict[str, list[int]]) -> list[tuple[int, str]]:
    options: list[tuple[int, str]] = []
    for kind, cols in pools.items():
        for c in cols:
            if c not in used:
                options.append((c, kind))
    random.shuffle(options)
    if not options:
        return []
    extra: list[tuple[int, str]] = []
    for c, kind in options[: random.randint(1, min(4, len(options)))]:
        if c not in used:
            used.add(c)
            extra.append((c, kind))
    return extra


def _sample_rows(limit_row: int, count: int) -> list[int] | None:
    if limit_row <= 0 or count > limit_row:
        return None
    rows = random.sample(range(limit_row), count)
    rows.sort()
    return rows


def _build_case(return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    h = random.randint(14, 24)
    w = random.randint(14, 26)

    guide = np.zeros((h, w), dtype=int)
    row_count = random.randint(2, 4)
    guide_rows = _pick_rows(h, row_count)
    if len(guide_rows) != row_count:
        raise ValueError("failed to space guide rows")

    primary_row = guide_rows[0]
    primary_width = random.randint(max(6, w // 3), w - 4)
    primary_c0 = random.randint(1, w - primary_width - 1)
    primary_c1 = primary_c0 + primary_width - 1
    gap_count = random.randint(1, min(3, primary_width // 3))
    gap_candidates = list(range(primary_c0 + 1, primary_c1))
    random.shuffle(gap_candidates)
    primary_gaps: set[int] = set()
    for c in gap_candidates:
        if all(abs(c - prev) >= 2 for prev in primary_gaps):
            primary_gaps.add(c)
            if len(primary_gaps) == gap_count:
                break
    if not primary_gaps:
        raise ValueError("primary guide needs a gap")
    _draw_interval(guide, primary_row, primary_c0, primary_c1, primary_gaps)
    solid_candidates = [c for c in range(primary_c0, primary_c1 + 1) if c not in primary_gaps]
    if not solid_candidates:
        raise ValueError("primary guide needs a solid column")
    outside_primary = [c for c in range(w) if c < primary_c0 or c > primary_c1]
    if not outside_primary:
        raise ValueError("primary guide must leave a void column")
    reserved_gap_col = random.choice(sorted(primary_gaps))
    reserved_solid_col = random.choice(solid_candidates)
    reserved_void_col = random.choice(outside_primary)

    intervals_by_row: dict[int, list[tuple[int, int]]] = {primary_row: [(primary_c0, primary_c1)]}
    for row in guide_rows[1:]:
        intervals: list[tuple[int, int]] = []
        target_intervals = random.randint(1, 2)
        for _ in range(target_intervals):
            width = random.randint(4, min(9, w - 2))
            span = _interval_without_touch(intervals, width, w, blocked_cols={reserved_void_col})
            if span is None:
                continue
            c0, c1 = span
            gap_count = random.randint(0, min(2, max(0, width - 2)))
            gap_positions: set[int] = set()
            candidates = list(range(c0 + 1, c1))
            random.shuffle(candidates)
            for c in candidates:
                if all(abs(c - prev) >= 2 for prev in gap_positions):
                    gap_positions.add(c)
                    if len(gap_positions) == gap_count:
                        break
            _draw_interval(guide, row, c0, c1, gap_positions)
            intervals.append((c0, c1))
        intervals_by_row[row] = intervals

    features = _feature_map(guide)
    gap_cols = [c for c in range(w) if features[c] and features[c][0][1] == "gap" and features[c][0][0] >= 2]
    solid_cols = [c for c in range(w) if features[c] and features[c][0][1] == "solid" and features[c][0][0] >= 2]
    void_cols = [c for c in range(w) if not features[c]]
    if reserved_gap_col not in gap_cols or reserved_solid_col not in solid_cols or reserved_void_col not in void_cols:
        raise ValueError("reserved columns lost their intended roles")
    if not gap_cols or not solid_cols or not void_cols:
        raise ValueError("missing required column types")

    used_cols: set[int] = set()
    assignments = [
        (reserved_gap_col, "gap_chain"),
        (reserved_solid_col, "solid"),
        (reserved_void_col, "void"),
    ]
    used_cols.update(c for c, _ in assignments)

    extra_pools = {
        "gap_chain": gap_cols,
        "solid": solid_cols,
        "void": void_cols,
    }
    assignments.extend(_choose_extra_columns(used_cols, extra_pools))

    grid = guide.copy()
    palette = random.sample(PIXEL_COLORS, random.randint(2, min(5, len(PIXEL_COLORS))))
    scenarios: list[dict] = []

    for col, kind in assignments:
        top_feature_row = features[col][0][0] if features[col] else h
        if kind == "gap_chain":
            count = random.randint(2, min(4, top_feature_row))
        elif kind == "solid":
            count = random.randint(1, min(3, top_feature_row))
        else:
            count = random.randint(1, min(2, h - 2))
        rows = _sample_rows(top_feature_row if kind != "void" else h - 1, count)
        if rows is None:
            raise ValueError("not enough vertical room for pixels")
        colors = [random.choice(palette) for _ in rows]
        for r, color in zip(rows, colors):
            grid[r, col] = color
        scenarios.append({"col": col, "kind": kind, "rows": rows, "colors": colors})

    expected = _simulate(grid)

    stats = {
        "gap_columns": sum(s["kind"] == "gap_chain" for s in scenarios),
        "solid_columns": sum(s["kind"] == "solid" for s in scenarios),
        "void_columns": sum(s["kind"] == "void" for s in scenarios),
        "filled_gaps": int(((expected == GREY) & (guide != GREY)).sum()),
        "survivors": int(np.count_nonzero((expected != BLACK) & (expected != GREY))),
        "input_pixels": int(np.count_nonzero((grid != BLACK) & (grid != GREY))),
    }

    if stats["filled_gaps"] < 1 or stats["survivors"] < 1:
        raise ValueError("degenerate sample")

    meta = {"guide": guide, "expected": expected, "scenarios": scenarios, "stats": stats}
    return (grid, meta) if return_meta else grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, meta = _build_case(return_meta=True)
        except ValueError:
            continue
        if not np.array_equal(main(grid), meta["expected"]):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")


if __name__ == "__main__":
    case = generate_input()
    print(case)
    print(main(case))
