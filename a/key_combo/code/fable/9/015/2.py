from __future__ import annotations

import math
from collections import deque
import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 500

HEIGHT_RANGE = (10, 16)
BAR_SIDE_RANGE = (7, 10)
TARGET_SIDE_RANGE = (8, 16)
ACTIVE_ROW_RANGE = (3, 5)
BLOCK_COUNT_RANGE = (2, 3)
BLOCK_HEIGHT_RANGE = (2, 4)
BLOCK_WIDTH_RANGE = (2, 5)
BAR_COLOR_COUNT = 4
BLOCK_COLOR_COUNT = 3
MAX_BAR_LENGTH = 6


def _iter_runs(row: np.ndarray, c0: int, c1: int, bg: int = BACKGROUND):
    c = int(c0)
    while c < int(c1):
        if int(row[c]) == bg:
            c += 1
            continue
        color = int(row[c])
        start = c
        while c < int(c1) and int(row[c]) == color:
            c += 1
        yield start, c - 1, color


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_monochrome(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            b = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": b,
                    "area": len(coords),
                }
            )
    return comps


def _analyze_bar_side(side: np.ndarray, bg: int = BACKGROUND) -> dict | None:
    h, _ = side.shape
    rows: dict[int, list[dict]] = {}
    for r in range(h):
        runs = [
            {
                "c0": c0,
                "c1": c1,
                "color": color,
                "length": c1 - c0 + 1,
            }
            for c0, c1, color in _iter_runs(side[r], 0, side.shape[1], bg)
        ]
        if len(runs) > 2:
            return None
        if runs:
            rows[r] = runs
    if not rows:
        return None
    return {"rows": rows, "mass": int(np.count_nonzero(side != bg))}


def _analyze_block_side(side: np.ndarray, bg: int = BACKGROUND) -> dict | None:
    comps = _components_monochrome(side, bg)
    if not comps:
        return None
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 2 or width < 2:
            return None
        if comp["area"] != height * width:
            return None
    return {"components": comps, "mass": int(np.count_nonzero(side != bg))}


def _extract_bars(grid: np.ndarray, sep_col: int, bars_on_left: bool, bg: int = BACKGROUND) -> dict[int, list[dict]]:
    h, w = grid.shape
    bars_by_row: dict[int, list[dict]] = {}
    for r in range(h):
        if bars_on_left:
            runs = list(_iter_runs(grid[r], 0, sep_col, bg))
        else:
            runs = list(_iter_runs(grid[r], sep_col + 1, w, bg))
        bars: list[dict] = []
        for c0, c1, color in runs:
            dist = (sep_col - 1 - c1) if bars_on_left else (c0 - (sep_col + 1))
            bars.append(
                {
                    "c0": c0,
                    "c1": c1,
                    "color": color,
                    "length": c1 - c0 + 1,
                    "dist": dist,
                }
            )
        if bars:
            bars.sort(key=lambda bar: (bar["dist"], bar["c0"]))
            bars_by_row[r] = bars
    return bars_by_row


def _find_layout(grid: np.ndarray, bg: int = BACKGROUND) -> dict:
    h, w = grid.shape
    candidates: list[dict] = []
    for c in range(1, w - 1):
        col = grid[:, c]
        vals = col[col != bg]
        if vals.size < h - 1:
            continue
        if len({int(v) for v in vals.tolist()}) != 1:
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]

        left_bars = _analyze_bar_side(left, bg)
        right_blocks = _analyze_block_side(right, bg)
        if left_bars is not None and right_blocks is not None:
            candidates.append(
                {
                    "sep_col": c,
                    "sep_color": int(vals[0]),
                    "bars_on_left": True,
                    "bar_info": left_bars,
                    "block_info": right_blocks,
                }
            )

        left_blocks = _analyze_block_side(left, bg)
        right_bars = _analyze_bar_side(right, bg)
        if left_blocks is not None and right_bars is not None:
            candidates.append(
                {
                    "sep_col": c,
                    "sep_color": int(vals[0]),
                    "bars_on_left": False,
                    "bar_info": right_bars,
                    "block_info": left_blocks,
                }
            )

    assert len(candidates) == 1, f"expected one valid separator layout, found {len(candidates)}"
    out = candidates[0]
    out["candidate_count"] = len(candidates)
    return out


def _sample_columns(sep_col: int, width: int, bars_on_left: bool, step: int, start_mode: str = "length") -> list[int]:
    if step <= 0:
        return []
    if start_mode == "length":
        first = sep_col + step if bars_on_left else sep_col - step
    elif start_mode == "adjacent":
        first = sep_col + 1 if bars_on_left else sep_col - 1
    else:
        raise ValueError(start_mode)

    cols: list[int] = []
    c = first
    while 0 <= c < width:
        cols.append(int(c))
        c += step if bars_on_left else -step
    return cols


def _apply_rule(
    grid: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bars_by_row: dict[int, list[dict]],
    *,
    paint_on_blank: bool = True,
    carve_in_solid: bool = True,
    start_mode: str = "length",
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    processed = np.zeros(g.shape, dtype=bool)
    _, w = g.shape
    events: list[dict] = []

    for r in sorted(bars_by_row):
        for bar_index, bar in enumerate(bars_by_row[r]):
            color = int(bar["color"])
            length = int(bar["length"])
            for c in _sample_columns(sep_col, w, bars_on_left, length, start_mode):
                if processed[r, c]:
                    continue
                processed[r, c] = True
                action = "paint" if int(g[r, c]) == BACKGROUND else "carve"
                if action == "paint":
                    if paint_on_blank:
                        out[r, c] = color
                else:
                    if carve_in_solid:
                        out[r, c] = BACKGROUND
                events.append(
                    {
                        "row": int(r),
                        "col": int(c),
                        "action": action,
                        "length": length,
                        "color": color,
                        "bar_index": int(bar_index),
                    }
                )
    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply each row bar's own-length period across the separator, painting blanks and carving holes in solid blocks.

    Input structure:
    - The background is black.
    - A near-solid vertical separator column splits the grid into two sides.
    - One side contains one or two horizontal monochrome bars on some rows; the other side contains several solid monochrome rectangles on black.

    Rule:
    - Find the separator column, then identify which side contains the row bars and which side contains the solid blocks.
    - For each bar of length L, look along the same row on the opposite side at separator-relative offsets L, 2L, 3L, ... until leaving the grid.
    - If a sampled cell is black, paint it with the bar color; if it lies inside a solid block, change it to black.
    - When a row has multiple bars, process the one closer to the separator first, and any cell already touched by an earlier bar on that row is skipped by later bars.
    - Keep the separator, the original bars, and the untouched parts of the solid blocks unchanged.

    Key insight:
    - The painted dots and carved holes are the same periodic trace seen on different backgrounds, and the nearer bar's claim must be resolved before farther bars can act.
    """
    g = np.asarray(grid, dtype=int)
    layout = _find_layout(g, BACKGROUND)
    bars_by_row = _extract_bars(g, layout["sep_col"], layout["bars_on_left"], BACKGROUND)
    out, _ = _apply_rule(g, layout["sep_col"], layout["bars_on_left"], bars_by_row)
    return out


def _choose_spaced_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int], list[int]]:
    colors = np.array(COLOR_POOL, dtype=int)
    rng.shuffle(colors)
    sep_color = int(colors[0])
    bar_colors = [int(x) for x in colors[1 : 1 + BAR_COLOR_COUNT]]
    block_colors = [int(x) for x in colors[1 + BAR_COLOR_COUNT : 1 + BAR_COLOR_COUNT + BLOCK_COLOR_COUNT]]
    return sep_color, bar_colors, block_colors


def _feasible_conflict_pairs(target_width: int, bar_capacity: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    max_len = min(MAX_BAR_LENGTH, target_width)
    for near_len in range(2, max_len + 1):
        for far_len in range(2, max_len + 1):
            if near_len == far_len:
                continue
            if math.lcm(near_len, far_len) > target_width:
                continue
            if near_len + far_len > bar_capacity:
                continue
            pairs.append((near_len, far_len))
    return pairs


def _sample_row_spec(
    rng: np.random.Generator,
    *,
    special: bool,
    bar_side_width: int,
    target_width: int,
    bar_colors: list[int],
) -> dict | None:
    sep_gap = int(rng.integers(0, 2))
    capacity = bar_side_width - sep_gap
    if capacity < 1:
        return None

    if special:
        inter_gap = int(rng.integers(0, 2))
        pairs = _feasible_conflict_pairs(target_width, capacity - inter_gap)
        if not pairs:
            return None
        near_len, far_len = pairs[int(rng.integers(0, len(pairs)))]
        row_colors = [int(x) for x in rng.choice(bar_colors, size=2, replace=False)]
        return {
            "sep_gap": sep_gap,
            "inter_gaps": [inter_gap],
            "bars": [
                {"length": near_len, "color": row_colors[0]},
                {"length": far_len, "color": row_colors[1]},
            ],
        }

    want_two = bool(rng.integers(0, 2) == 0)
    if want_two and capacity >= 2:
        for _ in range(40):
            inter_gap = int(rng.integers(0, 3))
            usable = capacity - inter_gap
            if usable < 2:
                continue
            len1 = int(rng.integers(1, min(MAX_BAR_LENGTH, target_width, usable - 1) + 1))
            len2_max = min(MAX_BAR_LENGTH, target_width, usable - len1)
            if len2_max < 1:
                continue
            len2 = int(rng.integers(1, len2_max + 1))
            row_colors = [int(x) for x in rng.choice(bar_colors, size=2, replace=False)]
            return {
                "sep_gap": sep_gap,
                "inter_gaps": [inter_gap],
                "bars": [
                    {"length": len1, "color": row_colors[0]},
                    {"length": len2, "color": row_colors[1]},
                ],
            }

    length = int(rng.integers(1, min(MAX_BAR_LENGTH, target_width, capacity) + 1))
    return {
        "sep_gap": sep_gap,
        "inter_gaps": [],
        "bars": [{"length": length, "color": int(rng.choice(bar_colors))}],
    }


def _render_bars(
    grid: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    row_specs: dict[int, dict],
) -> dict[int, list[dict]]:
    bars_by_row: dict[int, list[dict]] = {}
    for r, spec in row_specs.items():
        bars = spec["bars"]
        bars_by_row[r] = [{"length": int(bar["length"]), "color": int(bar["color"])} for bar in bars]
        if bars_on_left:
            cursor = sep_col - int(spec["sep_gap"])
            for idx, bar in enumerate(bars):
                length = int(bar["length"])
                color = int(bar["color"])
                start = cursor - length
                grid[r, start:cursor] = color
                if idx < len(spec["inter_gaps"]):
                    cursor = start - int(spec["inter_gaps"][idx])
                else:
                    cursor = start
        else:
            cursor = sep_col + 1 + int(spec["sep_gap"])
            for idx, bar in enumerate(bars):
                length = int(bar["length"])
                color = int(bar["color"])
                end = cursor + length
                grid[r, cursor:end] = color
                if idx < len(spec["inter_gaps"]):
                    cursor = end + int(spec["inter_gaps"][idx])
                else:
                    cursor = end
    return bars_by_row


def _choose_anchor_cells(
    rng: np.random.Generator,
    events: list[dict],
    block_count: int,
) -> list[tuple[int, int]]:
    cells = [(int(event["row"]), int(event["col"])) for event in events]
    order = rng.permutation(len(cells))
    anchors: list[tuple[int, int]] = []
    for idx in order:
        rr, cc = cells[int(idx)]
        if all(abs(rr - ar) > 1 or abs(cc - ac) > 1 for ar, ac in anchors):
            anchors.append((rr, cc))
        if len(anchors) == block_count:
            break
    return anchors


def _place_blocks(
    rng: np.random.Generator,
    grid: np.ndarray,
    anchors: list[tuple[int, int]],
    block_colors: list[int],
    *,
    bars_on_left: bool,
    sep_col: int,
) -> list[dict] | None:
    h, w = grid.shape
    target_lo = sep_col + 1 if bars_on_left else 0
    target_hi = w - 1 if bars_on_left else sep_col - 1
    occupied = np.zeros((h, w), dtype=bool)
    blocks: list[dict] = []

    for anchor, color in zip(anchors, block_colors):
        ar, ac = anchor
        placed = False
        for _ in range(80):
            height = int(rng.integers(BLOCK_HEIGHT_RANGE[0], BLOCK_HEIGHT_RANGE[1] + 1))
            width = int(rng.integers(BLOCK_WIDTH_RANGE[0], BLOCK_WIDTH_RANGE[1] + 1))
            r0_lo = max(0, ar - height + 1)
            r0_hi = min(h - height, ar)
            c0_lo = max(target_lo, ac - width + 1)
            c0_hi = min(target_hi - width + 1, ac)
            if r0_lo > r0_hi or c0_lo > c0_hi:
                continue
            r0 = int(rng.integers(r0_lo, r0_hi + 1))
            c0 = int(rng.integers(c0_lo, c0_hi + 1))
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + height + 1)
            cc0 = max(target_lo, c0 - 1)
            cc1 = min(target_hi + 1, c0 + width + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            grid[r0 : r0 + height, c0 : c0 + width] = int(color)
            occupied[r0 : r0 + height, c0 : c0 + width] = True
            blocks.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + height - 1,
                    "c1": c0 + width - 1,
                    "color": int(color),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return blocks


def _reverse_priority_bars(bars_by_row: dict[int, list[dict]]) -> dict[int, list[dict]]:
    return {r: list(reversed(bars)) for r, bars in bars_by_row.items()}


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    bars_by_row: dict[int, list[dict]],
) -> bool:
    layout = _find_layout(inp, BACKGROUND)
    if layout["sep_col"] != sep_col or layout["bars_on_left"] != bars_on_left:
        return False

    extracted = _extract_bars(inp, sep_col, bars_on_left, BACKGROUND)
    if sorted(extracted) != sorted(bars_by_row):
        return False
    for row in bars_by_row:
        want = [(bar["length"], bar["color"]) for bar in bars_by_row[row]]
        got = [(bar["length"], bar["color"]) for bar in extracted[row]]
        if want != got:
            return False

    if not np.array_equal(main(inp), expected):
        return False

    true_out, events = _apply_rule(inp, sep_col, bars_on_left, bars_by_row)
    if not np.array_equal(true_out, expected):
        return False

    paint_count = sum(event["action"] == "paint" for event in events)
    carve_count = sum(event["action"] == "carve" for event in events)
    if paint_count == 0 or carve_count == 0:
        return False
    if not any(len(bars) == 2 for bars in bars_by_row.values()):
        return False

    alt_priority, _ = _apply_rule(inp, sep_col, bars_on_left, _reverse_priority_bars(bars_by_row))
    if np.array_equal(alt_priority, expected):
        return False

    paint_only, _ = _apply_rule(inp, sep_col, bars_on_left, bars_by_row, carve_in_solid=False)
    if np.array_equal(paint_only, expected):
        return False

    carve_only, _ = _apply_rule(inp, sep_col, bars_on_left, bars_by_row, paint_on_blank=False)
    if np.array_equal(carve_only, expected):
        return False

    adjacent_start, _ = _apply_rule(inp, sep_col, bars_on_left, bars_by_row, start_mode="adjacent")
    if np.array_equal(adjacent_start, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    bar_side_width = int(rng.integers(BAR_SIDE_RANGE[0], BAR_SIDE_RANGE[1] + 1))
    target_width = int(rng.integers(TARGET_SIDE_RANGE[0], TARGET_SIDE_RANGE[1] + 1))
    w = bar_side_width + 1 + target_width
    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_width if bars_on_left else target_width

    max_active = min(ACTIVE_ROW_RANGE[1], (h + 1) // 2)
    if max_active < ACTIVE_ROW_RANGE[0]:
        return None
    active_count = int(rng.integers(ACTIVE_ROW_RANGE[0], max_active + 1))
    active_rows = _choose_spaced_rows(rng, h, active_count)
    special_row = int(active_rows[int(rng.integers(0, len(active_rows)))])

    sep_color, bar_colors, block_colors = _sample_palette(rng)

    grid = np.zeros((h, w), dtype=int)
    grid[:, sep_col] = sep_color
    inactive_rows = [r for r in range(h) if r not in set(active_rows)]
    if inactive_rows and bool(rng.integers(0, 4) == 0):
        grid[int(rng.choice(inactive_rows)), sep_col] = BACKGROUND

    row_specs: dict[int, dict] = {}
    for row in active_rows:
        spec = _sample_row_spec(
            rng,
            special=(row == special_row),
            bar_side_width=bar_side_width,
            target_width=target_width,
            bar_colors=bar_colors,
        )
        if spec is None:
            return None
        row_specs[int(row)] = spec

    bars_by_row = _render_bars(grid, sep_col, bars_on_left, row_specs)
    _, empty_events = _apply_rule(grid, sep_col, bars_on_left, bars_by_row)
    if len(empty_events) < 4:
        return None

    block_count = int(rng.integers(BLOCK_COUNT_RANGE[0], BLOCK_COUNT_RANGE[1] + 1))
    anchors = _choose_anchor_cells(rng, empty_events, block_count)
    if len(anchors) < block_count:
        return None

    blocks = _place_blocks(
        rng,
        grid,
        anchors,
        block_colors[:block_count],
        bars_on_left=bars_on_left,
        sep_col=sep_col,
    )
    if blocks is None:
        return None

    expected, _ = _apply_rule(grid, sep_col, bars_on_left, bars_by_row)
    if not _validate_case(grid, expected, sep_col, bars_on_left, bars_by_row):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/9/015/2")
