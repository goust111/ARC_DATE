from __future__ import annotations

import random

import numpy as np


BG = 0
BAR = 5
BRIDGE = 9
RANK_COLORS = {
    1: 1,  # blue
    2: 2,  # red
    3: 4,  # yellow
    4: 3,  # green
    5: 7,  # orange
}

_GEN_CALLS = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_bars(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    bars: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BAR:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BAR:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = len(coords)
            bh = r1 - r0 + 1
            bw = c1 - c0 + 1
            assert bh == 1 or bw == 1, "every object must be a 1-pixel-thick straight bar"
            assert area == max(bh, bw), "bar must fully occupy its bbox"
            orient = "H" if bh == 1 else "V"
            bars.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (r0, c0, r1, c1),
                    "length": area,
                    "orient": orient,
                }
            )
    return bars


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _gap_rect(a: dict, b: dict) -> tuple[int, int, int, int] | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    row0 = max(ar0, br0)
    row1 = min(ar1, br1)
    if row0 > row1:
        return None
    if ac1 < bc0:
        col0 = ac1 + 1
        col1 = bc0 - 1
    elif bc1 < ac0:
        col0 = bc1 + 1
        col1 = ac0 - 1
    else:
        return None
    if col0 > col1:
        return None
    return (row0, col0, row1, col1)


def _rect_blocked(rect: tuple[int, int, int, int], bars: list[dict], skip: set[int]) -> bool:
    row0, col0, row1, col1 = rect
    for idx, bar in enumerate(bars):
        if idx in skip:
            continue
        br0, bc0, br1, bc1 = bar["bbox"]
        if _ranges_overlap(row0, row1, br0, br1) and _ranges_overlap(col0, col1, bc0, bc1):
            return True
    return False


def _pair_facts(bars: list[dict]) -> list[dict]:
    ranked = sorted(bars, key=lambda bar: bar["length"], reverse=True)
    by_id = {id(bar): rank for rank, bar in enumerate(ranked, start=1)}
    facts: list[dict] = []
    for i, a in enumerate(bars):
        for j in range(i + 1, len(bars)):
            b = bars[j]
            rect = _gap_rect(a, b)
            blocked = rect is not None and _rect_blocked(rect, bars, {i, j})
            facts.append(
                {
                    "i": i,
                    "j": j,
                    "rank_i": by_id[id(a)],
                    "rank_j": by_id[id(b)],
                    "rank_diff": abs(by_id[id(a)] - by_id[id(b)]),
                    "row_overlap": _ranges_overlap(a["bbox"][0], a["bbox"][2], b["bbox"][0], b["bbox"][2]),
                    "separated": rect is not None,
                    "rect": rect,
                    "blocked": blocked,
                    "bridges": rect is not None and not blocked and abs(by_id[id(a)] - by_id[id(b)]) == 1,
                }
            )
    return facts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor five gray thin bars by length rank and bridge only unobstructed rank-adjacent horizontal gaps.

    Input structure:
    - The grid has a black background and exactly five gray 4-connected straight one-cell-thick bars.
    - Each bar is horizontal or vertical, all five lengths are distinct, and the bars do not touch.
    - Some bar pairs are horizontally separated while their row spans overlap.

    Rule:
    - Rank the gray bars by length from longest to shortest.
    - Recolor ranks 1..5 as blue, red, yellow, green, orange.
    - For each pair whose ranks differ by exactly 1, if their row spans overlap and their bboxes are horizontally separated, take the shared-row horizontal gap between the bboxes.
    - Fill that gap maroon only when no third bar bbox crosses the same gap columns while also overlapping the same shared rows.

    Key insight:
    - Length ranking branches into two uses at once: it determines both each bar's visible recolor and which spatially overlapping pairs are allowed to form bridges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BG, BAR}, "input must contain only black background and gray bars"

    bars = _component_bars(g)
    assert len(bars) == 5, "expected exactly five bars"
    lengths = [bar["length"] for bar in bars]
    assert len(set(lengths)) == 5, "bar lengths must be distinct"
    assert all(3 <= length <= 10 for length in lengths), "bar lengths must stay within the puzzle range"

    out = g.copy()
    ranked = sorted(bars, key=lambda bar: bar["length"], reverse=True)
    bar_indices = {id(bar): idx for idx, bar in enumerate(bars)}
    for rank, bar in enumerate(ranked, start=1):
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = RANK_COLORS[rank]

    for rank_idx in range(4):
        a = ranked[rank_idx]
        b = ranked[rank_idx + 1]
        rect = _gap_rect(a, b)
        if rect is None:
            continue
        if _rect_blocked(rect, bars, {bar_indices[id(a)], bar_indices[id(b)]}):
            continue
        row0, col0, row1, col1 = rect
        assert np.all(g[row0 : row1 + 1, col0 : col1 + 1] == BG), "bridge region must be empty in the input"
        out[row0 : row1 + 1, col0 : col1 + 1] = BRIDGE
    return out


def _bar_cells(r: int, c: int, length: int, orient: str) -> list[tuple[int, int]]:
    if orient == "H":
        return [(r, c + dc) for dc in range(length)]
    return [(r + dr, c) for dr in range(length)]


def _can_place(cells: list[tuple[int, int]], placed: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    for bar in placed:
        other = bar["cell_set"]
        if cell_set & other:
            return False
        for r, c in cell_set:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) in other:
                        return False
    return True


def _make_bar(r: int, c: int, length: int, orient: str, rank: int) -> dict:
    cells = _bar_cells(r, c, length, orient)
    rows = [p[0] for p in cells]
    cols = [p[1] for p in cells]
    return {
        "r": r,
        "c": c,
        "length": length,
        "orient": orient,
        "rank": rank,
        "cells": cells,
        "cell_set": set(cells),
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
    }


def _paint_bars(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for bar in bars:
        for r, c in bar["cells"]:
            grid[r, c] = BAR
    return grid


def _sorted_lengths() -> list[int]:
    return sorted(random.sample(range(3, 10), 5), reverse=True)


def _place_horizontal_extra(
    placed: list[dict],
    shape: tuple[int, int],
    row_choices: list[int],
    length: int,
    rank: int,
) -> dict | None:
    h, w = shape
    starts = [(r, c) for r in row_choices for c in range(0, w - length + 1)]
    random.shuffle(starts)
    for r, c in starts:
        bar = _make_bar(r, c, length, "H", rank)
        if _can_place(bar["cells"], placed, shape):
            return bar
    return None


def _place_vertical_extra(
    placed: list[dict],
    shape: tuple[int, int],
    col_choices: list[int],
    length: int,
    rank: int,
    top_min: int,
    top_max: int,
) -> dict | None:
    if top_min > top_max:
        return None
    starts = [(r, c) for c in col_choices for r in range(top_min, top_max + 1)]
    random.shuffle(starts)
    for r, c in starts:
        bar = _make_bar(r, c, length, "V", rank)
        if _can_place(bar["cells"], placed, shape):
            return bar
    return None


def _vertical_columns(width: int) -> tuple[int, int, int] | None:
    gap1 = random.randint(1, 4)
    gap2 = random.randint(1, 4)
    used = 3 + gap1 + gap2
    if used > width - 2:
        return None
    start = random.randint(1, width - used - 1)
    c1 = start
    c2 = c1 + gap1 + 1
    c3 = c2 + gap2 + 1
    return c1, c2, c3


def _vertical_star(shape: tuple[int, int], lengths: list[int]) -> list[dict] | None:
    h, w = shape
    l1, l2, l3, l4, l5 = lengths
    cols = _vertical_columns(w)
    if cols is None:
        return None
    c1, c2, c3 = cols

    if h - l1 - 5 < 2:
        return None
    t1 = random.randint(2, h - l1 - 3)
    b1 = t1 + l1 - 1

    cand2 = []
    for t2 in range(2, h - l2 - 2):
        b2 = t2 + l2 - 1
        if _ranges_overlap(t1, b1, t2, b2):
            cand2.append((t2, b2))
    cand3 = []
    for t3 in range(2, h - l3 - 2):
        b3 = t3 + l3 - 1
        if _ranges_overlap(t1, b1, t3, b3):
            cand3.append((t3, b3))

    pairs = []
    center1 = (t1 + b1) / 2
    for t2, b2 in cand2:
        if (t2 + b2) / 2 >= center1:
            continue
        for t3, b3 in cand3:
            if (t3 + b3) / 2 <= center1:
                continue
            if b2 + 1 < t3:
                pairs.append(((t2, b2), (t3, b3)))
    if not pairs:
        return None

    (t2, _b2), (t3, _b3) = random.choice(pairs)
    core_cols = (c1, c2, c3) if random.random() < 0.5 else (c3, c2, c1)
    placed: list[dict] = []
    for rank, (top, col, length) in (
        (1, (t1, core_cols[0], l1)),
        (2, (t2, core_cols[1], l2)),
        (3, (t3, core_cols[2], l3)),
    ):
        bar = _make_bar(top, col, length, "V", rank)
        if not _can_place(bar["cells"], placed, shape):
            return None
        placed.append(bar)

    min_top = min(bar["bbox"][0] for bar in placed)
    max_bot = max(bar["bbox"][2] for bar in placed)
    extra4 = _place_horizontal_extra(placed, shape, list(range(0, min_top - 1)), l4, 4)
    if extra4 is None:
        return None
    placed.append(extra4)
    extra5 = _place_horizontal_extra(placed, shape, list(range(max_bot + 2, h)), l5, 5)
    if extra5 is None:
        return None
    placed.append(extra5)
    return placed


def _vertical_chain(shape: tuple[int, int], lengths: list[int]) -> list[dict] | None:
    h, w = shape
    l1, l2, l3, l4, l5 = lengths
    cols = _vertical_columns(w)
    if cols is None:
        return None
    core_cols = list(cols)
    if random.random() < 0.5:
        core_cols.reverse()

    ovh = random.randint(2, 4)
    ov0 = random.randint(4, h - ovh - 5)
    ov1 = ov0 + ovh - 1

    placed: list[dict] = []
    for rank, col, length in zip((1, 2, 3), core_cols, (l1, l2, l3)):
        lo = max(2, ov1 - length + 1)
        hi = min(ov0, h - length - 3)
        if lo > hi:
            return None
        top = random.randint(lo, hi)
        bar = _make_bar(top, col, length, "V", rank)
        if not _can_place(bar["cells"], placed, shape):
            return None
        placed.append(bar)

    min_top = min(bar["bbox"][0] for bar in placed)
    extra4 = _place_horizontal_extra(placed, shape, list(range(0, min_top - 1)), l4, 4)
    if extra4 is None:
        return None
    placed.append(extra4)
    right_cols = list(range(max(bar["bbox"][3] for bar in placed[:3]) + 2, w))
    extra5 = _place_vertical_extra(
        placed,
        shape,
        right_cols,
        l5,
        5,
        max(2, ov0 - l5 + 1),
        min(ov1, h - l5 - 1),
    )
    if extra5 is None:
        return None
    placed.append(extra5)
    return placed


def _horizontal_chain(shape: tuple[int, int], lengths: list[int]) -> list[dict] | None:
    h, w = shape
    l1, l2, l3, l4, l5 = lengths
    gap1 = random.randint(1, 4)
    gap2 = random.randint(1, 4)
    used = l1 + l2 + l3 + gap1 + gap2
    if used > w - 2:
        return None
    row = random.randint(4, h - 5)
    start = random.randint(1, w - used - 1)
    cols = [start, start + l1 + gap1, start + l1 + gap1 + l2 + gap2]
    lengths_core = [l1, l2, l3]
    if random.random() < 0.5:
        cols.reverse()
        lengths_core = [l3, l2, l1]

    placed: list[dict] = []
    for rank, col, length in zip((1, 2, 3), cols, lengths_core):
        bar = _make_bar(row, col, length, "H", rank)
        if not _can_place(bar["cells"], placed, shape):
            return None
        placed.append(bar)

    right_cols = list(range(max(bar["bbox"][3] for bar in placed) + 2, w))
    extra4 = _place_horizontal_extra(placed, shape, [0, 1], l4, 4)
    if extra4 is None:
        return None
    placed.append(extra4)
    extra5 = _place_vertical_extra(
        placed,
        shape,
        right_cols,
        l5,
        5,
        max(2, row - l5 + 1),
        min(row, h - l5 - 1),
    )
    if extra5 is None:
        return None
    placed.append(extra5)
    return placed


def _build_input(template: str) -> np.ndarray | None:
    if template == "horizontal_chain":
        shape = (random.randint(16, 22), random.randint(22, 28))
    else:
        shape = (random.randint(18, 24), random.randint(16, 22))
    lengths = _sorted_lengths()
    if template == "vertical_star":
        bars = _vertical_star(shape, lengths)
    elif template == "vertical_chain":
        bars = _vertical_chain(shape, lengths)
    else:
        bars = _horizontal_chain(shape, lengths)
    if bars is None:
        return None
    random.shuffle(bars)
    return _paint_bars(shape, bars)


def _validate_semantics(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        bars = _component_bars(grid)
        if len(bars) != 5:
            return False
        lengths = [bar["length"] for bar in bars]
        if len(set(lengths)) != 5 or not all(3 <= length <= 10 for length in lengths):
            return False
        out = main(grid)
    except Exception:
        return False

    facts = _pair_facts(bars)
    bridges = [fact for fact in facts if fact["bridges"]]
    if not bridges:
        return False

    overlap_pairs = [fact for fact in facts if fact["row_overlap"] and fact["separated"]]
    if len(overlap_pairs) < 2:
        return False

    nonadj_unblocked = [
        fact
        for fact in facts
        if fact["rank_diff"] > 1 and fact["row_overlap"] and fact["separated"] and not fact["blocked"]
    ]
    if not nonadj_unblocked:
        return False

    adjacent_nonbridge = [
        fact
        for fact in facts
        if fact["rank_diff"] == 1 and (not fact["row_overlap"] or not fact["separated"] or fact["blocked"])
    ]
    if not adjacent_nonbridge:
        return False

    if int(np.sum(out == BRIDGE)) == 0:
        return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    templates = ("vertical_star", "vertical_chain", "horizontal_chain")
    preferred = templates[_GEN_CALLS % len(templates)]
    _GEN_CALLS += 1

    for _ in range(400):
        order = [preferred] + [tpl for tpl in templates if tpl != preferred]
        if random.random() < 0.5:
            order = [preferred] + random.sample([tpl for tpl in templates if tpl != preferred], 2)
        for template in order:
            grid = _build_input(template)
            if grid is not None and _validate_semantics(grid):
                return grid
    raise RuntimeError("failed to generate a valid ranked-bar bridging input")
