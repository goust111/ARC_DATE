from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
TEAL = 8

GRID_RANGE_BY_RECT_COUNT = {
    2: ((14, 18), (14, 18)),
    3: ((16, 20), (16, 20)),
    4: ((18, 22), (18, 22)),
}
RECT_COUNT_RANGE = (2, 4)
RECT_HEIGHT_RANGE = (5, 8)
RECT_WIDTH_RANGE = (5, 8)
EDGE_PAD = 1
MIN_RECT_GAP = 2
GAP_REPAIRED_LEN = 3
UNREPAIRED_GAP_LEN_RANGE = (1, 3)
DISTRACTOR_PAIR_RANGE = (1, 2)
RECT_PLACEMENT_TRIES = 400
PAIR_PLACEMENT_TRIES = 300
GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _gap_cells(rect: tuple[int, int, int, int], side: str, start: int, length: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return [(r0, c) for c in range(start, start + length)]
    if side == "bottom":
        return [(r1, c) for c in range(start, start + length)]
    if side == "left":
        return [(r, c0) for r in range(start, start + length)]
    return [(r, c1) for r in range(start, start + length)]


def _interval_options(rect: tuple[int, int, int, int], side: str, length: int) -> list[int]:
    r0, c0, r1, c1 = rect
    if side in ("top", "bottom"):
        lo = c0 + 1
        hi = c1 - length
    else:
        lo = r0 + 1
        hi = r1 - length
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _draw_connections(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape

    for r in range(h):
        cols = np.flatnonzero(out[r] == RED)
        if len(cols) == 2:
            a, b = map(int, cols)
            if b > a + 1:
                segment = out[r, a + 1 : b]
                segment[segment == BLACK] = TEAL

    for c in range(w):
        rows = np.flatnonzero(out[:, c] == RED)
        if len(rows) == 2:
            a, b = map(int, rows)
            if b > a + 1:
                segment = out[a + 1 : b, c]
                segment[segment == BLACK] = TEAL

    return out


def _black_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] != BLACK:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"cells": cells, "touches_border": touches_border})
    return comps


def _enclosed_black_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return [comp["cells"] for comp in _black_components(grid) if not comp["touches_border"]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect red marker pairs to close broken blue frames, then flood-fill the newly enclosed black regions yellow.

    Input structure:
    - A black background contains 2-4 separated blue rectangular frame fragments.
    - Each rectangle has 1-2 missing border segments; some missing segments contain two red markers aligned on that gap.
    - Additional red marker pairs may appear on otherwise empty rows or columns as distractor connections.

    Rule:
    - Preserve the input pixels, then in every row with exactly two red markers paint teal on the black cells strictly between them.
    - Do the same for every column with exactly two red markers.
    - After all teal connections are drawn, find every 4-connected black component and recolor it yellow iff it does not touch the grid border.
    - Leave blue frame pixels, red markers, teal connections, and border-reaching black background unchanged.

    Key insight:
    - The blue fragments alone are open; only after drawing the red-guided teal segments do some black regions become truly enclosed, so the fill must be computed from the post-connection topology.
    """
    connected = _draw_connections(np.asarray(grid, dtype=int))
    out = connected.copy()
    for comp in _black_components(connected):
        if not comp["touches_border"]:
            for r, c in comp["cells"]:
                out[r, c] = YELLOW
    return out


def _rect_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = MIN_RECT_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _sample_rectangles(h: int, w: int, n_rects: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(RECT_PLACEMENT_TRIES):
        rects = []
        dims = [
            (
                random.randint(*RECT_HEIGHT_RANGE),
                random.randint(*RECT_WIDTH_RANGE),
            )
            for _ in range(n_rects)
        ]
        dims.sort(reverse=True)
        ok = True
        for rh, rw in dims:
            placed = False
            for _ in range(RECT_PLACEMENT_TRIES):
                if rh + 2 * EDGE_PAD > h or rw + 2 * EDGE_PAD > w:
                    ok = False
                    break
                r0 = random.randint(EDGE_PAD, h - rh - EDGE_PAD)
                c0 = random.randint(EDGE_PAD, w - rw - EDGE_PAD)
                rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if all(_rect_separated(rect, other) for other in rects):
                    rects.append(rect)
                    placed = True
                    break
            if not ok or not placed:
                ok = False
                break
        if ok and len(rects) == n_rects:
            random.shuffle(rects)
            return rects
    return None


def _pattern_list(n_rects: int) -> list[str]:
    patterns = ["open1"]
    if n_rects == 2:
        patterns.append(random.choice(["closed1", "closed2"]))
    elif n_rects == 3:
        patterns.extend(["closed2", random.choice(["closed1", "open0"])])
    else:
        patterns.extend(["closed2", "closed1", random.choice(["open0", "closed1"])])
    random.shuffle(patterns)
    return patterns


def _gap_plan_for_pattern(pattern: str) -> list[bool]:
    if pattern == "closed1":
        return [True]
    if pattern == "closed2":
        return [True, True]
    if pattern == "open0":
        return [False] if random.random() < 0.5 else [False, False]
    return [True, False]


def _add_horizontal_pair(
    pair_row: int,
    c0: int,
    c1: int,
    used_pair_rows: set[int],
    used_pair_cols: set[int],
    used_red_rows: set[int],
    used_red_cols: set[int],
) -> bool:
    if pair_row in used_pair_rows or pair_row in used_red_rows:
        return False
    if c0 in used_red_cols or c1 in used_red_cols or c0 in used_pair_cols or c1 in used_pair_cols:
        return False
    used_pair_rows.add(pair_row)
    used_red_cols.add(c0)
    used_red_cols.add(c1)
    return True


def _add_vertical_pair(
    pair_col: int,
    r0: int,
    r1: int,
    used_pair_rows: set[int],
    used_pair_cols: set[int],
    used_red_rows: set[int],
    used_red_cols: set[int],
) -> bool:
    if pair_col in used_pair_cols or pair_col in used_red_cols:
        return False
    if r0 in used_red_rows or r1 in used_red_rows or r0 in used_pair_rows or r1 in used_pair_rows:
        return False
    used_pair_cols.add(pair_col)
    used_red_rows.add(r0)
    used_red_rows.add(r1)
    return True


def _choose_gaps(
    rect: tuple[int, int, int, int],
    pattern: str,
    used_pair_rows: set[int],
    used_pair_cols: set[int],
    used_red_rows: set[int],
    used_red_cols: set[int],
) -> list[dict] | None:
    sides = ["top", "bottom", "left", "right"]
    plan = _gap_plan_for_pattern(pattern)

    for _ in range(160):
        gap_specs = []
        random.shuffle(sides)
        chosen_sides = sides[: len(plan)]
        valid = True
        tmp_pair_rows = set(used_pair_rows)
        tmp_pair_cols = set(used_pair_cols)
        tmp_red_rows = set(used_red_rows)
        tmp_red_cols = set(used_red_cols)

        kinds = plan[:]
        random.shuffle(kinds)
        for side, repaired in zip(chosen_sides, kinds):
            if repaired:
                length = GAP_REPAIRED_LEN
            else:
                max_len = 3
                if side in ("top", "bottom"):
                    max_len = min(max_len, rect[3] - rect[1] - 1)
                else:
                    max_len = min(max_len, rect[2] - rect[0] - 1)
                if max_len < 1:
                    valid = False
                    break
                length = random.randint(1, max_len)

            starts = _interval_options(rect, side, length)
            random.shuffle(starts)
            picked = None
            for start in starts:
                cells = _gap_cells(rect, side, start, length)
                if repaired:
                    first = cells[0]
                    last = cells[-1]
                    if side in ("top", "bottom"):
                        if not _add_horizontal_pair(
                            first[0],
                            first[1],
                            last[1],
                            tmp_pair_rows,
                            tmp_pair_cols,
                            tmp_red_rows,
                            tmp_red_cols,
                        ):
                            continue
                    else:
                        if not _add_vertical_pair(
                            first[1],
                            first[0],
                            last[0],
                            tmp_pair_rows,
                            tmp_pair_cols,
                            tmp_red_rows,
                            tmp_red_cols,
                        ):
                            continue
                picked = {
                    "side": side,
                    "cells": cells,
                    "repaired": repaired,
                }
                break
            if picked is None:
                valid = False
                break
            gap_specs.append(picked)

        if valid:
            used_pair_rows.clear()
            used_pair_rows.update(tmp_pair_rows)
            used_pair_cols.clear()
            used_pair_cols.update(tmp_pair_cols)
            used_red_rows.clear()
            used_red_rows.update(tmp_red_rows)
            used_red_cols.clear()
            used_red_cols.update(tmp_red_cols)
            return gap_specs
    return None


def _render_rectangles(
    h: int,
    w: int,
    rects: list[tuple[int, int, int, int]],
    patterns: list[str],
) -> tuple[np.ndarray, list[dict], dict] | None:
    grid = np.zeros((h, w), dtype=int)
    used_pair_rows: set[int] = set()
    used_pair_cols: set[int] = set()
    used_red_rows: set[int] = set()
    used_red_cols: set[int] = set()
    rect_meta: list[dict] = []
    pairs: list[dict] = []

    for rect, pattern in zip(rects, patterns):
        gap_specs = _choose_gaps(rect, pattern, used_pair_rows, used_pair_cols, used_red_rows, used_red_cols)
        if gap_specs is None:
            return None

        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE
        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE

        gap_cells = set()
        repaired_cells = set()
        for gap in gap_specs:
            cells = gap["cells"]
            for rr, cc in cells:
                grid[rr, cc] = BLACK
                gap_cells.add((rr, cc))
            if gap["repaired"]:
                (r_a, c_a), (r_b, c_b) = cells[0], cells[-1]
                grid[r_a, c_a] = RED
                grid[r_b, c_b] = RED
                repaired_cells.update({(r_a, c_a), (r_b, c_b)})
                pairs.append(
                    {
                        "kind": "repair",
                        "orientation": "h" if r_a == r_b else "v",
                        "endpoints": ((r_a, c_a), (r_b, c_b)),
                        "rect": rect,
                    }
                )

        rect_meta.append(
            {
                "rect": rect,
                "pattern": pattern,
                "closed": pattern.startswith("closed"),
                "gap_cells": gap_cells,
                "repaired_endpoint_cells": repaired_cells,
            }
        )

    usage = {
        "used_pair_rows": used_pair_rows,
        "used_pair_cols": used_pair_cols,
        "used_red_rows": used_red_rows,
        "used_red_cols": used_red_cols,
    }
    return grid, rect_meta, {"pairs": pairs, **usage}


def _forbidden_for_distractors(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in rects:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return forbidden


def _place_distractors(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    pairs_meta: dict,
) -> bool:
    h, w = grid.shape
    forbidden = _forbidden_for_distractors(grid.shape, rects)
    used_pair_rows = pairs_meta["used_pair_rows"]
    used_pair_cols = pairs_meta["used_pair_cols"]
    used_red_rows = pairs_meta["used_red_rows"]
    used_red_cols = pairs_meta["used_red_cols"]
    pair_list = pairs_meta["pairs"]
    target = random.randint(*DISTRACTOR_PAIR_RANGE)

    for _ in range(target):
        placed = False
        for _ in range(PAIR_PLACEMENT_TRIES):
            orientation = random.choice(["h", "v"])
            if orientation == "h":
                row = random.randrange(h)
                if row in used_pair_rows or row in used_red_rows:
                    continue
                segment_len = random.randint(3, min(8, w))
                start = random.randint(0, w - segment_len)
                end = start + segment_len - 1
                if forbidden[row, start : end + 1].any():
                    continue
                if not np.all(grid[row, start : end + 1] == BLACK):
                    continue
                if start in used_red_cols or end in used_red_cols or start in used_pair_cols or end in used_pair_cols:
                    continue
                if not _add_horizontal_pair(row, start, end, used_pair_rows, used_pair_cols, used_red_rows, used_red_cols):
                    continue
                grid[row, start] = RED
                grid[row, end] = RED
                pair_list.append(
                    {
                        "kind": "distractor",
                        "orientation": "h",
                        "endpoints": ((row, start), (row, end)),
                        "rect": None,
                    }
                )
                placed = True
                break
            else:
                col = random.randrange(w)
                if col in used_pair_cols or col in used_red_cols:
                    continue
                segment_len = random.randint(3, min(8, h))
                start = random.randint(0, h - segment_len)
                end = start + segment_len - 1
                if forbidden[start : end + 1, col].any():
                    continue
                if not np.all(grid[start : end + 1, col] == BLACK):
                    continue
                if start in used_red_rows or end in used_red_rows or start in used_pair_rows or end in used_pair_rows:
                    continue
                if not _add_vertical_pair(col, start, end, used_pair_rows, used_pair_cols, used_red_rows, used_red_cols):
                    continue
                grid[start, col] = RED
                grid[end, col] = RED
                pair_list.append(
                    {
                        "kind": "distractor",
                        "orientation": "v",
                        "endpoints": ((start, col), (end, col)),
                        "rect": None,
                    }
                )
                placed = True
                break
        if not placed:
            return False
    return True


def _marker_usage_is_valid(grid: np.ndarray) -> bool:
    coords = [tuple(map(int, x)) for x in np.argwhere(grid == RED)]
    row_counts = {r: int(np.sum(grid[r] == RED)) for r in range(grid.shape[0])}
    col_counts = {c: int(np.sum(grid[:, c] == RED)) for c in range(grid.shape[1])}
    if any(count > 2 for count in row_counts.values()):
        return False
    if any(count > 2 for count in col_counts.values()):
        return False
    for r, c in coords:
        participates = 0
        if row_counts[r] == 2:
            participates += 1
        if col_counts[c] == 2:
            participates += 1
        if participates != 1:
            return False
    return True


def _yellow_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] != YELLOW:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == YELLOW:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _validate_instance(grid: np.ndarray, rect_meta: list[dict], pairs: list[dict]) -> bool:
    if not _marker_usage_is_valid(grid):
        return False

    connected = _draw_connections(grid)
    output = main(grid)

    if _enclosed_black_components(grid):
        return False

    enclosed_after_connect = _enclosed_black_components(connected)
    if not enclosed_after_connect:
        return False

    yellow_after_output = _yellow_components(output)
    closed_rects = [meta for meta in rect_meta if meta["closed"]]
    open_rects = [meta for meta in rect_meta if not meta["closed"]]
    if len(enclosed_after_connect) != len(closed_rects):
        return False
    if len(yellow_after_output) != len(closed_rects):
        return False

    for meta in closed_rects:
        r0, c0, r1, c1 = meta["rect"]
        interior = output[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not np.all(interior == YELLOW):
            return False

    for meta in open_rects:
        r0, c0, r1, c1 = meta["rect"]
        interior = output[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.all(interior == YELLOW):
            return False
        if not np.any(interior == BLACK):
            return False

    repair_pairs = [pair for pair in pairs if pair["kind"] == "repair"]
    distractor_pairs = [pair for pair in pairs if pair["kind"] == "distractor"]
    if not repair_pairs or not distractor_pairs:
        return False

    rect_gap_cells = set()
    for meta in rect_meta:
        rect_gap_cells |= meta["gap_cells"]

    for pair in distractor_pairs:
        (r0, c0), (r1, c1) = pair["endpoints"]
        if r0 == r1:
            segment = {(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)}
        else:
            segment = {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}
        if segment & rect_gap_cells:
            return False

    if np.array_equal(output, grid):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    (row_range, col_range) = GRID_RANGE_BY_RECT_COUNT[n_rects]
    h = random.randint(*row_range)
    w = random.randint(*col_range)
    rects = _sample_rectangles(h, w, n_rects)
    if rects is None:
        return None

    patterns = _pattern_list(n_rects)
    rendered = _render_rectangles(h, w, rects, patterns)
    if rendered is None:
        return None
    grid, rect_meta, pair_state = rendered

    if not _place_distractors(grid, rects, pair_state):
        return None

    if not _validate_instance(grid, rect_meta, pair_state["pairs"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for puzzle 5")
