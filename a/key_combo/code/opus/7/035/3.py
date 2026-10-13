from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BARRIER = 5
DOMINO_COLORS = (1, 2, 3, 4, 7)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            r1 = int(coords[:, 0].max())
            c0 = int(coords[:, 1].min())
            c1 = int(coords[:, 1].max())
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _barriers(grid: np.ndarray) -> list[dict]:
    comps = _components(grid, {BARRIER})
    out: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        area = int(comp["area"])
        if r0 == r1:
            length = c1 - c0 + 1
            assert area == length and 3 <= length <= 5, "invalid horizontal barrier"
            out.append(
                {
                    "orientation": "h",
                    "row": r0,
                    "start": c0,
                    "end": c1,
                }
            )
        elif c0 == c1:
            length = r1 - r0 + 1
            assert area == length and 3 <= length <= 5, "invalid vertical barrier"
            out.append(
                {
                    "orientation": "v",
                    "col": c0,
                    "start": r0,
                    "end": r1,
                }
            )
        else:
            raise AssertionError("barriers must be straight segments")
    assert out, "expected at least one barrier"
    return out


def _dominoes(grid: np.ndarray, bg: int) -> list[dict]:
    colors = set(int(v) for v in np.unique(grid)) - {int(bg), BARRIER}
    comps = _components(grid, colors)
    out: list[dict] = []
    for idx, comp in enumerate(comps):
        coords = comp["coords"]
        assert int(comp["area"]) == 2, "every non-barrier object must be a domino"
        (r0, c0), (r1, c1) = sorted((tuple(map(int, rc)) for rc in coords))
        if r0 == r1 and abs(c0 - c1) == 1:
            out.append(
                {
                    "id": idx,
                    "color": int(comp["color"]),
                    "orientation": "h",
                    "row": r0,
                    "c0": min(c0, c1),
                    "c1": max(c0, c1),
                    "cells": ((r0, c0), (r1, c1)),
                }
            )
        elif c0 == c1 and abs(r0 - r1) == 1:
            out.append(
                {
                    "id": idx,
                    "color": int(comp["color"]),
                    "orientation": "v",
                    "col": c0,
                    "r0": min(r0, r1),
                    "r1": max(r0, r1),
                    "cells": ((r0, c0), (r1, c1)),
                }
            )
        else:
            raise AssertionError("invalid domino shape")
    assert out, "expected at least one domino"
    return out


def _collect_pairs(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    barriers = _barriers(g)
    dominoes = _dominoes(g, bg)
    paired_ids: set[int] = set()
    pairs: list[tuple[int, int]] = []
    by_id = {int(dom["id"]): dom for dom in dominoes}

    for barrier in barriers:
        if barrier["orientation"] == "v":
            col = int(barrier["col"])
            for row in range(int(barrier["start"]), int(barrier["end"]) + 1):
                left = [
                    dom
                    for dom in dominoes
                    if dom["orientation"] == "h" and int(dom["row"]) == row and int(dom["c1"]) < col
                ]
                right = [
                    dom
                    for dom in dominoes
                    if dom["orientation"] == "h" and int(dom["row"]) == row and int(dom["c0"]) > col
                ]
                if len(left) == 1 and len(right) == 1:
                    a = int(left[0]["id"])
                    b = int(right[0]["id"])
                    assert a not in paired_ids and b not in paired_ids, "domino paired twice"
                    paired_ids.add(a)
                    paired_ids.add(b)
                    pairs.append((a, b))
        else:
            row = int(barrier["row"])
            for col in range(int(barrier["start"]), int(barrier["end"]) + 1):
                top = [
                    dom
                    for dom in dominoes
                    if dom["orientation"] == "v" and int(dom["col"]) == col and int(dom["r1"]) < row
                ]
                bottom = [
                    dom
                    for dom in dominoes
                    if dom["orientation"] == "v" and int(dom["col"]) == col and int(dom["r0"]) > row
                ]
                if len(top) == 1 and len(bottom) == 1:
                    a = int(top[0]["id"])
                    b = int(bottom[0]["id"])
                    assert a not in paired_ids and b not in paired_ids, "domino paired twice"
                    paired_ids.add(a)
                    paired_ids.add(b)
                    pairs.append((a, b))

    return dominoes, pairs, [by_id[dom_id] for dom_id in sorted(paired_ids)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap domino colors across barriers only when the domino orientation matches the crossing direction.

    Input structure:
    - The background is black.
    - Grey pixels form disconnected straight barrier segments, each horizontal or vertical.
    - Every other foreground object is a same-color 2-cell domino, either horizontal or vertical.

    Rule:
    - For each vertical barrier row, pair exactly one horizontal domino on the left with exactly one horizontal domino on the right.
    - For each horizontal barrier column, pair exactly one vertical domino above with exactly one vertical domino below.
    - Each valid pair swaps colors in place, while every domino that never joins a valid pair is erased; barriers stay fixed.

    Key insight:
    - Pairing is gated by orientation: only dominoes perpendicular to a barrier can exchange across it, so orientation-parallel dominoes are removed even if they sit nearby.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    dominoes, pairs, paired = _collect_pairs(g)

    out = np.where(g == BARRIER, BARRIER, BG).astype(int)
    by_id = {int(dom["id"]): dom for dom in dominoes}
    for a, b in pairs:
        dom_a = by_id[a]
        dom_b = by_id[b]
        for r, c in dom_a["cells"]:
            out[int(r), int(c)] = int(dom_b["color"])
        for r, c in dom_b["cells"]:
            out[int(r), int(c)] = int(dom_a["color"])

    assert len(paired) % 2 == 0
    return out


def _chebyshev_far_enough(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r0, c0 in a:
        for r1, c1 in b:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    candidate = set((int(r), int(c)) for r, c in cells)
    for r, c in candidate:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in candidate and int(grid[nr, nc]) != BG:
                return False
    return True


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _segment_cells(orientation: str, anchor: int, start: int, length: int) -> list[tuple[int, int]]:
    if orientation == "v":
        return [(start + k, anchor) for k in range(length)]
    return [(anchor, start + k) for k in range(length)]


def _segments_separated(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    return _chebyshev_far_enough(set(a), set(b))


def _place_vertical_barrier(grid: np.ndarray) -> dict | None:
    h, w = grid.shape
    for _ in range(80):
        length = random.randint(3, 5)
        col = random.randint(4, w - 5)
        start = random.randint(2, h - length - 3)
        cells = _segment_cells("v", col, start, length)
        if _can_place(grid, cells):
            _paint(grid, cells, BARRIER)
            return {"orientation": "v", "col": col, "start": start, "end": start + length - 1, "cells": cells}
    return None


def _place_horizontal_barrier(grid: np.ndarray, other_cells: list[tuple[int, int]]) -> dict | None:
    h, w = grid.shape
    other = set(other_cells)
    for _ in range(120):
        length = random.randint(3, 5)
        row = random.randint(4, h - 5)
        start = random.randint(2, w - length - 3)
        cells = _segment_cells("h", row, start, length)
        if not _segments_separated(cells, list(other)):
            continue
        if _can_place(grid, cells):
            _paint(grid, cells, BARRIER)
            return {"orientation": "h", "row": row, "start": start, "end": start + length - 1, "cells": cells}
    return None


def _horizontal_options(row: int, barrier_col: int, side: str, width: int) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []
    if side == "left":
        starts = list(range(0, barrier_col - 2))
        valid = [c for c in starts if c + 1 <= barrier_col - 2]
    else:
        starts = list(range(barrier_col + 2, width - 1))
        valid = starts
    random.shuffle(valid)
    for c0 in valid:
        options.append([(row, c0), (row, c0 + 1)])
    return options


def _vertical_options(col: int, barrier_row: int, side: str, height: int) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []
    if side == "top":
        starts = list(range(0, barrier_row - 2))
        valid = [r0 for r0 in starts if r0 + 1 <= barrier_row - 2]
    else:
        starts = list(range(barrier_row + 2, height - 1))
        valid = starts
    random.shuffle(valid)
    for r0 in valid:
        options.append([(r0, col), (r0 + 1, col)])
    return options


def _place_from_options(grid: np.ndarray, options: list[list[tuple[int, int]]], color: int) -> list[tuple[int, int]] | None:
    for cells in options:
        if _can_place(grid, cells):
            _paint(grid, cells, color)
            return cells
    return None


def _distinct_pair_colors() -> tuple[int, int]:
    a, b = random.sample(DOMINO_COLORS, 2)
    return int(a), int(b)


def _choose_two(values: list[int]) -> tuple[int, int]:
    a, b = random.sample(values, 2)
    return int(a), int(b)


def _place_near_vertical_barrier(grid: np.ndarray, barrier: dict, mandatory: bool) -> bool:
    h, w = grid.shape
    rows = list(range(int(barrier["start"]), int(barrier["end"])))
    if not rows:
        return not mandatory
    random.shuffle(rows)
    sides = ["left", "right"]
    random.shuffle(sides)
    cols = []
    for side in sides:
        if side == "left":
            cols.extend(range(max(0, int(barrier["col"]) - 4), int(barrier["col"]) - 1))
        else:
            cols.extend(range(int(barrier["col"]) + 2, min(w, int(barrier["col"]) + 5)))
    random.shuffle(cols)
    for row0 in rows:
        for col in cols:
            cells = [(row0, col), (row0 + 1, col)]
            if _can_place(grid, cells):
                _paint(grid, cells, random.choice(DOMINO_COLORS))
                return True
    return not mandatory


def _place_near_horizontal_barrier(grid: np.ndarray, barrier: dict, mandatory: bool) -> bool:
    h, w = grid.shape
    cols = list(range(int(barrier["start"]), int(barrier["end"])))
    if not cols:
        return not mandatory
    random.shuffle(cols)
    rows = []
    for side in random.sample(["top", "bottom"], 2):
        if side == "top":
            rows.extend(range(max(0, int(barrier["row"]) - 4), int(barrier["row"]) - 1))
        else:
            rows.extend(range(int(barrier["row"]) + 2, min(h, int(barrier["row"]) + 5)))
    random.shuffle(rows)
    for col0 in cols:
        for row in rows:
            cells = [(row, col0), (row, col0 + 1)]
            if _can_place(grid, cells):
                _paint(grid, cells, random.choice(DOMINO_COLORS))
                return True
    return not mandatory


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(11, 15)
    w = random.randint(11, 15)
    grid = np.full((h, w), BG, dtype=int)

    vbar = _place_vertical_barrier(grid)
    if vbar is None:
        return None
    hbar = _place_horizontal_barrier(grid, vbar["cells"])
    if hbar is None:
        return None

    v_rows = list(range(int(vbar["start"]), int(vbar["end"]) + 1))
    h_cols = list(range(int(hbar["start"]), int(hbar["end"]) + 1))
    pair_row, unpaired_row = _choose_two(v_rows)
    pair_col, unpaired_col = _choose_two(h_cols)

    left_color, right_color = _distinct_pair_colors()
    if _place_from_options(grid, _horizontal_options(pair_row, int(vbar["col"]), "left", w), left_color) is None:
        return None
    if _place_from_options(grid, _horizontal_options(pair_row, int(vbar["col"]), "right", w), right_color) is None:
        return None

    lone_side = random.choice(["left", "right"])
    lone_color = random.choice(DOMINO_COLORS)
    if _place_from_options(grid, _horizontal_options(unpaired_row, int(vbar["col"]), lone_side, w), lone_color) is None:
        return None

    top_color, bottom_color = _distinct_pair_colors()
    if _place_from_options(grid, _vertical_options(pair_col, int(hbar["row"]), "top", h), top_color) is None:
        return None
    if _place_from_options(grid, _vertical_options(pair_col, int(hbar["row"]), "bottom", h), bottom_color) is None:
        return None

    lone_side = random.choice(["top", "bottom"])
    lone_color = random.choice(DOMINO_COLORS)
    if _place_from_options(grid, _vertical_options(unpaired_col, int(hbar["row"]), lone_side, h), lone_color) is None:
        return None

    if random.random() < 0.9 and not _place_near_vertical_barrier(grid, vbar, mandatory=False):
        return None
    if random.random() < 0.9 and not _place_near_horizontal_barrier(grid, hbar, mandatory=False):
        return None

    dominoes, pairs, paired = _collect_pairs(grid)
    if len(pairs) < 2:
        return None
    if len(dominoes) - len(paired) < 2:
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-3 input")
