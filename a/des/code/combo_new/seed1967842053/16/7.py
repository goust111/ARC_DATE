import random
from itertools import product

import numpy as np


BLACK = 0
COLOR_ORDER = (1, 2, 3, 4, 5, 6, 7, 8, 9)
COLOR_RANK = {c: i for i, c in enumerate(COLOR_ORDER)}
OPPOSITE = {"L": "R", "R": "L", "T": "B", "B": "T"}


def _shape_cells(side, top, left, h, w, pattern):
    cells = []
    if side == "R":
        for i, bit in enumerate(pattern):
            for c in range(left, left + w - 1):
                cells.append((top + i, c))
            if bit:
                cells.append((top + i, left + w - 1))
                cells.append((top + i, left + w))
    elif side == "L":
        for i, bit in enumerate(pattern):
            if bit:
                for c in range(left - 1, left + w):
                    cells.append((top + i, c))
            else:
                for c in range(left + 1, left + w):
                    cells.append((top + i, c))
    elif side == "B":
        for j, bit in enumerate(pattern):
            for r in range(top, top + h - 1):
                cells.append((r, left + j))
            if bit:
                cells.append((top + h - 1, left + j))
                cells.append((top + h, left + j))
    elif side == "T":
        for j, bit in enumerate(pattern):
            if bit:
                for r in range(top - 1, top + h):
                    cells.append((r, left + j))
            else:
                for r in range(top + 1, top + h):
                    cells.append((r, left + j))
    return cells


def _components(grid):
    rows, cols = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    parts = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == BLACK or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] != BLACK:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append({"color": color, "cells": cells})
    return parts


def _parse_candidates(part):
    cells = part["cells"]
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    rel = {(r - min_r, c - min_c) for r, c in cells}
    candidates = []

    if width >= 4 and height >= 3:
        h, w = height, width - 1
        pattern = tuple((i, w) in rel for i in range(h))
        expected = set()
        for i, bit in enumerate(pattern):
            for j in range(w - 1):
                expected.add((i, j))
            if bit:
                expected.add((i, w - 1))
                expected.add((i, w))
        if expected == rel and any(pattern) and not all(pattern):
            candidates.append(
                {
                    **part,
                    "side": "R",
                    "top": min_r,
                    "left": min_c,
                    "h": h,
                    "w": w,
                    "pattern": pattern,
                }
            )

        pattern = tuple((i, 0) in rel for i in range(h))
        expected = set()
        for i, bit in enumerate(pattern):
            if bit:
                for j in range(w + 1):
                    expected.add((i, j))
            else:
                for j in range(2, w + 1):
                    expected.add((i, j))
        if expected == rel and any(pattern) and not all(pattern):
            candidates.append(
                {
                    **part,
                    "side": "L",
                    "top": min_r,
                    "left": min_c + 1,
                    "h": h,
                    "w": w,
                    "pattern": pattern,
                }
            )

    if height >= 4 and width >= 3:
        h, w = height - 1, width
        pattern = tuple((h, j) in rel for j in range(w))
        expected = set()
        for j, bit in enumerate(pattern):
            for i in range(h - 1):
                expected.add((i, j))
            if bit:
                expected.add((h - 1, j))
                expected.add((h, j))
        if expected == rel and any(pattern) and not all(pattern):
            candidates.append(
                {
                    **part,
                    "side": "B",
                    "top": min_r,
                    "left": min_c,
                    "h": h,
                    "w": w,
                    "pattern": pattern,
                }
            )

        pattern = tuple((0, j) in rel for j in range(w))
        expected = set()
        for j, bit in enumerate(pattern):
            if bit:
                for i in range(h + 1):
                    expected.add((i, j))
            else:
                for i in range(2, h + 1):
                    expected.add((i, j))
        if expected == rel and any(pattern) and not all(pattern):
            candidates.append(
                {
                    **part,
                    "side": "T",
                    "top": min_r + 1,
                    "left": min_c,
                    "h": h,
                    "w": w,
                    "pattern": pattern,
                }
            )

    return candidates


def _compatible(a, b):
    if OPPOSITE.get(a["side"]) != b["side"]:
        return False
    if a["h"] != b["h"] or a["w"] != b["w"]:
        return False
    if len(a["pattern"]) != len(b["pattern"]):
        return False
    return all(x != y for x, y in zip(a["pattern"], b["pattern"]))


def _one_matching(n, edges):
    edge_set = {frozenset(edge) for edge in edges}

    def rec(unused):
        if not unused:
            return []
        first = min(unused)
        for second in sorted(unused - {first}):
            if frozenset((first, second)) not in edge_set:
                continue
            rest = rec(unused - {first, second})
            if rest is not None:
                return [(first, second)] + rest
        return None

    return rec(set(range(n)))


def _choose_parts(parts):
    parsed = [_parse_candidates(part) for part in parts]
    if any(not choices for choices in parsed):
        raise ValueError("input contains an unrecognized jagged fragment")
    for selected in product(*parsed):
        edges = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                if _compatible(selected[i], selected[j]):
                    edges.append((i, j))
        matching = _one_matching(len(selected), edges)
        if matching is not None:
            return selected, matching
    raise ValueError("input contains no complete complementary pairing")


def _later_color(a, b):
    return a if COLOR_RANK.get(a, -1) >= COLOR_RANK.get(b, -1) else b


def _shifted(part, new_top, new_left):
    dr = new_top - part["top"]
    dc = new_left - part["left"]
    return [(r + dr, c + dc) for r, c in part["cells"]]


def _paint_pair(out, a, b):
    color = _later_color(a["color"], b["color"])
    if {a["side"], b["side"]} == {"R", "L"}:
        left_piece = a if a["side"] == "R" else b
        right_piece = b if a["side"] == "R" else a
        if left_piece["left"] <= right_piece["left"]:
            placements = [
                (left_piece, left_piece["top"], left_piece["left"]),
                (right_piece, left_piece["top"], left_piece["left"] + left_piece["w"]),
            ]
        else:
            placements = [
                (right_piece, right_piece["top"], right_piece["left"]),
                (left_piece, right_piece["top"], right_piece["left"] - left_piece["w"]),
            ]
    else:
        upper_piece = a if a["side"] == "B" else b
        lower_piece = b if a["side"] == "B" else a
        if upper_piece["top"] <= lower_piece["top"]:
            placements = [
                (upper_piece, upper_piece["top"], upper_piece["left"]),
                (lower_piece, upper_piece["top"] + upper_piece["h"], upper_piece["left"]),
            ]
        else:
            placements = [
                (lower_piece, lower_piece["top"], lower_piece["left"]),
                (upper_piece, lower_piece["top"] - upper_piece["h"], lower_piece["left"]),
            ]

    rows, cols = out.shape
    for part, top, left in placements:
        for r, c in _shifted(part, top, left):
            if 0 <= r < rows and 0 <= c < cols:
                out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary jagged rectangular fragments and redraw each joined pair with the later priority color.

    Input structure:
    The grid has black background and an even number of separated, single-color rectangular fragments.  Each fragment
    has exactly one jagged edge on its top, bottom, left, or right side, with one-pixel teeth and one-pixel bites.

    Rule:
    Match fragments whose jagged edges are opposite-facing and whose tooth/bite patterns are exact complements.  For
    each matched pair, translate the farther separated fragment along the jagged-edge normal until the two profiles
    interlock without overlap or gaps.  Repaint the whole joined pair using whichever of the two fragment colors appears
    later in the fixed sequence blue, red, green, yellow, grey, magenta, orange, azure, maroon.

    Key insight:
    Shape matching determines which pieces belong together and where they meet; the final color is a separate priority
    rule over the two matched colors rather than a geometric or numeric property of the fragments.
    """
    out = np.zeros_like(grid)
    parts = _components(grid)
    if not parts:
        return out
    selected, matching = _choose_parts(parts)
    for i, j in matching:
        _paint_pair(out, selected[i], selected[j])
    return out


def _expanded(cells):
    around = set()
    for r, c in cells:
        around.add((r, c))
        around.add((r - 1, c))
        around.add((r + 1, c))
        around.add((r, c - 1))
        around.add((r, c + 1))
    return around


def _inside(cells, rows, cols):
    return all(0 <= r < rows and 0 <= c < cols for r, c in cells)


def _random_pattern(length):
    while True:
        pattern = tuple(random.choice((False, True)) for _ in range(length))
        if any(pattern) and not all(pattern):
            return pattern


def generate_input() -> np.ndarray:
    orient = random.choice(("H", "V"))
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    gap = random.randint(3, 8)
    colors = random.sample(COLOR_ORDER, 2)

    if orient == "H":
        rows = random.randint(h + 2, 30)
        cols = random.randint(2 * w + gap + 1, 30)
        top = random.randint(1, rows - h)
        left = random.randint(1, cols - (2 * w + gap))
        pattern = _random_pattern(h)
        entries = [
            (_shape_cells("R", top, left, h, w, pattern), colors[0]),
            (_shape_cells("L", top, left + w + gap, h, w, tuple(not x for x in pattern)), colors[1]),
        ]
    else:
        rows = random.randint(2 * h + gap + 1, 30)
        cols = random.randint(w + 2, 30)
        top = random.randint(1, rows - (2 * h + gap))
        left = random.randint(1, cols - w)
        pattern = _random_pattern(w)
        entries = [
            (_shape_cells("B", top, left, h, w, pattern), colors[0]),
            (_shape_cells("T", top + h + gap, left, h, w, tuple(not x for x in pattern)), colors[1]),
        ]

    grid = np.zeros((rows, cols), dtype=int)
    for cells, color in entries:
        for r, c in cells:
            grid[r, c] = color
    return grid
