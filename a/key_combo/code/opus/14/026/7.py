from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

INPUT_COLORS = (RED, GREEN, ORANGE, MAROON)
COUNT_TO_FILL = {0: BLUE, 1: RED, 2: YELLOW, 3: PINK}
GEN_RETRIES = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _opening_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(f"bad side: {side}")


def _direction_from_opening(side: str) -> tuple[int, int]:
    return {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]


def _detect_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for comp in _components(grid):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            raise ValueError("cup bbox too small")
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        side_present = {
            "top": bool(np.all(sub[0, :] == color)),
            "bottom": bool(np.all(sub[-1, :] == color)),
            "left": bool(np.all(sub[:, 0] == color)),
            "right": bool(np.all(sub[:, -1] == color)),
        }
        missing = [side for side, present in side_present.items() if not present]
        if len(missing) != 1:
            raise ValueError("expected exactly one missing side")
        opening = missing[0]
        interior = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(grid[r, c]) == BG
        ]
        if not interior:
            raise ValueError("cup interior is empty")
        cups.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "opening": opening,
                "direction": _direction_from_opening(opening),
                "opening_cells": _opening_cells((r0, c0, r1, c1), opening),
                "interior": interior,
            }
        )
    cups.sort(key=lambda cup: cup["bbox"])
    return cups


def _project_from_opening(
    blockers: np.ndarray,
    opening_cells: list[tuple[int, int]],
    direction: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = blockers.shape
    dr, dc = direction
    cells: set[tuple[int, int]] = set()
    for r, c in opening_cells:
        rr = r
        cc = c
        while 0 <= rr < h and 0 <= cc < w and int(blockers[rr, cc]) == BG:
            cells.add((rr, cc))
            rr += dr
            cc += dc
    return cells


def _analyze_cups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    cups = _detect_cups(g)
    blockers = g.copy()
    for cup in cups:
        for r, c in cup["interior"]:
            blockers[r, c] = TEAL
    for cup in cups:
        cup["line_cells"] = _project_from_opening(blockers, cup["opening_cells"], cup["direction"])
        cup["cross_with"] = set()
    for i in range(len(cups)):
        for j in range(i + 1, len(cups)):
            if cups[i]["line_cells"] & cups[j]["line_cells"]:
                cups[i]["cross_with"].add(j)
                cups[j]["cross_with"].add(i)
    for cup in cups:
        cup["cross_count"] = len(cup["cross_with"])
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each cup interior, project teal rays from its open side, and recolor the interior by how many other cups its rays intersect.

    Input structure:
    - The background is black.
    - There are 3 or 4 separated cup-like objects, each a one-pixel-thick rectangular outline in a distinct color.
    - Every cup has exactly one fully missing side, which defines its opening and projection direction.

    Rule:
    - For each cup, identify its bounding box and temporarily fill the black interior cells with teal.
    - From every black cell on the missing side, extend a teal ray straight outward in the opening direction until the next cell would be non-black or leave the grid.
    - For each cup, count how many other cups have at least one overlapping projected ray cell with it.
    - Recolor that cup's filled interior by the fixed palette: 0 overlaps -> blue, 1 -> red, 2 -> yellow, 3 -> pink; keep the outline unchanged and keep all projected rays teal.

    Key insight:
    - A cup's final interior color is not local to that cup: it is determined by the global ray-intersection pattern created by all cups together.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cups = _analyze_cups(g)
    for cup in cups:
        for r, c in cup["interior"]:
            out[r, c] = TEAL
    for cup in cups:
        for r, c in cup["line_cells"]:
            out[r, c] = TEAL
    for cup in cups:
        fill = COUNT_TO_FILL[cup["cross_count"]]
        for r, c in cup["interior"]:
            out[r, c] = fill
    return out


def _rand_size() -> tuple[int, int]:
    return random.randint(3, 4), random.randint(4, 5)


def _stamp_cup(grid: np.ndarray, spec: dict) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    h = int(spec["h"])
    w = int(spec["w"])
    color = int(spec["color"])
    opening = str(spec["opening"])
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
        raise ValueError("cup out of bounds")
    if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BG):
        raise ValueError("cup overlap")
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if opening == "top":
        grid[r0, c0 : c1 + 1] = BG
    elif opening == "bottom":
        grid[r1, c0 : c1 + 1] = BG
    elif opening == "left":
        grid[r0 : r1 + 1, c0] = BG
    elif opening == "right":
        grid[r0 : r1 + 1, c1] = BG
    else:
        raise ValueError(f"bad opening: {opening}")


def _make_specs_chain(include_isolated: bool) -> tuple[list[dict], list[int]] | None:
    size_count = 4 if include_isolated else 3
    sizes = [_rand_size() for _ in range(size_count)]
    (ha, wa), (hb, wb), (hc, wc) = sizes[:3]
    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    row_gap1 = random.randint(2, 4)
    row_gap2 = random.randint(2, 4)
    col_gap1 = random.randint(2, 4)
    col_gap2 = random.randint(2, 4)
    right_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)

    a = {"r0": top_margin, "c0": left_margin + wb + col_gap1, "h": ha, "w": wa, "opening": "bottom"}
    b = {"r0": a["r0"] + ha + row_gap1, "c0": left_margin, "h": hb, "w": wb, "opening": "right"}
    c = {"r0": b["r0"] + hb + row_gap2, "c0": a["c0"] + wa + col_gap2, "h": hc, "w": wc, "opening": "top"}

    specs = [a, b, c]
    width_edge = c["c0"] + wc + right_margin
    if include_isolated:
        hd, wd = sizes[3]
        d = {
            "r0": top_margin,
            "c0": left_margin,
            "h": hd,
            "w": wd,
            "opening": "left",
        }
        specs.append(d)
        width_edge = max(width_edge, d["c0"] + wd + right_margin)
    height_edge = c["r0"] + hc + bottom_margin

    if height_edge > 25 or width_edge > 26:
        return None
    expected = [0, 1, 1, 2] if include_isolated else [1, 1, 2]
    return specs, expected


def _make_specs_hub() -> tuple[list[dict], list[int]] | None:
    (h1, w1), (h2, w2), (h3, w3), (hb, wb) = [_rand_size() for _ in range(4)]
    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    top_rows = [top_margin + random.randint(0, 1) for _ in range(3)]
    row_gap = random.randint(2, 3)
    col_gap1 = random.randint(1, 2)
    col_gap2 = random.randint(1, 2)
    col_gap3 = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)

    b = {
        "r0": max(top_rows[0] + h1, top_rows[1] + h2, top_rows[2] + h3) + row_gap,
        "c0": left_margin,
        "h": hb,
        "w": wb,
        "opening": "right",
    }
    a = {"r0": top_rows[0], "c0": left_margin + wb + col_gap1, "h": h1, "w": w1, "opening": "bottom"}
    c = {"r0": top_rows[1], "c0": a["c0"] + w1 + col_gap2, "h": h2, "w": w2, "opening": "bottom"}
    d = {"r0": top_rows[2], "c0": c["c0"] + w2 + col_gap3, "h": h3, "w": w3, "opening": "bottom"}

    specs = [a, b, c, d]
    height_edge = b["r0"] + hb + bottom_margin
    width_edge = d["c0"] + w3 + right_margin
    if height_edge > 25 or width_edge > 28:
        return None
    return specs, [1, 1, 1, 3]


def _paint_specs(specs: list[dict]) -> np.ndarray:
    max_r = max(int(spec["r0"]) + int(spec["h"]) for spec in specs)
    max_c = max(int(spec["c0"]) + int(spec["w"]) for spec in specs)
    grid = np.zeros((max_r + 1, max_c + 1), dtype=int)
    colors = random.sample(INPUT_COLORS, len(specs))
    order = list(range(len(specs)))
    random.shuffle(order)
    for idx, spec_idx in enumerate(order):
        specs[spec_idx]["color"] = colors[idx]
    for spec in specs:
        _stamp_cup(grid, spec)
    return grid


def _pad_grid(grid: np.ndarray) -> np.ndarray:
    row_budget = max(0, min(6, 30 - grid.shape[0]))
    col_budget = max(0, min(6, 30 - grid.shape[1]))
    top = random.randint(0, row_budget)
    bottom = random.randint(0, row_budget - top)
    left = random.randint(0, col_budget)
    right = random.randint(0, col_budget - left)
    out = np.zeros((grid.shape[0] + top + bottom, grid.shape[1] + left + right), dtype=int)
    out[top : top + grid.shape[0], left : left + grid.shape[1]] = grid
    return out


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k=k)
    return np.asarray(out, dtype=int)


def _pair_count(cups: list[dict]) -> int:
    total = 0
    for i in range(len(cups)):
        for j in range(i + 1, len(cups)):
            if cups[i]["line_cells"] & cups[j]["line_cells"]:
                total += 1
    return total


def _semantic_ok(grid: np.ndarray, expected_counts: list[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        cups = _analyze_cups(grid)
    except Exception:
        return False
    if len(cups) != len(expected_counts):
        return False
    if sorted(cup["cross_count"] for cup in cups) != sorted(expected_counts):
        return False
    if len({cup["opening"] for cup in cups}) < 2:
        return False
    if _pair_count(cups) < 1:
        return False
    if len({cup["cross_count"] for cup in cups}) < 2:
        return False
    if any(not cup["line_cells"] for cup in cups):
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    for cup in cups:
        fill = COUNT_TO_FILL[cup["cross_count"]]
        for r, c in cup["interior"]:
            if int(out[r, c]) != fill:
                return False
        for r, c in cup["line_cells"]:
            if int(out[r, c]) != TEAL:
                return False
        r0, c0, r1, c1 = cup["bbox"]
        outline_cells = (
            [(r0, c) for c in range(c0, c1 + 1)]
            + [(r1, c) for c in range(c0, c1 + 1)]
            + [(r, c0) for r in range(r0, r1 + 1)]
            + [(r, c1) for r in range(r0, r1 + 1)]
        )
        for r, c in outline_cells:
            if (r, c) not in cup["opening_cells"] and int(out[r, c]) != int(grid[r, c]):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    family = random.choice(("chain3", "isolated4", "hub4"))
    if family == "chain3":
        built = _make_specs_chain(include_isolated=False)
    elif family == "isolated4":
        built = _make_specs_chain(include_isolated=True)
    else:
        built = _make_specs_hub()
    if built is None:
        return None
    specs, expected_counts = built
    try:
        grid = _paint_specs([dict(spec) for spec in specs])
    except ValueError:
        return None
    grid = _pad_grid(grid)
    grid = _transform_grid(grid)
    if not _semantic_ok(grid, expected_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-ray puzzle")
