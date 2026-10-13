from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Support both the strict 5-cell generator shape and the 4-cell reading of the
# natural-language description.
ARROW_TEMPLATES = {
    "up": (
        frozenset({(0, 0), (1, -1), (1, 0), (1, 1)}),
        frozenset({(0, 0), (1, 0), (2, -1), (2, 0), (2, 1)}),
    ),
    "down": (
        frozenset({(0, 0), (-1, -1), (-1, 0), (-1, 1)}),
        frozenset({(0, 0), (-1, 0), (-2, -1), (-2, 0), (-2, 1)}),
    ),
    "left": (
        frozenset({(0, 0), (-1, 1), (0, 1), (1, 1)}),
        frozenset({(0, 0), (0, 1), (-1, 2), (0, 2), (1, 2)}),
    ),
    "right": (
        frozenset({(0, 0), (-1, -1), (0, -1), (1, -1)}),
        frozenset({(0, 0), (0, -1), (-1, -2), (0, -2), (1, -2)}),
    ),
}

MIN_STRIPE_LEN = 5

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
MIN_STRIPE_FRAC = 0.55
TARGETED_COUNT_RANGE = (1, 4)
PLAIN_COUNT_RANGE = (0, 2)
PAIR_PLACE_RETRIES = 200
PLAIN_PLACE_RETRIES = 120
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": arr})
    return comps


def _detect_stripe(coords: np.ndarray) -> dict | None:
    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})
    n = len(coords)
    if len(rows) == 1 and n >= MIN_STRIPE_LEN:
        c0, c1 = cols[0], cols[-1]
        if n == c1 - c0 + 1:
            return {"orientation": "h", "row": rows[0], "c0": c0, "c1": c1}
    if len(cols) == 1 and n >= MIN_STRIPE_LEN:
        r0, r1 = rows[0], rows[-1]
        if n == r1 - r0 + 1:
            return {"orientation": "v", "col": cols[0], "r0": r0, "r1": r1}
    return None


def _detect_arrow(coords: np.ndarray) -> dict | None:
    coord_set = {tuple(map(int, rc)) for rc in coords}
    for direction, variants in ARROW_TEMPLATES.items():
        for tip in coord_set:
            tr, tc = tip
            for rels in variants:
                translated = {(tr + dr, tc + dc) for dr, dc in rels}
                if translated == coord_set:
                    return {"direction": direction, "tip": tip}
    return None


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], dict[int, dict]]:
    arrows: list[dict] = []
    stripes_by_color: dict[int, dict] = {}
    arrow_colors: set[int] = set()

    for comp in _components(grid):
        color = int(comp["color"])
        stripe = _detect_stripe(comp["coords"])
        if stripe is not None:
            assert color not in stripes_by_color, "each color may own at most one stripe"
            stripe["color"] = color
            stripe["coords"] = comp["coords"]
            stripes_by_color[color] = stripe
            continue

        arrow = _detect_arrow(comp["coords"])
        if arrow is not None:
            assert color not in arrow_colors, "same color cannot have two arrows"
            arrow["color"] = color
            arrow["coords"] = comp["coords"]
            arrows.append(arrow)
            arrow_colors.add(color)
            continue

        raise AssertionError("unexpected foreground component shape")

    assert arrows, "expected at least one arrow"
    for arrow in arrows:
        color = int(arrow["color"])
        assert color in stripes_by_color, "each arrow must have a matching stripe"
        stripe = stripes_by_color[color]
        if arrow["direction"] in ("up", "down"):
            assert stripe["orientation"] == "h", "vertical arrows must target horizontal stripes"
        else:
            assert stripe["orientation"] == "v", "horizontal arrows must target vertical stripes"
    return arrows, stripes_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each colored arrow and punch a centered 3-cell gap into the matching
    same-color stripe along the arrow's ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    arrows, stripes_by_color = _parse_scene(g)

    out = g.copy()
    for arrow in arrows:
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for arrow in arrows:
        color = int(arrow["color"])
        tip_r, tip_c = map(int, arrow["tip"])
        stripe = stripes_by_color[color]
        direction = arrow["direction"]

        if stripe["orientation"] == "h":
            impact_r = int(stripe["row"])
            impact_c = tip_c
            assert stripe["c0"] < impact_c < stripe["c1"], "gap center must stay inside the stripe"
            if direction == "up":
                assert impact_r < tip_r
            else:
                assert direction == "down"
                assert impact_r > tip_r
            cols = [impact_c - 1, impact_c, impact_c + 1]
            assert cols[0] >= 0 and cols[-1] < w
            assert all(int(g[impact_r, c]) == color for c in cols)
            out[impact_r, cols] = BG
        else:
            impact_r = tip_r
            impact_c = int(stripe["col"])
            assert stripe["r0"] < impact_r < stripe["r1"], "gap center must stay inside the stripe"
            if direction == "left":
                assert impact_c < tip_c
            else:
                assert direction == "right"
                assert impact_c > tip_c
            rows = [impact_r - 1, impact_r, impact_r + 1]
            assert rows[0] >= 0 and rows[-1] < h
            assert all(int(g[r, impact_c]) == color for r in rows)
            out[rows, impact_c] = BG

    return out


def _arrow_cells(tip: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    tr, tc = tip
    rels = max(ARROW_TEMPLATES[direction], key=len)
    return {(tr + dr, tc + dc) for dr, dc in rels}


def _long_lengths(dim: int) -> list[int]:
    lo = max(MIN_STRIPE_LEN + 2, int(np.ceil(dim * MIN_STRIPE_FRAC)))
    hi = dim - 2
    return list(range(lo, hi + 1))


def _all_in_bounds(cells: Iterable[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _free(cells: Iterable[tuple[int, int]], blocked: np.ndarray) -> bool:
    return all(not bool(blocked[r, c]) for r, c in cells)


def _add_blocked(blocked: np.ndarray, cells: Iterable[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        blocked[r0:r1, c0:c1] = True


def _ray_clear_cells(tip: tuple[int, int], impact: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = tip
    ir, ic = impact
    dr = 0 if ir == tr else (1 if ir > tr else -1)
    dc = 0 if ic == tc else (1 if ic > tc else -1)
    cur_r, cur_c = tr + dr, tc + dc
    cells: set[tuple[int, int]] = set()
    while (cur_r, cur_c) != (ir, ic):
        cells.add((cur_r, cur_c))
        cur_r += dr
        cur_c += dc
    return cells


def _place_targeted_pair(h: int, w: int, blocked: np.ndarray, color: int) -> dict | None:
    h_lengths = _long_lengths(w)
    v_lengths = _long_lengths(h)
    if not h_lengths or not v_lengths:
        return None

    for _ in range(PAIR_PLACE_RETRIES):
        stripe_orientation = random.choice(("h", "v"))
        if stripe_orientation == "h":
            direction = random.choice(("up", "down"))
            length = random.choice(h_lengths)
            c0 = random.randint(1, w - length - 1)
            c1 = c0 + length - 1
            impact_c = random.randint(c0 + 1, c1 - 1)
            if direction == "up":
                stripe_row = random.randint(1, h - 5)
                tip_r = random.randint(stripe_row + 2, h - 3)
            else:
                stripe_row = random.randint(4, h - 2)
                tip_r = random.randint(2, stripe_row - 2)
            tip = (tip_r, impact_c)
            impact = (stripe_row, impact_c)
            stripe_cells = {(stripe_row, c) for c in range(c0, c1 + 1)}
            stripe_meta = {"orientation": "h", "row": stripe_row, "c0": c0, "c1": c1}
        else:
            direction = random.choice(("left", "right"))
            length = random.choice(v_lengths)
            r0 = random.randint(1, h - length - 1)
            r1 = r0 + length - 1
            impact_r = random.randint(r0 + 1, r1 - 1)
            if direction == "left":
                stripe_col = random.randint(1, w - 5)
                tip_c = random.randint(stripe_col + 2, w - 3)
            else:
                stripe_col = random.randint(4, w - 2)
                tip_c = random.randint(2, stripe_col - 2)
            tip = (impact_r, tip_c)
            impact = (impact_r, stripe_col)
            stripe_cells = {(r, stripe_col) for r in range(r0, r1 + 1)}
            stripe_meta = {"orientation": "v", "col": stripe_col, "r0": r0, "r1": r1}

        arrow_cells = _arrow_cells(tip, direction)
        corridor = _ray_clear_cells(tip, impact)
        reserved = stripe_cells | arrow_cells | corridor
        if len(stripe_cells & arrow_cells) != 0:
            continue
        if not _all_in_bounds(reserved, h, w):
            continue
        if not _free(reserved, blocked):
            continue

        _add_blocked(blocked, stripe_cells | arrow_cells)
        for r, c in corridor:
            blocked[r, c] = True

        return {
            "color": color,
            "direction": direction,
            "tip": tip,
            "impact": impact,
            "stripe": stripe_meta,
            "arrow_cells": arrow_cells,
            "stripe_cells": stripe_cells,
        }

    return None


def _place_plain_stripe(h: int, w: int, blocked: np.ndarray, color: int) -> dict | None:
    h_lengths = _long_lengths(w)
    v_lengths = _long_lengths(h)
    if not h_lengths or not v_lengths:
        return None

    for _ in range(PLAIN_PLACE_RETRIES):
        orientation = random.choice(("h", "v"))
        if orientation == "h":
            length = random.choice(h_lengths)
            row = random.randint(1, h - 2)
            c0 = random.randint(1, w - length - 1)
            c1 = c0 + length - 1
            cells = {(row, c) for c in range(c0, c1 + 1)}
            meta = {"orientation": "h", "row": row, "c0": c0, "c1": c1}
        else:
            length = random.choice(v_lengths)
            col = random.randint(1, w - 2)
            r0 = random.randint(1, h - length - 1)
            r1 = r0 + length - 1
            cells = {(r, col) for r in range(r0, r1 + 1)}
            meta = {"orientation": "v", "col": col, "r0": r0, "r1": r1}

        if not _free(cells, blocked):
            continue

        _add_blocked(blocked, cells)
        return {"color": color, "stripe": meta, "stripe_cells": cells}

    return None


def _render_input(h: int, w: int, targeted: list[dict], plain: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for item in plain:
        color = int(item["color"])
        for r, c in item["stripe_cells"]:
            grid[r, c] = color
    for item in targeted:
        color = int(item["color"])
        for r, c in item["stripe_cells"] | item["arrow_cells"]:
            grid[r, c] = color
    return grid


def _expected_output(grid: np.ndarray, targeted: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in targeted:
        rows, cols = zip(*item["arrow_cells"])
        out[list(rows), list(cols)] = BG
    for item in targeted:
        sr, sc = item["impact"]
        if item["stripe"]["orientation"] == "h":
            out[sr, sc - 1 : sc + 2] = BG
        else:
            out[sr - 1 : sr + 2, sc] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    max_targeted = TARGETED_COUNT_RANGE[1] if min(h, w) >= 18 else 3
    n_targeted = random.randint(TARGETED_COUNT_RANGE[0], max_targeted)
    n_plain = random.randint(*PLAIN_COUNT_RANGE)
    total_colors = n_targeted + n_plain
    if total_colors > len(COLOR_POOL):
        return None

    colors = random.sample(COLOR_POOL, total_colors)
    blocked = np.zeros((h, w), dtype=bool)
    targeted: list[dict] = []
    plain: list[dict] = []

    for color in colors[:n_targeted]:
        item = _place_targeted_pair(h, w, blocked, color)
        if item is None:
            return None
        targeted.append(item)

    for color in colors[n_targeted:]:
        item = _place_plain_stripe(h, w, blocked, color)
        if item is None:
            return None
        plain.append(item)

    grid = _render_input(h, w, targeted, plain)
    expected = _expected_output(grid, targeted)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    parsed_arrows, parsed_stripes = _parse_scene(grid)
    if len(parsed_arrows) != len(targeted):
        return None
    if set(parsed_stripes) != {int(x["color"]) for x in targeted + plain}:
        return None
    if np.array_equal(grid, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-to-stripe puzzle")
