from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
MAROON = 9

MAX_GENERATE_ATTEMPTS = 1200
HEIGHT_RANGE = (13, 19)
WIDTH_RANGE = (13, 19)
VISIBLE_FIXED_COUNTS = (2, 3)
TARGET_YELLOW_COUNTS = (1, 2)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds((h, w), nr, nc) or seen[nr, nc] or int(g[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _sign(x: int) -> int:
    return 1 if x > 0 else -1


def _line_color(g: np.ndarray, coords: list[tuple[int, int]]) -> int:
    colors = {int(g[r, c]) for r, c in coords}
    line_colors = sorted(colors - {GREY, MAROON})
    assert len(line_colors) == 1, "each rod must have exactly one line color"
    return line_colors[0]


def _parse_rod(g: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    h, w = g.shape
    coord_set = set(coords)
    color = _line_color(g, coords)
    marker_by_color = {
        GREY: [p for p in coords if int(g[p]) == GREY],
        MAROON: [p for p in coords if int(g[p]) == MAROON],
    }
    assert len(marker_by_color[GREY]) == 1, "rod must contain one grey endpoint marker"
    assert len(marker_by_color[MAROON]) == 1, "rod must contain one maroon hinge marker"
    endpoint = marker_by_color[GREY][0]
    hinge = marker_by_color[MAROON][0]

    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        orientation = "horizontal"
        r = rows[0]
        min_c, max_c = cols[0], cols[-1]
        assert cols == list(range(min_c, max_c + 1)), "horizontal rod must be contiguous"
        assert endpoint[0] == hinge[0] == r, "horizontal rod markers must share one row"
        assert endpoint[1] in (min_c, max_c), "endpoint must sit at one end of the rod"
        assert min_c < hinge[1] < max_c, "hinge must be interior"

        fixed_sign = _sign(endpoint[1] - hinge[1])
        fixed_cells = [(r, c) for c in range(min(endpoint[1], hinge[1]) + 1, max(endpoint[1], hinge[1]))]
        tail_cells = (
            [(r, c) for c in range(hinge[1] + 1, max_c + 1)]
            if fixed_sign < 0
            else [(r, c) for c in range(min_c, hinge[1])]
        )
        fixed_step = (0, fixed_sign)
        tail_step = (0, -fixed_sign)
        up_space = hinge[0]
        down_space = h - 1 - hinge[0]
        assert up_space != down_space, "hinge must have a unique larger fold side"
        fold_step = (-1, 0) if up_space > down_space else (1, 0)
    else:
        assert len(cols) == 1, "rod must be straight"
        orientation = "vertical"
        c = cols[0]
        min_r, max_r = rows[0], rows[-1]
        assert rows == list(range(min_r, max_r + 1)), "vertical rod must be contiguous"
        assert endpoint[1] == hinge[1] == c, "vertical rod markers must share one column"
        assert endpoint[0] in (min_r, max_r), "endpoint must sit at one end of the rod"
        assert min_r < hinge[0] < max_r, "hinge must be interior"

        fixed_sign = _sign(endpoint[0] - hinge[0])
        fixed_cells = [(r, c) for r in range(min(endpoint[0], hinge[0]) + 1, max(endpoint[0], hinge[0]))]
        tail_cells = (
            [(r, c) for r in range(hinge[0] + 1, max_r + 1)]
            if fixed_sign < 0
            else [(r, c) for r in range(min_r, hinge[0])]
        )
        fixed_step = (fixed_sign, 0)
        tail_step = (-fixed_sign, 0)
        left_space = hinge[1]
        right_space = w - 1 - hinge[1]
        assert left_space != right_space, "hinge must have a unique larger fold side"
        fold_step = (0, -1) if left_space > right_space else (0, 1)

    assert color in (RED, BLUE), "expected one red rod and one blue rod"
    if color == RED:
        assert orientation == "horizontal", "red rod must be horizontal"
    else:
        assert orientation == "vertical", "blue rod must be vertical"

    assert len(fixed_cells) >= 2, "hinge-to-endpoint side must expose at least two line pixels"
    assert len(tail_cells) >= 2, "moving tail must expose at least two line pixels"
    for p in fixed_cells + tail_cells:
        assert p in coord_set and int(g[p]) == color, "rod body must be monochrome away from markers"

    return {
        "color": color,
        "orientation": orientation,
        "endpoint": endpoint,
        "hinge": hinge,
        "fixed_cells": fixed_cells,
        "tail_cells": tail_cells,
        "fixed_step": fixed_step,
        "tail_step": tail_step,
        "fold_step": fold_step,
    }


def _parse_rods(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg)
    assert len(comps) == 2, "expected exactly two marked rods"
    rods = [_parse_rod(g, comp) for comp in comps]
    colors = {int(rod["color"]) for rod in rods}
    assert colors == {RED, BLUE}, "expected one red rod and one blue rod"
    red = next(rod for rod in rods if int(rod["color"]) == RED)
    blue = next(rod for rod in rods if int(rod["color"]) == BLUE)
    assert red["hinge"][0] != blue["hinge"][0], "hinges should not share a row"
    assert red["hinge"][1] != blue["hinge"][1], "hinges should not share a column"
    return [red, blue]


def _ray_cells(origin: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r, c = origin
    dr, dc = step
    out: set[tuple[int, int]] = set()
    while 0 <= r < h and 0 <= c < w:
        out.add((r, c))
        r += dr
        c += dc
    return out


def _render_from_rods(
    rods: list[dict[str, object]],
    shape: tuple[int, int],
    *,
    reverse_fixed: set[int] | None = None,
    reverse_fold: set[int] | None = None,
) -> np.ndarray:
    reverse_fixed = reverse_fixed or set()
    reverse_fold = reverse_fold or set()

    occupied: dict[int, set[tuple[int, int]]] = {}
    for rod in rods:
        color = int(rod["color"])
        fixed_step = tuple(int(v) for v in rod["fixed_step"])
        fold_step = tuple(int(v) for v in rod["fold_step"])
        if color in reverse_fixed:
            fixed_step = (-fixed_step[0], -fixed_step[1])
        if color in reverse_fold:
            fold_step = (-fold_step[0], -fold_step[1])
        cells = _ray_cells(tuple(rod["hinge"]), fixed_step, shape) | _ray_cells(tuple(rod["hinge"]), fold_step, shape)
        occupied[color] = cells

    out = np.full(shape, BG, dtype=int)
    for color, cells in occupied.items():
        for r, c in cells:
            out[r, c] = color

    for r, c in occupied[RED] & occupied[BLUE]:
        out[r, c] = YELLOW

    for rod in rods:
        out[tuple(rod["endpoint"])] = GREY
        out[tuple(rod["hinge"])] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the red and blue rods into L-shapes, extend their two rays, and mark red-blue crossings yellow.

    Input structure:
    - The background is black.
    - Exactly two 4-connected multicolor rods appear: one horizontal red rod and one
      vertical blue rod.
    - Each rod contains one grey endpoint marker at a rod end and one maroon hinge
      marker at an interior cell; all remaining cells of that rod use the rod color.

    Rule:
    - For each rod, the side from the grey endpoint to the maroon hinge is the fixed arm,
      and the opposite side is the moving tail.
    - Rotate the moving tail 90 degrees around the hinge toward the perpendicular side
      with more open space to the grid edge, producing an L-shape.
    - Extend both L-shape arms from the hinge along their own directions until the grid
      boundary, keeping red and blue colors on their respective rays.
    - Any cell reached by both a red ray and a blue ray becomes yellow.
    - Grey endpoint markers and maroon hinge markers remain visible on top of the rays.

    Key insight:
    - The grey marker chooses which original half of each rod stays fixed, while the hinge
      position chooses the unique larger fold side; together they determine the two ray
      directions whose crossings must be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rods = _parse_rods(g)
    return _render_from_rods(rods, g.shape)


def _deterministic_tail_len(
    color: int,
    hinge: tuple[int, int],
    endpoint: tuple[int, int],
    shape: tuple[int, int],
    tail_capacity: int,
    fold_capacity: int,
) -> int:
    max_len = min(3, tail_capacity, fold_capacity)
    assert max_len >= 2, "tail must have room for at least two cells"
    if max_len == 2:
        return 2
    h, w = shape
    seed = 31 * h + 17 * w + 13 * hinge[0] + 11 * hinge[1] + 7 * endpoint[0] + 5 * endpoint[1] + color
    return 2 + (seed % (max_len - 1))


def _choose_red_row(h: int) -> int:
    up_candidates = [r for r in range(2, h - 2) if r > h - 1 - r]
    down_candidates = [r for r in range(2, h - 2) if r < h - 1 - r]
    branch = random.choice(["up", "down"])
    pool = up_candidates if branch == "up" else down_candidates
    return random.choice(pool)


def _choose_blue_col(w: int) -> int:
    left_candidates = [c for c in range(2, w - 2) if c > w - 1 - c]
    right_candidates = [c for c in range(2, w - 2) if c < w - 1 - c]
    branch = random.choice(["left", "right"])
    pool = left_candidates if branch == "left" else right_candidates
    return random.choice(pool)


def _rod_input_cells(
    color: int,
    hinge: tuple[int, int],
    fixed_step: tuple[int, int],
    fixed_count: int,
    tail_len: int,
) -> dict[str, object]:
    hr, hc = hinge
    fdr, fdc = fixed_step
    endpoint = (hr + fdr * (fixed_count + 1), hc + fdc * (fixed_count + 1))
    fixed_cells = [(hr + fdr * k, hc + fdc * k) for k in range(1, fixed_count + 1)]
    tdr, tdc = -fdr, -fdc
    tail_cells = [(hr + tdr * k, hc + tdc * k) for k in range(1, tail_len + 1)]
    cells = {hinge, endpoint, *fixed_cells, *tail_cells}
    return {
        "color": color,
        "hinge": hinge,
        "endpoint": endpoint,
        "fixed_cells": fixed_cells,
        "tail_cells": tail_cells,
        "cells": cells,
    }


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _build_candidate(shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape

    red_row = _choose_red_row(h)
    blue_col = _choose_blue_col(w)
    red_col = random.randint(4, w - 5)
    blue_row = random.randint(4, h - 5)
    if red_row == blue_row or red_col == blue_col:
        return None

    red_fixed_step = (0, random.choice([-1, 1]))
    blue_fixed_step = (random.choice([-1, 1]), 0)
    red_fixed_count = random.choice(VISIBLE_FIXED_COUNTS)
    blue_fixed_count = random.choice(VISIBLE_FIXED_COUNTS)

    red_endpoint = (red_row, red_col + red_fixed_step[1] * (red_fixed_count + 1))
    blue_endpoint = (blue_row + blue_fixed_step[0] * (blue_fixed_count + 1), blue_col)
    if not in_bounds(shape, *red_endpoint) or not in_bounds(shape, *blue_endpoint):
        return None

    red_tail_capacity = red_col if red_fixed_step[1] > 0 else (w - 1 - red_col)
    blue_tail_capacity = blue_row if blue_fixed_step[0] > 0 else (h - 1 - blue_row)
    red_fold_capacity = red_row if red_row > h - 1 - red_row else (h - 1 - red_row)
    blue_fold_capacity = blue_col if blue_col > w - 1 - blue_col else (w - 1 - blue_col)

    red_tail_len = _deterministic_tail_len(RED, (red_row, red_col), red_endpoint, shape, red_tail_capacity, red_fold_capacity)
    blue_tail_len = _deterministic_tail_len(BLUE, (blue_row, blue_col), blue_endpoint, shape, blue_tail_capacity, blue_fold_capacity)

    red = _rod_input_cells(RED, (red_row, red_col), red_fixed_step, red_fixed_count, red_tail_len)
    blue = _rod_input_cells(BLUE, (blue_row, blue_col), blue_fixed_step, blue_fixed_count, blue_tail_len)

    if not red["cells"].isdisjoint(blue["cells"]):
        return None
    if _expanded(set(red["cells"]), shape) & set(blue["cells"]):
        return None
    if _expanded(set(blue["cells"]), shape) & set(red["cells"]):
        return None

    inp = np.full(shape, BG, dtype=int)
    for r, c in red["fixed_cells"] + red["tail_cells"]:
        inp[r, c] = RED
    for r, c in blue["fixed_cells"] + blue["tail_cells"]:
        inp[r, c] = BLUE
    inp[red["endpoint"]] = GREY
    inp[blue["endpoint"]] = GREY
    inp[red["hinge"]] = MAROON
    inp[blue["hinge"]] = MAROON

    yellow_target = random.choice(TARGET_YELLOW_COUNTS)
    out = main(inp)
    if int(np.count_nonzero(out == YELLOW)) != yellow_target:
        return None
    if not _semantic_ok(inp):
        return None
    return inp


def _semantic_ok(inp: np.ndarray) -> bool:
    try:
        rods = _parse_rods(inp)
    except AssertionError:
        return False

    correct = _render_from_rods(rods, inp.shape)
    if not np.array_equal(correct, main(inp)):
        return False

    if int(np.count_nonzero(correct == YELLOW)) < 1:
        return False

    for rod in rods:
        color = int(rod["color"])
        tail_capacity = len(rod["tail_cells"])
        endpoint = tuple(rod["endpoint"])
        hinge = tuple(rod["hinge"])
        fixed_count = len(rod["fixed_cells"])
        if fixed_count not in VISIBLE_FIXED_COUNTS:
            return False
        visible_tail_capacity = 0
        if color == RED:
            visible_tail_capacity = hinge[1] if int(rod["fixed_step"][1]) > 0 else (inp.shape[1] - 1 - hinge[1])
            fold_capacity = hinge[0] if int(rod["fold_step"][0]) < 0 else (inp.shape[0] - 1 - hinge[0])
        else:
            visible_tail_capacity = hinge[0] if int(rod["fixed_step"][0]) > 0 else (inp.shape[0] - 1 - hinge[0])
            fold_capacity = hinge[1] if int(rod["fold_step"][1]) < 0 else (inp.shape[1] - 1 - hinge[1])
        expected_tail = _deterministic_tail_len(color, hinge, endpoint, inp.shape, visible_tail_capacity, fold_capacity)
        if tail_capacity != expected_tail:
            return False

    alternatives = [
        _render_from_rods(rods, inp.shape, reverse_fold={RED}),
        _render_from_rods(rods, inp.shape, reverse_fold={BLUE}),
        _render_from_rods(rods, inp.shape, reverse_fixed={RED}),
        _render_from_rods(rods, inp.shape, reverse_fixed={BLUE}),
    ]
    if any(np.array_equal(correct, alt) for alt in alternatives):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        shape = (
            random.randint(*HEIGHT_RANGE),
            random.randint(*WIDTH_RANGE),
        )
        grid = _build_candidate(shape)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
