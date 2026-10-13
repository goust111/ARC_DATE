from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

SEED_PALETTE = (BLUE, RED, YELLOW, ORANGE)
PATHISH = {GREY, GREEN, MAROON}
NON_SEED_COLORS = {BG, GREEN, GREY, MAROON}
GEN_RETRIES = 400

DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SEED_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _single_segment(flags: list[bool]) -> tuple[int, int] | None:
    idx = [i for i, flag in enumerate(flags) if flag]
    if not idx:
        return None
    if idx != list(range(idx[0], idx[-1] + 1)):
        return None
    return idx[0], idx[-1]


def _component_of_color(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = grid.shape
    sr, sc = start
    color = int(grid[sr, sc])
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _find_hub(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(H - 2):
        for c0 in range(W - 2):
            if not np.all(g[r0 : r0 + 3, c0 : c0 + 3] == GREY):
                continue
            if r0 == 0 or c0 == 0 or r0 + 2 >= H - 1 or c0 + 2 >= W - 1:
                continue
            ok = True
            for side in ("top", "bottom", "left", "right"):
                if side == "top":
                    flags = [int(g[r0 - 1, c0 + dc]) in PATHISH for dc in range(3)]
                elif side == "bottom":
                    flags = [int(g[r0 + 3, c0 + dc]) in PATHISH for dc in range(3)]
                elif side == "left":
                    flags = [int(g[r0 + dr, c0 - 1]) in PATHISH for dr in range(3)]
                else:
                    flags = [int(g[r0 + dr, c0 + 3]) in PATHISH for dr in range(3)]
                if _single_segment(flags) is None:
                    ok = False
                    break
            if ok:
                candidates.append((r0, c0))
    assert len(candidates) == 1, "expected one 3x3 grey hub"
    return candidates[0]


def _extract_branch(
    grid: np.ndarray, hub_r0: int, hub_c0: int, direction: str
) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    if direction == "top":
        start_flags = [int(g[hub_r0 - 1, hub_c0 + dc]) in PATHISH for dc in range(3)]
        seg = _single_segment(start_flags)
        assert seg is not None
        c0 = hub_c0 + seg[0]
        c1 = hub_c0 + seg[1]
        rows = []
        r = hub_r0 - 1
        while r >= 0 and all(int(g[r, c]) in PATHISH for c in range(c0, c1 + 1)):
            rows.append(r)
            r -= 1
        assert rows
        cells = {(rr, cc) for rr in rows for cc in range(c0, c1 + 1)}
        frontier = {(rows[-1], cc) for cc in range(c0, c1 + 1)}
        outward = (-1, 0)
    elif direction == "bottom":
        start_flags = [int(g[hub_r0 + 3, hub_c0 + dc]) in PATHISH for dc in range(3)]
        seg = _single_segment(start_flags)
        assert seg is not None
        c0 = hub_c0 + seg[0]
        c1 = hub_c0 + seg[1]
        rows = []
        r = hub_r0 + 3
        while r < H and all(int(g[r, c]) in PATHISH for c in range(c0, c1 + 1)):
            rows.append(r)
            r += 1
        assert rows
        cells = {(rr, cc) for rr in rows for cc in range(c0, c1 + 1)}
        frontier = {(rows[-1], cc) for cc in range(c0, c1 + 1)}
        outward = (1, 0)
    elif direction == "left":
        start_flags = [int(g[hub_r0 + dr, hub_c0 - 1]) in PATHISH for dr in range(3)]
        seg = _single_segment(start_flags)
        assert seg is not None
        r0 = hub_r0 + seg[0]
        r1 = hub_r0 + seg[1]
        cols = []
        c = hub_c0 - 1
        while c >= 0 and all(int(g[r, c]) in PATHISH for r in range(r0, r1 + 1)):
            cols.append(c)
            c -= 1
        assert cols
        cells = {(rr, cc) for rr in range(r0, r1 + 1) for cc in cols}
        frontier = {(rr, cols[-1]) for rr in range(r0, r1 + 1)}
        outward = (0, -1)
    else:
        start_flags = [int(g[hub_r0 + dr, hub_c0 + 3]) in PATHISH for dr in range(3)]
        seg = _single_segment(start_flags)
        assert seg is not None
        r0 = hub_r0 + seg[0]
        r1 = hub_r0 + seg[1]
        cols = []
        c = hub_c0 + 3
        while c < W and all(int(g[r, c]) in PATHISH for r in range(r0, r1 + 1)):
            cols.append(c)
            c += 1
        assert cols
        cells = {(rr, cc) for rr in range(r0, r1 + 1) for cc in cols}
        frontier = {(rr, cols[-1]) for rr in range(r0, r1 + 1)}
        outward = (0, 1)

    control = [(r, c) for (r, c) in cells if int(g[r, c]) in (GREEN, MAROON)]
    assert len(control) == 1, "each branch must have exactly one control pixel"
    control_rc = control[0]
    open_branch = int(g[control_rc]) == GREEN

    seed_starts = []
    for r, c in frontier:
        nr = r + outward[0]
        nc = c + outward[1]
        if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) not in NON_SEED_COLORS:
            seed_starts.append((nr, nc))
    assert seed_starts, "branch frontier must touch a seed region"

    seed_component = _component_of_color(g, seed_starts[0])
    seed_color = int(g[seed_starts[0]])
    assert all(int(g[r, c]) == seed_color for r, c in seed_component)

    width = len(frontier)
    return {
        "direction": direction,
        "cells": cells,
        "frontier": frontier,
        "width": width,
        "control": control_rc,
        "open": open_branch,
        "seed_color": seed_color,
        "seed_component": seed_component,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate seed colors through a gated path network and color the hub center by the strongest arriving branch.

    Input structure:
    - A black background contains a central 3x3 solid grey hub.
    - Four straight grey branches extend from the hub's top, bottom, left, and right edges.
    - Each branch contains exactly one green or maroon control pixel and ends next to a same-color seed region.

    Rule:
    - Find the hub and the four orthogonal branches attached to its sides.
    - If a branch control pixel is green, recolor that whole branch to its seed-region color up to the hub edge.
    - If a branch control pixel is maroon, keep the control pixel maroon and keep the rest of that branch grey.
    - Recolor only the hub center pixel to the successful seed color whose open branch occupies the most hub-adjacent edge cells; keep the other 8 hub pixels grey.

    Key insight:
    - The control pixel is a gate on color propagation: topology says which seed can reach the hub, and branch width at the hub resolves the final competition among the successful arrivals.
    """
    g = np.asarray(grid, dtype=int)
    hub_r0, hub_c0 = _find_hub(g)
    branches = [_extract_branch(g, hub_r0, hub_c0, direction) for direction in ("top", "bottom", "left", "right")]

    out = g.copy()
    for branch in branches:
        cells = branch["cells"]
        control_r, control_c = branch["control"]
        if branch["open"]:
            for r, c in cells:
                out[r, c] = int(branch["seed_color"])
        else:
            for r, c in cells:
                out[r, c] = GREY
            out[control_r, control_c] = MAROON

    open_branches = [branch for branch in branches if branch["open"]]
    assert open_branches, "at least one branch must be open"
    max_width = max(int(branch["width"]) for branch in open_branches)
    winners = [branch for branch in open_branches if int(branch["width"]) == max_width]
    assert len(winners) == 1, "open branches must have a unique maximal hub contact"
    out[hub_r0 + 1, hub_c0 + 1] = int(winners[0]["seed_color"])
    return out


def _span_for_width(width: int) -> tuple[int, int]:
    if width == 1:
        return random.choice(((0, 0), (1, 1), (2, 2)))
    if width == 2:
        return random.choice(((0, 1), (1, 2)))
    return 0, 2


def _branch_cells(direction: str, hub_r0: int, hub_c0: int, span: tuple[int, int], length: int) -> set[tuple[int, int]]:
    a, b = span
    cells: set[tuple[int, int]] = set()
    if direction == "top":
        for d in range(length):
            r = hub_r0 - 1 - d
            for c in range(hub_c0 + a, hub_c0 + b + 1):
                cells.add((r, c))
    elif direction == "bottom":
        for d in range(length):
            r = hub_r0 + 3 + d
            for c in range(hub_c0 + a, hub_c0 + b + 1):
                cells.add((r, c))
    elif direction == "left":
        for d in range(length):
            c = hub_c0 - 1 - d
            for r in range(hub_r0 + a, hub_r0 + b + 1):
                cells.add((r, c))
    else:
        for d in range(length):
            c = hub_c0 + 3 + d
            for r in range(hub_r0 + a, hub_r0 + b + 1):
                cells.add((r, c))
    return cells


def _frontier_cells(direction: str, hub_r0: int, hub_c0: int, span: tuple[int, int], length: int) -> list[tuple[int, int]]:
    a, b = span
    if direction == "top":
        return [(hub_r0 - length, hub_c0 + c) for c in range(a, b + 1)]
    if direction == "bottom":
        return [(hub_r0 + 2 + length, hub_c0 + c) for c in range(a, b + 1)]
    if direction == "left":
        return [(hub_r0 + r, hub_c0 - length) for r in range(a, b + 1)]
    return [(hub_r0 + r, hub_c0 + 2 + length) for r in range(a, b + 1)]


def _place_seed(
    grid: np.ndarray,
    direction: str,
    frontier: list[tuple[int, int]],
    color: int,
) -> set[tuple[int, int]] | None:
    H, W = grid.shape
    if direction in ("top", "bottom"):
        base_axis = min(c for _r, c in frontier)
        width = len(frontier)
        adj_row = frontier[0][0] + (-1 if direction == "top" else 1)
    else:
        base_axis = min(r for r, _c in frontier)
        width = len(frontier)
        adj_col = frontier[0][1] + (-1 if direction == "left" else 1)

    shape_choices = list(SEED_SHAPES)
    random.shuffle(shape_choices)
    for shape in shape_choices:
        lat_values = [c for _d, c in shape]
        depth0 = {c for d, c in shape if d == 0}
        shape_min = min(lat_values)
        shape_max = max(lat_values)
        shifts = list(range(-shape_max, width - shape_min))
        random.shuffle(shifts)
        for shift in shifts:
            if not any(0 <= c + shift < width for c in depth0):
                continue
            cells = set()
            ok = True
            for depth, lateral in shape:
                axis = lateral + shift
                if direction == "top":
                    r = adj_row - depth
                    c = base_axis + axis
                elif direction == "bottom":
                    r = adj_row + depth
                    c = base_axis + axis
                elif direction == "left":
                    r = base_axis + axis
                    c = adj_col - depth
                else:
                    r = base_axis + axis
                    c = adj_col + depth
                if not (0 <= r < H and 0 <= c < W) or int(grid[r, c]) != BG:
                    ok = False
                    break
                cells.add((r, c))
            if not ok:
                continue
            touch = False
            frontier_set = set(frontier)
            for r, c in cells:
                if any((nr, nc) in frontier_set for nr, nc in neighbors4(r, c)):
                    touch = True
                    break
            if not touch:
                continue
            for r, c in cells:
                grid[r, c] = color
            return cells
    return None


def _manual_output(grid: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for branch in spec["branches"]:
        for r, c in branch["cells"]:
            out[r, c] = branch["seed_color"] if branch["open"] else GREY
        if not branch["open"]:
            cr, cc = branch["control"]
            out[cr, cc] = MAROON
    hr0, hc0 = spec["hub"]
    out[hr0 + 1, hc0 + 1] = spec["winner_color"]
    return out


def _build_attempt() -> tuple[np.ndarray, dict[str, object]] | None:
    H = random.randint(17, 24)
    W = random.randint(17, 24)
    lengths = {direction: random.randint(3, 5) for direction in DIRS}

    open_count = random.choice((2, 3))
    open_dirs = set(random.sample(tuple(DIRS), open_count))
    states = {direction: direction in open_dirs for direction in DIRS}

    open_widths = random.sample((1, 2, 3), open_count)
    random.shuffle(open_widths)
    widths: dict[str, int] = {}
    for direction, width in zip(open_dirs, open_widths):
        widths[direction] = width
    for direction in DIRS:
        if direction not in widths:
            widths[direction] = random.randint(1, 3)

    max_open_width = max(widths[d] for d in open_dirs)
    if sum(1 for d in open_dirs if widths[d] == max_open_width) != 1:
        return None

    top_need = lengths["top"] + 2
    bottom_need = lengths["bottom"] + 2
    left_need = lengths["left"] + 2
    right_need = lengths["right"] + 2
    r0_min = top_need
    r0_max = H - 5 - bottom_need
    c0_min = left_need
    c0_max = W - 5 - right_need
    if r0_min > r0_max or c0_min > c0_max:
        return None

    hub_r0 = random.randint(r0_min, r0_max)
    hub_c0 = random.randint(c0_min, c0_max)
    grid = np.full((H, W), BG, dtype=int)
    grid[hub_r0 : hub_r0 + 3, hub_c0 : hub_c0 + 3] = GREY

    seed_colors = list(SEED_PALETTE)
    random.shuffle(seed_colors)
    branches = []

    for direction, seed_color in zip(("top", "bottom", "left", "right"), seed_colors):
        width = widths[direction]
        length = lengths[direction]
        span = _span_for_width(width)
        cells = _branch_cells(direction, hub_r0, hub_c0, span, length)
        if any(int(grid[r, c]) != BG for r, c in cells):
            return None
        for r, c in cells:
            grid[r, c] = GREY

        frontier = _frontier_cells(direction, hub_r0, hub_c0, span, length)
        if _place_seed(grid, direction, frontier, seed_color) is None:
            return None

        control_depth = 0 if width == 3 else random.randint(1, length - 2)
        if direction == "top":
            control_band = [(hub_r0 - 1 - control_depth, c) for c in range(hub_c0 + span[0], hub_c0 + span[1] + 1)]
        elif direction == "bottom":
            control_band = [(hub_r0 + 3 + control_depth, c) for c in range(hub_c0 + span[0], hub_c0 + span[1] + 1)]
        elif direction == "left":
            control_band = [(r, hub_c0 - 1 - control_depth) for r in range(hub_r0 + span[0], hub_r0 + span[1] + 1)]
        else:
            control_band = [(r, hub_c0 + 3 + control_depth) for r in range(hub_r0 + span[0], hub_r0 + span[1] + 1)]
        control = random.choice(control_band)
        grid[control] = GREEN if states[direction] else MAROON

        branches.append(
            {
                "direction": direction,
                "open": states[direction],
                "width": width,
                "cells": cells,
                "control": control,
                "seed_color": seed_color,
            }
        )

    winner = max((b for b in branches if b["open"]), key=lambda b: b["width"])
    if sum(1 for b in branches if b["open"] and b["width"] == winner["width"]) != 1:
        return None

    spec = {"hub": (hub_r0, hub_c0), "branches": branches, "winner_color": winner["seed_color"]}
    return grid, spec


def _semantic_validate(inp: np.ndarray, spec: dict[str, object]) -> bool:
    out = main(inp)
    expected = _manual_output(inp, spec)
    if not np.array_equal(out, expected):
        return False

    hub_r0, hub_c0 = spec["hub"]
    if not np.all(out[hub_r0 : hub_r0 + 3, hub_c0 : hub_c0 + 3] == np.array(
        [[GREY, GREY, GREY], [GREY, spec["winner_color"], GREY], [GREY, GREY, GREY]]
    )):
        return False

    for branch in spec["branches"]:
        colors = {int(out[r, c]) for r, c in branch["cells"]}
        if branch["open"]:
            if colors != {int(branch["seed_color"])}:
                return False
        else:
            expected_colors = {GREY, MAROON}
            if colors - expected_colors:
                return False
            if int(out[branch["control"]]) != MAROON:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_attempt()
        if built is None:
            continue
        inp, spec = built
        if _semantic_validate(inp, spec):
            return inp
    raise RuntimeError("failed to generate a valid gated path-network input")
