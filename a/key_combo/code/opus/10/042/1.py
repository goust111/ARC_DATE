from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
WALL = 6
ANCHOR = 7

ROD_COLORS = (BLUE, GREEN)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (13, 18)
GRID_W_RANGE = (13, 18)
ROD_COUNT_CHOICES = (3, 4)
MAX_GENERATION_ATTEMPTS = 600
PAIR_ATTEMPTS = 250
EXTRA_ROD_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _add(a: tuple[int, int], b: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return a[0] + b[0] * k, a[1] + b[1] * k


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _line_step(orientation: str, fixed_sign: int) -> tuple[int, int]:
    if orientation == "h":
        return 0, fixed_sign
    return fixed_sign, 0


def _perp_dirs(orientation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if orientation == "h":
        return (-1, 0), (1, 0)
    return (0, -1), (0, 1)


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return -direction[0], -direction[1]


def _rod_from_folded(
    shape: tuple[int, int],
    orientation: str,
    hinge: tuple[int, int],
    fixed_sign: int,
    fixed_red_len: int,
    tail_len: int,
    fold_dir: tuple[int, int],
    line_color: int,
) -> dict[str, object] | None:
    assert orientation in ("h", "v")
    assert line_color in ROD_COLORS
    assert 0 <= fixed_red_len <= 3
    assert 1 <= tail_len <= 4
    assert 5 <= fixed_red_len + tail_len + 2 <= 7
    if fold_dir not in _perp_dirs(orientation):
        return None

    fixed_step = _line_step(orientation, fixed_sign)
    tail_step = _opposite(fixed_step)
    endpoint = _add(hinge, fixed_step, fixed_red_len + 1)
    fixed_line = [_add(hinge, fixed_step, k) for k in range(1, fixed_red_len + 1)]
    tail_input = [_add(hinge, tail_step, k) for k in range(1, tail_len + 1)]
    folded = [_add(hinge, fold_dir, k) for k in range(1, tail_len + 1)]

    required = [hinge, endpoint, *fixed_line, *tail_input, *folded]
    if any(not _in_bounds(shape, r, c) for r, c in required):
        return None

    free_end = tail_input[-1]
    chosen_start = _add(free_end, fold_dir)
    blocked_dir = _opposite(fold_dir)
    blocked_start = _add(free_end, blocked_dir)
    if not _in_bounds(shape, *chosen_start) or not _in_bounds(shape, *blocked_start):
        return None

    return {
        "orientation": orientation,
        "hinge": hinge,
        "endpoint": endpoint,
        "fixed_sign": fixed_sign,
        "fixed_step": fixed_step,
        "fixed_line": fixed_line,
        "tail_input": tail_input,
        "tail_len": tail_len,
        "fixed_red_len": fixed_red_len,
        "folded": folded,
        "fold_dir": fold_dir,
        "blocked_dir": blocked_dir,
        "free_end": free_end,
        "chosen_start": chosen_start,
        "blocked_start": blocked_start,
        "line_color": line_color,
        "input_cells": {hinge, endpoint, *fixed_line, *tail_input},
        "input_line_cells": {*fixed_line, *tail_input},
        "output_nonmarker": {*fixed_line, *folded},
    }


def _parse_rods(grid: np.ndarray) -> list[dict[str, object]]:
    rod_mask = np.isin(grid, [BLUE, RED, GREEN, GREY])
    comps = _components(rod_mask)
    rods: list[dict[str, object]] = []
    for comp in comps:
        cells = {(int(r), int(c)) for r, c in comp}
        values = {int(grid[r, c]) for r, c in cells}
        line_colors = sorted(v for v in values if v in ROD_COLORS)
        assert len(line_colors) == 1, "each rod must use exactly one line color"
        line_color = line_colors[0]

        hinges = [(r, c) for r, c in cells if int(grid[r, c]) == RED]
        endpoints = [(r, c) for r, c in cells if int(grid[r, c]) == GREY]
        assert len(hinges) == 1 and len(endpoints) == 1, "each rod needs one hinge and one endpoint"
        hinge = hinges[0]
        endpoint = endpoints[0]

        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        assert len(rows) == 1 or len(cols) == 1, "each rod must be straight"
        orientation = "h" if len(rows) == 1 else "v"
        fixed_sign = (endpoint[1] - hinge[1]) if orientation == "h" else (endpoint[0] - hinge[0])
        fixed_sign = 1 if fixed_sign > 0 else -1

        fixed_step = _line_step(orientation, fixed_sign)
        if orientation == "h":
            dist = abs(endpoint[1] - hinge[1])
            assert endpoint[0] == hinge[0] and dist >= 1
        else:
            dist = abs(endpoint[0] - hinge[0])
            assert endpoint[1] == hinge[1] and dist >= 1

        fixed_line = []
        for k in range(1, dist):
            pos = _add(hinge, fixed_step, k)
            assert pos in cells and int(grid[pos]) == line_color
            fixed_line.append(pos)

        tail_step = _opposite(fixed_step)
        tail_input: list[tuple[int, int]] = []
        k = 1
        while True:
            pos = _add(hinge, tail_step, k)
            if pos not in cells:
                break
            assert int(grid[pos]) == line_color
            tail_input.append(pos)
            k += 1
        assert tail_input, "each rod must have a free segment"

        expected = {hinge, endpoint, *fixed_line, *tail_input}
        assert expected == cells, "rod cells must form endpoint-fixed segment plus free tail"
        assert 5 <= len(expected) <= 7, "rod length must match the task description"

        rods.append(
            {
                "orientation": orientation,
                "hinge": hinge,
                "endpoint": endpoint,
                "fixed_sign": fixed_sign,
                "fixed_step": fixed_step,
                "fixed_line": fixed_line,
                "tail_input": tail_input,
                "tail_len": len(tail_input),
                "fixed_red_len": len(fixed_line),
                "free_end": tail_input[-1],
                "line_color": line_color,
                "input_cells": expected,
                "input_line_cells": {*fixed_line, *tail_input},
                "perp_dirs": _perp_dirs(orientation),
            }
        )

    assert 3 <= len(rods) <= 4, "expected 3-4 rods"
    rods.sort(key=lambda rod: (rod["hinge"], rod["endpoint"]))  # type: ignore[arg-type]
    return rods


def _reachable(grid: np.ndarray, start: tuple[int, int], anchor: tuple[int, int]) -> bool:
    if not _in_bounds(grid.shape, *start):
        return False
    if int(grid[start]) not in (BG, ANCHOR):
        return False
    if start == anchor:
        return True

    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) not in (BG, ANCHOR):
                continue
            if (nr, nc) == anchor:
                return True
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def _pick_fold_dir(grid: np.ndarray, rod: dict[str, object], anchor: tuple[int, int]) -> tuple[int, int]:
    options = []
    for direction in _perp_dirs(rod["orientation"]):  # type: ignore[arg-type]
        start = _add(rod["free_end"], direction)  # type: ignore[arg-type]
        if _reachable(grid, start, anchor):
            options.append(direction)
    assert len(options) == 1, "each rod must have exactly one anchor-reachable fold direction"
    return options[0]


def _render_output_from_rods(
    grid: np.ndarray,
    rods: list[dict[str, object]],
    fold_dirs: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    line_cells = set()
    for rod in rods:
        line_cells.update(rod["input_line_cells"])  # type: ignore[arg-type]
    for r, c in line_cells:
        out[r, c] = BG

    occupancy: dict[tuple[int, int], list[int]] = {}
    for rod, fold_dir in zip(rods, fold_dirs):
        line_color = int(rod["line_color"])
        for pos in rod["fixed_line"]:  # type: ignore[assignment]
            occupancy.setdefault(pos, []).append(line_color)
        hinge = rod["hinge"]  # type: ignore[assignment]
        tail_len = int(rod["tail_len"])
        for k in range(1, tail_len + 1):
            pos = _add(hinge, fold_dir, k)
            occupancy.setdefault(pos, []).append(line_color)

    for pos, colors in occupancy.items():
        assert int(out[pos]) not in (RED, GREY, WALL, ANCHOR), "folded pixels cannot overwrite markers, walls, or anchor"
        out[pos] = YELLOW if len(colors) >= 2 else colors[0]
    return out


def _space_counterexample(shape: tuple[int, int], rod: dict[str, object], fold_dir: tuple[int, int]) -> bool:
    h, w = shape
    hinge = rod["hinge"]  # type: ignore[assignment]
    if rod["orientation"] == "h":
        up = hinge[0]
        down = h - 1 - hinge[0]
        if up == down:
            return False
        smaller = (-1, 0) if up < down else (1, 0)
        return fold_dir == smaller
    left = hinge[1]
    right = w - 1 - hinge[1]
    if left == right:
        return False
    smaller = (0, -1) if left < right else (0, 1)
    return fold_dir == smaller


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked rod toward the anchor-reachable side, then recolor fold overlaps yellow.

    Input structure:
    - The grid has a black background with 3-4 straight rods, magenta wall pixels, and one orange anchor pixel.
    - Each rod is a 5-7 cell horizontal or vertical segment whose line cells are all blue or all green.
    - Every rod contains one grey endpoint marker at one end and one red hinge marker inside the rod.

    Rule:
    - Treat the grey-marker side as fixed and the cells on the far side of the red hinge as the free tail.
    - From the free-end cell, test the two perpendicular sides by 4-connected flood-fill through only black cells and the orange anchor; fold toward the unique side that can reach the anchor.
    - Rotate the free tail 90 degrees around the hinge into that chosen perpendicular ray, keep the fixed segment and both markers in place, and after all rods are folded recolor every coordinate occupied by two or more rods' non-marker cells to yellow.

    Key insight:
    - Fold direction is decided by reachability topology created by the wall barriers, not by which geometric side has more open space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    anchors = np.argwhere(g == ANCHOR)
    assert len(anchors) == 1, "expected exactly one anchor"
    anchor = tuple(map(int, anchors[0]))

    rods = _parse_rods(g)
    fold_dirs = [_pick_fold_dir(g, rod, anchor) for rod in rods]
    return _render_output_from_rods(g, rods, fold_dirs)


def _render_input(shape: tuple[int, int], rods: list[dict[str, object]], walls: set[tuple[int, int]], anchor: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in walls:
        grid[r, c] = WALL
    grid[anchor] = ANCHOR
    for rod in rods:
        for r, c in rod["fixed_line"]:  # type: ignore[assignment]
            grid[r, c] = int(rod["line_color"])
        for r, c in rod["tail_input"]:  # type: ignore[assignment]
            grid[r, c] = int(rod["line_color"])
        grid[rod["hinge"]] = RED  # type: ignore[index]
        grid[rod["endpoint"]] = GREY  # type: ignore[index]
    return grid


def _render_expected_output(
    shape: tuple[int, int],
    rods: list[dict[str, object]],
    walls: set[tuple[int, int]],
    anchor: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in walls:
        grid[r, c] = WALL
    grid[anchor] = ANCHOR
    for rod in rods:
        grid[rod["hinge"]] = RED  # type: ignore[index]
        grid[rod["endpoint"]] = GREY  # type: ignore[index]

    occupancy: dict[tuple[int, int], list[int]] = {}
    for rod in rods:
        line_color = int(rod["line_color"])
        for pos in rod["fixed_line"]:  # type: ignore[assignment]
            occupancy.setdefault(pos, []).append(line_color)
        for pos in rod["folded"]:  # type: ignore[assignment]
            occupancy.setdefault(pos, []).append(line_color)
    for pos, colors in occupancy.items():
        grid[pos] = YELLOW if len(colors) >= 2 else colors[0]
    return grid


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _compatible(
    shape: tuple[int, int],
    existing: list[dict[str, object]],
    candidate: dict[str, object],
    allowed_overlap: set[tuple[int, int]],
) -> bool:
    cand_input = set(candidate["input_cells"])  # type: ignore[arg-type]
    cand_output = set(candidate["output_nonmarker"])  # type: ignore[arg-type]
    cand_markers = {candidate["hinge"], candidate["endpoint"]}  # type: ignore[list-item]
    cand_margin = _expanded(cand_input, shape)

    for rod in existing:
        other_input = set(rod["input_cells"])  # type: ignore[arg-type]
        other_output = set(rod["output_nonmarker"])  # type: ignore[arg-type]
        other_markers = {rod["hinge"], rod["endpoint"]}  # type: ignore[list-item]
        if cand_input & other_input:
            return False
        if cand_margin & other_input:
            return False
        if _expanded(other_input, shape) & cand_input:
            return False
        if (cand_output & other_output) - allowed_overlap:
            return False
        if cand_output & other_markers:
            return False
        if other_output & cand_markers:
            return False
        if cand_markers & other_markers:
            return False
    return True


def _wall_segment_for_rod(
    shape: tuple[int, int],
    rod: dict[str, object],
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    blocked_start = rod["blocked_start"]  # type: ignore[assignment]
    if blocked_start in occupied:
        return None
    segment = {blocked_start}
    axis_a = rod["fixed_step"]  # type: ignore[assignment]
    axis_b = _opposite(axis_a)
    options = [axis_a, axis_b]
    random.shuffle(options)
    extra_len = random.randint(0, 2)
    pos = blocked_start
    direction = options[0]
    for _ in range(extra_len):
        pos = _add(pos, direction)
        if not _in_bounds(shape, *pos) or pos in occupied:
            break
        segment.add(pos)
    return segment


def _sample_total_geometry(require_long_tail: bool) -> tuple[int, int]:
    if require_long_tail:
        tail_len = random.randint(2, 4)
        fixed_red_len = random.randint(1, min(3, 5 - tail_len))
    else:
        total = random.randint(5, 7)
        low_tail = max(1, total - 5)
        high_tail = min(4, total - 2)
        tail_len = random.randint(low_tail, high_tail)
        fixed_red_len = total - 2 - tail_len
    return fixed_red_len, tail_len


def _sample_overlap_pair(shape: tuple[int, int]) -> tuple[list[dict[str, object]], set[tuple[int, int]]] | None:
    h, w = shape
    for _ in range(PAIR_ATTEMPTS):
        overlap = (random.randint(4, h - 5), random.randint(4, w - 5))
        rods: list[dict[str, object]] = []
        for orientation in ("h", "v"):
            fixed_red_len, tail_len = _sample_total_geometry(require_long_tail=True)
            fold_dir = random.choice(_perp_dirs(orientation))
            delta = random.randint(1, tail_len)
            hinge = _add(overlap, _opposite(fold_dir), delta)
            fixed_sign = random.choice((-1, 1))
            line_color = random.choice(ROD_COLORS)
            rod = _rod_from_folded(
                shape,
                orientation,
                hinge,
                fixed_sign,
                fixed_red_len,
                tail_len,
                fold_dir,
                line_color,
            )
            if rod is None:
                break
            rods.append(rod)
        if len(rods) != 2:
            continue
        if not _compatible(shape, [rods[0]], rods[1], {overlap}):
            continue
        if overlap not in rods[0]["output_nonmarker"] or overlap not in rods[1]["output_nonmarker"]:
            continue
        return rods, {overlap}
    return None


def _sample_extra_rod(
    shape: tuple[int, int],
    existing: list[dict[str, object]],
    allowed_overlap: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(EXTRA_ROD_ATTEMPTS):
        orientation = random.choice(("h", "v"))
        fixed_red_len, tail_len = _sample_total_geometry(require_long_tail=False)
        fold_dir = random.choice(_perp_dirs(orientation))
        hinge = (random.randint(1, h - 2), random.randint(1, w - 2))
        fixed_sign = random.choice((-1, 1))
        line_color = random.choice(ROD_COLORS)
        rod = _rod_from_folded(
            shape,
            orientation,
            hinge,
            fixed_sign,
            fixed_red_len,
            tail_len,
            fold_dir,
            line_color,
        )
        if rod is None:
            continue
        if not _compatible(shape, existing, rod, allowed_overlap):
            continue
        return rod
    return None


def _reachable_region(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    if int(grid[start]) != BG:
        return set()
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in seen and int(grid[nr, nc]) == BG:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _choose_anchor(grid: np.ndarray, rods: list[dict[str, object]]) -> tuple[int, int] | None:
    reachable_sets = [_reachable_region(grid, rod["chosen_start"]) for rod in rods]  # type: ignore[arg-type]
    if not reachable_sets or any(not region for region in reachable_sets):
        return None
    common = set.intersection(*reachable_sets)
    if not common:
        return None

    rod_cells = set()
    for rod in rods:
        rod_cells.update(rod["input_cells"])  # type: ignore[arg-type]

    candidates = []
    for cell in common:
        if cell in rod_cells:
            continue
        if any(nb in rod_cells for nb in _neighbors4(*cell)):
            continue
        edge_dist = min(cell[0], cell[1], grid.shape[0] - 1 - cell[0], grid.shape[1] - 1 - cell[1])
        candidates.append((edge_dist, random.random(), cell))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _semantic_ok(
    inp: np.ndarray,
    expected_out: np.ndarray,
    planned_rods: list[dict[str, object]],
    overlap_cells: set[tuple[int, int]],
) -> bool:
    try:
        parsed = _parse_rods(inp)
        anchors = np.argwhere(inp == ANCHOR)
        if len(anchors) != 1:
            return False
        anchor = tuple(map(int, anchors[0]))
        fold_dirs = [_pick_fold_dir(inp, rod, anchor) for rod in parsed]
        out = main(inp)
    except Exception:
        return False

    if not np.array_equal(out, expected_out):
        return False
    if not np.any(out == YELLOW):
        return False
    yellow_cells = {tuple(map(int, rc)) for rc in np.argwhere(out == YELLOW)}
    if not overlap_cells.issubset(yellow_cells):
        return False
    if not any(_space_counterexample(inp.shape, rod, fold_dir) for rod, fold_dir in zip(parsed, fold_dirs)):
        return False
    for rod, fold_dir in zip(parsed, fold_dirs):
        blocked_start = _add(rod["free_end"], _opposite(fold_dir))  # type: ignore[arg-type]
        if not _in_bounds(inp.shape, *blocked_start) or int(inp[blocked_start]) != WALL:
            return False
    if len(planned_rods) != len(parsed):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    rod_count = random.choice(ROD_COUNT_CHOICES)

    pair = _sample_overlap_pair(shape)
    if pair is None:
        return None
    rods, overlap_cells = pair

    while len(rods) < rod_count:
        extra = _sample_extra_rod(shape, rods, overlap_cells)
        if extra is None:
            return None
        rods.append(extra)

    occupied = set()
    for rod in rods:
        occupied.update(rod["input_cells"])  # type: ignore[arg-type]
        occupied.update(rod["output_nonmarker"])  # type: ignore[arg-type]

    walls: set[tuple[int, int]] = set()
    for rod in rods:
        segment = _wall_segment_for_rod(shape, rod, occupied | walls)
        if segment is None:
            return None
        walls.update(segment)

    provisional = np.full(shape, BG, dtype=int)
    for r, c in walls:
        provisional[r, c] = WALL
    for rod in rods:
        for r, c in rod["input_cells"]:  # type: ignore[assignment]
            provisional[r, c] = 9  # temporary blocker

    anchor = _choose_anchor(np.where(provisional == 9, BG + 9, provisional), rods)
    if anchor is None:
        return None

    inp = _render_input(shape, rods, walls, anchor)
    expected_out = _render_expected_output(shape, rods, walls, anchor)
    if not _semantic_ok(inp, expected_out, rods, overlap_cells):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reachability-fold puzzle")
