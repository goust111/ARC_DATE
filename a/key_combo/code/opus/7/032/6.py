from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
ALIGN_DIRS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

SPRITE_LIBRARY = [
    frozenset({(0, 0), (-1, 0), (1, 0), (0, -1), (1, 1)}),
    frozenset({(0, 0), (-1, 0), (1, 0), (0, 1), (-1, -1)}),
    frozenset({(0, 0), (0, -1), (0, 1), (1, 0), (-1, 1)}),
    frozenset({(0, 0), (-1, 0), (0, -1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, -1), (0, 1), (-1, 0), (1, -1)}),
    frozenset({(0, 0), (-1, 0), (0, 1), (1, 0), (-1, -1)}),
    frozenset({(0, 0), (-1, 0), (1, 0), (-1, -1), (1, 1)}),
    frozenset({(0, 0), (-1, 0), (1, 0), (-1, 1), (1, -1)}),
    frozenset({(0, 0), (0, -1), (0, 1), (-1, -1), (1, 1)}),
    frozenset({(0, 0), (0, -1), (0, 1), (-1, 1), (1, -1)}),
]

GRID_MIN = 15
GRID_MAX = 24
SPRITE_COUNT_CHOICES = (3, 4)
INPUT_LINE_DIST_RANGE = (4, 8)
GEN_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 120


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(r: int, c: int, conn8: bool) -> list[tuple[int, int]]:
    dirs = DIR8 if conn8 else DIR4
    return [(r + dr, c + dc) for dr, dc in dirs]


def _components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, conn8):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _sprite_cells(anchor: tuple[int, int], sprite: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in sprite}


def _shortest_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    r_step = _sign(tr - r)
    c_step = _sign(tc - c)
    while r != tr and c != tc:
        r += r_step
        c += c_step
        path.append((r, c))
    while r != tr:
        r += r_step
        path.append((r, c))
    while c != tc:
        c += c_step
        path.append((r, c))
    return path


def _diag_run_length(path: list[tuple[int, int]]) -> int:
    longest = 0
    run_steps = 0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        if abs(r1 - r0) == 1 and abs(c1 - c0) == 1:
            run_steps += 1
        else:
            if run_steps:
                longest = max(longest, run_steps + 1)
                run_steps = 0
    if run_steps:
        longest = max(longest, run_steps + 1)
    return longest


def _line_choice_key(a: tuple[int, int], b: tuple[int, int], diag_len: int) -> tuple[int, int, int, int, int]:
    p0, p1 = sorted((a, b))
    return (-diag_len, p0[0], p0[1], p1[0], p1[1])


def _select_best_line(
    anchors: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], int]:
    best = None
    for a, b in combinations(sorted(anchors), 2):
        path = _shortest_line(a, b)
        diag_len = _diag_run_length(path)
        candidate = (_line_choice_key(a, b, diag_len), a, b, path, diag_len)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _key, a, b, path, diag_len = best
    return a, b, path, diag_len


def _recover_blue_endpoints(grid: np.ndarray) -> list[tuple[int, int]]:
    blue_mask = grid == BLUE
    comps = _components(blue_mask, conn8=True)
    assert len(comps) == 1, "expected one blue line component"
    blue_set = set(comps[0])
    endpoints = [
        cell
        for cell in blue_set
        if sum((nr, nc) in blue_set for nr, nc in _neighbors(cell[0], cell[1], True)) <= 1
    ]
    assert len(endpoints) == 2, "expected a single non-branching blue line"
    return sorted(endpoints)


def _recover_sprite(
    grid: np.ndarray, endpoints: list[tuple[int, int]]
) -> tuple[frozenset[tuple[int, int]], list[tuple[int, int]]]:
    comps = _components(grid == GREEN, conn8=False)
    assert comps, "expected green sprite pixels"
    full_size = max(len(comp) for comp in comps)
    full_components = [comp for comp in comps if len(comp) == full_size]
    assert full_components, "expected at least one intact sprite"

    sprite = None
    intact_anchors: list[tuple[int, int]] = []
    for comp in full_components:
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0
        anchor = ((r0 + r1) // 2, (c0 + c1) // 2)
        comp_set = set(comp)
        assert anchor in comp_set, "intact sprite must include its anchor"
        rel = frozenset((r - anchor[0], c - anchor[1]) for r, c in comp)
        if sprite is None:
            sprite = rel
        else:
            assert rel == sprite, "all intact sprites must share one template"
        intact_anchors.append(anchor)

    assert sprite is not None
    anchors = sorted(set(intact_anchors + endpoints))
    assert 3 <= len(anchors) <= 4, "expected 3 or 4 sprite anchors"
    return sprite, anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap sprite and line roles, then keep only the anchor-connection whose diagonal run is longest.

    Input structure:
    - A black background contains 3 or 4 identical green sprite bodies placed apart from one another.
    - A single blue straight segment connects the anchor centers of exactly two sprites; that blue line may overwrite those two anchor pixels.
    - Every sprite uses the same small irregular 4-connected template, and each anchor is the center pixel of the sprite bbox.

    Rule:
    - Recover all sprite anchors and erase the original drawing.
    - Connect every pair of anchors by a blue 8-connected shortest path.
    - Re-stamp the green sprite template at the two endpoints of the original input line.
    - For each anchor-to-anchor path, measure the longest contiguous diagonal subsegment; horizontal or vertical paths score 0.
    - Keep only the path with the global maximum diagonal score, breaking ties by the lexicographically smallest endpoint of the pair, then by the other endpoint.

    Key insight:
    - The blue lines to compare are not the input line itself: they are newly created from the recovered anchor set, and the longest diagonal run among those generated paths decides the final survivor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    endpoints = _recover_blue_endpoints(g)
    sprite, anchors = _recover_sprite(g, endpoints)
    _best_a, _best_b, best_path, _diag_len = _select_best_line(anchors)

    out = np.full_like(g, BLACK)
    for r, c in best_path:
        out[r, c] = BLUE
    for anchor in endpoints:
        for dr, dc in sprite:
            rr, cc = anchor[0] + dr, anchor[1] + dc
            assert _in_bounds(out.shape, rr, cc)
            out[rr, cc] = GREEN
    return out


def _is_aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return dr == 0 or dc == 0 or abs(dr) == abs(dc)


def _unit_dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _sign(b[0] - a[0]), _sign(b[1] - a[1])


def _allowed_departures(sprite: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    return {direction for direction in ALIGN_DIRS if direction not in sprite}


def _sprite_fits(shape: tuple[int, int], anchor: tuple[int, int], sprite: frozenset[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, anchor[0] + dr, anchor[1] + dc) for dr, dc in sprite)


def _sprites_separate(
    anchor: tuple[int, int],
    anchors: list[tuple[int, int]],
    sprite: frozenset[tuple[int, int]],
) -> bool:
    new_cells = _sprite_cells(anchor, sprite)
    expanded = {(r + dr, c + dc) for r, c in new_cells for dr, dc in DIR8 + [(0, 0)]}
    for other in anchors:
        if expanded & _sprite_cells(other, sprite):
            return False
    return True


def _all_sprite_cells(
    anchors: list[tuple[int, int]], sprite: frozenset[tuple[int, int]]
) -> dict[tuple[int, int], tuple[int, int]]:
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for anchor in anchors:
        for cell in _sprite_cells(anchor, sprite):
            out[cell] = anchor
    return out


def _line_clear_for_input(
    a: tuple[int, int],
    b: tuple[int, int],
    sprite_cells: dict[tuple[int, int], tuple[int, int]],
) -> bool:
    if not _is_aligned(a, b):
        return False
    path = _shortest_line(a, b)
    if _unit_dir(a, b) != _unit_dir(path[0], path[1]):
        return False
    for cell in path[1:-1]:
        if cell in sprite_cells:
            return False
    return True


def _build_input_grid(
    shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    sprite: frozenset[tuple[int, int]],
    input_pair: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for anchor in anchors:
        for r, c in _sprite_cells(anchor, sprite):
            grid[r, c] = GREEN
    for r, c in _shortest_line(*input_pair):
        grid[r, c] = BLUE
    return grid


def _choose_input_pair(
    anchors: list[tuple[int, int]],
    sprite: frozenset[tuple[int, int]],
    prefer_pair: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    sprite_cells = _all_sprite_cells(anchors, sprite)
    candidates = []
    for a, b in combinations(anchors, 2):
        if not _is_aligned(a, b):
            continue
        step_ab = _unit_dir(a, b)
        step_ba = _unit_dir(b, a)
        departures = _allowed_departures(sprite)
        if step_ab not in departures or step_ba not in departures:
            continue
        if _line_clear_for_input(a, b, sprite_cells):
            candidates.append((a, b))
    if not candidates:
        return None
    if prefer_pair is not None:
        preferred = tuple(sorted(prefer_pair))
        for pair in candidates:
            if tuple(sorted(pair)) == preferred:
                return pair
    return random.choice(candidates)


def _union_overlap_score(anchors: list[tuple[int, int]]) -> tuple[int, int]:
    counts: dict[tuple[int, int], int] = {}
    total = 0
    for a, b in combinations(anchors, 2):
        for cell in _shortest_line(a, b):
            counts[cell] = counts.get(cell, 0) + 1
            total += 1
    overlap = sum(v - 1 for v in counts.values() if v > 1)
    return overlap, total


def _input_pair_candidates(
    shape: tuple[int, int], sprite: frozenset[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    direction = random.choice([(0, 1), (1, 0), (1, 1), (1, -1)])
    distance = random.randint(*INPUT_LINE_DIST_RANGE)
    dr, dc = direction
    if dc == -1:
        c0_min = 1 + distance
        c0_max = shape[1] - 2
    else:
        c0_min = 1
        c0_max = shape[1] - 2 - distance * abs(dc)
    r0_min = 1
    r0_max = shape[0] - 2 - distance * abs(dr)
    if r0_min > r0_max or c0_min > c0_max:
        raise ValueError("no room for input pair")
    a = (random.randint(r0_min, r0_max), random.randint(c0_min, c0_max))
    b = (a[0] + dr * distance, a[1] + dc * distance)
    assert _sprite_fits(shape, a, sprite) and _sprite_fits(shape, b, sprite)
    return a, b, direction


def _generate_anchor_layout(
    shape: tuple[int, int], sprite: frozenset[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    departures = _allowed_departures(sprite)
    for _ in range(PLACEMENT_ATTEMPTS):
        try:
            first, second, direction = _input_pair_candidates(shape, sprite)
        except ValueError:
            continue
        if direction not in departures or (-direction[0], -direction[1]) not in departures:
            continue
        anchors = [first, second]
        sprite_cells = _all_sprite_cells(anchors, sprite)
        if not _line_clear_for_input(first, second, sprite_cells):
            continue

        candidates = [
            (r, c)
            for r in range(1, shape[0] - 1)
            for c in range(1, shape[1] - 1)
            if _sprite_fits(shape, (r, c), sprite)
        ]
        random.shuffle(candidates)
        for anchor in candidates:
            if len(anchors) == count:
                break
            if not _sprites_separate(anchor, anchors, sprite):
                continue
            if any(cell in _shortest_line(first, second) for cell in _sprite_cells(anchor, sprite)):
                continue
            anchors.append(anchor)
        if len(anchors) == count:
            return sorted(anchors)
    return None


def _line_summary(
    anchors: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    return [(a, b, _diag_run_length(_shortest_line(a, b))) for a, b in combinations(sorted(anchors), 2)]


def _valid_generated_instance(
    grid: np.ndarray,
    anchors: list[tuple[int, int]],
    input_pair: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    _best_a, _best_b, _path, best_diag = _select_best_line(anchors)
    if best_diag <= 0:
        return False

    summaries = _line_summary(anchors)
    max_diag = max(diag for _, _, diag in summaries)
    ties = [(a, b) for a, b, diag in summaries if diag == max_diag]
    wants_tie = random.random() < 0.35
    if wants_tie != (len(ties) > 1):
        return False

    overlap, total = _union_overlap_score(anchors)
    if total == 0 or overlap > total // 3:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False

    # Keep outputs visually sparse enough to read.
    if np.count_nonzero(out == BLUE) > (grid.shape[0] * grid.shape[1]) // 3:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
        sprite = random.choice(SPRITE_LIBRARY)
        count = random.choice(SPRITE_COUNT_CHOICES)
        anchors = _generate_anchor_layout(shape, sprite, count)
        if anchors is None:
            continue

        best_pair = _select_best_line(anchors)[:2]
        use_best_as_input = random.random() < 0.45
        preferred_pair = best_pair if use_best_as_input and _is_aligned(*best_pair) else None
        input_pair = _choose_input_pair(anchors, sprite, preferred_pair)
        if input_pair is None:
            continue

        grid = _build_input_grid(shape, anchors, sprite, input_pair)
        if _valid_generated_instance(grid, anchors, input_pair):
            return grid
    raise RuntimeError("failed to generate valid input")
