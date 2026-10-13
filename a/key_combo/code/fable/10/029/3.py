from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MAGENTA_SOURCE = PINK
CYAN_SOURCE = TEAL

Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components8(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _classify_diagonal_segment(coords: list[Coord]) -> tuple[str, int]:
    assert len(coords) >= 2, "each red diagonal segment must contain at least two pixels"
    sums = {r + c for r, c in coords}
    diffs = {r - c for r, c in coords}

    if len(sums) == 1 and len(diffs) > 1:
        ordered = sorted(coords)
        for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
            assert (r1 - r0, c1 - c0) == (1, -1), "slash segment must be contiguous"
        return "slash", next(iter(sums))

    if len(diffs) == 1 and len(sums) > 1:
        ordered = sorted(coords)
        for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
            assert (r1 - r0, c1 - c0) == (1, 1), "backslash segment must be contiguous"
        return "backslash", next(iter(diffs))

    raise AssertionError("red components must be straight 45-degree diagonal segments")


def _find_hidden_center(grid: np.ndarray) -> Coord:
    red_components = _components8(grid == RED)
    assert len(red_components) == 2, "expected exactly two red diagonal segments"

    slash_sum: int | None = None
    backslash_diff: int | None = None
    for comp in red_components:
        kind, constant = _classify_diagonal_segment(comp)
        if kind == "slash":
            assert slash_sum is None, "expected exactly one slash segment"
            slash_sum = constant
        else:
            assert backslash_diff is None, "expected exactly one backslash segment"
            backslash_diff = constant

    assert slash_sum is not None and backslash_diff is not None, "need one segment of each diagonal orientation"
    r_num = slash_sum + backslash_diff
    c_num = slash_sum - backslash_diff
    assert r_num % 2 == 0 and c_num % 2 == 0, "extended diagonals must intersect at a grid cell"
    center = (r_num // 2, c_num // 2)
    assert _in_bounds(grid.shape, *center), "hidden intersection must lie inside the grid"
    assert int(grid[center]) == _get_bg(grid), "the hidden center cell must be uncolored"
    return center


def _rotate_offset(offset: Coord, color: int) -> Coord:
    dr, dc = offset
    if color == MAGENTA_SOURCE:
        return dc, -dr
    if color == CYAN_SOURCE:
        return -dc, dr
    raise AssertionError("unexpected source color")


def _target_for(center: Coord, source: Coord, color: int) -> Coord:
    dr = source[0] - center[0]
    dc = source[1] - center[1]
    rr, rc = _rotate_offset((dr, dc), color)
    return center[0] + rr, center[1] + rc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the two red diagonals to recover the missing intersection, then rotate magenta and cyan sources around it into orange targets.

    Input structure:
    - A black background contains exactly two short red 45-degree diagonal segments, one "/"-oriented and one "\\"-oriented.
    - The extended diagonals intersect at a single in-bounds grid cell that is left black in the input.
    - Additional isolated single-pixel sources are magenta and cyan.

    Rule:
    - Infer the hidden intersection point O by extending the two red diagonal segments along their diagonals.
    - Remove every magenta and cyan source pixel.
    - Rotate each magenta source 90 degrees clockwise around O and each cyan source 90 degrees counterclockwise around O, preserving the full row/column offset from O.
    - Paint every in-bounds landing cell orange; out-of-bounds landings disappear; keep the red segments unchanged.

    Key insight:
    - The rotation center is not drawn, so the solver must first construct it from the intersection of the two diagonal extensions.
    - Source color determines rotation direction, and the output recolors both branches to the same orange target color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = set(map(int, np.unique(g)))
    assert colors <= {bg, RED, MAGENTA_SOURCE, CYAN_SOURCE}, "unexpected colors present"

    center = _find_hidden_center(g)
    magenta_sources = [tuple(map(int, rc)) for rc in np.argwhere(g == MAGENTA_SOURCE)]
    cyan_sources = [tuple(map(int, rc)) for rc in np.argwhere(g == CYAN_SOURCE)]
    assert magenta_sources and cyan_sources, "both source colors must appear"

    out = g.copy()
    out[(g == MAGENTA_SOURCE) | (g == CYAN_SOURCE)] = bg

    for color, sources in ((MAGENTA_SOURCE, magenta_sources), (CYAN_SOURCE, cyan_sources)):
        for source in sources:
            target = _target_for(center, source, color)
            if _in_bounds(g.shape, *target):
                assert int(out[target]) == bg, "rotated targets must land on empty cells"
                out[target] = ORANGE

    return out


GRID_H_RANGE = (14, 25)
GRID_W_RANGE = (14, 25)
CENTER_MARGIN = 3
SEGMENT_LENGTH_RANGE = (2, 5)
SEGMENT_GAP_RANGE = (1, 4)
MAGENTA_COUNT_RANGE = (2, 4)
CYAN_COUNT_RANGE = (2, 4)
MAX_SOURCE_RADIUS = 8
MAX_OOB_TARGETS = 2
GEN_ATTEMPTS = 500


def _max_steps(shape: tuple[int, int], center: Coord, direction: Coord) -> int:
    r, c = center
    dr, dc = direction
    steps = 0
    while _in_bounds(shape, r + dr * (steps + 1), c + dc * (steps + 1)):
        steps += 1
    return steps


def _choose_segment(shape: tuple[int, int], center: Coord, directions: tuple[Coord, Coord]) -> tuple[list[Coord], int]:
    options: list[tuple[Coord, int, int]] = []
    for direction in directions:
        reach = _max_steps(shape, center, direction)
        for gap in range(SEGMENT_GAP_RANGE[0], min(SEGMENT_GAP_RANGE[1], reach - 1) + 1):
            max_len = min(SEGMENT_LENGTH_RANGE[1], reach - gap + 1)
            for length in range(SEGMENT_LENGTH_RANGE[0], max_len + 1):
                options.append((direction, gap, length))
    assert options, "center must allow both diagonal segments"
    direction, gap, length = random.choice(options)
    cells = [(center[0] + direction[0] * k, center[1] + direction[1] * k) for k in range(gap, gap + length)]
    return cells, gap


def _touches_any(cell: Coord, others: set[Coord]) -> bool:
    r, c = cell
    for rr, cc in others:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def _render_expected(
    shape: tuple[int, int],
    slash_cells: list[Coord],
    backslash_cells: list[Coord],
    magenta_sources: list[Coord],
    cyan_sources: list[Coord],
    center: Coord,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BLACK, dtype=int)
    for r, c in slash_cells + backslash_cells:
        inp[r, c] = RED
    for r, c in magenta_sources:
        inp[r, c] = MAGENTA_SOURCE
    for r, c in cyan_sources:
        inp[r, c] = CYAN_SOURCE

    out = inp.copy()
    out[(out == MAGENTA_SOURCE) | (out == CYAN_SOURCE)] = BLACK
    for source in magenta_sources:
        target = _target_for(center, source, MAGENTA_SOURCE)
        if _in_bounds(shape, *target):
            out[target] = ORANGE
    for source in cyan_sources:
        target = _target_for(center, source, CYAN_SOURCE)
        if _in_bounds(shape, *target):
            out[target] = ORANGE
    return inp, out


def _candidate_specs(shape: tuple[int, int], center: Coord, color: int, want_oob: bool, forbidden: set[Coord]) -> list[tuple[Coord, Coord | None, Coord | None]]:
    candidates: list[tuple[Coord, Coord | None, Coord | None]] = []
    limit = min(MAX_SOURCE_RADIUS, max(shape) - 1)
    for dr in range(-limit, limit + 1):
        for dc in range(-limit, limit + 1):
            if dr == 0 and dc == 0:
                continue
            if abs(dr) == abs(dc):
                continue
            source = (center[0] + dr, center[1] + dc)
            if not _in_bounds(shape, *source):
                continue
            if source in forbidden:
                continue

            target = _target_for(center, source, color)
            target_in_bounds = _in_bounds(shape, *target)
            if want_oob != (not target_in_bounds):
                continue
            if target_in_bounds and target in forbidden:
                continue

            wrong_color = CYAN_SOURCE if color == MAGENTA_SOURCE else MAGENTA_SOURCE
            wrong_target = _target_for(center, source, wrong_color)
            wrong_target_value: Coord | None = wrong_target if _in_bounds(shape, *wrong_target) else None
            candidates.append((source, target if target_in_bounds else None, wrong_target_value))
    random.shuffle(candidates)
    return candidates


def _build_requirements() -> list[tuple[int, bool]]:
    magenta_count = random.randint(*MAGENTA_COUNT_RANGE)
    cyan_count = random.randint(*CYAN_COUNT_RANGE)
    reqs = [(MAGENTA_SOURCE, False)] * magenta_count + [(CYAN_SOURCE, False)] * cyan_count

    oob_goal = random.choices([0, 1, 2], weights=[3, 3, 1], k=1)[0]
    oob_goal = min(oob_goal, len(reqs) - 2)
    if oob_goal == 0:
        random.shuffle(reqs)
        return reqs

    indices = list(range(len(reqs)))
    random.shuffle(indices)
    chosen: list[int] = []
    magenta_oob = 0
    cyan_oob = 0
    for idx in indices:
        color = reqs[idx][0]
        magenta_left = sum(1 for i, (c, _) in enumerate(reqs) if c == MAGENTA_SOURCE and i not in chosen)
        cyan_left = sum(1 for i, (c, _) in enumerate(reqs) if c == CYAN_SOURCE and i not in chosen)
        if color == MAGENTA_SOURCE and magenta_left <= 1:
            continue
        if color == CYAN_SOURCE and cyan_left <= 1:
            continue
        chosen.append(idx)
        if color == MAGENTA_SOURCE:
            magenta_oob += 1
        else:
            cyan_oob += 1
        if len(chosen) == oob_goal:
            break

    for idx in chosen:
        color, _ = reqs[idx]
        reqs[idx] = (color, True)
    random.shuffle(reqs)
    return reqs


def _select_sources(
    shape: tuple[int, int],
    center: Coord,
    red_cells: set[Coord],
    requirements: list[tuple[int, bool]],
) -> list[tuple[int, Coord, Coord | None, Coord | None]] | None:
    chosen: list[tuple[int, Coord, Coord | None, Coord | None]] = []
    occupied_input = set(red_cells)
    occupied_targets: set[Coord] = set()

    for color, want_oob in requirements:
        found = False
        for source, target, wrong_target in _candidate_specs(shape, center, color, want_oob, red_cells):
            if source in occupied_input or source in occupied_targets:
                continue
            if _touches_any(source, occupied_input):
                continue
            if target is not None:
                if target in occupied_targets or target in occupied_input:
                    continue
                if _touches_any(target, red_cells):
                    continue
            if wrong_target is not None and wrong_target in red_cells:
                wrong_target = None

            chosen.append((color, source, target, wrong_target))
            occupied_input.add(source)
            if target is not None:
                occupied_targets.add(target)
            found = True
            break
        if not found:
            return None
    return chosen


def _has_direction_witnesses(chosen: list[tuple[int, Coord, Coord | None, Coord | None]], red_cells: set[Coord]) -> bool:
    correct_targets = {target for _color, _source, target, _wrong in chosen if target is not None}
    visible_by_color = {MAGENTA_SOURCE: 0, CYAN_SOURCE: 0}
    witness_by_color = {MAGENTA_SOURCE: False, CYAN_SOURCE: False}

    for color, _source, target, wrong_target in chosen:
        if target is not None:
            visible_by_color[color] += 1
        if wrong_target is not None and wrong_target not in correct_targets and wrong_target not in red_cells:
            witness_by_color[color] = True

    return (
        visible_by_color[MAGENTA_SOURCE] >= 1
        and visible_by_color[CYAN_SOURCE] >= 1
        and witness_by_color[MAGENTA_SOURCE]
        and witness_by_color[CYAN_SOURCE]
    )


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    center = (
        random.randint(CENTER_MARGIN, h - CENTER_MARGIN - 1),
        random.randint(CENTER_MARGIN, w - CENTER_MARGIN - 1),
    )

    slash_cells, slash_gap = _choose_segment(shape, center, ((-1, 1), (1, -1)))
    backslash_cells, backslash_gap = _choose_segment(shape, center, ((-1, -1), (1, 1)))
    if max(slash_gap, backslash_gap) < 2:
        return None

    red_cells = set(slash_cells) | set(backslash_cells)
    requirements = _build_requirements()
    chosen = _select_sources(shape, center, red_cells, requirements)
    if chosen is None or not _has_direction_witnesses(chosen, red_cells):
        return None

    magenta_sources = [source for color, source, _target, _wrong in chosen if color == MAGENTA_SOURCE]
    cyan_sources = [source for color, source, _target, _wrong in chosen if color == CYAN_SOURCE]
    inp, expected = _render_expected(shape, slash_cells, backslash_cells, magenta_sources, cyan_sources, center)

    if int(inp[center]) != BLACK:
        return None
    if set(map(int, np.unique(inp))) != {BLACK, RED, MAGENTA_SOURCE, CYAN_SOURCE}:
        return None
    if len(_components8(inp == RED)) != 2:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-intersection rotation puzzle")
