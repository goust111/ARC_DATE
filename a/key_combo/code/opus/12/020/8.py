from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SPRITE_COLORS = (BLUE, RED, GREEN)
MARKER_COUNT_RANGE = (3, 5)
GRID_H_RANGE = (12, 19)
GRID_W_RANGE = (14, 21)
SHIFT_CHOICES = (-3, -2, -1, 1, 2, 3)
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _single_component(grid: np.ndarray, color: int) -> np.ndarray:
    pts = np.argwhere(grid == color)
    assert len(pts) > 0, f"missing color {color}"
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == color and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    assert len(seen) == len(pts), f"color {color} must form one 4-connected component"
    return np.array(sorted(seen), dtype=int)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _move_rows(coords: np.ndarray, dr: int) -> np.ndarray:
    moved = coords.copy()
    moved[:, 0] += int(dr)
    return moved


def _paint(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def _solve_with_template(grid: np.ndarray, template_color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps = {color: _single_component(g, color) for color in SPRITE_COLORS}
    template = comps[template_color]
    template_shape = _normalize(template)
    template_top = int(template[:, 0].min())

    out = g.copy()
    for color in (RED, GREEN):
        if color == template_color:
            continue
        assert _normalize(comps[color]) == template_shape, "sprites must match by translation"
        out[g == color] = BG
        moved = _move_rows(comps[color], template_top - int(comps[color][:, 0].min()))
        assert moved[:, 0].min() >= 0 and moved[:, 0].max() < g.shape[0]
        _paint(out, moved, color)

    markers = np.argwhere(out == YELLOW)
    h, w = out.shape
    for r, c in markers:
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if out[rr, cc] == BG:
                    out[rr, cc] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the red and green sprite copies to the blue template, then expand yellow marker halos on background only.

    Input structure:
    - A black background contains exactly three 4-connected monochrome sprites: one blue, one red, and one green.
    - The three sprites have the same shape up to translation and occupy disjoint horizontal spans.
    - Several isolated yellow marker pixels lie elsewhere on the grid.

    Rule:
    - Identify the blue sprite as the alignment template and keep it fixed.
    - Move the red and green sprites vertically so their row coordinates match the blue sprite's corresponding rows, while keeping each sprite's columns unchanged; clear their old locations to black.
    - After alignment, for every yellow marker, inspect its clipped 3x3 Chebyshev neighborhood and fill each black cell in that neighborhood with yellow.
    - Never overwrite any non-black cell, so sprite pixels and marker centers block the halo.

    Key insight:
    - The yellow halos depend on the post-alignment sprite positions, so the blue-guided vertical shift must be applied before expanding the marker neighborhoods.
    """
    return _solve_with_template(grid, BLUE)


def _halo_only(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    markers = np.argwhere(out == YELLOW)
    h, w = out.shape
    for r, c in markers:
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if out[rr, cc] == BG:
                    out[rr, cc] = YELLOW
    return out


def _align_only(grid: np.ndarray, template_color: int = BLUE) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps = {color: _single_component(g, color) for color in SPRITE_COLORS}
    template = comps[template_color]
    template_shape = _normalize(template)
    template_top = int(template[:, 0].min())
    out = g.copy()
    for color in SPRITE_COLORS:
        if color == template_color:
            continue
        assert _normalize(comps[color]) == template_shape
        out[g == color] = BG
        moved = _move_rows(comps[color], template_top - int(comps[color][:, 0].min()))
        _paint(out, moved, color)
    return out


def _shape_cells() -> list[tuple[int, int]] | None:
    h, w = random.choice([(2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)])
    target = random.randint(5, min(8, h * w - 1))

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return None

    r0 = min(rows)
    c0 = min(cols)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    hh = max(r for r, _ in norm) + 1
    ww = max(c for _, c in norm) + 1
    if len(norm) == hh * ww:
        return None
    return norm


def _instantiate(shape: list[tuple[int, int]], top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _sample_sprite_layout(h: int, w: int, shape: list[tuple[int, int]]):
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1

    red_shift, green_shift = random.sample(SHIFT_CHOICES, 2)
    min_top = max(0, -min(0, red_shift, green_shift))
    max_top = min(h - shape_h, h - shape_h - max(0, red_shift, green_shift))
    if min_top > max_top:
        return None
    blue_top = random.randint(min_top, max_top)

    gap1 = random.randint(2, 4)
    gap2 = random.randint(2, 4)
    required = 3 * shape_w + gap1 + gap2
    if required > w:
        return None
    left_margin = random.randint(0, w - required)
    starts = [
        left_margin,
        left_margin + shape_w + gap1,
        left_margin + 2 * shape_w + gap1 + gap2,
    ]
    colors_in_order = random.sample(list(SPRITE_COLORS), 3)
    final_lefts = {color: starts[idx] for idx, color in enumerate(colors_in_order)}
    final_tops = {BLUE: blue_top, RED: blue_top, GREEN: blue_top}
    input_tops = {BLUE: blue_top, RED: blue_top + red_shift, GREEN: blue_top + green_shift}

    final_coords = {color: _instantiate(shape, final_tops[color], final_lefts[color]) for color in SPRITE_COLORS}
    input_coords = {color: _instantiate(shape, input_tops[color], final_lefts[color]) for color in SPRITE_COLORS}
    return input_coords, final_coords, final_lefts, (red_shift, green_shift)


def _sprite_sets(coords_by_color: dict[int, np.ndarray]) -> dict[int, set[tuple[int, int]]]:
    return {color: set(map(tuple, coords.tolist())) for color, coords in coords_by_color.items()}


def _halo_cells(center: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    pts = set()
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if _in_bounds(shape, rr, cc):
                pts.add((rr, cc))
    return pts


def _candidate_markers(
    shape: tuple[int, int],
    input_sets: dict[int, set[tuple[int, int]]],
    final_sets: dict[int, set[tuple[int, int]]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    all_input = set().union(*input_sets.values())
    all_final = set().union(*final_sets.values())
    all_blocked = all_input | all_final
    moved_input = input_sets[RED] | input_sets[GREEN]
    moved_final = final_sets[RED] | final_sets[GREEN]
    moved_delta = moved_input ^ moved_final

    interaction = []
    quiet = []
    general = []
    h, w = shape
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            if pos in all_blocked:
                continue
            halo = _halo_cells(pos, shape)
            if halo & all_final:
                general.append(pos)
            if halo & moved_delta:
                interaction.append(pos)
            if not (halo & all_final):
                quiet.append(pos)
    random.shuffle(interaction)
    random.shuffle(quiet)
    random.shuffle(general)
    return interaction, quiet, general


def _far_enough(markers: list[tuple[int, int]], pos: tuple[int, int], min_dist: int = 2) -> bool:
    return all(_chebyshev(pos, other) >= min_dist for other in markers)


def _pick_markers(
    grid_shape: tuple[int, int],
    input_sets: dict[int, set[tuple[int, int]]],
    final_sets: dict[int, set[tuple[int, int]]],
) -> list[tuple[int, int]] | None:
    interaction, quiet, general = _candidate_markers(grid_shape, input_sets, final_sets)
    if not interaction or len(general) < 2:
        return None

    target = random.randint(*MARKER_COUNT_RANGE)
    markers: list[tuple[int, int]] = []

    for pos in interaction:
        if _far_enough(markers, pos):
            markers.append(pos)
            break
    if not markers:
        return None

    if target >= 4:
        for pos in interaction[1:]:
            if _far_enough(markers, pos):
                markers.append(pos)
                break

    if target >= 3:
        quiet_edge = [p for p in quiet if p[0] in (0, grid_shape[0] - 1) or p[1] in (0, grid_shape[1] - 1)]
        random.shuffle(quiet_edge)
        for pos in quiet_edge + quiet:
            if _far_enough(markers, pos):
                markers.append(pos)
                break

    for pool in (general, quiet):
        for pos in pool:
            if len(markers) >= target:
                break
            if _far_enough(markers, pos):
                markers.append(pos)
        if len(markers) >= target:
            break

    if len(markers) < target:
        return None
    return markers


def _build_grid(
    h: int,
    w: int,
    input_coords: dict[int, np.ndarray],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for color, coords in input_coords.items():
        _paint(grid, coords, color)
    for r, c in markers:
        grid[r, c] = YELLOW
    return grid


def _marker_effect_diff(inp: np.ndarray, solved: np.ndarray) -> bool:
    aligned = _align_only(inp, BLUE)
    markers = [tuple(map(int, p)) for p in np.argwhere(inp == YELLOW)]
    for marker in markers:
        halo = _halo_cells(marker, inp.shape)
        aligned_fill = {p for p in halo if int(aligned[p]) == BG}
        solved_fill = {p for p in halo if int(solved[p]) == YELLOW and int(aligned[p]) == BG}
        unaligned_fill = {p for p in halo if int(inp[p]) == BG}
        if solved_fill == aligned_fill and aligned_fill != unaligned_fill:
            return True
    return False


def _valid_layout(inp: np.ndarray, solved: np.ndarray, shifts: tuple[int, int]) -> bool:
    if np.array_equal(inp, solved):
        return False
    if np.array_equal(solved, _halo_only(inp)):
        return False
    if np.array_equal(solved, _align_only(inp, BLUE)):
        return False
    if np.array_equal(solved, _solve_with_template(inp, RED)):
        return False
    if np.array_equal(solved, _solve_with_template(inp, GREEN)):
        return False
    if shifts[0] == 0 or shifts[1] == 0 or shifts[0] == shifts[1]:
        return False
    if not _marker_effect_diff(inp, solved):
        return False

    blue_box = _bbox(_single_component(inp, BLUE))
    red_box = _bbox(_single_component(inp, RED))
    green_box = _bbox(_single_component(inp, GREEN))
    spans = sorted([(blue_box[1], blue_box[3]), (red_box[1], red_box[3]), (green_box[1], green_box[3])])
    for (_, c1), (n0, _) in zip(spans, spans[1:]):
        if n0 - c1 < 3:
            return False

    markers = np.argwhere(inp == YELLOW)
    if not (3 <= len(markers) <= 5):
        return False
    if any(int(inp[r, c]) != YELLOW for r, c in markers):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = _shape_cells()
    if shape is None:
        return None

    layout = _sample_sprite_layout(h, w, shape)
    if layout is None:
        return None
    input_coords, final_coords, _final_lefts, shifts = layout

    input_sets = _sprite_sets(input_coords)
    final_sets = _sprite_sets(final_coords)
    markers = _pick_markers((h, w), input_sets, final_sets)
    if markers is None:
        return None

    grid = _build_grid(h, w, input_coords, markers)
    expected = _build_grid(h, w, final_coords, markers)
    expected = _halo_only(expected)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if not _valid_layout(grid, solved, shifts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
