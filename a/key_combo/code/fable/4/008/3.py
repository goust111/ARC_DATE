from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
ORANGE = 7
TEAL = 8

TURN_COLORS = (BLUE, RED)
ALLOWED_INPUT_COLORS = {BLACK, BLUE, RED, TEAL}

MIN_STEP = 2
GRID_H_RANGE = (12, 26)
GRID_W_RANGE = (12, 26)
HIT_COUNT_RANGE = (4, 8)
DISTRACTOR_COUNT_RANGE = (3, 9)
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_interior(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 < r < h - 1 and 0 < c < w - 1


def _initial_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    dirs: list[tuple[int, int]] = []
    if r == 0:
        dirs.append((1, 0))
    if r == h - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == w - 1:
        dirs.append((0, -1))
    assert len(dirs) == 1, "teal source must be on exactly one edge, not a corner"
    return dirs[0]


def _turn(direction: tuple[int, int], color: int) -> tuple[int, int]:
    dr, dc = direction
    if int(color) == RED:
        return dc, -dr
    if int(color) == BLUE:
        return -dc, dr
    raise AssertionError("only red and blue seeds turn the ray")


def _ray_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    out: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return out
        out.append((r, c))


def _open_segment(
    start: tuple[int, int], end: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    out: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if (r, c) == end:
            return out
        out.append((r, c))


def _source_options(h: int, w: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    opts: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for c in range(1, w - 1):
        opts.append(((0, c), (1, 0)))
        opts.append(((h - 1, c), (-1, 0)))
    for r in range(1, h - 1):
        opts.append(((r, 0), (0, 1)))
        opts.append(((r, w - 1), (0, -1)))
    return opts


def _trace_hits(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= ALLOWED_INPUT_COLORS, "unexpected input color"

    source = np.argwhere(g == TEAL)
    assert len(source) == 1, "expected exactly one teal fire source"
    r0, c0 = map(int, source[0])
    direction = _initial_direction(g.shape, r0, c0)

    pos = (r0, c0)
    visited: set[tuple[int, int]] = set()
    hits: list[tuple[int, int, int]] = []
    for _ in range(g.size + 1):
        hit: tuple[int, int, int] | None = None
        for r, c in _ray_cells(g.shape, pos, direction):
            color = int(g[r, c])
            if color in TURN_COLORS:
                hit = (r, c, color)
                break
        if hit is None:
            return hits

        r, c, color = hit
        assert (r, c) not in visited, "a seed may be hit at most once"
        visited.add((r, c))
        hits.append(hit)
        pos = (r, c)
        direction = _turn(direction, color)

    raise AssertionError("ray trace did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Light the hidden ray's hit seeds orange while leaving the invisible ray unpainted.

    Input structure:
    - The background is black, with scattered single-pixel red and blue seeds.
    - Exactly one teal seed sits on a non-corner edge cell and is the ignition source.
    - No visible line marks the ray path; only seed pixels are present.

    Rule:
    - Start from the teal source and move perpendicular to that edge into the grid.
    - The ray travels straight until the nearest red or blue seed in its direction.
    - Each hit seed is recolored orange in the output, then its original color turns
      the ray ninety degrees: red clockwise, blue counterclockwise.
    - Stop when the ray leaves the grid; all unhit seeds and the teal source stay unchanged.

    Key insight:
    - The red and blue pixels are not just targets but turn instructions for an
      invisible causal chain, so one missed turn changes all later orange seeds.
    """
    g = np.asarray(grid, dtype=int)
    hits = _trace_hits(g)

    out = g.copy()
    for r, c, _color in hits:
        out[r, c] = ORANGE
    return out


def _choose_turn_colors(rng: random.Random, n_hits: int) -> list[int]:
    colors = [rng.choice(TURN_COLORS) for _ in range(n_hits)]
    red_pos, blue_pos = rng.sample(range(n_hits - 1), 2)
    colors[red_pos] = RED
    colors[blue_pos] = BLUE
    return colors


def _target_candidates(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden_targets: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for dist, cell in enumerate(_ray_cells(shape, pos, direction), start=1):
        r, c = cell
        if cell in occupied:
            break
        if dist < MIN_STEP:
            continue
        if not _is_interior(shape, r, c):
            continue
        if cell in forbidden_targets:
            continue
        candidates.append(cell)
    return candidates


def _make_chain(
    rng: random.Random, h: int, w: int
) -> tuple[
    tuple[int, int],
    list[tuple[int, int, int]],
    list[tuple[int, int]],
    set[tuple[int, int]],
] | None:
    n_hits = rng.randint(*HIT_COUNT_RANGE)
    colors = _choose_turn_colors(rng, n_hits)

    source, direction = rng.choice(_source_options(h, w))
    shape = (h, w)
    pos = source
    occupied = {source}
    forbidden_targets: set[tuple[int, int]] = set()
    hits: list[tuple[int, int, int]] = []
    incoming_dirs: list[tuple[int, int]] = []

    for color in colors:
        candidates = _target_candidates(shape, pos, direction, occupied, forbidden_targets)
        if not candidates:
            return None

        # Prefer moderate gaps so examples stay readable but still varied.
        nearby = [p for p in candidates if len(_open_segment(pos, p, direction)) <= 8]
        target = rng.choice(nearby or candidates)
        forbidden_targets.update(_open_segment(pos, target, direction))
        incoming_dirs.append(direction)
        hits.append((target[0], target[1], int(color)))
        occupied.add(target)
        pos = target
        direction = _turn(direction, int(color))

    final_ray = _ray_cells(shape, pos, direction)
    if any(cell in occupied for cell in final_ray):
        return None
    forbidden_targets.update(final_ray)

    if {RED, BLUE} - {color for *_rc, color in hits[:-1]}:
        return None
    return source, hits, incoming_dirs, forbidden_targets


def _too_close(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    return any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in occupied)


def _place_shadow_distractor(
    rng: random.Random,
    grid: np.ndarray,
    hits: list[tuple[int, int, int]],
    incoming_dirs: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    indices = list(range(len(hits) - 1))
    rng.shuffle(indices)
    for idx in indices:
        r, c, _color = hits[idx]
        direction = incoming_dirs[idx]
        ray = _ray_cells(grid.shape, (r, c), direction)
        rng.shuffle(ray)
        for cell in ray:
            rr, cc = cell
            if not _is_interior(grid.shape, rr, cc):
                continue
            if cell in occupied or cell in forbidden or _too_close(cell, occupied):
                continue
            grid[rr, cc] = rng.choice(TURN_COLORS)
            occupied.add(cell)
            return cell
    return None


def _safe_distractor_cells(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cell = (r, c)
            if cell in occupied or cell in forbidden:
                continue
            if _too_close(cell, occupied):
                continue
            cells.append(cell)
    return cells


def _has_shadow_distractor(
    grid: np.ndarray,
    hits: list[tuple[int, int, int]],
    incoming_dirs: list[tuple[int, int]],
) -> bool:
    hit_cells = {(r, c) for r, c, _color in hits}
    for (r, c, _color), direction in zip(hits[:-1], incoming_dirs[:-1]):
        for cell in _ray_cells(grid.shape, (r, c), direction):
            rr, cc = cell
            if int(grid[rr, cc]) in TURN_COLORS and cell not in hit_cells:
                return True
    return False


def _semantic_check(
    grid: np.ndarray,
    expected_hits: list[tuple[int, int, int]],
    incoming_dirs: list[tuple[int, int]],
) -> bool:
    try:
        traced = _trace_hits(grid)
    except AssertionError:
        return False
    if traced != expected_hits:
        return False
    if len(traced) < HIT_COUNT_RANGE[0]:
        return False
    if {RED, BLUE} - {color for *_rc, color in traced[:-1]}:
        return False
    if not _has_shadow_distractor(grid, traced, incoming_dirs):
        return False

    out = main(grid)
    hit_cells = {(r, c) for r, c, _color in traced}
    all_seed_cells = {
        (int(r), int(c))
        for r, c in np.argwhere((grid == RED) | (grid == BLUE))
    }
    dormant = all_seed_cells - hit_cells
    if len(dormant) < 3:
        return False
    if not np.all(out[grid == TEAL] == TEAL):
        return False
    if not np.all(out[grid == BLACK] == BLACK):
        return False
    for r, c in hit_cells:
        if int(out[r, c]) != ORANGE:
            return False
    for r, c in dormant:
        if int(out[r, c]) != int(grid[r, c]):
            return False
    changed = {(int(r), int(c)) for r, c in np.argwhere(out != grid)}
    return changed == hit_cells


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    chain = _make_chain(rng, h, w)
    if chain is None:
        return None

    source, hits, incoming_dirs, forbidden = chain
    grid = np.zeros((h, w), dtype=int)
    grid[source] = TEAL
    occupied = {source}
    for r, c, color in hits:
        grid[r, c] = color
        occupied.add((r, c))

    if _place_shadow_distractor(rng, grid, hits, incoming_dirs, occupied, forbidden) is None:
        return None

    distractor_total = rng.randint(*DISTRACTOR_COUNT_RANGE)
    while len([cell for cell in occupied if cell != source and cell not in {(r, c) for r, c, _ in hits}]) < distractor_total:
        candidates = _safe_distractor_cells(grid, occupied, forbidden)
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        grid[r, c] = rng.choice(TURN_COLORS)
        occupied.add((r, c))

    if not _semantic_check(grid, hits, incoming_dirs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-turn ignition chain input")
