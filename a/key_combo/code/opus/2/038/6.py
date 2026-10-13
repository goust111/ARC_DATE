from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
MAIN = 1
MARKER = 2
EDGE = 3
ODD_EDGE = 7

SIDES = ("top", "bottom", "left", "right")
DIRECTION_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_H_RANGE = (12, 16)
GRID_W_RANGE = (14, 18)
MAIN_SIZE_RANGE = (17, 25)
EMBEDDED_COUNT_RANGE = (1, 4)
TOTAL_MARKERS_RANGE = (3, 6)
GEN_ATTEMPTS = 400

Coord = tuple[int, int]

_PARITY_TOGGLE = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _paint_edge(grid: np.ndarray, side: str, color: int) -> None:
    if side == "top":
        grid[0, :] = color
    elif side == "bottom":
        grid[-1, :] = color
    elif side == "left":
        grid[:, 0] = color
    else:
        grid[:, -1] = color


def _detect_edge(grid: np.ndarray, bg: int) -> tuple[str, int, int, int]:
    candidates: list[tuple[str, int, int, int]] = []
    if int(grid[0, 0]) != bg and np.all(grid[0, :] == grid[0, 0]):
        candidates.append(("top", int(grid[0, 0]), -1, 0))
    if int(grid[-1, 0]) != bg and np.all(grid[-1, :] == grid[-1, 0]):
        candidates.append(("bottom", int(grid[-1, 0]), 1, 0))
    if int(grid[0, 0]) != bg and np.all(grid[:, 0] == grid[0, 0]):
        candidates.append(("left", int(grid[0, 0]), 0, -1))
    if int(grid[0, -1]) != bg and np.all(grid[:, -1] == grid[0, -1]):
        candidates.append(("right", int(grid[0, -1]), 0, 1))

    assert len(candidates) == 1, "expected exactly one solid non-background edge line"
    return candidates[0]


def _blue_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    return sum(
        1
        for rr, cc in neighbors4(r, c)
        if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == MAIN
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair embedded red defects, recolor the cue edge by repair-count parity, and extend isolated markers toward that edge.

    Input structure:
    - A black background contains exactly one full green edge line on one border.
    - One larger blue 4-connected object sits away from that edge line.
    - Red marker pixels are either embedded defects inside the blue object or isolated pixels on the background.

    Rule:
    - Recolor each embedded red marker to blue and count the number of such repairs K.
    - If K is odd, recolor the entire green edge line to orange; if K is even, keep the edge line green.
    - Keep each isolated red marker in place and paint the adjacent black cell one step toward the edge line red; do not overwrite non-black cells.

    Key insight:
    - The red markers split into two roles by local blue-neighbor context, and the repair count feeds back into the final appearance of the directional edge cue itself.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    side, edge_color, dr, dc = _detect_edge(g, bg)

    assert bg == BG, "expected black background"
    assert edge_color == EDGE, "expected a green cue edge in the input"

    out = g.copy()
    embedded: list[Coord] = []
    isolated: list[Coord] = []

    for r0, c0 in np.argwhere(g == MARKER):
        r, c = int(r0), int(c0)
        blue_neighbors = _blue_neighbors(g, r, c)
        if blue_neighbors >= 3:
            embedded.append((r, c))
        else:
            assert blue_neighbors == 0, "markers must be either embedded or isolated"
            isolated.append((r, c))

    for r, c in embedded:
        out[r, c] = MAIN

    _paint_edge(out, side, ODD_EDGE if len(embedded) % 2 == 1 else EDGE)

    for r, c in isolated:
        nr, nc = r + dr, c + dc
        assert in_bounds(g.shape, nr, nc), "isolated-marker extension must remain in bounds"
        if int(g[nr, nc]) == BG:
            out[nr, nc] = MARKER

    return out


def _allowed_main_cells(h: int, w: int, side: str) -> set[Coord]:
    r0, r1 = 2, h - 2
    c0, c1 = 2, w - 2
    if side == "top":
        r0 = 3
    elif side == "bottom":
        r1 = h - 3
    elif side == "left":
        c0 = 3
    else:
        c1 = w - 3
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _degree_in(cells: set[Coord], cell: Coord) -> int:
    r, c = cell
    return sum((rr, cc) in cells for rr, cc in neighbors4(r, c))


def _grow_blob(rng: random.Random, h: int, w: int, side: str, size: int) -> set[Coord] | None:
    allowed = _allowed_main_cells(h, w, side)
    if len(allowed) < size:
        return None

    start = rng.choice(tuple(allowed))
    blob = {start}
    frontier = {(rr, cc) for rr, cc in neighbors4(*start) if (rr, cc) in allowed}
    center_r = sum(r for r, _ in allowed) / len(allowed)
    center_c = sum(c for _, c in allowed) / len(allowed)

    while len(blob) < size and frontier:
        choices = tuple(frontier)
        weights = []
        for cell in choices:
            r, c = cell
            degree_bias = (_degree_in(blob, cell) + 0.35) ** 3
            dist = abs(r - center_r) + abs(c - center_c)
            weights.append(degree_bias / (1.0 + 0.12 * dist))
        chosen = rng.choices(choices, weights=weights, k=1)[0]
        frontier.remove(chosen)
        blob.add(chosen)
        for rr, cc in neighbors4(*chosen):
            if (rr, cc) in allowed and (rr, cc) not in blob:
                frontier.add((rr, cc))

    if len(blob) != size:
        return None

    rows = [r for r, _ in blob]
    cols = [c for _, c in blob]
    bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    if bbox_area == len(blob):
        return None
    return blob


def _is_connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(cells)


def _embedded_order(rng: random.Random, blob: set[Coord]) -> list[Coord]:
    candidates = [cell for cell in blob if _degree_in(blob, cell) >= 3]
    rng.shuffle(candidates)
    picked: list[Coord] = []
    removed: set[Coord] = set()

    for cell in candidates:
        if any((rr, cc) in removed for rr, cc in neighbors4(*cell)):
            continue
        trial = removed | {cell}
        if _is_connected(blob - trial):
            picked.append(cell)
            removed = trial
    return picked


def _choose_embedded_sites(
    rng: random.Random, blob: set[Coord], want_odd: bool
) -> list[Coord] | None:
    ordered = _embedded_order(rng, blob)
    feasible = [
        k
        for k in range(EMBEDDED_COUNT_RANGE[0], min(EMBEDDED_COUNT_RANGE[1], len(ordered)) + 1)
        if (k % 2 == 1) == want_odd
    ]
    if not feasible:
        return None
    count = rng.choice(feasible)
    return ordered[:count]


def _well_separated(pair: tuple[Coord, Coord], used: set[Coord]) -> bool:
    for r, c in pair:
        for rr, cc in used:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
    return True


def _candidate_isolated_pairs(
    rng: random.Random, grid: np.ndarray, blob: set[Coord], dr: int, dc: int
) -> list[tuple[Coord, Coord]]:
    h, w = grid.shape
    pairs: list[tuple[Coord, Coord]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            if any((rr, cc) in blob for rr, cc in neighbors4(r, c)):
                continue
            tr, tc = r + dr, c + dc
            if not in_bounds(grid.shape, tr, tc):
                continue
            if int(grid[tr, tc]) != BG:
                continue
            pairs.append(((r, c), (tr, tc)))

    rng.shuffle(pairs)
    selected: list[tuple[Coord, Coord]] = []
    used: set[Coord] = set()
    for pair in pairs:
        marker, target = pair
        if marker in used or target in used:
            continue
        if not _well_separated(pair, used):
            continue
        selected.append(pair)
        used.add(marker)
        used.add(target)
    return selected


def _gen_attempt(rng: random.Random, want_odd: bool) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    side = rng.choice(SIDES)
    dr, dc = DIRECTION_BY_SIDE[side]

    base = np.full((h, w), BG, dtype=int)
    _paint_edge(base, side, EDGE)

    blob = _grow_blob(rng, h, w, side, rng.randint(*MAIN_SIZE_RANGE))
    if blob is None:
        return None
    for r, c in blob:
        base[r, c] = MAIN

    embedded_sites = _choose_embedded_sites(rng, blob, want_odd)
    if embedded_sites is None:
        return None

    candidate_pairs = _candidate_isolated_pairs(rng, base, blob, dr, dc)
    min_isolated = max(1, TOTAL_MARKERS_RANGE[0] - len(embedded_sites))
    max_isolated = min(TOTAL_MARKERS_RANGE[1] - len(embedded_sites), len(candidate_pairs))
    if max_isolated < min_isolated:
        return None
    isolated_count = rng.randint(min_isolated, max_isolated)
    isolated_pairs = candidate_pairs[:isolated_count]

    inp = base.copy()
    for r, c in embedded_sites:
        inp[r, c] = MARKER
    for (r, c), _target in isolated_pairs:
        inp[r, c] = MARKER

    expected = base.copy()
    if len(embedded_sites) % 2 == 1:
        _paint_edge(expected, side, ODD_EDGE)
    for (r, c), (tr, tc) in isolated_pairs:
        expected[r, c] = MARKER
        expected[tr, tc] = MARKER

    if not _is_connected(set(zip(*np.where(inp == MAIN)))):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _PARITY_TOGGLE

    rng = random.Random()
    want_odd = (_PARITY_TOGGLE % 2) == 0
    _PARITY_TOGGLE += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng, want_odd=want_odd)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
