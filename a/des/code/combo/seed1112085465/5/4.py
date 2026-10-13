from __future__ import annotations

from collections import Counter, deque

import numpy as np
import random


AZURE = 8
GRAY = 5
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
MAROON = 9

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_TO_NAME = {
    (-1, 0): "N",
    (1, 0): "S",
    (0, -1): "W",
    (0, 1): "E",
}
NEIGHBOR_COLOR = {
    0: RED,
    1: BLUE,
    2: GREEN,
    3: YELLOW,
    4: MAROON,
}


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or not bool(mask[r, c]):
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(np.asarray(coords, dtype=int))

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by how many other objects sit one background
    pixel away in the four cardinal directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    components = _find_components(g != bg)
    if not components:
        return g.copy()

    owner = np.full(g.shape, -1, dtype=int)
    for comp_id, coords in enumerate(components):
        owner[coords[:, 0], coords[:, 1]] = comp_id

    height, width = g.shape
    neighbors: list[set[int]] = [set() for _ in components]
    for comp_id, coords in enumerate(components):
        for r, c in coords:
            for dr, dc in DIRS4:
                mr = int(r + dr)
                mc = int(c + dc)
                tr = int(r + 2 * dr)
                tc = int(c + 2 * dc)
                if not (0 <= tr < height and 0 <= tc < width):
                    continue
                if int(g[mr, mc]) != bg:
                    continue
                other = int(owner[tr, tc])
                if other >= 0 and other != comp_id:
                    neighbors[comp_id].add(other)

    out = g.copy()
    for comp_id, coords in enumerate(components):
        degree = min(len(neighbors[comp_id]), 4)
        out[coords[:, 0], coords[:, 1]] = NEIGHBOR_COLOR[degree]
    return out


# --------------- generator (o4-mini) ---------------

LATTICE_SIZE = 5
MAX_LATTICE_SPAN = 5
TILE = 5
STEP = TILE + 1
PATCH_CENTER = (2, 2)
PATCH_EXTRA_CELLS = 6
BORDER_RANGE = (1, 2)
GENERATION_ATTEMPTS = 500

BASE_MOTIFS_BY_DEGREE = {
    0: [((0, 0),)],
    1: [((0, 0), (0, 1))],
    2: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ],
    3: [((0, 1), (1, 0), (1, 1), (1, 2))],
    4: [((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))],
}
ALL_BASE_MOTIFS = [motif for motifs in BASE_MOTIFS_BY_DEGREE.values() for motif in motifs]


def _normalize_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _rotate_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_cells([(c, -r) for r, c in cells])


def _flip_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize_cells([(r, -c) for r, c in cells])


def _randomize_motif(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    out = cells
    for _ in range(random.randrange(4)):
        out = _rotate_cells(out)
    if random.random() < 0.5:
        out = _flip_cells(out)
    return out


def _choose_motif_for_degree(degree: int) -> tuple[tuple[int, int], ...]:
    return _randomize_motif(random.choice(BASE_MOTIFS_BY_DEGREE[degree]))


def _choose_motifs() -> list[tuple[tuple[int, int], ...]]:
    focus_degree = random.randint(0, 4)
    other_degrees = [d for d in range(5) if d != focus_degree]
    motif_degrees = [focus_degree] + random.sample(other_degrees, random.randint(1, 2))
    motifs = [_choose_motif_for_degree(degree) for degree in motif_degrees]

    if random.random() < 0.35:
        motifs.append(_randomize_motif(random.choice(ALL_BASE_MOTIFS)))

    motifs.sort(key=len, reverse=True)
    return motifs


def _place_motifs(motifs: list[tuple[tuple[int, int], ...]]) -> set[tuple[int, int]] | None:
    occupied: set[tuple[int, int]] = set()

    for motif in motifs:
        max_r = max(r for r, _c in motif)
        max_c = max(c for _r, c in motif)
        candidates: list[set[tuple[int, int]]] = []

        for r0 in range(LATTICE_SIZE - max_r):
            for c0 in range(LATTICE_SIZE - max_c):
                placed = {(r0 + r, c0 + c) for r, c in motif}
                if placed & occupied:
                    continue

                accidental_contact = False
                for r, c in placed:
                    for dr, dc in DIRS4:
                        nbr = (r + dr, c + dc)
                        if nbr in occupied and nbr not in placed:
                            accidental_contact = True
                            break
                    if accidental_contact:
                        break
                if accidental_contact:
                    continue

                merged = occupied | placed
                rows = [r for r, _c in merged]
                cols = [c for _r, c in merged]
                if max(rows) - min(rows) + 1 <= MAX_LATTICE_SPAN and max(cols) - min(cols) + 1 <= MAX_LATTICE_SPAN:
                    candidates.append(placed)

        if not candidates:
            return None
        occupied |= random.choice(candidates)

    return occupied


def _required_patch_cells(directions: set[str]) -> set[tuple[int, int]]:
    shape = {PATCH_CENTER}
    if "N" in directions:
        shape |= {(1, 2), (0, 2)}
    if "S" in directions:
        shape |= {(3, 2), (4, 2)}
    if "W" in directions:
        shape |= {(2, 1), (2, 0)}
    if "E" in directions:
        shape |= {(2, 3), (2, 4)}
    return shape


def _generate_patch(directions: set[str]) -> set[tuple[int, int]]:
    shape = _required_patch_cells(directions)
    target_area = min(TILE * TILE, len(shape) + random.randint(0, PATCH_EXTRA_CELLS))

    while len(shape) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in shape:
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < TILE and 0 <= cc < TILE and (rr, cc) not in shape:
                    frontier.append((rr, cc))
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            break

        weights = []
        for r, c in frontier:
            support = sum((r + dr, c + dc) in shape for dr, dc in DIRS4)
            touches_border = int(r in (0, TILE - 1) or c in (0, TILE - 1))
            weights.append((support + 1) ** 2 + touches_border)
        shape.add(random.choices(frontier, weights=weights, k=1)[0])

    return shape


def _node_directions(node: tuple[int, int], occupied: set[tuple[int, int]]) -> set[str]:
    r, c = node
    directions = set()
    for dr, dc in DIRS4:
        if (r + dr, c + dc) in occupied:
            directions.add(DIR_TO_NAME[(dr, dc)])
    return directions


def _render_lattice(occupied: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r, _c in occupied]
    cols = [c for _r, c in occupied]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)

    inner_h = (max_r - min_r + 1) * STEP - 1
    inner_w = (max_c - min_c + 1) * STEP - 1
    border_top = random.randint(*BORDER_RANGE)
    border_bottom = random.randint(*BORDER_RANGE)
    border_left = random.randint(*BORDER_RANGE)
    border_right = random.randint(*BORDER_RANGE)

    inp = np.full((border_top + inner_h + border_bottom, border_left + inner_w + border_right), AZURE, dtype=int)
    out = inp.copy()

    for node in sorted(occupied):
        directions = _node_directions(node, occupied)
        patch = _generate_patch(directions)
        degree = min(len(directions), 4)
        color = NEIGHBOR_COLOR[degree]

        r0 = border_top + (node[0] - min_r) * STEP
        c0 = border_left + (node[1] - min_c) * STEP
        rr = [r0 + r for r, _c in patch]
        cc = [c0 + c for _r, c in patch]
        inp[rr, cc] = GRAY
        out[rr, cc] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    motifs = _choose_motifs()
    occupied = _place_motifs(motifs)
    if occupied is None:
        return None

    inp, out = _render_lattice(occupied)
    counts = Counter(map(int, inp.ravel()))
    if counts[AZURE] <= counts[GRAY]:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
