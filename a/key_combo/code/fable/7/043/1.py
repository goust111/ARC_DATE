import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

GEN_RETRIES = 400


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_cells(shape: tuple[int, int], center: tuple[int, int], dist: int) -> list[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    if dist == 0:
        return [(cr, cc)]

    cells: list[tuple[int, int]] = []
    top = cr - dist
    bottom = cr + dist
    left = cc - dist
    right = cc + dist

    for c in range(left, right + 1):
        if 0 <= top < h and 0 <= c < w:
            cells.append((top, c))
        if 0 <= bottom < h and 0 <= c < w:
            cells.append((bottom, c))
    for r in range(top + 1, bottom):
        if 0 <= r < h and 0 <= left < w:
            cells.append((r, left))
        if 0 <= r < h and 0 <= right < w:
            cells.append((r, right))
    return cells


def _recover_generator(component: list[tuple[int, int]], grid: np.ndarray) -> tuple[tuple[int, int], list[int]]:
    candidates: list[tuple[tuple[int, int], list[int]]] = []
    comp_cells = {(int(r), int(c)) for r, c in component}

    for center in component:
        cr, cc = center
        by_dist: dict[int, int] = {}
        valid = True
        max_dist = 0
        for r, c in comp_cells:
            dist = max(abs(r - cr), abs(c - cc))
            max_dist = max(max_dist, dist)
            color = int(grid[r, c])
            prev = by_dist.get(dist)
            if prev is None:
                by_dist[dist] = color
            elif prev != color:
                valid = False
                break
        if not valid:
            continue
        if set(by_dist) != set(range(max_dist + 1)):
            continue
        sequence = [int(by_dist[d]) for d in range(max_dist + 1)]
        if len(set(sequence)) != len(sequence):
            continue
        if sequence[0] != int(grid[cr, cc]):
            continue
        candidates.append(((cr, cc), sequence))

    assert len(candidates) == 1, "expected a unique concentric generator center"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a cropped concentric square-ring palette outward from the phase indicated by an isolated seed pixel.

    Input structure:
    - A black background contains one multi-color generator object made of a center pixel and successive Chebyshev square rings, possibly cropped by the grid border.
    - A second non-black object is a single isolated seed pixel.
    - The seed color matches one non-center ring color of the generator palette.

    Rule:
    - Split the non-black cells into the generator component and the isolated seed.
    - Recover the generator's color sequence by finding the unique center whose equal-Chebyshev-distance layers are monochrome and form consecutive distances.
    - Let the seed color appear at palette index p; around the seed, paint ring d with palette color c[p+d] for every d that still fits inside the recovered suffix, but only on original background cells.

    Key insight:
    - The visible generator object is a cropped sample of a full concentric-ring palette, and the seed color does not start a new palette: it selects the phase from which the existing palette suffix must continue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    nonblack = g != BLACK
    comps = _components8(nonblack)
    assert len(comps) == 2, "expected exactly one generator component and one isolated seed"

    sizes = sorted((len(comp), i) for i, comp in enumerate(comps))
    assert sizes[0][0] == 1 and sizes[1][0] > 1, "expected one singleton seed and one larger generator"

    seed = comps[sizes[0][1]][0]
    generator = comps[sizes[1][1]]
    center, sequence = _recover_generator(generator, g)

    seed_color = int(g[seed])
    assert seed_color in sequence[1:], "seed color must match a non-center ring"
    phase = sequence.index(seed_color)

    out = g.copy()
    for dist in range(1, len(sequence) - phase):
        color = int(sequence[phase + dist])
        for r, c in _ring_cells(g.shape, seed, dist):
            if out[r, c] == BLACK:
                out[r, c] = color
    return out


def _draw_generator(grid: np.ndarray, center: tuple[int, int], sequence: list[int]) -> None:
    for dist, color in enumerate(sequence):
        for r, c in _ring_cells(grid.shape, center, dist):
            grid[r, c] = int(color)


def _seed_candidates(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK:
                continue
            isolated = True
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BLACK:
                    isolated = False
                    break
            if isolated:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _valid_generator_center(center: tuple[int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    r, c = center
    return 0 <= r < h and 0 <= c < w


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 24)
    w = random.randint(10, 24)
    grid = np.zeros((h, w), dtype=int)

    center = (random.randrange(h), random.randrange(w))
    if not _valid_generator_center(center, grid.shape):
        return None

    max_ring = max(center[0], h - 1 - center[0], center[1], w - 1 - center[1])
    if max_ring < 2:
        return None

    ring_count = random.randint(2, min(6, max_ring, 8))
    sequence = random.sample(PALETTE, ring_count + 1)
    _draw_generator(grid, center, sequence)

    phase = random.randint(1, ring_count - 1)
    for seed in _seed_candidates(grid):
        candidate = grid.copy()
        candidate[seed] = int(sequence[phase])
        try:
            out = main(candidate)
        except AssertionError:
            continue
        if np.array_equal(out, candidate):
            continue
        return candidate
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid phase-ring-extension input")
