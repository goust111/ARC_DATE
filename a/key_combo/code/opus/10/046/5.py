from __future__ import annotations

import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
YELLOW = 4
BLACK = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _diagonal_tile_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((2 * h, 2 * w), dtype=bool)
    for r, c in np.argwhere(grid == BLUE):
        rr = int(r)
        cc = int(c)
        while rr < 2 * h and cc < 2 * w:
            out[rr, cc] = True
            rr += 1
            cc += 1
    return out


def _enclosed_background_regions(blue_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = blue_mask.shape
    bg = ~blue_mask
    seen = np.zeros((h, w), dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            region: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                region.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                regions.append(region)
    return regions


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    blue_mask = _diagonal_tile_mask(g)
    out = np.zeros(blue_mask.shape, dtype=int)
    out[blue_mask] = BLUE
    for region in _enclosed_background_regions(blue_mask):
        fill = RED if len(region) % 2 == 0 else YELLOW
        for r, c in region:
            out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the blue line motif along the down-right diagonal and parity-fill the enclosed holes it creates.

    Input structure:
    - The input is a black grid containing only blue foreground pixels.
    - The blue pixels form one or more open 4-connected line fragments rather than a closed frame.
    - Any enclosed regions are not present in the input itself; they appear only after diagonal tiling.

    Rule:
    - Create a doubled black canvas.
    - For every blue input pixel at `(r, c)`, paint blue at `(r+k, c+k)` for all nonnegative `k` that stay in bounds.
    - Find every 4-connected black region in the doubled canvas that does not touch the canvas border.
    - Fill each enclosed region red when its area is even, or yellow when its area is odd.

    Key insight:
    - The input lines do not need to enclose anything on their own; the closed holes are emergent artifacts of repeating the motif along the diagonal, so tiling must happen before flood-fill and parity analysis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 0 < g.shape[0] <= 15 and 0 < g.shape[1] <= 15, "input must stay within ARC limits after doubling"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE}), "expected only black and blue in the input"
    assert np.any(g == BLUE), "expected at least one blue pixel"
    return _solve(g)


def _degree_map(grid: np.ndarray) -> dict[tuple[int, int], int]:
    deg: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(grid == BLUE):
        rr = int(r)
        cc = int(c)
        deg[(rr, cc)] = sum(
            1
            for nr, nc in _neighbors4(rr, cc)
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == BLUE
        )
    return deg


def _is_open_path_component(comp: list[tuple[int, int]]) -> bool:
    cells = set(comp)
    degrees = []
    edge_count = 0
    for r, c in comp:
        d = 0
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                d += 1
        degrees.append(d)
        edge_count += d
    edge_count //= 2
    endpoints = sum(d == 1 for d in degrees)
    return edge_count == len(comp) - 1 and max(degrees, default=0) <= 2 and endpoints == 2


def _random_path(grid: np.ndarray, length: int, rng: random.Random) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    starts = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BLACK]
    rng.shuffle(starts)
    for start in starts:
        r0, c0 = start
        grid[r0, c0] = BLUE
        path = [start]
        while len(path) < length:
            r, c = path[-1]
            candidates = []
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] != BLACK:
                    continue
                # Keep each fragment as a simple path and prevent contact with earlier blue cells.
                touching = [
                    (ar, ac)
                    for ar, ac in _neighbors4(nr, nc)
                    if 0 <= ar < h and 0 <= ac < w and grid[ar, ac] == BLUE
                ]
                if touching != [(r, c)] and set(touching) != {(r, c)}:
                    continue
                candidates.append((nr, nc))
            if not candidates:
                for rr, cc in path:
                    grid[rr, cc] = BLACK
                break
            step = rng.choice(candidates)
            grid[step] = BLUE
            path.append(step)
        else:
            return path
    return None


def _random_transform(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    turns = rng.randrange(4)
    out = np.rot90(grid, turns)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=int)


def _semantic_ok(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 15:
        return False
    if set(map(int, np.unique(grid))) - {BLACK, BLUE}:
        return False

    blue_mask = grid == BLUE
    components = _components(blue_mask)
    if len(components) != 2:
        return False
    sizes = sorted(len(comp) for comp in components)
    if not (5 <= sum(sizes) <= 10) or min(sizes) < 2:
        return False
    if not all(_is_open_path_component(comp) for comp in components):
        return False
    if _enclosed_background_regions(blue_mask):
        return False

    out = _solve(grid)
    if out.shape != (2 * grid.shape[0], 2 * grid.shape[1]):
        return False

    blue_out = out == BLUE
    holes = _enclosed_background_regions(blue_out)
    if len(holes) < 2:
        return False
    for region in holes:
        want = RED if len(region) % 2 == 0 else YELLOW
        if any(out[r, c] != want for r, c in region):
            return False
    if np.any((out != BLACK) & (out != BLUE) & (out != RED) & (out != YELLOW)):
        return False
    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    side = rng.randint(7, 10)
    grid = np.zeros((side, side), dtype=int)

    total = rng.randint(5, 10)
    len1 = rng.randint(2, total - 2)
    len2 = total - len1
    if len2 < 2:
        return None

    path_lengths = [len1, len2]
    rng.shuffle(path_lengths)
    for length in path_lengths:
        if _random_path(grid, length, rng) is None:
            return None

    grid = _random_transform(grid, rng)
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(800):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
