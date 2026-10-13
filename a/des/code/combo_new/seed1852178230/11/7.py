from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _component_cells(grid: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    sr, sc = start
    color = int(grid[sr, sc])
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    cells: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c, h, w):
            if seen[nr, nc] or int(grid[nr, nc]) != color:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))
    return cells


def _decode_pointer(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = map(int, grid.shape)
    assert int(grid[r, c]) == BLACK, "pointer cell must be black"

    neighbor_values = [int(grid[nr, nc]) for nr, nc in _neighbors4(r, c, h, w)]
    assert all(v != BLACK for v in neighbor_values), "pointer neighbors must remain colored"

    counts = Counter(neighbor_values)
    owner_color, owner_count = counts.most_common(1)[0]
    cross_neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if int(grid[nr, nc]) != owner_color]

    assert owner_count == 3, "pointer must keep three neighbors in its own region"
    assert len(cross_neighbors) == 1, "pointer must have exactly one cross-region neighbor"

    tr, tc = cross_neighbors[0]
    return owner_color, int(grid[tr, tc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each colored region with the color of the neighboring region touched by its black pointer.

    Input structure:
    - The whole grid is partitioned into 4-connected monochrome regions whose colors are all non-black.
    - Each region contains exactly one black pointer pixel in its interior, replacing one cell of that region.
    - Every pointer keeps three 4-neighbors in its own region and has exactly one 4-neighbor in a different region.

    Rule:
    - For each black pointer, use its unique cross-region neighbor to read the target region's input color.
    - Recover the pointer's own region as the monochrome 4-connected component of its surrounding color.
    - Repaint that entire region, including the black pointer cell, with the target color.
    - All regions update simultaneously, so every borrowed color is taken from the input grid.

    Key insight:
    - The black pixel is not noise: its lone cross-boundary touch identifies which adjacent region donates the new color, while its three same-color neighbors identify which region is being recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = map(int, g.shape)
    pointer_info: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in np.argwhere(g == BLACK):
        pointer_info[(int(r), int(c))] = _decode_pointer(g, int(r), int(c))
    assert pointer_info, "expected at least one pointer"

    out = g.copy()
    seen = np.zeros((h, w), dtype=bool)
    claimed_pointers: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BLACK:
                continue

            cells = _component_cells(g, (r, c), seen)
            color = int(g[r, c])

            adjacent_pointers = {
                (nr, nc)
                for cr, cc in cells
                for nr, nc in _neighbors4(cr, cc, h, w)
                if int(g[nr, nc]) == BLACK and pointer_info[(nr, nc)][0] == color
            }
            assert len(adjacent_pointers) == 1, "every region must own exactly one pointer"

            pointer = next(iter(adjacent_pointers))
            target_color = pointer_info[pointer][1]
            claimed_pointers.add(pointer)

            for cr, cc in cells:
                out[cr, cc] = target_color
            out[pointer] = target_color

    assert claimed_pointers == set(pointer_info), "all pointers must be matched to exactly one region"
    return out


def _grow_partition(mh: int, mw: int, n_regions: int, rng: random.Random) -> np.ndarray | None:
    owner = np.full((mh, mw), -1, dtype=int)
    seeds = rng.sample([(r, c) for r in range(mh) for c in range(mw)], n_regions)
    sizes = [0] * n_regions
    frontier = [set() for _ in range(n_regions)]

    for rid, (r, c) in enumerate(seeds):
        owner[r, c] = rid
        sizes[rid] = 1
        frontier[rid].add((r, c))

    remaining = mh * mw - n_regions
    while remaining > 0:
        choices: list[tuple[int, list[tuple[int, int]]]] = []
        weights: list[float] = []
        for rid in range(n_regions):
            boundary = set()
            for r, c in frontier[rid]:
                for nr, nc in _neighbors4(r, c, mh, mw):
                    if int(owner[nr, nc]) == -1:
                        boundary.add((nr, nc))
            if boundary:
                choices.append((rid, list(boundary)))
                weights.append(1.0 / sizes[rid])

        if not choices:
            return None

        choice_idx = rng.choices(range(len(choices)), weights=weights, k=1)[0]
        rid, boundary = choices[choice_idx]
        nr, nc = rng.choice(boundary)
        owner[nr, nc] = rid
        sizes[rid] += 1
        frontier[rid].add((nr, nc))
        remaining -= 1

    return owner


def _expand_partition(owner: np.ndarray, colors: list[int], cell_h: int, cell_w: int) -> np.ndarray:
    mh, mw = map(int, owner.shape)
    grid = np.empty((mh * cell_h, mw * cell_w), dtype=int)
    for mr in range(mh):
        for mc in range(mw):
            color = colors[int(owner[mr, mc])]
            r0 = mr * cell_h
            c0 = mc * cell_w
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = color
    return grid


def _pointer_candidates(grid: np.ndarray, color: int) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != color:
                continue

            neighbor_values = [int(grid[nr, nc]) for nr, nc in _neighbors4(r, c, h, w)]
            same_count = sum(v == color for v in neighbor_values)
            cross = [(nr, nc, int(grid[nr, nc])) for nr, nc in _neighbors4(r, c, h, w) if int(grid[nr, nc]) != color]
            if same_count == 3 and len(cross) == 1:
                nr, nc, target = cross[0]
                candidates.append(((r, c), (nr, nc), target))
    return candidates


def _insert_pointers(grid: np.ndarray, colors: list[int], rng: random.Random) -> np.ndarray | None:
    pointer_positions: set[tuple[int, int]] = set()
    forbidden_pointer_cells: set[tuple[int, int]] = set()
    chosen: dict[int, tuple[tuple[int, int], tuple[int, int], int]] = {}

    region_order = colors[:]
    rng.shuffle(region_order)
    region_order.sort(key=lambda color: len(_pointer_candidates(grid, color)))

    for color in region_order:
        options = [
            candidate
            for candidate in _pointer_candidates(grid, color)
            if candidate[0] not in forbidden_pointer_cells and candidate[1] not in pointer_positions
        ]
        if not options:
            return None

        pos, target, target_color = rng.choice(options)
        chosen[color] = (pos, target, target_color)
        pointer_positions.add(pos)
        forbidden_pointer_cells.add(pos)
        forbidden_pointer_cells.add(target)

    out = grid.copy()
    for pos, _target, _target_color in chosen.values():
        out[pos] = BLACK
    return out


def _validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert 1 <= h <= 30 and 1 <= w <= 30

    pointer_info: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in np.argwhere(g == BLACK):
        pointer_info[(int(r), int(c))] = _decode_pointer(g, int(r), int(c))
    assert len(pointer_info) >= 3

    seen = np.zeros((h, w), dtype=bool)
    matched: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BLACK:
                continue

            component = _component_cells(g, (r, c), seen)
            color = int(g[r, c])
            nearby = {
                (nr, nc)
                for cr, cc in component
                for nr, nc in _neighbors4(cr, cc, h, w)
                if int(g[nr, nc]) == BLACK and pointer_info[(nr, nc)][0] == color
            }
            assert len(nearby) == 1
            matched |= nearby

    assert matched == set(pointer_info)
    out = main(g)
    assert out.shape == g.shape
    assert not np.any(out == BLACK)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.randint(3, 5)
    cell_w = rng.randint(3, 5)
    mh = rng.randint(3, min(6, 30 // cell_h))
    mw = rng.randint(3, min(6, 30 // cell_w))

    max_regions = min(8, (mh * mw) // 2)
    if max_regions < 3:
        return None
    n_regions = rng.randint(3, max_regions)

    owner = _grow_partition(mh, mw, n_regions, rng)
    if owner is None:
        return None

    colors = rng.sample(PALETTE, n_regions)
    grid = _expand_partition(owner, colors, cell_h, cell_w)
    with_pointers = _insert_pointers(grid, colors, rng)
    if with_pointers is None:
        return None

    _validate_generated_grid(with_pointers)
    return with_pointers


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-region puzzle")
