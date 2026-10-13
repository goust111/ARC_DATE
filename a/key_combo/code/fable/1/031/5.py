from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
TAIL_MARKER = 5
BODY_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_K_RANGE = (3, 7)
GEN_GRID_SIDE_RANGE = (18, 30)
GEN_RECT_SIDE_RANGE = (3, 7)
GEN_PADDING_CHOICES = (1, 2)
GEN_BORDER = 1
GEN_MAX_FOREGROUND_RATIO = 0.42
GEN_MONOTONE_AREA_PROB = 0.25
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _render_nested_rings(colors: list[int]) -> np.ndarray:
    size = 2 * len(colors) - 1
    out = np.full((size, size), BG, dtype=int)
    for depth, color in enumerate(colors):
        out[depth : size - depth, depth : size - depth] = int(color)
    return out


def _parse_blocks(grid: np.ndarray) -> list[tuple[int, int]]:
    bg = _get_bg(grid)
    blocks: list[tuple[int, int]] = []

    for coords in _components(grid != bg):
        values = [int(grid[r, c]) for r, c in coords]
        counts = Counter(values)
        assert len(counts) == 2, "each block must contain one body color plus one marker color"

        singles = [color for color, count in counts.items() if count == 1]
        bodies = [color for color, count in counts.items() if count > 1]
        assert len(singles) == 1 and len(bodies) == 1, "expected exactly one singleton marker pixel"

        body = int(bodies[0])
        marker = int(singles[0])
        assert body != TAIL_MARKER, "grey is reserved for the terminal marker"
        blocks.append((body, marker))

    assert blocks, "expected at least one block"
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the pointer-defined block chain into nested square rings from outermost to center.

    Input structure:
    - A black background contains several separated connected blocks.
    - Each block is mostly one body color and contains exactly one singleton marker
      pixel of a different color.
    - Every non-terminal marker color matches another block's body color; the tail
      marker is grey.

    Rule:
    - Treat each block's dominant color as its body color and its singleton color as
      the pointer target.
    - The chain head is the unique body color that is never referenced by any
      non-grey marker; follow pointers until the grey terminal marker.
    - Output a (2k-1)x(2k-1) grid for the k-block chain, filling the outer ring with
      the head body color, then each inner ring with the next body color, and the
      center with the tail body color.

    Key insight:
    - Geometric size and placement do not determine the nesting order; only the
      internal marker-to-body color links define the color chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    blocks = _parse_blocks(g)
    body_colors = [body for body, _marker in blocks]
    assert len(body_colors) == len(set(body_colors)), "body colors must be unique"

    next_by_body = {body: marker for body, marker in blocks}
    referenced = {marker for _body, marker in blocks if marker != TAIL_MARKER}
    assert referenced <= set(body_colors), "every non-terminal marker must reference a body color"

    heads = [body for body in body_colors if body not in referenced]
    assert len(heads) == 1, "expected one chain head"

    chain: list[int] = []
    seen: set[int] = set()
    current = int(heads[0])
    while True:
        assert current not in seen, "pointer chain must be acyclic"
        seen.add(current)
        chain.append(current)

        nxt = int(next_by_body[current])
        if nxt == TAIL_MARKER:
            break
        assert nxt in next_by_body, "pointer must reference an existing body color"
        current = nxt

    assert len(chain) == len(blocks), "pointer chain must visit every block exactly once"
    return _render_nested_rings(chain)


def _areas_monotone(areas: list[int]) -> bool:
    return all(a < b for a, b in zip(areas, areas[1:])) or all(a > b for a, b in zip(areas, areas[1:]))


def _dimension_pool() -> dict[int, list[tuple[int, int]]]:
    pool: dict[int, list[tuple[int, int]]] = {}
    side_min, side_max = GEN_RECT_SIDE_RANGE
    for h in range(side_min, side_max + 1):
        for w in range(side_min, side_max + 1):
            pool.setdefault(h * w, []).append((h, w))
    return pool


GEN_AREA_TO_DIMS = _dimension_pool()


def _choose_dims(k: int) -> list[tuple[int, int]]:
    monotone = random.random() < GEN_MONOTONE_AREA_PROB
    unique_areas = list(GEN_AREA_TO_DIMS)

    if monotone:
        areas = sorted(random.sample(unique_areas, k), reverse=random.choice([False, True]))
        return [random.choice(GEN_AREA_TO_DIMS[area]) for area in areas]

    for _ in range(100):
        dims = [random.choice(GEN_AREA_TO_DIMS[random.choice(unique_areas)]) for _ in range(k)]
        if not _areas_monotone([h * w for h, w in dims]):
            return dims

    areas = sorted(random.sample(unique_areas, k))
    random.shuffle(areas)
    return [random.choice(GEN_AREA_TO_DIMS[area]) for area in areas]


def _marker_cell(h: int, w: int) -> tuple[int, int]:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    return random.choice(interior)


def _make_patch(h: int, w: int, body: int, marker: int) -> np.ndarray:
    patch = np.full((h, w), int(body), dtype=int)
    mr, mc = _marker_cell(h, w)
    patch[mr, mc] = int(marker)
    return patch


def _candidate_positions(occupied: np.ndarray, h: int, w: int, padding: int, border: int) -> list[tuple[int, int]]:
    rows, cols = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(border, rows - h - border + 1):
        for c0 in range(border, cols - w - border + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(rows, r0 + h + padding)
            cc1 = min(cols, c0 + w + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_patches(rows: int, cols: int, patches: list[np.ndarray], padding: int) -> np.ndarray | None:
    grid = np.full((rows, cols), BG, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    order = sorted(range(len(patches)), key=lambda idx: patches[idx].size, reverse=True)
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        positions = _candidate_positions(occupied, h, w, padding, GEN_BORDER)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        occupied[r0 : r0 + h, c0 : c0 + w] = True

    return grid


def _expected_from_chain(chain: list[int]) -> np.ndarray:
    return _render_nested_rings(chain)


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(*GEN_K_RANGE)
    chain = random.sample(BODY_PALETTE, k)
    dims = _choose_dims(k)

    patches = []
    total_area = 0
    for i, (body, (h, w)) in enumerate(zip(chain, dims)):
        marker = chain[i + 1] if i + 1 < k else TAIL_MARKER
        patch = _make_patch(h, w, body, marker)
        patches.append(patch)
        total_area += h * w

    rows = random.randint(*GEN_GRID_SIDE_RANGE)
    cols = random.randint(*GEN_GRID_SIDE_RANGE)
    if total_area >= rows * cols * GEN_MAX_FOREGROUND_RATIO:
        return None

    grid = _place_patches(rows, cols, patches, padding=random.choice(GEN_PADDING_CHOICES))
    if grid is None:
        return None

    expected = _expected_from_chain(chain)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested pointer-chain input")
