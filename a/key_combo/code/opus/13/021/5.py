from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np


BACKGROUND = 0
SHAPE_COLOR = 1
FILL_COLOR = 3
AXIS_COLOR = 5

GRID_HALVES = (5, 6)
MAX_GENERATION_ATTEMPTS = 200
MAX_SOURCE_CELLS = 8

BASE_PATTERNS: tuple[frozenset[tuple[int, int]], ...] = (
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}),
    frozenset({(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}),
    frozenset({(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (2, 1), (2, 2), (3, 1)}),
    frozenset({(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (2, 1), (2, 2), (3, 2)}),
    frozenset({(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (2, 2), (2, 3), (3, 2)}),
    frozenset({(0, 1), (0, 2), (0, 3), (1, 1), (1, 3), (2, 2), (2, 3), (3, 3)}),
)

CARDINAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in CARDINAL_STEPS:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coords)


def _mirror_coords(source: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    q = source.shape[0]
    k = q
    rr, cc = np.nonzero(source)
    return [
        (rr, cc),
        (rr, 2 * k - cc),
        (2 * k - rr, cc),
        (2 * k - rr, 2 * k - cc),
    ]


def _render_mirrored(source: np.ndarray, shape_color: int) -> np.ndarray:
    q = source.shape[0]
    n = 2 * q + 1
    out = np.full((n, n), BACKGROUND, dtype=int)
    for rr, cc in _mirror_coords(source):
        out[rr, cc] = int(shape_color)
    return out


def _fill_enclosed_pockets(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    zeros = out == BACKGROUND
    for comp in _components(zeros):
        rows = comp[:, 0]
        cols = comp[:, 1]
        touches_border = np.any(rows == 0) or np.any(rows == out.shape[0] - 1) or np.any(cols == 0) or np.any(cols == out.shape[1] - 1)
        if not touches_border:
            out[rows, cols] = FILL_COLOR
    return out


def _source_shape_from_grid(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd square grid"

    row_candidates = [r for r in range(h) if np.all(g[r, :] == AXIS_COLOR)]
    col_candidates = [c for c in range(w) if np.all(g[:, c] == AXIS_COLOR)]
    assert len(row_candidates) == 1 and len(col_candidates) == 1, "expected exactly one full central cross"
    k = row_candidates[0]
    assert col_candidates[0] == k and k == h // 2, "cross must lie on the central row and column"

    payload_mask = (g != BACKGROUND) & (g != AXIS_COLOR)
    allowed = np.zeros_like(payload_mask, dtype=bool)
    allowed[:k, :k] = True
    assert np.array_equal(payload_mask, payload_mask & allowed), "the lone source shape must lie entirely in the upper-left quadrant"

    payload_colors = {int(v) for v in np.unique(g[payload_mask])}
    assert len(payload_colors) == 1, "expected a single source color"
    shape_color = payload_colors.pop()
    assert FILL_COLOR not in set(map(int, np.unique(g))), "fill color must be absent from the input"

    source = (g[:k, :k] == shape_color)
    cell_count = int(source.sum())
    assert 4 <= cell_count <= 8, "expected a small 4-connected source shape"
    assert len(_components(source)) == 1, "source shape must be 4-connected"

    edge_row = int(source[k - 1, :].sum())
    edge_col = int(source[:, k - 1].sum())
    assert edge_row > 0 or edge_col > 0, "source shape must touch the mirror boundary"
    assert edge_row < k and edge_col < k, "source shape must not completely fill the boundary row or column"
    return source, shape_color, k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the upper-left shape across the central cross and flood-fill the enclosed pockets that appear only after mirroring.

    Input structure:
    - The grid is an odd square on black background with one full grey row and one full grey column crossing at the center.
    - A single 4-connected source shape of one non-grey color appears only in the upper-left quadrant.
    - The source touches the quadrant edge next to the cross, while the other three quadrants are empty.

    Rule:
    - Use the central row and column as reflection axes for the upper-left source shape.
    - Copy the shape into the other three quadrants by horizontal, vertical, and 180-degree mirroring.
    - Erase the grey cross to black, then find every black 4-connected region that does not touch the outer border.
    - Flood-fill all such enclosed background pockets with green, keeping the mirrored shape unchanged.

    Key insight:
    - The enclosed regions are not present in the input at all; they emerge only after mentally completing the four mirrored copies and then checking which background areas become sealed off.
    """
    source, shape_color, _k = _source_shape_from_grid(grid)
    mirrored = _render_mirrored(source, shape_color)
    return _fill_enclosed_pockets(mirrored)


def _transpose_shape(shape: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(c, r) for r, c in shape}


def _anchor_to_corner(shape: set[tuple[int, int]], q: int) -> set[tuple[int, int]]:
    max_r = max(r for r, _c in shape)
    max_c = max(c for _r, c in shape)
    dr = q - 1 - max_r
    dc = q - 1 - max_c
    return {(r + dr, c + dc) for r, c in shape}


def _frontier(shape: set[tuple[int, int]], q: int) -> list[tuple[int, int]]:
    cells = set()
    for r, c in shape:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < q and 0 <= nc < q and (nr, nc) not in shape and nr < q - 1 and nc < q - 1:
                cells.add((nr, nc))
    return sorted(cells)


def _shape_to_input(shape: set[tuple[int, int]], q: int) -> np.ndarray:
    n = 2 * q + 1
    grid = np.full((n, n), BACKGROUND, dtype=int)
    grid[q, :] = AXIS_COLOR
    grid[:, q] = AXIS_COLOR
    for r, c in shape:
        grid[r, c] = SHAPE_COLOR
    return grid


def _is_valid_source(shape: set[tuple[int, int]], q: int) -> bool:
    if not (4 <= len(shape) <= MAX_SOURCE_CELLS):
        return False
    if not _is_connected(shape):
        return False
    if not any(r == q - 1 or c == q - 1 for r, c in shape):
        return False
    if sum(r == q - 1 for r, _c in shape) >= q:
        return False
    if sum(c == q - 1 for _r, c in shape) >= q:
        return False
    out = main(_shape_to_input(shape, q))
    return bool(np.any(out == FILL_COLOR))


@lru_cache(maxsize=None)
def _candidate_shapes(q: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for pattern in BASE_PATTERNS:
        for orient in (set(pattern), _transpose_shape(pattern)):
            anchored = _anchor_to_corner(orient, q)
            if _is_valid_source(anchored, q):
                variants.add(tuple(sorted(anchored)))
            if len(anchored) < MAX_SOURCE_CELLS:
                for tail in _frontier(anchored, q):
                    trial = set(anchored)
                    trial.add(tail)
                    if _is_valid_source(trial, q):
                        variants.add(tuple(sorted(trial)))
    assert variants, f"no valid source shapes for q={q}"
    return tuple(sorted(variants))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        q = int(rng.choice(GRID_HALVES))
        shape = set(_candidate_shapes(q)[int(rng.integers(len(_candidate_shapes(q))))])
        grid = _shape_to_input(shape, q)
        out = main(grid)
        if np.any(out == FILL_COLOR):
            return grid
    raise RuntimeError("generate_input: failed to sample a valid mirrored-pocket instance")
