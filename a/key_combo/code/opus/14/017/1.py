from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

MASK_SIZE = 11
INPUT_SHAPE = (MASK_SIZE, MASK_SIZE * 2 + 1)
SEPARATOR_COL = MASK_SIZE
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            queue = deque([(r, c)])
            labels[r, c] = comp_id
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        queue.append((nr, nc))
    return labels, comp_id


def _find_separator(grid: np.ndarray) -> int:
    separator_cols = np.flatnonzero(np.all(grid == RED, axis=0))
    assert len(separator_cols) == 1, "expected one full-height red separator column"
    return int(separator_cols[0])


def _extract_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape and left.shape == (MASK_SIZE, MASK_SIZE), "expected two 11x11 masks"
    assert set(np.unique(left)).issubset({BLACK, ORANGE})
    assert set(np.unique(right)).issubset({BLACK, ORANGE})
    return left == ORANGE, right == ORANGE


def _region_graph(wall: np.ndarray, labels: np.ndarray, n_regions: int) -> dict[int, set[int]]:
    h, w = wall.shape
    graph = {region_id: set() for region_id in range(1, n_regions + 1)}
    for r, c in np.argwhere(wall):
        if 0 < r < h - 1:
            a = int(labels[r - 1, c])
            b = int(labels[r + 1, c])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)
        if 0 < c < w - 1:
            a = int(labels[r, c - 1])
            b = int(labels[r, c + 1])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)
    return graph


def _color_from_wall(wall: np.ndarray, *, min_regions: int = 3) -> np.ndarray:
    wall = np.asarray(wall, dtype=bool)
    assert wall.shape == (MASK_SIZE, MASK_SIZE)
    assert not wall[MASK_SIZE - 1, 0], "lower-left output anchor must be background"

    labels, n_regions = _label_components(~wall)
    assert n_regions >= min_regions, "wall must create enough background regions"
    start = int(labels[MASK_SIZE - 1, 0])
    assert start > 0

    graph = _region_graph(wall, labels, n_regions)
    assert all(graph.values()), "every background region must touch another region across a wall"

    parity = {start: 0}
    queue = deque([start])
    while queue:
        region = queue.popleft()
        for neighbor in sorted(graph[region]):
            expected = 1 - parity[region]
            if neighbor in parity:
                assert parity[neighbor] == expected, "region adjacency graph must be bipartite"
            else:
                parity[neighbor] = expected
                queue.append(neighbor)

    assert len(parity) == n_regions, "all background regions must be reachable from the lower-left region"

    out = np.full((MASK_SIZE, MASK_SIZE), GREY, dtype=int)
    out[~wall] = BLUE
    for region_id, region_parity in parity.items():
        out[labels == region_id] = GREEN if region_parity == 0 else BLUE
    out[wall] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    OR-merge the two separated masks into walls, then alternately color the wall-separated regions.

    Input structure:
    - The input is an 11x23 grid split by one full-height red separator column.
    - The left and right 11x11 halves contain only black background and orange one-cell-wide mask fragments.
    - Each half is interpreted in the same coordinate system after removing the separator.

    Rule:
    - Locate the red separator and align the two orange masks cell-by-cell.
    - Build a wall mask by taking the pixelwise OR of the orange pixels from both halves.
    - Output an 11x11 grid where wall cells become grey.
    - Flood-fill the remaining 4-connected background regions, start from the region containing the lower-left cell, color it green, and flip between green and blue across every wall adjacency.

    Key insight:
    - The orange pixels are not copied directly; their OR-union creates the wall topology, and that topology alone determines the alternating region coloring.
    """
    left_mask, right_mask = _extract_masks(grid)
    wall = left_mask | right_mask
    return _color_from_wall(wall, min_regions=3)


def _valid_positions(count: int) -> list[tuple[int, ...]]:
    coords = []
    for combo in itertools.combinations(range(1, MASK_SIZE - 1), count):
        if all(b - a >= 2 for a, b in zip(combo, combo[1:])):
            coords.append(combo)
    return coords


POSITION_CHOICES = {2: _valid_positions(2), 3: _valid_positions(3)}


def _build_comb_wall(attach: str, prongs: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    wall = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
    shared = np.zeros_like(wall)

    if attach == "top":
        wall[0, prongs[0] : prongs[-1] + 1] = True
        shared[0, prongs[0] : prongs[-1] + 1] = True
        for c in prongs:
            wall[:, c] = True
    elif attach == "bottom":
        wall[-1, prongs[0] : prongs[-1] + 1] = True
        shared[-1, prongs[0] : prongs[-1] + 1] = True
        for c in prongs:
            wall[:, c] = True
    elif attach == "left":
        wall[prongs[0] : prongs[-1] + 1, 0] = True
        shared[prongs[0] : prongs[-1] + 1, 0] = True
        for r in prongs:
            wall[r, :] = True
    else:
        assert attach == "right"
        wall[prongs[0] : prongs[-1] + 1, -1] = True
        shared[prongs[0] : prongs[-1] + 1, -1] = True
        for r in prongs:
            wall[r, :] = True

    return wall, shared


def _split_wall_masks(attach: str, prongs: tuple[int, ...], shared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_mask = shared.copy()
    right_mask = shared.copy()
    start_left = 0 if attach in {"top", "left"} else 1

    if attach in {"top", "bottom"}:
        for index, c in enumerate(prongs):
            owner_left = (index + start_left) % 2 == 0
            if owner_left:
                left_mask[:, c] = True
            else:
                right_mask[:, c] = True
    else:
        for index, r in enumerate(prongs):
            owner_left = (index + start_left) % 2 == 0
            if owner_left:
                left_mask[r, :] = True
            else:
                right_mask[r, :] = True

    return left_mask, right_mask


def _assemble_input(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    grid = np.zeros(INPUT_SHAPE, dtype=int)
    grid[:, SEPARATOR_COL] = RED
    grid[:, :MASK_SIZE][left_mask] = ORANGE
    grid[:, SEPARATOR_COL + 1 :][right_mask] = ORANGE
    return grid


def _mask_connected(mask: np.ndarray) -> bool:
    _, n = _label_components(mask)
    return n == 1


def _validate_generated(grid: np.ndarray) -> bool:
    left_mask, right_mask = _extract_masks(grid)
    if not _mask_connected(left_mask) or not _mask_connected(right_mask):
        return False

    wall = left_mask | right_mask
    if not _mask_connected(wall):
        return False
    if wall[MASK_SIZE - 1, 0]:
        return False

    left_only = left_mask & ~right_mask
    right_only = right_mask & ~left_mask
    if not left_only.any() or not right_only.any():
        return False

    try:
        full_out = _color_from_wall(wall, min_regions=3)
        left_out = _color_from_wall(left_mask, min_regions=2)
        right_out = _color_from_wall(right_mask, min_regions=2)
    except AssertionError:
        return False

    if np.array_equal(full_out, left_out) or np.array_equal(full_out, right_out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    attach = random.choice(["top", "bottom", "left", "right"])
    prong_count = random.choice([2, 3])
    prongs = random.choice(POSITION_CHOICES[prong_count])

    wall, shared = _build_comb_wall(attach, prongs)
    left_mask, right_mask = _split_wall_masks(attach, prongs, shared)
    if not np.array_equal(wall, left_mask | right_mask):
        return None

    grid = _assemble_input(left_mask, right_mask)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the OR-wall alternating-region task")
