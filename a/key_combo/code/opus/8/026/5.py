from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, in_bounds, neighbors4, paste


TILE_SIZE = 4
TILE_REPETITIONS = 3
MAX_GENERATION_ATTEMPTS = 400
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _rotate_cw(tile: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(tile, -int(quarter_turns) % 4).copy()


def _rotation_rule(i: int, j: int) -> int:
    return (int(i) + int(j)) % 4


def _render_with_rule(tile: np.ndarray, rotation_fn) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    h, w = tile.shape
    out = np.full((h * TILE_REPETITIONS, w * TILE_REPETITIONS), Color.BLACK, dtype=int)
    red_positions: list[tuple[int, int]] = []

    for i in range(TILE_REPETITIONS):
        for j in range(TILE_REPETITIONS):
            rotated = _rotate_cw(tile, rotation_fn(i, j))
            r0 = i * h
            c0 = j * w
            paste(out, rotated, r0, c0)

            red_coords = np.argwhere(rotated == Color.RED)
            assert red_coords.shape[0] == 1, "each rotated tile must contain exactly one red marker"
            rr, cc = map(int, red_coords[0])
            red_positions.append((r0 + rr, c0 + cc))

    for r, c in red_positions:
        for dr, dc in DIAGONALS:
            nr = r + dr
            nc = c + dc
            if in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == Color.BLACK:
                out[nr, nc] = Color.YELLOW

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the input sprite into a 3x3 canvas, rotate each copy by its tile position, and emit yellow diagonal marks from the rotated red marker.

    Input structure:
    - The input is a square tile on a black background, typically 4x4.
    - Blue pixels form one 4-connected foreground shape.
    - Exactly one red marker pixel touches that blue shape by a cardinal edge.

    Rule:
    - Build a 3x3 tiling of the input tile.
    - Tile copy (i, j) uses a clockwise rotation of `(i + j) mod 4` quarter-turns before placement.
    - After all rotated tiles are placed, every red marker paints its four diagonal neighbors yellow.
    - Yellow only replaces black background cells; blue and red pixels stay unchanged, and out-of-bounds diagonal marks are discarded.

    Key insight:
    - The copy position controls rotation, so the single red marker lands at different local coordinates across the tiled canvas and therefore emits a non-uniform yellow corner pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square tile"

    red_coords = np.argwhere(g == Color.RED)
    assert red_coords.shape[0] == 1, "expected exactly one red marker"
    assert np.count_nonzero(g == Color.BLUE) > 0, "expected blue foreground pixels"
    assert len(components_mask(g == Color.BLUE, connectivity=4)) == 1, "blue shape must be 4-connected"

    red_r, red_c = map(int, red_coords[0])
    assert any(
        in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == Color.BLUE
        for nr, nc in neighbors4(red_r, red_c)
    ), "red marker must touch the blue shape"

    return _render_with_rule(g, _rotation_rule)


def _connected_blue_mask(cell_count: int) -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    seed = (random.randrange(TILE_SIZE), random.randrange(TILE_SIZE))
    mask[seed] = True

    while int(mask.sum()) < int(cell_count):
        frontier = []
        for r, c in np.argwhere(mask):
            for nr, nc in neighbors4(int(r), int(c)):
                if in_bounds(mask.shape, nr, nc) and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return mask
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True

    return mask


def _red_candidates(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    corners = []
    edges = []
    for r in range(TILE_SIZE):
        for c in range(TILE_SIZE):
            if mask[r, c]:
                continue
            if not any(
                in_bounds(mask.shape, nr, nc) and mask[nr, nc]
                for nr, nc in neighbors4(r, c)
            ):
                continue
            if r in {0, TILE_SIZE - 1} and c in {0, TILE_SIZE - 1}:
                corners.append((r, c))
            elif r in {0, TILE_SIZE - 1} or c in {0, TILE_SIZE - 1}:
                edges.append((r, c))
    return corners, edges


def _has_full_nonzero_line(tile: np.ndarray) -> bool:
    return bool(np.any(np.all(tile != Color.BLACK, axis=0)) or np.any(np.all(tile != Color.BLACK, axis=1)))


def _all_rotations_distinct(tile: np.ndarray) -> bool:
    variants = {_rotate_cw(tile, k).tobytes() for k in range(4)}
    return len(variants) == 4


def _base_canvas(tile: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    tile = np.asarray(tile, dtype=int)
    h, w = tile.shape
    base = np.full((h * TILE_REPETITIONS, w * TILE_REPETITIONS), Color.BLACK, dtype=int)
    red_positions: list[tuple[int, int]] = []
    for i in range(TILE_REPETITIONS):
        for j in range(TILE_REPETITIONS):
            rotated = _rotate_cw(tile, _rotation_rule(i, j))
            r0 = i * h
            c0 = j * w
            paste(base, rotated, r0, c0)
            rr, cc = map(int, np.argwhere(rotated == Color.RED)[0])
            red_positions.append((r0 + rr, c0 + cc))
    return base, red_positions


def _verify_output_semantics(tile: np.ndarray, out: np.ndarray) -> bool:
    base, red_positions = _base_canvas(tile)
    if out.shape != base.shape:
        return False

    expected_yellow = set()
    for r, c in red_positions:
        for dr, dc in DIAGONALS:
            nr = r + dr
            nc = c + dc
            if in_bounds(base.shape, nr, nc) and int(base[nr, nc]) == Color.BLACK:
                expected_yellow.add((nr, nc))

    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            observed = int(out[r, c])
            base_color = int(base[r, c])
            if (r, c) in expected_yellow:
                if observed != Color.YELLOW:
                    return False
            elif observed != base_color:
                return False

    return True


def _shortcut_outputs(tile: np.ndarray) -> list[np.ndarray]:
    return [
        _render_with_rule(tile, lambda _i, _j: 0),
        _render_with_rule(tile, lambda i, _j: i),
        _render_with_rule(tile, lambda _i, j: j),
        _render_with_rule(tile, lambda i, j: (2 * (i + j)) % 4),
        _render_with_rule(tile, lambda i, j: (2 * i + j) % 4),
    ]


def _valid_input(tile: np.ndarray) -> bool:
    if tile.shape != (TILE_SIZE, TILE_SIZE):
        return False
    if not set(map(int, np.unique(tile))).issubset({Color.BLACK, Color.BLUE, Color.RED}):
        return False
    if np.count_nonzero(tile == Color.RED) != 1:
        return False

    blue_mask = tile == Color.BLUE
    blue_count = int(blue_mask.sum())
    if blue_count < 5 or blue_count > 8:
        return False
    if len(components_mask(blue_mask, connectivity=4)) != 1:
        return False
    if _has_full_nonzero_line(tile):
        return False
    if not _all_rotations_distinct(tile):
        return False

    red_r, red_c = map(int, np.argwhere(tile == Color.RED)[0])
    if not any(
        in_bounds(tile.shape, nr, nc) and int(tile[nr, nc]) == Color.BLUE
        for nr, nc in neighbors4(red_r, red_c)
    ):
        return False

    out = main(tile)
    if out.shape != (TILE_SIZE * TILE_REPETITIONS, TILE_SIZE * TILE_REPETITIONS):
        return False
    if int(np.count_nonzero(out == Color.YELLOW)) < 6:
        return False
    if not _verify_output_semantics(tile, out):
        return False
    if any(np.array_equal(out, shortcut) for shortcut in _shortcut_outputs(tile)):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    blue_count = random.randint(5, 8)
    blue_mask = _connected_blue_mask(blue_count)
    if int(blue_mask.sum()) != blue_count:
        return None

    corner_candidates, edge_candidates = _red_candidates(blue_mask)
    if corner_candidates and random.random() < 0.55:
        red_r, red_c = random.choice(corner_candidates)
    elif edge_candidates:
        red_r, red_c = random.choice(edge_candidates)
    elif corner_candidates:
        red_r, red_c = random.choice(corner_candidates)
    else:
        return None

    tile = np.full((TILE_SIZE, TILE_SIZE), Color.BLACK, dtype=int)
    tile[blue_mask] = Color.BLUE
    tile[red_r, red_c] = Color.RED

    if not _valid_input(tile):
        return None
    return tile


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        tile = _generate_attempt()
        if tile is not None:
            return tile
    raise RuntimeError("failed to generate a valid tile for puzzle 5")
