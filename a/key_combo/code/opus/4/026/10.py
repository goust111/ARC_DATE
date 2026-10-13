from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5

CORNER_BOUNDS = {
    0: lambda m: (slice(0, m), slice(0, m)),
    1: lambda m: (slice(0, m), slice(m + 1, 2 * m + 1)),
    2: lambda m: (slice(m + 1, 2 * m + 1), slice(m + 1, 2 * m + 1)),
    3: lambda m: (slice(m + 1, 2 * m + 1), slice(0, m)),
}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_MAX_ATTEMPTS = 400


def _tile_slices(mid: int, corner: int) -> tuple[slice, slice]:
    return CORNER_BOUNDS[corner](mid)


def _extract_tiles(grid: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd square grid"
    mid = h // 2
    assert np.all(g[mid, :] == BLACK) and np.all(g[:, mid] == BLACK), "expected a black cross separator"

    tiles: dict[int, np.ndarray] = {}
    missing: list[int] = []
    for corner in range(4):
        rs, cs = _tile_slices(mid, corner)
        tile = g[rs, cs].copy()
        tiles[corner] = tile
        if np.all(tile == GRAY):
            missing.append(corner)

    assert len(missing) == 1, "expected exactly one gray placeholder tile"
    return tiles, missing[0]


def _render_tile(base_red: np.ndarray, base_blue: np.ndarray, rot: int, swapped: bool) -> np.ndarray:
    red = np.rot90(base_red, rot)
    blue = np.rot90(base_blue, rot)
    if swapped:
        red, blue = blue, red
    tile = np.zeros(base_red.shape, dtype=int)
    tile[blue] = BLUE
    tile[red] = RED
    return tile


def _derive_base(tile: np.ndarray, rot: int, swapped: bool) -> tuple[np.ndarray, np.ndarray] | None:
    obs_red = tile == RED
    obs_blue = tile == BLUE
    if swapped:
        obs_red, obs_blue = obs_blue, obs_red
    base_red = np.rot90(obs_red, -rot)
    base_blue = np.rot90(obs_blue, -rot)
    if np.any(base_red & base_blue):
        return None
    if np.count_nonzero(base_red) < 3 or np.count_nonzero(base_blue) < 3:
        return None
    return base_red, base_blue


def _all_rotations_distinct(mask: np.ndarray) -> bool:
    seen = {np.rot90(mask, k).tobytes() for k in range(4)}
    return len(seen) == 4


def _find_valid_solutions(
    grid: np.ndarray,
) -> list[tuple[np.ndarray, dict[str, object]]]:
    tiles, missing_corner = _extract_tiles(grid)
    visible = [corner for corner in range(4) if corner != missing_corner]
    anchor = visible[0]
    valid: list[tuple[np.ndarray, dict[str, object]]] = []

    for rotation_by_corner in itertools.permutations(range(4), 4):
        base = _derive_base(
            tiles[anchor],
            rotation_by_corner[anchor],
            swapped=bool(anchor % 2),
        )
        if base is None:
            continue
        base_red, base_blue = base
        shape = base_red | base_blue
        if not _all_rotations_distinct(shape):
            continue

        ok = True
        for corner in visible:
            predicted = _render_tile(
                base_red,
                base_blue,
                rotation_by_corner[corner],
                swapped=bool(corner % 2),
            )
            if not np.array_equal(predicted, tiles[corner]):
                ok = False
                break
        if not ok:
            continue

        output = _render_tile(
            base_red,
            base_blue,
            rotation_by_corner[missing_corner],
            swapped=bool(missing_corner % 2),
        )
        valid.append(
            (
                output,
                {
                    "base_red": base_red,
                    "base_blue": base_blue,
                    "rotation_by_corner": rotation_by_corner,
                    "missing_corner": missing_corner,
                },
            )
        )
    return valid


def _infer_unique_output(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    valid = _find_valid_solutions(grid)
    assert valid, "no valid rotation/color interpretation found"

    unique: dict[bytes, tuple[np.ndarray, dict[str, object]]] = {}
    for output, info in valid:
        unique.setdefault(output.tobytes(), (output, info))
    assert len(unique) == 1, "visible tiles allow multiple distinct outputs"
    return next(iter(unique.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the missing corner tile from three rotated two-color variants, apply the corner-parity color swap, and return the cropped missing tile.

    Input structure:
    - The grid is an odd square split by a one-cell black cross into four equal corner tiles.
    - Three corner tiles contain the same black-background two-color pattern under three different 90-degree rotations.
    - The remaining corner tile is a solid gray placeholder.

    Rule:
    - Treat red and blue together as one foreground mask and identify which three members of a four-rotation family are visible.
    - Recover a base two-color pattern that explains the visible tiles when each corner uses its own rotation and odd-numbered corners swap red and blue.
    - Generate the missing fourth rotation, apply the same corner-parity swap rule for the gray corner, and output only that corner tile.

    Key insight:
    - Shape rotation alone is insufficient: the missing geometry comes from the absent rotation family member, while the final red/blue placement is controlled separately by whether the target corner index is even or odd.
    """
    output, _info = _infer_unique_output(grid)
    return output


def _connected_growth(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    mask[start] = True
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cell = random.choice(frontier)
        cells.add(cell)
        mask[cell] = True
    return mask


def _choose_two_seeds(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    best_pair = None
    best_dist = -1
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            a = tuple(map(int, coords[i]))
            b = tuple(map(int, coords[j]))
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            if dist > best_dist:
                best_dist = dist
                best_pair = (a, b)
    assert best_pair is not None
    return best_pair


def _partition_colors(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(shape)
    if len(coords) < 6:
        return None

    for _ in range(30):
        seed_red, seed_blue = _choose_two_seeds(coords)
        labels = -np.ones(shape.shape, dtype=int)
        labels[seed_red] = 0
        labels[seed_blue] = 1
        q = deque([seed_red, seed_blue] if random.random() < 0.5 else [seed_blue, seed_red])
        while q:
            r, c = q.popleft()
            neighbors = DIRS4[:]
            random.shuffle(neighbors)
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape.shape[0] and 0 <= nc < shape.shape[1] and shape[nr, nc] and labels[nr, nc] == -1:
                    labels[nr, nc] = labels[r, c]
                    q.append((nr, nc))

        if np.any((labels == -1) & shape):
            continue
        red = labels == 0
        blue = labels == 1
        if np.count_nonzero(red) >= 3 and np.count_nonzero(blue) >= 3:
            return red, blue
    return None


def _sample_base_tile(tile_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        bh = random.randint(3, tile_size)
        bw = random.randint(3, tile_size)
        min_area = max(6, (bh * bw) // 3)
        max_area = min(bh * bw - 1, (2 * bh * bw) // 3 + 1)
        if min_area > max_area:
            continue

        shape_bbox = _connected_growth(bh, bw, random.randint(min_area, max_area))
        if np.count_nonzero(shape_bbox) < 6:
            continue
        if len({np.rot90(shape_bbox, k).tobytes() for k in range(4)}) != 4:
            continue

        colors = _partition_colors(shape_bbox)
        if colors is None:
            continue
        red_bbox, blue_bbox = colors

        top = random.randint(0, tile_size - bh)
        left = random.randint(0, tile_size - bw)
        red = np.zeros((tile_size, tile_size), dtype=bool)
        blue = np.zeros((tile_size, tile_size), dtype=bool)
        red[top : top + bh, left : left + bw] = red_bbox
        blue[top : top + bh, left : left + bw] = blue_bbox
        shape = red | blue

        if not _all_rotations_distinct(shape):
            continue
        if np.any(red & blue):
            continue
        if len({np.rot90(red, k).tobytes() for k in range(4)}) < 2:
            continue
        if len({np.rot90(blue, k).tobytes() for k in range(4)}) < 2:
            continue
        return red, blue
    return None


def _build_input(
    base_red: np.ndarray,
    base_blue: np.ndarray,
    rotation_by_corner: tuple[int, int, int, int],
    missing_corner: int,
) -> np.ndarray:
    tile_size = int(base_red.shape[0])
    grid = np.zeros((2 * tile_size + 1, 2 * tile_size + 1), dtype=int)
    for corner in range(4):
        rs, cs = _tile_slices(tile_size, corner)
        if corner == missing_corner:
            grid[rs, cs] = GRAY
        else:
            grid[rs, cs] = _render_tile(
                base_red,
                base_blue,
                rotation_by_corner[corner],
                swapped=bool(corner % 2),
            )
    return grid


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
) -> bool:
    try:
        inferred, info = _infer_unique_output(grid)
    except AssertionError:
        return False
    if not np.array_equal(inferred, expected):
        return False

    base_red = info["base_red"]
    base_blue = info["base_blue"]
    rotation_by_corner = info["rotation_by_corner"]
    missing_corner = int(info["missing_corner"])
    missing_rot = int(rotation_by_corner[missing_corner])

    wrong_swap = _render_tile(base_red, base_blue, missing_rot, swapped=not bool(missing_corner % 2))
    if np.array_equal(wrong_swap, expected):
        return False

    wrong_rot = _render_tile(base_red, base_blue, (missing_rot + 1) % 4, swapped=bool(missing_corner % 2))
    if np.array_equal(wrong_rot, expected):
        return False

    visible_tiles, _ = _extract_tiles(grid)
    visible_shapes = {
        np.argwhere((visible_tiles[i] == RED) | (visible_tiles[i] == BLUE)).tobytes()
        for i in range(4)
        if i != missing_corner
    }
    if len(visible_shapes) != 3:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    tile_size = random.choice([5, 7])
    base = _sample_base_tile(tile_size)
    if base is None:
        return None
    base_red, base_blue = base

    rotation_by_corner = tuple(random.sample(range(4), 4))
    missing_corner = random.randrange(4)
    grid = _build_input(base_red, base_blue, rotation_by_corner, missing_corner)
    expected = _render_tile(
        base_red,
        base_blue,
        rotation_by_corner[missing_corner],
        swapped=bool(missing_corner % 2),
    )
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
