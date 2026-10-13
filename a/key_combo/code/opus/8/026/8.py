from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
PINK = 6

PALETTE = [3, 4, 7, 8, 9]
TEMPLATE_LIBRARY = [
    np.array([[1, 0], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 0]], dtype=bool),
    np.array([[0, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0]], dtype=bool),
    np.array([[0, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
]
MAX_ATTEMPTS = 400


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(values.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expand(index_grid: np.ndarray, mask_by_color: dict[int, np.ndarray]) -> np.ndarray:
    side_h, side_w = index_grid.shape
    out = np.zeros((side_h * 2, side_w * 2), dtype=int)
    for r in range(side_h):
        for c in range(side_w):
            color = int(index_grid[r, c])
            mask = mask_by_color[color]
            block = out[r * 2 : (r + 1) * 2, c * 2 : (c + 1) * 2]
            block[mask] = color
    return out


def _parse(grid: np.ndarray) -> tuple[int, int, list[int], dict[int, np.ndarray], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 8 and g.shape[1] >= 6

    assert int(g[0, 0]) == GREY
    top = g[0, 1:]
    left = g[1:, 0]
    assert np.count_nonzero(top == RED) == 1
    assert np.count_nonzero(left == BLUE) == 1
    assert np.all(top[(top != RED)] == GREY)
    assert np.all(left[(left != BLUE)] == GREY)
    col_shift = int(np.flatnonzero(top == RED)[0])
    row_shift = int(np.flatnonzero(left == BLUE)[0])

    payload = g[1:, 1:]
    row_runs = _runs(np.any(payload != BG, axis=1))
    assert len(row_runs) == 3
    (sr0, sr1), (mr0, mr1), (ir0, ir1) = row_runs
    assert sr1 - sr0 + 1 == 2
    assert mr1 - mr0 + 1 == 2
    assert ir1 - ir0 + 1 in (3, 4)
    assert sr0 == 0

    swatch_band = payload[sr0 : sr1 + 1, :]
    mask_band = payload[mr0 : mr1 + 1, :]
    index_band = payload[ir0 : ir1 + 1, :]

    swatch_runs = _runs(np.any(swatch_band != BG, axis=0))
    assert len(swatch_runs) in (2, 3)
    colors: list[int] = []
    mask_by_color: dict[int, np.ndarray] = {}
    for c0, c1 in swatch_runs:
        assert c1 - c0 + 1 == 2
        swatch = swatch_band[:, c0 : c1 + 1]
        swatch_colors = set(map(int, swatch.ravel()))
        assert len(swatch_colors) == 1
        color = int(next(iter(swatch_colors)))
        assert color not in {BG, BLUE, RED, GREY, PINK}
        colors.append(color)

        mask_patch = mask_band[:, c0 : c1 + 1]
        assert mask_patch.shape == (2, 2)
        assert set(map(int, np.unique(mask_patch))).issubset({BG, PINK})
        mask = mask_patch == PINK
        assert mask.any()
        mask_by_color[color] = mask

    assert len(set(colors)) == len(colors)
    assert len({mask.tobytes() for mask in mask_by_color.values()}) == len(colors)

    index_box = _bbox(index_band != BG)
    assert index_box is not None
    r0, c0, r1, c1 = index_box
    assert r0 == 0 and r1 == index_band.shape[0] - 1
    index_grid = index_band[:, c0 : c1 + 1]
    assert index_grid.shape[0] == index_grid.shape[1]
    assert index_grid.shape[0] in (3, 4)
    assert np.all(index_grid != BG)
    assert set(map(int, np.unique(index_grid))).issubset(set(colors))
    assert set(colors).issubset(set(map(int, np.unique(index_grid))))
    return row_shift, col_shift, colors, mask_by_color, index_grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cyclically shift the lower index grid, then enlarge each shifted pixel with its legend-selected stencil.

    Input structure:
    - The top row is a grey control row with one red marker, and the left column is a grey control column with one blue marker.
    - After removing that border, the remaining area has an upper legend zone and a lower square index grid.
    - The legend zone contains 2-3 solid 2x2 color swatches and 2-3 aligned 2x2 magenta masks; their left-to-right order defines the pairing.

    Rule:
    - Read the red marker column as the cyclic right-shift amount and the blue marker row as the cyclic downward-shift amount.
    - Remove the control border, extract the lower square index grid, and roll it by those two offsets.
    - For each shifted index pixel, replace it with a 2x2 output block whose colored cells are given by the paired magenta mask for that pixel's legend color.

    Key insight:
    - The border markers do not choose output colors directly; they first reorder the color-index grid, and that reordered color layout indirectly changes which stencil lands at each enlarged output position.
    """
    row_shift, col_shift, _colors, mask_by_color, index_grid = _parse(grid)
    shifted = np.roll(index_grid, shift=(row_shift, col_shift), axis=(0, 1))
    return _expand(shifted, mask_by_color)


def _random_templates(count: int) -> list[np.ndarray]:
    while True:
        masks = [m.copy() for m in random.sample(TEMPLATE_LIBRARY, count)]
        if len({m.tobytes() for m in masks}) != count:
            continue
        if len({int(m.sum()) for m in masks}) == 1 and all(np.array_equal(m, masks[0]) for m in masks[1:]):
            continue
        return masks


def _random_index(side: int, colors: list[int]) -> np.ndarray:
    flat = [random.choice(colors) for _ in range(side * side)]
    for pos, color in zip(random.sample(range(side * side), len(colors)), colors):
        flat[pos] = color
    return np.array(flat, dtype=int).reshape(side, side)


def _build_input(
    colors: list[int],
    masks: list[np.ndarray],
    index_grid: np.ndarray,
    row_shift: int,
    col_shift: int,
) -> np.ndarray:
    count = len(colors)
    side = index_grid.shape[0]
    width = count * 2 + (count - 1)
    height = 2 + 1 + 2 + 1 + side

    payload = np.zeros((height, width), dtype=int)
    for i, (color, mask) in enumerate(zip(colors, masks)):
        c0 = i * 3
        payload[0:2, c0 : c0 + 2] = int(color)
        payload[3:5, c0 : c0 + 2] = np.where(mask, PINK, BG)

    index_c0 = random.randint(0, width - side)
    payload[6 : 6 + side, index_c0 : index_c0 + side] = index_grid

    grid = np.full((height + 1, width + 1), GREY, dtype=int)
    grid[1:, 1:] = payload
    grid[0, 1 + col_shift] = RED
    grid[1 + row_shift, 0] = BLUE
    return grid


def _all_shift_outputs_unique(index_grid: np.ndarray, mask_by_color: dict[int, np.ndarray]) -> bool:
    seen: set[bytes] = set()
    side = index_grid.shape[0]
    for row_shift in range(side):
        for col_shift in range(side):
            rolled = np.roll(index_grid, shift=(row_shift, col_shift), axis=(0, 1))
            key = _expand(rolled, mask_by_color).tobytes()
            if key in seen:
                return False
            seen.add(key)
    return True


def _is_semantic_instance(grid: np.ndarray) -> bool:
    try:
        row_shift, col_shift, colors, mask_by_color, index_grid = _parse(grid)
        side = index_grid.shape[0]
        if not (0 <= row_shift < side and 0 <= col_shift < side):
            return False
        if row_shift == 0 and col_shift == 0:
            return False
        if len(colors) not in (2, 3):
            return False
        if len({mask.tobytes() for mask in mask_by_color.values()}) != len(colors):
            return False
        if len({int(mask.sum()) for mask in mask_by_color.values()}) == 1:
            return False

        shifted = np.roll(index_grid, shift=(row_shift, col_shift), axis=(0, 1))
        if np.array_equal(shifted, index_grid):
            return False

        static_out = _expand(index_grid, mask_by_color)
        shifted_out = _expand(shifted, mask_by_color)
        if np.array_equal(static_out, shifted_out):
            return False
        if not _all_shift_outputs_unique(index_grid, mask_by_color):
            return False
        if not np.array_equal(main(grid), shifted_out):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([2, 3])
    side = random.choice([3, 4])
    colors = random.sample(PALETTE, count)
    masks = _random_templates(count)
    index_grid = _random_index(side, colors)
    row_shift, col_shift = random.choice(
        [(r, c) for r in range(side) for c in range(side) if (r, c) != (0, 0)]
    )
    grid = _build_input(colors, masks, index_grid, row_shift, col_shift)
    if not _is_semantic_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cyclic-shift stencil instance")
