import random
from collections import Counter

import numpy as np


BG_COLOR = 7
SEP_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 8, 9)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator(grid: np.ndarray) -> tuple[int, int, int]:
    row_candidates: dict[int, list[int]] = {}
    col_candidates: dict[int, list[int]] = {}

    for r in range(grid.shape[0]):
        if np.all(grid[r] == grid[r, 0]):
            row_candidates.setdefault(int(grid[r, 0]), []).append(r)
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == grid[0, c]):
            col_candidates.setdefault(int(grid[0, c]), []).append(c)

    colors = set(row_candidates) & set(col_candidates)
    valid = [(color, row_candidates[color], col_candidates[color]) for color in colors]
    valid = [item for item in valid if len(item[1]) == 1 and len(item[2]) == 1]
    assert len(valid) == 1, "expected exactly one full separator row and column"

    color, rows, cols = valid[0]
    return int(color), int(rows[0]), int(cols[0])


def _foreground_mask(patch: np.ndarray, bg: int, sep: int) -> np.ndarray:
    return (patch != bg) & (patch != sep)


def _rotation_to_reference(reference_mask: np.ndarray, other_mask: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(np.rot90(other_mask, k), reference_mask)]
    assert len(matches) == 1, "rotation relative to the reference must be unique"
    return int(matches[0])


def _overlay(base: np.ndarray, layer: np.ndarray, bg: int, sep: int) -> None:
    mask = _foreground_mask(layer, bg, sep)
    conflict = mask & (base != bg) & (base != layer)
    assert not np.any(conflict), "aligned quadrants disagree on foreground colors"
    base[mask] = layer[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize four rotated quadrant copies to the top-left orientation and merge them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep, split_r, split_c = _find_separator(g)
    bg = _most_common_color(g)
    assert bg != sep

    top = g[:split_r, :]
    bottom = g[split_r + 1 :, :]
    assert top.shape[0] == bottom.shape[0] and top.shape[0] > 0

    left_width = split_c
    right_width = g.shape[1] - split_c - 1
    assert left_width == right_width and left_width > 0

    tl = g[:split_r, :split_c]
    tr = g[:split_r, split_c + 1 :]
    bl = g[split_r + 1 :, :split_c]
    br = g[split_r + 1 :, split_c + 1 :]
    qh, qw = tl.shape
    assert tr.shape == bl.shape == br.shape == (qh, qw)
    assert qh == qw, "90-degree rotations require square quadrants"

    reference_mask = _foreground_mask(tl, bg, sep)
    assert np.any(reference_mask), "reference quadrant must contain a foreground pattern"

    aligned = [tl]
    for quad in (tr, bl, br):
        rot = _rotation_to_reference(reference_mask, _foreground_mask(quad, bg, sep))
        aligned.append(np.rot90(quad, rot))

    out = np.full((qh, qw), bg, dtype=int)
    for quad in aligned:
        _overlay(out, quad, bg, sep)
    return out


def _neighbors(cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _grow_mask(size: int, area: int) -> np.ndarray:
    inner_cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    seed = random.choice(inner_cells)
    cells = {seed}
    frontier = set(_neighbors(seed, size)) & set(inner_cells)

    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nxt in _neighbors(cell, size):
            if nxt in cells or nxt not in inner_cells:
                continue
            frontier.add(nxt)

    if len(cells) != area:
        return np.zeros((size, size), dtype=bool)

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _all_rotations_distinct(mask: np.ndarray) -> bool:
    keys = {np.rot90(mask, k).tobytes() for k in range(4)}
    return len(keys) == 4


def _valid_template(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return False
    if rows[0] == 0 or cols[0] == 0 or rows[-1] == mask.shape[0] - 1 or cols[-1] == mask.shape[1] - 1:
        return False
    return _all_rotations_distinct(mask)


def _random_template(size: int) -> np.ndarray | None:
    min_area = max(5, size - 1)
    max_area = min(size * size // 2, size * size - 8)
    if min_area > max_area:
        return None

    for _ in range(80):
        area = random.randint(min_area, max_area)
        mask = _grow_mask(size, area)
        if not _valid_template(mask):
            continue

        color_count = random.randint(2, min(4, len(PALETTE)))
        colors = random.sample(PALETTE, color_count)
        patch = np.full((size, size), BG_COLOR, dtype=int)
        coords = list(zip(*np.where(mask)))
        random.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            patch[r, c] = colors[idx % color_count]

        if len({np.rot90(mask, k).tobytes() for k in range(4)}) == 4:
            return patch
    return None


def _build_grid(template: np.ndarray, rotations: tuple[int, int, int]) -> np.ndarray:
    size = template.shape[0]
    grid = np.full((size * 2 + 1, size * 2 + 1), BG_COLOR, dtype=int)
    grid[size, :] = SEP_COLOR
    grid[:, size] = SEP_COLOR

    quads = [
        template,
        np.rot90(template, rotations[0]),
        np.rot90(template, rotations[1]),
        np.rot90(template, rotations[2]),
    ]
    grid[:size, :size] = quads[0]
    grid[:size, size + 1 :] = quads[1]
    grid[size + 1 :, :size] = quads[2]
    grid[size + 1 :, size + 1 :] = quads[3]
    return grid


def _expected_output(template: np.ndarray) -> np.ndarray:
    return template.copy()


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(6, 11)
    template = _random_template(size)
    if template is None:
        return None

    rotations = tuple(random.sample((1, 2, 3), 3))
    grid = _build_grid(template, rotations)
    expected = _expected_output(template)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
