from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


Coord = tuple[int, int]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _coord_set(mask: np.ndarray) -> set[Coord]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _all_in_bounds(coords: set[Coord], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _pairwise_disjoint(*coord_sets: set[Coord]) -> bool:
    seen: set[Coord] = set()
    for coords in coord_sets:
        if seen & coords:
            return False
        seen |= coords
    return True


def _reflect_horizontal(coords: set[Coord], row_sum: int) -> set[Coord]:
    return {(int(row_sum - r), int(c)) for r, c in coords}


def _reflect_vertical(coords: set[Coord], col_sum: int) -> set[Coord]:
    return {(int(r), int(col_sum - c)) for r, c in coords}


def _reflect_both(coords: set[Coord], row_sum: int, col_sum: int) -> set[Coord]:
    return {(int(row_sum - r), int(col_sum - c)) for r, c in coords}


def _horizontal_candidates(
    red: set[Coord], residue: set[Coord], shape: tuple[int, int]
) -> list[tuple[int, set[Coord]]]:
    red_by_col: dict[int, list[int]] = {}
    for r, c in red:
        red_by_col.setdefault(int(c), []).append(int(r))

    sums = {
        int(r_red + r_res)
        for r_res, c in residue
        for r_red in red_by_col.get(int(c), [])
    }
    out: list[tuple[int, set[Coord]]] = []
    for row_sum in sorted(sums):
        full = _reflect_horizontal(red, row_sum)
        if _all_in_bounds(full, shape) and residue <= full:
            out.append((row_sum, full))
    return out


def _vertical_candidates(
    red: set[Coord], residue: set[Coord], shape: tuple[int, int]
) -> list[tuple[int, set[Coord]]]:
    red_by_row: dict[int, list[int]] = {}
    for r, c in red:
        red_by_row.setdefault(int(r), []).append(int(c))

    sums = {
        int(c_red + c_res)
        for r, c_res in residue
        for c_red in red_by_row.get(int(r), [])
    }
    out: list[tuple[int, set[Coord]]] = []
    for col_sum in sorted(sums):
        full = _reflect_vertical(red, col_sum)
        if _all_in_bounds(full, shape) and residue <= full:
            out.append((col_sum, full))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the two hidden mirror axes from blue/green residues, complete all mirrors, and draw the yellow double-reflection.

    Input structure:
    - Black background with one complete red shape, one blue residue, and one green residue.
    - The blue cells are a partial reflection of the red shape across an unseen horizontal axis.
    - The green cells are a partial reflection of the red shape across an unseen vertical axis.

    Rule:
    - Match red and blue cells by shared columns to infer the horizontal mirror axis, and match red and green cells by shared rows to infer the vertical mirror axis.
    - Complete the full blue and green reflections of the red shape across those two axes.
    - Reflect the red shape across both inferred axes to paint the full yellow fourth copy, while preserving the red original.

    Key insight:
    - The axes are not marked directly; they are recoverable only from coordinate-consistent residue/original reflection pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    red = _coord_set(g == RED)
    blue_residue = _coord_set(g == BLUE)
    green_residue = _coord_set(g == GREEN)
    assert red and blue_residue and green_residue

    horizontal = _horizontal_candidates(red, blue_residue, g.shape)
    vertical = _vertical_candidates(red, green_residue, g.shape)
    solutions = []

    for row_sum, blue_full in horizontal:
        if red & blue_full:
            continue
        for col_sum, green_full in vertical:
            yellow_full = _reflect_both(red, row_sum, col_sum)
            if not _all_in_bounds(yellow_full, g.shape):
                continue
            if not _pairwise_disjoint(red, blue_full, green_full, yellow_full):
                continue
            solutions.append((blue_full, green_full, yellow_full))

    assert len(solutions) == 1, "residues must determine exactly one pair of mirror axes"
    blue_full, green_full, yellow_full = solutions[0]

    out = np.full(g.shape, bg, dtype=int)
    for coords, color in (
        (red, RED),
        (blue_full, BLUE),
        (green_full, GREEN),
        (yellow_full, YELLOW),
    ):
        for r, c in coords:
            out[r, c] = color
    return out


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _has_forbidden_symmetry(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(np.fliplr(mask)))
    )


def _frontier(cells: set[Coord], h: int, w: int) -> list[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr, dc in N4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                out.add((rr, cc))
    return list(out)


def _random_shape_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    min_cells = max(4, int(np.ceil(area * 0.35)))
    max_cells = min(area - 1, int(np.floor(area * 0.72)))
    if min_cells > max_cells:
        return None

    for _ in range(120):
        target = random.randint(min_cells, max_cells)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            choices = _frontier(cells, h, w)
            if not choices:
                break
            cells.add(random.choice(choices))
        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not _touches_all_edges(mask):
            continue
        if _mask_bbox(mask) != (0, 0, h - 1, w - 1):
            continue
        if _has_forbidden_symmetry(mask):
            continue
        return mask
    return None


def _translate(mask: np.ndarray, top: int, left: int) -> set[Coord]:
    return {(int(top + r), int(left + c)) for r, c in np.argwhere(mask)}


def _find_unique_residue(
    full_coords: set[Coord],
    red_coords: set[Coord],
    shape: tuple[int, int],
    mode: str,
) -> set[Coord] | None:
    cells = list(full_coords)
    n = len(cells)
    if n <= 1:
        return None

    lo = max(1, n // 3)
    hi = n - 1
    for _ in range(80):
        k = random.randint(lo, hi)
        subset = set(random.sample(cells, k))
        if mode == "h":
            count = len(_horizontal_candidates(red_coords, subset, shape))
        else:
            count = len(_vertical_candidates(red_coords, subset, shape))
        if count == 1:
            return subset

    random.shuffle(cells)
    subset: set[Coord] = set()
    for cell in cells[:-1]:
        subset.add(cell)
        if mode == "h":
            count = len(_horizontal_candidates(red_coords, subset, shape))
        else:
            count = len(_vertical_candidates(red_coords, subset, shape))
        if count == 1:
            return subset
    return None


def _paint(coords: set[Coord], color: int, grid: np.ndarray) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _gen_attempt() -> np.ndarray | None:
    shape_h = random.randint(3, 7)
    shape_w = random.randint(3, 7)
    mask = _random_shape_mask(shape_h, shape_w)
    if mask is None:
        return None

    gap_h = random.randint(1, 8)
    gap_w = random.randint(1, 8)
    core_h = 2 * shape_h + gap_h
    core_w = 2 * shape_w + gap_w
    if core_h > 30 or core_w > 30:
        return None

    rem_h = 30 - core_h
    rem_w = 30 - core_w
    pad_top = random.randint(0, min(4, rem_h))
    pad_bottom = random.randint(0, min(4, rem_h - pad_top))
    pad_left = random.randint(0, min(4, rem_w))
    pad_right = random.randint(0, min(4, rem_w - pad_left))

    grid_h = core_h + pad_top + pad_bottom
    grid_w = core_w + pad_left + pad_right
    if grid_h > 30 or grid_w > 30:
        return None

    red_top = pad_top
    red_left = pad_left
    blue_top = red_top + shape_h + gap_h
    green_left = red_left + shape_w + gap_w
    row_sum = red_top + blue_top + shape_h - 1
    col_sum = red_left + green_left + shape_w - 1

    red = _translate(mask, red_top, red_left)
    blue_full = _reflect_horizontal(red, row_sum)
    green_full = _reflect_vertical(red, col_sum)
    yellow_full = _reflect_both(red, row_sum, col_sum)
    if not _pairwise_disjoint(red, blue_full, green_full, yellow_full):
        return None

    blue_residue = _find_unique_residue(blue_full, red, (grid_h, grid_w), "h")
    if blue_residue is None:
        return None
    green_residue = _find_unique_residue(green_full, red, (grid_h, grid_w), "v")
    if green_residue is None:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    _paint(red, RED, inp)
    _paint(blue_residue, BLUE, inp)
    _paint(green_residue, GREEN, inp)

    expected = np.full((grid_h, grid_w), BG, dtype=int)
    _paint(red, RED, expected)
    _paint(blue_full, BLUE, expected)
    _paint(green_full, GREEN, expected)
    _paint(yellow_full, YELLOW, expected)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residue-double-axis sample")
