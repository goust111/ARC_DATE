from __future__ import annotations

import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate every compartment's sprite to match the uniquely recolored sample compartment and repaint all separators with the sample border color.

    Input structure:
    - The grid is split by one-cell-thick separator rows and columns into an MxN array of equal rectangular compartments.
    - Separator pixels use one common border color except around one sample compartment whose full perimeter uses a different border color.
    - Each compartment interior contains one copy of the same non-symmetric sprite, but different compartments may show different 0/90/180/270 degree rotations.

    Rule:
    - Detect the sample compartment from the unique perimeter color and read both its border color and its sprite orientation.
    - For every compartment, compare its interior sprite against the sample sprite under the four right-angle rotations and choose the rotation that makes them match.
    - Replace the interior of each compartment with that rotated sprite, positioned exactly where the sample sprite sits inside its compartment.
    - Recolor every separator row and separator column to the sample border color.

    Key insight:
    - The unusual border color identifies a reference compartment; its perimeter gives the target separator color, and its sprite gives the target rotation that all other compartments must align to.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _most_common_color(g)
    sep_rows, sep_cols = _find_separator_indices(g, bg)
    region_rows = len(sep_rows) - 1
    region_cols = len(sep_cols) - 1
    assert region_rows >= 2 and region_cols >= 2

    sep_mask = np.zeros_like(g, dtype=bool)
    sep_mask[sep_rows, :] = True
    sep_mask[:, sep_cols] = True
    template_color = _least_common_color(g[sep_mask])

    template_rc = _find_template_region(g, sep_rows, sep_cols, template_color)
    tr, tc = template_rc
    template_interior = _region_interior(g, sep_rows, sep_cols, tr, tc)
    template_sprite, template_r0, template_c0 = _crop_non_bg(template_interior, bg)

    out = g.copy()
    out[sep_mask] = template_color

    for r in range(region_rows):
        for c in range(region_cols):
            interior = _region_interior(g, sep_rows, sep_cols, r, c)
            sprite, _, _ = _crop_non_bg(interior, bg)

            matched = None
            for k in range(4):
                rotated = np.rot90(sprite, k)
                if np.array_equal(rotated, template_sprite):
                    matched = rotated
                    break
            assert matched is not None, "every compartment sprite must match the sample up to rotation"

            rr0 = sep_rows[r] + 1
            rr1 = sep_rows[r + 1]
            cc0 = sep_cols[c] + 1
            cc1 = sep_cols[c + 1]
            out[rr0:rr1, cc0:cc1] = bg
            sh, sw = matched.shape
            out[rr0 + template_r0 : rr0 + template_r0 + sh, cc0 + template_c0 : cc0 + template_c0 + sw] = matched

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")


def _most_common_color(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    return int(values[np.argmax(counts)])


def _least_common_color(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    return int(values[np.argmin(counts)])


def _find_separator_indices(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    sep_rows = np.flatnonzero(np.all(grid != bg, axis=1))
    sep_cols = np.flatnonzero(np.all(grid != bg, axis=0))
    assert len(sep_rows) >= 3 and len(sep_cols) >= 3
    assert int(sep_rows[0]) == 0 and int(sep_cols[0]) == 0
    assert int(sep_rows[-1]) == grid.shape[0] - 1 and int(sep_cols[-1]) == grid.shape[1] - 1

    row_steps = np.diff(sep_rows)
    col_steps = np.diff(sep_cols)
    assert np.all(row_steps == row_steps[0]) and np.all(col_steps == col_steps[0])
    assert int(row_steps[0]) >= 2 and int(col_steps[0]) >= 2
    return sep_rows, sep_cols


def _region_interior(grid: np.ndarray, sep_rows: np.ndarray, sep_cols: np.ndarray, r: int, c: int) -> np.ndarray:
    return grid[sep_rows[r] + 1 : sep_rows[r + 1], sep_cols[c] + 1 : sep_cols[c + 1]]


def _crop_non_bg(arr: np.ndarray, bg: int) -> tuple[np.ndarray, int, int]:
    pts = np.argwhere(arr != bg)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy(), int(r0), int(c0)


def _perimeter_is_color(
    grid: np.ndarray,
    sep_rows: np.ndarray,
    sep_cols: np.ndarray,
    r: int,
    c: int,
    color: int,
) -> bool:
    r0, r1 = int(sep_rows[r]), int(sep_rows[r + 1])
    c0, c1 = int(sep_cols[c]), int(sep_cols[c + 1])
    return (
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _find_template_region(
    grid: np.ndarray,
    sep_rows: np.ndarray,
    sep_cols: np.ndarray,
    template_color: int,
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(len(sep_rows) - 1):
        for c in range(len(sep_cols) - 1):
            if _perimeter_is_color(grid, sep_rows, sep_cols, r, c, template_color):
                candidates.append((r, c))
    assert len(candidates) == 1
    return candidates[0]


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    max_cell = min((30 - (rows + 1)) // rows, (30 - (cols + 1)) // cols, 8)
    if max_cell < 5:
        return None
    cell = rng.randint(5, max_cell)

    palette = list(range(10))
    bg, border_color, template_color, sprite_color = rng.sample(palette, 4)

    sprite_mask = _make_sprite_mask(rng, cell)
    if sprite_mask is None:
        return None

    height = rows * cell + rows + 1
    width = cols * cell + cols + 1
    grid = np.full((height, width), bg, dtype=int)

    sep_rows = np.arange(0, height, cell + 1, dtype=int)
    sep_cols = np.arange(0, width, cell + 1, dtype=int)
    grid[sep_rows, :] = border_color
    grid[:, sep_cols] = border_color

    template_r = rng.randrange(rows)
    template_c = rng.randrange(cols)
    _paint_region_perimeter(grid, sep_rows, sep_cols, template_r, template_c, template_color)

    target_rotation = rng.randrange(4)
    rotations = np.array([[rng.randrange(4) for _ in range(cols)] for _ in range(rows)], dtype=int)
    rotations[template_r, template_c] = target_rotation
    others = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != (template_r, template_c)]
    if all(int(rotations[r, c]) == target_rotation for r, c in others):
        rr, cc = rng.choice(others)
        rotations[rr, cc] = (target_rotation + rng.choice((1, 2, 3))) % 4

    for r in range(rows):
        for c in range(cols):
            interior = np.full((cell, cell), bg, dtype=int)
            rotated_mask = np.rot90(sprite_mask, int(rotations[r, c]))
            sh, sw = rotated_mask.shape
            top = (cell - sh) // 2
            left = (cell - sw) // 2
            interior[top : top + sh, left : left + sw][rotated_mask] = sprite_color

            rr0 = sep_rows[r] + 1
            rr1 = sep_rows[r + 1]
            cc0 = sep_cols[c] + 1
            cc1 = sep_cols[c + 1]
            grid[rr0:rr1, cc0:cc1] = interior

    if _most_common_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def _paint_region_perimeter(
    grid: np.ndarray,
    sep_rows: np.ndarray,
    sep_cols: np.ndarray,
    r: int,
    c: int,
    color: int,
) -> None:
    r0, r1 = int(sep_rows[r]), int(sep_rows[r + 1])
    c0, c1 = int(sep_cols[c]), int(sep_cols[c + 1])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _make_sprite_mask(rng: random.Random, cell: int) -> np.ndarray | None:
    max_h = min(cell - 2, 4)
    max_w = min(cell - 2, 4)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(200):
        h = rng.randint(2, max_h)
        w = rng.randint(2, max_w)
        total = h * w
        min_cells = 3
        max_cells = min(total - 1, 7)
        if max_cells < min_cells:
            continue
        cells = rng.randint(min_cells, max_cells)
        mask = np.zeros((h, w), dtype=bool)

        seed = (rng.randrange(h), rng.randrange(w))
        mask[seed] = True
        frontier = {seed}
        while int(mask.sum()) < cells:
            fr, fc = rng.choice(tuple(frontier))
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = fr + dr, fc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.discard((fr, fc))
                if not frontier:
                    break
                continue
            chosen = rng.choice(nbrs)
            mask[chosen] = True
            frontier.add(chosen)

        if int(mask.sum()) != cells:
            continue

        cropped = _crop_mask(mask)
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if np.any(cropped.sum(axis=0) == cropped.shape[0]) or np.any(cropped.sum(axis=1) == cropped.shape[1]):
            continue
        rotations = [np.rot90(cropped, k) for k in range(4)]
        unique = []
        for rot in rotations:
            if not any(np.array_equal(rot, prev) for prev in unique):
                unique.append(rot)
        if len(unique) != 4:
            continue
        return cropped

    return None


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()
