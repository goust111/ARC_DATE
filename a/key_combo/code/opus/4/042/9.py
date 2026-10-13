from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
GREY = 5
SPRITE_PALETTE = (2, 3, 4, 6, 7, 8, 9)
CELL_INNER = 3
CELL_STRIDE = CELL_INNER + 1
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _flood_from_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    reached = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not reached[r, c]:
                reached[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not reached[r, c]:
                reached[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                q.append((nr, nc))
    return reached


def _table_counts(grid: np.ndarray) -> dict[int, int]:
    grey_box = bbox(grid == GREY)
    assert grey_box is not None, "expected a grey table"
    r0, c0, r1, c1 = grey_box
    table = grid[r0 : r1 + 1, c0 : c1 + 1]
    full_rows = [r for r in range(table.shape[0]) if np.all(table[r] == GREY)]
    full_cols = [c for c in range(table.shape[1]) if np.all(table[:, c] == GREY)]
    assert len(full_rows) >= 3 and len(full_cols) >= 4, "expected a multi-cell table"

    counts: dict[int, int] = {}
    for ra, rb in zip(full_rows, full_rows[1:]):
        for ca, cb in zip(full_cols, full_cols[1:]):
            if rb - ra <= 1 or cb - ca <= 1:
                continue
            cell = table[ra + 1 : rb, ca + 1 : cb]
            colors = {int(v) for v in cell.ravel()} - {BG, GREY}
            assert len(colors) <= 1, "each table cell may contain at most one sprite color"
            if not colors:
                continue
            color = colors.pop()
            counts[color] = counts.get(color, 0) + 1
    assert 3 <= len(counts) <= 4, "expected 3-4 sprite colors"
    assert len(set(counts.values())) == len(counts), "sprite frequencies must be unique"
    return counts


def _shape_holes(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    blue_box = bbox(grid == BLUE)
    assert blue_box is not None, "expected one blue shape"
    shape = crop_bbox(grid, blue_box)
    black = shape == BG
    exterior = _flood_from_boundary(black)
    holes_mask = black & ~exterior
    holes = _components(holes_mask, connectivity=4)
    assert 3 <= len(holes) <= 4, "expected 3-4 enclosed holes"
    areas = [int(coords.shape[0]) for coords in holes]
    assert len(set(areas)) == len(areas), "hole areas must be unique"
    return shape, holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort table colors by sprite frequency, then fill the blue shape's holes by matching that rank to hole area rank.

    Input structure:
    - Black background contains two separated regions: a grey grid table above and a blue walled shape below.
    - Table cells may contain one solid 2x2 sprite of a non-grey, non-blue color, or be empty.
    - The lower blue shape encloses 3-4 black holes whose areas are all different.

    Rule:
    - Count how many table cells contain each sprite color, ignoring grey grid lines and empty cells.
    - Sort the colors from least frequent to most frequent.
    - Crop the lower blue shape to its bounding box, find black regions inside it that are not 4-connected to that crop boundary, and sort those holes by area from smallest to largest.
    - Fill the i-th smallest hole with the i-th least frequent color, preserving the blue walls.

    Key insight:
    - The upper table and lower shape are independent structures; the hidden coupling is a rank alignment between color frequency order and enclosed-hole area order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    counts = _table_counts(g)
    colors = [color for color, _count in sorted(counts.items(), key=lambda item: item[1])]

    shape, holes = _shape_holes(g)
    holes = sorted(holes, key=lambda coords: int(coords.shape[0]))
    assert len(colors) == len(holes), "color count must match hole count"

    out = shape.copy()
    for color, coords in zip(colors, holes):
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _first_seen_order(table_patch: np.ndarray) -> list[int]:
    seen: list[int] = []
    for rr in range((table_patch.shape[0] - 1) // CELL_STRIDE):
        for cc in range((table_patch.shape[1] - 1) // CELL_STRIDE):
            r0 = rr * CELL_STRIDE + 1
            c0 = cc * CELL_STRIDE + 1
            cell = table_patch[r0 : r0 + CELL_INNER, c0 : c0 + CELL_INNER]
            colors = {int(v) for v in cell.ravel()} - {BG}
            if colors:
                color = next(iter(colors))
                if color not in seen:
                    seen.append(color)
    return seen


def _make_table(n_colors: int) -> tuple[np.ndarray, list[int], dict[int, int]]:
    for _ in range(200):
        rows = random.randint(2, 3)
        cols = random.randint(max(4, n_colors), 5)
        cell_count = rows * cols
        count_choices = [
            combo
            for combo in __import__("itertools").combinations(range(1, 6), n_colors)
            if sum(combo) <= cell_count - 1
        ]
        if not count_choices:
            continue
        counts_sorted = list(random.choice(count_choices))
        random.shuffle(counts_sorted)
        colors = random.sample(SPRITE_PALETTE, n_colors)
        counts = dict(zip(colors, counts_sorted))
        count_order = [color for color, _count in sorted(counts.items(), key=lambda item: item[1])]
        if count_order == sorted(count_order):
            continue

        table = np.full((rows * CELL_STRIDE + 1, cols * CELL_STRIDE + 1), BG, dtype=int)
        table[0::CELL_STRIDE, :] = GREY
        table[:, 0::CELL_STRIDE] = GREY

        cells = [(r, c) for r in range(rows) for c in range(cols)]
        occupied = random.sample(cells, sum(counts.values()))
        cursor = 0
        for color in random.sample(colors, len(colors)):
            for r, c in occupied[cursor : cursor + counts[color]]:
                ir = r * CELL_STRIDE + 1 + random.randint(0, 1)
                ic = c * CELL_STRIDE + 1 + random.randint(0, 1)
                table[ir : ir + 2, ic : ic + 2] = color
            cursor += counts[color]

        if _table_counts(table) != counts:
            continue
        if _first_seen_order(table) == count_order:
            continue
        return table, count_order, counts
    raise RuntimeError("failed to build a non-degenerate table")


def _shape_patch_four() -> np.ndarray:
    options = [
        (h1, h2, w1, w2)
        for h1 in (1, 2, 3)
        for h2 in (1, 2, 3)
        for w1 in (1, 2, 3, 4)
        for w2 in (1, 2, 3, 4)
        if h1 != h2 and w1 != w2 and len({h1 * w1, h1 * w2, h2 * w1, h2 * w2}) == 4
    ]
    h1, h2, w1, w2 = random.choice(options)
    h = h1 + h2 + 3
    w = w1 + w2 + 3
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = BLUE
    patch[-1, :] = BLUE
    patch[:, 0] = BLUE
    patch[:, -1] = BLUE
    patch[h1 + 1, 1:-1] = BLUE
    patch[1:-1, w1 + 1] = BLUE
    return patch


def _shape_patch_three() -> np.ndarray:
    options = [
        (h1, h2, w_left, w_right)
        for h1 in (1, 2, 3)
        for h2 in (1, 2, 3)
        for w_left in (1, 2, 3)
        for w_right in (1, 2, 3, 4)
        if len({h1 * w_left, h2 * w_left, (h1 + h2 + 1) * w_right}) == 3
    ]
    h1, h2, w_left, w_right = random.choice(options)
    h = h1 + h2 + 3
    w = w_left + w_right + 3
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = BLUE
    patch[-1, :] = BLUE
    patch[:, 0] = BLUE
    patch[:, -1] = BLUE
    patch[1:-1, w_left + 1] = BLUE
    patch[h1 + 1, 1 : w_left + 2] = BLUE
    return patch


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    out = np.rot90(patch, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=int, copy=True)


def _spatial_orders(holes: list[np.ndarray]) -> list[list[int]]:
    meta = []
    for idx, coords in enumerate(holes):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        center_r = float(coords[:, 0].mean())
        center_c = float(coords[:, 1].mean())
        meta.append((idx, int(r0), int(c0), int(r1), int(c1), center_r, center_c))
    return [
        [idx for idx, *_rest in sorted(meta, key=lambda item: (item[6], item[5]))],
        [idx for idx, *_rest in sorted(meta, key=lambda item: (item[5], item[6]))],
        [idx for idx, *_rest in sorted(meta, key=lambda item: (item[1], item[2]))],
    ]


def _make_shape(n_holes: int) -> tuple[np.ndarray, list[np.ndarray]]:
    for _ in range(200):
        patch = _shape_patch_three() if n_holes == 3 else _shape_patch_four()
        patch = _transform_patch(patch)
        shape, holes = _shape_holes(patch)
        if not np.array_equal(shape, patch):
            continue
        area_order = [
            idx for idx, _coords in sorted(enumerate(holes), key=lambda item: int(item[1].shape[0]))
        ]
        if any(area_order == order for order in _spatial_orders(holes)):
            continue
        blue_area = int(np.sum(patch == BLUE))
        if not (20 <= blue_area <= 35):
            continue
        return patch, holes
    raise RuntimeError("failed to build a non-degenerate shape")


def _expected_output(shape_patch: np.ndarray, color_order: list[int]) -> np.ndarray:
    shape, holes = _shape_holes(shape_patch)
    holes = sorted(holes, key=lambda coords: int(coords.shape[0]))
    assert len(color_order) == len(holes)
    out = shape.copy()
    for color, coords in zip(color_order, holes):
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _assemble(table_patch: np.ndarray, shape_patch: np.ndarray) -> np.ndarray | None:
    top_margin = random.randint(0, 1)
    left_table = random.randint(0, 3)
    sep = random.randint(2, 3)
    shape_row = top_margin + table_patch.shape[0] + sep
    left_shape = random.randint(0, 4)
    width = max(left_table + table_patch.shape[1], left_shape + shape_patch.shape[1]) + random.randint(0, 2)
    height = shape_row + shape_patch.shape[0] + random.randint(0, 1)
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[top_margin : top_margin + table_patch.shape[0], left_table : left_table + table_patch.shape[1]] = table_patch
    grid[shape_row : shape_row + shape_patch.shape[0], left_shape : left_shape + shape_patch.shape[1]] = shape_patch
    return grid


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, counts: dict[int, int]) -> bool:
    if get_bg(inp) != BG:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    actual_counts = _table_counts(inp)
    if actual_counts != counts:
        return False
    color_order = [color for color, _count in sorted(actual_counts.items(), key=lambda item: item[1])]
    if color_order == sorted(color_order):
        return False

    shape, holes = _shape_holes(inp)
    if shape.shape != expected.shape:
        return False
    if len(color_order) != len(holes):
        return False

    area_order = [idx for idx, _coords in sorted(enumerate(holes), key=lambda item: int(item[1].shape[0]))]
    if any(area_order == order for order in _spatial_orders(holes)):
        return False

    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    n_holes = random.choice([3, 4])
    table_patch, color_order, counts = _make_table(n_holes)
    shape_patch, _holes = _make_shape(n_holes)
    inp = _assemble(table_patch, shape_patch)
    if inp is None:
        return None
    expected = _expected_output(shape_patch, color_order)
    if not _semantic_checks(inp, expected, counts):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
