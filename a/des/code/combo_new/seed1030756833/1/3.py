from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ARC_MAX_SIZE = 30
GEN_RETRIES = 300


def _segments(line: np.ndarray, bg: int = BACKGROUND) -> list[tuple[int, int, int]]:
    """Return (color, start, end) runs for a fully colored ruler line."""
    vals = [int(x) for x in line.tolist()]
    assert vals and all(v != int(bg) for v in vals), "ruler must be non-background"

    runs: list[tuple[int, int, int]] = []
    start = 0
    cur = vals[0]
    for i, val in enumerate(vals[1:], start=1):
        if val == cur:
            continue
        runs.append((int(cur), int(start), int(i)))
        start = i
        cur = val
    runs.append((int(cur), int(start), int(len(vals))))
    return runs


def _crop_non_bg(patch: np.ndarray, bg: int = BACKGROUND) -> np.ndarray | None:
    coords = np.argwhere(patch != int(bg))
    if coords.size == 0:
        return None
    colors = set(map(int, patch[patch != int(bg)].tolist()))
    assert len(colors) == 1, "each partition must contain one object color"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _pack_groups(groups: list[list[np.ndarray]], bg: int = BACKGROUND) -> np.ndarray:
    present = [group for group in groups if group]
    assert present, "expected at least one non-empty partition"

    group_sizes: list[tuple[int, int]] = []
    for group in present:
        height = sum(int(item.shape[0]) for item in group) + len(group) - 1
        width = max(int(item.shape[1]) for item in group)
        group_sizes.append((int(height), int(width)))

    out_h = max(h for h, _w in group_sizes)
    out_w = sum(w for _h, w in group_sizes) + len(group_sizes) - 1
    out = np.full((int(out_h), int(out_w)), int(bg), dtype=int)

    c = 0
    for group, (_height, width) in zip(present, group_sizes, strict=False):
        r = 0
        for item in group:
            h, w = map(int, item.shape)
            out[r : r + h, c : c + w] = item
            r += h + 1
        c += int(width) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop ruler-addressed partition objects and stack them by column-ruler color.

    Input structure:
    - The background is black.
    - The first row and first column are contiguous colored ruler segments; the
      first row gives internal column widths and the first column gives internal
      row heights.
    - Inside the ruler-defined rectangular partitions, some cells contain a
      single-color object and the remaining cells are all background.

    Rule:
    - Parse the top and left rulers, then inspect every internal partition.
    - Crop each non-empty partition to the minimal bounding box of its object.
    - Group cropped objects by their top-ruler segment color, preserving left-to-right
      ruler order; within each group, stack objects in top-to-bottom row-ruler order
      with one background row between neighboring crops.
    - Place the non-empty groups left to right with one background column between
      neighboring groups, left-aligning objects within each group and top-aligning
      the groups.

    Key insight:
    - The edge rulers define a two-dimensional address system; the output ignores
      empty addresses, crops the addressed objects, then repacks them by the column
      ruler colors rather than by their original absolute positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3, "expected rulers plus an internal area"

    col_runs = _segments(g[0, 1:], BACKGROUND)
    row_runs = _segments(g[1:, 0], BACKGROUND)
    assert sum(end - start for _color, start, end in col_runs) == width - 1
    assert sum(end - start for _color, start, end in row_runs) == height - 1

    col_colors = [color for color, _start, _end in col_runs]
    row_colors = [color for color, _start, _end in row_runs]
    assert len(col_colors) == len(set(col_colors)), "column ruler colors must be unique"
    assert len(row_colors) == len(set(row_colors)), "row ruler colors must be unique"

    internal = g[1:, 1:]
    groups: list[list[np.ndarray]] = [[] for _ in col_runs]
    for row_color, rs, re in row_runs:
        assert int(row_color) != BACKGROUND
        for col_i, (col_color, cs, ce) in enumerate(col_runs):
            assert int(col_color) != BACKGROUND
            cell = internal[int(rs) : int(re), int(cs) : int(ce)]
            cropped = _crop_non_bg(cell, BACKGROUND)
            if cropped is not None:
                groups[col_i].append(cropped)

    out = _pack_groups(groups, BACKGROUND)
    assert max(out.shape) <= ARC_MAX_SIZE, "output grid exceeds ARC size"
    return out


# --------------- generator ---------------

COL_COUNT_RANGE = (2, 4)
ROW_COUNT_RANGE = (2, 4)
PARTITION_SIZE_RANGE = (4, 7)
OBJECT_COUNT_RANGE = (2, 8)


def _rand_sizes(count: int) -> list[int]:
    return [random.randint(*PARTITION_SIZE_RANGE) for _ in range(int(count))]


def _colored_runs(colors: list[int], sizes: list[int]) -> np.ndarray:
    vals: list[int] = []
    for color, size in zip(colors, sizes, strict=False):
        vals.extend([int(color)] * int(size))
    return np.array(vals, dtype=int)


def _shape_mask(max_h: int, max_w: int) -> np.ndarray:
    max_h = int(max_h)
    max_w = int(max_w)
    shape_types = ["rect", "line", "ell", "tee", "frame", "poly"]
    random.shuffle(shape_types)

    for kind in shape_types:
        if kind == "rect":
            h = random.randint(1, min(4, max_h))
            w = random.randint(1, min(4, max_w))
            return np.ones((h, w), dtype=bool)

        if kind == "line":
            if max_w >= 2 and (max_h == 1 or random.random() < 0.5):
                w = random.randint(2, min(5, max_w))
                return np.ones((1, w), dtype=bool)
            if max_h >= 2:
                h = random.randint(2, min(5, max_h))
                return np.ones((h, 1), dtype=bool)

        if kind == "ell" and max_h >= 2 and max_w >= 2:
            h = random.randint(2, min(4, max_h))
            w = random.randint(2, min(4, max_w))
            mask = np.zeros((h, w), dtype=bool)
            orient = random.choice(["tl", "tr", "bl", "br"])
            if orient in {"tl", "tr"}:
                mask[0, :] = True
            else:
                mask[-1, :] = True
            if orient in {"tl", "bl"}:
                mask[:, 0] = True
            else:
                mask[:, -1] = True
            return mask

        if kind == "tee" and max_h >= 3 and max_w >= 3:
            h = random.randint(3, min(5, max_h))
            w = random.randint(3, min(5, max_w))
            mask = np.zeros((h, w), dtype=bool)
            mid_r = h // 2
            mid_c = w // 2
            orient = random.choice(["up", "down", "left", "right"])
            if orient == "up":
                mask[0, :] = True
                mask[:, mid_c] = True
            elif orient == "down":
                mask[-1, :] = True
                mask[:, mid_c] = True
            elif orient == "left":
                mask[:, 0] = True
                mask[mid_r, :] = True
            else:
                mask[:, -1] = True
                mask[mid_r, :] = True
            return mask

        if kind == "frame" and max_h >= 3 and max_w >= 3:
            h = random.randint(3, min(5, max_h))
            w = random.randint(3, min(5, max_w))
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            return mask

        if kind == "poly":
            h = random.randint(2, min(5, max_h))
            w = random.randint(2, min(5, max_w))
            target = random.randint(3, min(h * w, 8))
            cells = {(random.randrange(h), random.randrange(w))}
            while len(cells) < target:
                r, c = random.choice(tuple(cells))
                options = [
                    (r + dr, c + dc)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= r + dr < h and 0 <= c + dc < w
                ]
                cells.add(random.choice(options))
            rows = [r for r, _c in cells]
            cols = [c for _r, c in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in cells:
                mask[r - r0, c - c0] = True
            return mask

    return np.ones((1, 1), dtype=bool)


def _place_object(grid: np.ndarray, cell: tuple[int, int, int, int], color: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, cell)
    room_h = int(r1 - r0)
    room_w = int(c1 - c0)
    max_h = max(1, room_h - 2)
    max_w = max(1, room_w - 2)
    mask = _shape_mask(max_h, max_w)
    h, w = map(int, mask.shape)

    top_min = r0 + 1 if room_h - h >= 2 else r0
    left_min = c0 + 1 if room_w - w >= 2 else c0
    top_max = r1 - h - (1 if room_h - h >= 2 else 0)
    left_max = c1 - w - (1 if room_w - w >= 2 else 0)
    top = random.randint(int(top_min), int(top_max))
    left = random.randint(int(left_min), int(left_max))

    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    patch[mask] = int(color)
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)
    return patch


def _expected_from_crops(
    crops_by_cell: dict[tuple[int, int], np.ndarray],
    rows: int,
    cols: int,
) -> np.ndarray:
    groups: list[list[np.ndarray]] = []
    for c in range(int(cols)):
        group = []
        for r in range(int(rows)):
            crop = crops_by_cell.get((r, c))
            if crop is not None:
                group.append(crop)
        groups.append(group)
    return _pack_groups(groups, BACKGROUND)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_COUNT_RANGE)
    cols = random.randint(*COL_COUNT_RANGE)
    row_sizes = _rand_sizes(rows)
    col_sizes = _rand_sizes(cols)
    if 1 + sum(row_sizes) > ARC_MAX_SIZE or 1 + sum(col_sizes) > ARC_MAX_SIZE:
        return None

    ruler_colors = random.sample(range(1, 10), rows + cols)
    col_colors = ruler_colors[:cols]
    row_colors = ruler_colors[cols:]
    object_palette = [c for c in range(1, 10) if c not in ruler_colors]
    if not object_palette:
        object_palette = list(range(1, 10))

    grid = np.full((1 + sum(row_sizes), 1 + sum(col_sizes)), BACKGROUND, dtype=int)
    grid[0, 1:] = _colored_runs(col_colors, col_sizes)
    grid[1:, 0] = _colored_runs(row_colors, row_sizes)

    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    max_objects = min(OBJECT_COUNT_RANGE[1], len(all_cells))
    min_objects = min(max(OBJECT_COUNT_RANGE[0], min(rows, cols)), max_objects)
    n_objects = random.randint(min_objects, max_objects)

    chosen = set(random.sample(all_cells, n_objects))
    if cols > 1 and not any(any((r, c) in chosen for r in range(rows)) for c in range(cols)):
        return None

    row_starts = np.cumsum([0] + row_sizes[:-1]).astype(int).tolist()
    col_starts = np.cumsum([0] + col_sizes[:-1]).astype(int).tolist()

    crops_by_cell: dict[tuple[int, int], np.ndarray] = {}
    for r, c in sorted(chosen):
        rr = 1 + int(row_starts[r])
        cc = 1 + int(col_starts[c])
        cell = (rr, cc, rr + int(row_sizes[r]), cc + int(col_sizes[c]))
        color = random.choice(object_palette)
        patch = _place_object(grid, cell, color)
        cropped = _crop_non_bg(patch, BACKGROUND)
        assert cropped is not None
        crops_by_cell[(int(r), int(c))] = cropped

    expected = _expected_from_crops(crops_by_cell, rows, cols)
    if max(expected.shape) > ARC_MAX_SIZE:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ruler-stack input")
