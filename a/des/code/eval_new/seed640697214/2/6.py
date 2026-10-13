from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_is(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> bool:
    if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
        return False
    top = grid[r0, c0 : c1 + 1]
    bottom = grid[r1, c0 : c1 + 1]
    left = grid[r0 : r1 + 1, c0]
    right = grid[r0 : r1 + 1, c1]
    return bool(
        np.all(top == color)
        and np.all(bottom == color)
        and np.all(left == color)
        and np.all(right == color)
    )


def _marker_row_and_order(grid: np.ndarray) -> tuple[int, list[int]]:
    color_rows = [
        r
        for r in range(grid.shape[0])
        if np.any((grid[r] != BG) & (grid[r] != FRAME))
    ]
    assert color_rows
    marker_row = int(max(color_rows))
    cols = [c for c in range(grid.shape[1]) if int(grid[marker_row, c]) not in (BG, FRAME)]
    order = [int(grid[marker_row, c]) for c in cols]
    assert order
    assert len(order) == len(set(order))
    return marker_row, order


def _count_layers(grid: np.ndarray, seed_bbox: tuple[int, int, int, int], marker_row: int) -> int:
    r0, c0, r1, c1 = seed_bbox
    max_offset = min(r0, c0, marker_row - 1 - r1, grid.shape[1] - 1 - c1)
    assert max_offset >= 1

    base_offset = None
    for offset in range(1, max_offset + 1):
        if _border_is(grid, r0 - offset, c0 - offset, r1 + offset, c1 + offset, FRAME):
            base_offset = offset
            break
    assert base_offset is not None

    layers = 1
    offset = base_offset + 2
    while offset <= max_offset:
        if not _border_is(grid, r0 - (offset - 1), c0 - (offset - 1), r1 + (offset - 1), c1 + (offset - 1), BG):
            break
        if not _border_is(grid, r0 - offset, c0 - offset, r1 + offset, c1 + offset, FRAME):
            break
        layers += 1
        offset += 2
    return int(layers)


def _scale_pattern(pattern: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(pattern, factor, axis=0), factor, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each nested grey frame count as a scale factor, then enlarge and reorder the seed patterns.

    Input structure:
    - Black background with several separated groups of concentric grey rectangular frames.
    - Inside each group, a non-black non-grey seed pattern sits at the center area and uses one visible color.
    - A bottom marker row contains one pixel per group; each marker color matches that group's seed color and the left-to-right marker order encodes the output order.

    Rule:
    - For each marker color, locate the matching seed pattern above the marker row.
    - Count how many grey frame layers surround that seed pattern.
    - Integer-scale the seed pattern by that layer count, turning each source pixel into an N x N block.
    - Output only the enlarged patterns, arranged left-to-right by the marker-row order, with one black column between them and all tops aligned.

    Key insight:
    - The concentric frame depth is not decoration: the number of grey layers is the enlargement factor, while the bottom markers independently specify the display order.
    """
    g = np.asarray(grid, dtype=int)
    marker_row, order = _marker_row_and_order(g)

    scaled_patterns: list[np.ndarray] = []
    for color in order:
        seed_mask = g[:marker_row, :] == int(color)
        assert bool(seed_mask.any())
        bbox = _bbox_from_mask(seed_mask)
        r0, c0, r1, c1 = bbox
        pattern = np.where(seed_mask[r0 : r1 + 1, c0 : c1 + 1], int(color), BG)
        factor = _count_layers(g, bbox, marker_row)
        scaled_patterns.append(_scale_pattern(pattern, factor))

    out_h = max(pattern.shape[0] for pattern in scaled_patterns)
    out_w = sum(pattern.shape[1] for pattern in scaled_patterns) + len(scaled_patterns) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for pattern in scaled_patterns:
        ph, pw = pattern.shape
        out[:ph, col : col + pw] = pattern
        col += pw + 1
    return out


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < height:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < width:
        out.append((r, c + 1))
    return out


def _touches_all_sides(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    return 0 in rows and height - 1 in rows and 0 in cols and width - 1 in cols


def _random_seed_mask(height: int, width: int) -> np.ndarray | None:
    for _ in range(120):
        area = random.randint(max(2, (height * width) // 2), height * width)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, height, width):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) < 2 or not _touches_all_sides(cells, height, width):
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _build_group(color: int, layers: int, seed_mask: np.ndarray, inner_margin: int) -> tuple[np.ndarray, np.ndarray]:
    sh, sw = seed_mask.shape
    seed_offset = 2 * (layers - 1) + 1 + inner_margin
    gh = sh + 2 * seed_offset
    gw = sw + 2 * seed_offset
    group = np.full((gh, gw), BG, dtype=int)

    for layer in range(layers):
        inset = 2 * layer
        r0, c0 = inset, inset
        r1, c1 = gh - 1 - inset, gw - 1 - inset
        group[r0, c0 : c1 + 1] = FRAME
        group[r1, c0 : c1 + 1] = FRAME
        group[r0 : r1 + 1, c0] = FRAME
        group[r0 : r1 + 1, c1] = FRAME

    seed = np.where(seed_mask, int(color), BG)
    group_slice = group[seed_offset : seed_offset + sh, seed_offset : seed_offset + sw]
    group_slice[seed_mask] = int(color)
    return group, seed


def _compose_expected(order: list[int], scaled_by_color: dict[int, np.ndarray]) -> np.ndarray:
    patterns = [scaled_by_color[int(color)] for color in order]
    out_h = max(pattern.shape[0] for pattern in patterns)
    out_w = sum(pattern.shape[1] for pattern in patterns) + len(patterns) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for pattern in patterns:
        ph, pw = pattern.shape
        out[:ph, col : col + pw] = pattern
        col += pw + 1
    return out


def _candidate_row_partitions(count: int) -> list[list[list[int]]]:
    if count == 2:
        return [[[0, 1]], [[0], [1]]]
    assert count == 3
    return [[[0, 1, 2]], [[0, 1], [2]], [[0], [1, 2]]]


def _layout_groups(group_arrays: list[np.ndarray]) -> tuple[list[tuple[int, int]], int, int] | None:
    count = len(group_arrays)
    order = list(range(count))
    random.shuffle(order)
    shuffled = [group_arrays[i] for i in order]

    templates = _candidate_row_partitions(count)
    random.shuffle(templates)
    for rows in templates:
        h_gaps = [random.randint(1, 3) for _ in rows]
        v_gap = random.randint(1, 3)

        row_heights = [max(shuffled[i].shape[0] for i in row) for row in rows]
        row_widths = [
            sum(shuffled[i].shape[1] for i in row) + h_gaps[idx] * (len(row) - 1)
            for idx, row in enumerate(rows)
        ]
        body_h = sum(row_heights) + v_gap * (len(rows) - 1)
        body_w = max(row_widths)
        if body_h > 27 or body_w > 30:
            continue

        positions_shuffled: list[tuple[int, int]] = [(-1, -1)] * count
        top = 0
        for row_idx, row in enumerate(rows):
            row_h = row_heights[row_idx]
            row_w = row_widths[row_idx]
            left = random.randint(0, max(0, body_w - row_w))
            cursor = left
            for idx in row:
                gh, gw = shuffled[idx].shape
                r = top + random.randint(0, max(0, row_h - gh))
                positions_shuffled[idx] = (r, cursor)
                cursor += gw + h_gaps[row_idx]
            top += row_h + v_gap

        positions: list[tuple[int, int]] = [(-1, -1)] * count
        for shuffled_idx, original_idx in enumerate(order):
            positions[original_idx] = positions_shuffled[shuffled_idx]
        return positions, body_h, body_w
    return None


def _marker_columns(width: int, count: int) -> list[int] | None:
    for _ in range(80):
        cols = sorted(random.sample(range(width), count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.randint(2, 3)
    colors = random.sample(PALETTE, count)

    groups: list[np.ndarray] = []
    scaled_by_color: dict[int, np.ndarray] = {}
    for color in colors:
        layers = random.randint(1, 3)
        seed_h = random.randint(2, 3)
        seed_w = random.randint(2, 3)
        inner_margin = 0 if layers == 3 else random.randint(0, 1)
        seed_mask = _random_seed_mask(seed_h, seed_w)
        if seed_mask is None:
            return None
        group, seed = _build_group(color, layers, seed_mask, inner_margin)
        groups.append(group)
        scaled_by_color[int(color)] = _scale_pattern(seed, layers)

    expected_width = sum(scaled_by_color[color].shape[1] for color in colors) + count - 1
    expected_height = max(scaled_by_color[color].shape[0] for color in colors)
    if expected_width > 30 or expected_height > 30:
        return None

    layout = _layout_groups(groups)
    if layout is None:
        return None
    positions, body_h, body_w = layout

    marker_gap = random.randint(1, 2)
    top_margin = random.randint(0, 2)
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, max(0, 30 - (body_w + left_margin)))
    height = top_margin + body_h + marker_gap + 1
    width = left_margin + body_w + right_margin
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for group, (r0, c0) in zip(groups, positions):
        gh, gw = group.shape
        rr = top_margin + r0
        cc = left_margin + c0
        target = grid[rr : rr + gh, cc : cc + gw]
        assert not bool(np.any((target != BG) & (group != BG)))
        target[group != BG] = group[group != BG]

    marker_order = colors[:]
    random.shuffle(marker_order)
    marker_cols = _marker_columns(width, count)
    if marker_cols is None:
        return None
    marker_row = height - 1
    for color, col in zip(marker_order, marker_cols):
        grid[marker_row, col] = int(color)

    expected = _compose_expected(marker_order, scaled_by_color)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        grid, expected = case
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid nested-frame scaling puzzle")
