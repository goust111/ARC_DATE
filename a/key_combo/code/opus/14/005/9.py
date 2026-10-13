from __future__ import annotations

import random
from collections import deque

import numpy as np


PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
LAYOUTS = ((3, 4), (4, 3))
REGION_SIZE_RANGE = (4, 6)
MAX_ATTEMPTS = 400

SHAPES_BY_WIDTH = {
    1: [
        ((0, 0), (1, 0)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)),
    ],
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (0, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ],
    4: [
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)),
    ],
}

ROW_COVER_COLS = {
    (3, 4): (0, 1, 2),
    (4, 3): (0, 1, 2, 0),
}

COL_COVER_ROWS = {
    (3, 4): {0: (1, 2), 1: (0, 2), 2: (0, 1), 3: (0, 1)},
    (4, 3): {0: (1, 2), 1: (0, 2), 2: (0, 3)},
}


def _full_zero_dividers(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == 0)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 0)]


def _intervals(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for divider in sorted(dividers):
        if divider > start:
            spans.append((start, divider))
        start = divider + 1
    if start < length:
        spans.append((start, length))
    return spans


def _bbox_width(coords: np.ndarray) -> int:
    return int(coords[:, 1].max() - coords[:, 1].min() + 1)


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _normalize(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _flip_h(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_c = max(c for _, c in cells)
    return _normalize(tuple((r, max_c - c) for r, c in cells))


def _flip_v(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_r = max(r for r, _ in cells)
    return _normalize(tuple((max_r - r, c) for r, c in cells))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return (
        max(r for r, _ in cells) + 1,
        max(c for _, c in cells) + 1,
    )


def _shape_variants(cells: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = {
        _normalize(cells),
        _flip_h(cells),
        _flip_v(cells),
        _flip_v(_flip_h(cells)),
    }
    return list(variants)


def _candidate_shapes(region_h: int, region_w: int, bbox_w: int, area: int | None = None) -> list[tuple[tuple[int, int], ...]]:
    candidates: list[tuple[tuple[int, int], ...]] = []
    for base in SHAPES_BY_WIDTH[bbox_w]:
        if area is not None and len(base) != area:
            continue
        for variant in _shape_variants(base):
            h, w = _shape_dims(variant)
            if h <= region_h and w <= region_w:
                candidates.append(variant)
    unique = list(dict.fromkeys(candidates))
    return unique


def _available_areas(region_h: int, region_w: int, bbox_w: int) -> set[int]:
    return {len(shape) for shape in _candidate_shapes(region_h, region_w, bbox_w)}


def _place_shape(region_h: int, region_w: int, bbox_w: int, area: int | None = None) -> list[tuple[int, int]] | None:
    candidates = _candidate_shapes(region_h, region_w, bbox_w, area)
    if not candidates:
        return None
    shape = random.choice(candidates)
    shape_h, shape_w = _shape_dims(shape)
    row_margin = 1 if region_h - shape_h >= 2 else 0
    col_margin = 1 if region_w - shape_w >= 2 else 0
    top_min = row_margin
    top_max = region_h - shape_h - row_margin
    left_min = col_margin
    left_max = region_w - shape_w - col_margin
    if top_max < top_min:
        top_min, top_max = 0, region_h - shape_h
    if left_max < left_min:
        left_min, left_max = 0, region_w - shape_w
    top = random.randint(top_min, top_max)
    left = random.randint(left_min, left_max)
    return [(top + r, left + c) for r, c in shape]


def _make_row_cover(region_h: int, region_w: int) -> list[tuple[int, int]]:
    col = random.randint(0, region_w - 1)
    return [(r, col) for r in range(region_h)]


def _place_col_cover(region_h: int, region_w: int, side: int, area: int | None = None) -> list[tuple[int, int]] | None:
    if region_w < 4:
        return None
    left = 0 if side == 0 or region_w == 4 else region_w - 4
    candidates = _candidate_shapes(region_h, region_w, 4, area)
    candidates = [shape for shape in candidates if max(c for _, c in shape) + 1 == 4]
    if not candidates:
        return None
    shape = random.choice(candidates)
    shape_h, _shape_w = _shape_dims(shape)
    top = random.randint(0, region_h - shape_h)
    return [(top + r, left + c) for r, c in shape]


def _expected_output(colors: np.ndarray, widths: np.ndarray, areas: np.ndarray) -> np.ndarray:
    row_count, col_count = colors.shape
    out = np.zeros((row_count, col_count), dtype=int)
    best_area = np.full((row_count, col_count), -1, dtype=int)
    best_order = np.full((row_count, col_count), row_count * col_count + 1, dtype=int)
    order = 0
    for r in range(row_count):
        for c in range(col_count):
            dst_c = int((c + widths[r, c]) % col_count)
            area = int(areas[r, c])
            if area > best_area[r, dst_c] or (area == best_area[r, dst_c] and order < best_order[r, dst_c]):
                out[r, dst_c] = int(colors[r, c])
                best_area[r, dst_c] = area
                best_order[r, dst_c] = order
            order += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each partition to one pixel, then shift by the source object's bounding-box width.

    Input structure:
    - The grid is split by full black divider rows and columns into a 3x4 or 4x3 array of rectangular regions.
    - Each region contains exactly one monochromatic 4-connected foreground object on black background.
    - The object's original bounding-box width and area are encoded by its shape inside that region.

    Rule:
    - Detect the full black divider lines and recover the rectangular region layout.
    - For each region, read the object's color, bounding-box width, and area, then compress that whole region to one colored pixel.
    - In the compressed grid, shift each pixel right by its own original bounding-box width with row-wise wraparound.
    - If multiple shifted pixels land on one cell, keep the one with larger original area; if the areas tie, keep the earlier region in row-major order.

    Key insight:
    - The compressed grid keeps only one pixel per region, so the needed shift amount must be extracted before compression from the object's destroyed fine-scale width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans = _intervals(g.shape[0], _full_zero_dividers(g, axis=0))
    col_spans = _intervals(g.shape[1], _full_zero_dividers(g, axis=1))
    assert row_spans and col_spans

    out = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    best_area = np.full(out.shape, -1, dtype=int)
    best_order = np.full(out.shape, out.size + 1, dtype=int)

    order = 0
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            region = g[r0:r1, c0:c1]
            coords = np.argwhere(region != 0)
            assert coords.size > 0, "every region must contain one foreground object"
            colors = np.unique(region[region != 0])
            assert len(colors) == 1, "every region must be monochromatic"
            area = int(coords.shape[0])
            width = _bbox_width(coords)
            dst_c = int((cc + width) % out.shape[1])
            if area > best_area[rr, dst_c] or (area == best_area[rr, dst_c] and order < best_order[rr, dst_c]):
                out[rr, dst_c] = int(colors[0])
                best_area[rr, dst_c] = area
                best_order[rr, dst_c] = order
            order += 1
    return out


def _color_grid(row_count: int, col_count: int) -> np.ndarray:
    total = row_count * col_count
    pool: list[int] = []
    while len(pool) < total:
        chunk = list(PALETTE)
        random.shuffle(chunk)
        pool.extend(chunk)
    return np.array(pool[:total], dtype=int).reshape(row_count, col_count)


def _collision_options(col_count: int, max_widths: list[int]) -> list[tuple[int, int, int, int, int]]:
    options = []
    for c1 in range(col_count):
        for c2 in range(c1 + 1, col_count):
            for w1 in range(1, max_widths[c1] + 1):
                for w2 in range(1, max_widths[c2] + 1):
                    dst1 = (c1 + w1) % col_count
                    if dst1 == (c2 + w2) % col_count:
                        options.append((c1, w1, c2, w2, dst1))
    return options


def _row_has_collision(widths_row: np.ndarray) -> bool:
    col_count = widths_row.shape[0]
    dests = [(c + int(widths_row[c])) % col_count for c in range(col_count)]
    return len(set(dests)) < len(dests)


def _gen_attempt() -> np.ndarray | None:
    row_count, col_count = random.choice(LAYOUTS)
    region_heights = [random.randint(*REGION_SIZE_RANGE) for _ in range(row_count)]
    region_widths = [random.randint(*REGION_SIZE_RANGE) for _ in range(col_count)]

    grid_h = sum(region_heights) + (row_count - 1)
    grid_w = sum(region_widths) + (col_count - 1)
    if grid_h > 30 or grid_w > 30:
        return None

    roles: list[list[tuple[str, int]]] = [[("random", -1) for _ in range(col_count)] for _ in range(row_count)]
    for r, c in enumerate(ROW_COVER_COLS[(row_count, col_count)]):
        roles[r][c] = ("row_cover", 0)
    for c in range(col_count):
        cover_rows = COL_COVER_ROWS[(row_count, col_count)][c]
        needed = 1 if region_widths[c] == 4 else 2
        for side in range(needed):
            r = cover_rows[side]
            if roles[r][c][0] != "random":
                return None
            roles[r][c] = ("col_cover", side)

    widths = np.zeros((row_count, col_count), dtype=int)
    for r in range(row_count):
        for c in range(col_count):
            role, _side = roles[r][c]
            if role == "row_cover":
                widths[r, c] = 1
            elif role == "col_cover":
                widths[r, c] = 4
            else:
                widths[r, c] = random.randint(1, min(4, region_widths[c]))

    want_wrap = random.random() < 0.65
    want_collision = random.random() < 0.65
    want_tie = want_collision and random.random() < 0.45
    collision_meta: tuple[int, int, int, int, int, bool] | None = None

    def allowed_widths(r: int, c: int) -> list[int]:
        role, _side = roles[r][c]
        if role == "row_cover":
            return [1]
        if role == "col_cover":
            return [4]
        return list(range(1, min(4, region_widths[c]) + 1))

    if want_wrap:
        wrap_choices = [
            (r, c, w)
            for r in range(row_count)
            for c in range(col_count)
            for w in allowed_widths(r, c)
            if c + w >= col_count
        ]
        if wrap_choices:
            r, c, w = random.choice(wrap_choices)
            widths[r, c] = w

    if want_collision:
        row_choices = []
        for r in range(row_count):
            options = []
            for c1 in range(col_count):
                for c2 in range(c1 + 1, col_count):
                    if roles[r][c1][0] == "row_cover" or roles[r][c2][0] == "row_cover":
                        continue
                    for w1 in allowed_widths(r, c1):
                        for w2 in allowed_widths(r, c2):
                            dst1 = (c1 + w1) % col_count
                            if dst1 == (c2 + w2) % col_count:
                                options.append((c1, w1, c2, w2, dst1))
            if options:
                row_choices.append((r, options))
        if row_choices:
            row, options = random.choice(row_choices)
            c1, w1, c2, w2, dst = random.choice(options)
            widths[row, c1] = w1
            widths[row, c2] = w2
            collision_meta = (row, c1, c2, dst, random.randint(0, 1), want_tie)
        else:
            want_collision = False

    flat_widths = set(int(v) for v in widths.ravel())
    if len(flat_widths) < 2:
        cells = [(r, c) for r in range(row_count) for c in range(col_count)]
        random.shuffle(cells)
        for r, c in cells:
            choices = [w for w in allowed_widths(r, c) if w != widths[r, c]]
            if choices:
                widths[r, c] = random.choice(choices)
                break

    colors = _color_grid(row_count, col_count)
    if collision_meta is not None:
        row, c1, c2, _dst, _branch, _want_tie = collision_meta
        if colors[row, c1] == colors[row, c2]:
            colors[row, c2] = random.choice([color for color in PALETTE if color != int(colors[row, c1])])

    area_targets = np.full((row_count, col_count), -1, dtype=int)
    if collision_meta is not None:
        row, c1, c2, _dst, larger_branch, is_tie = collision_meta
        if roles[row][c1][0] == "row_cover" or roles[row][c2][0] == "row_cover":
            return None
        areas1 = _available_areas(region_heights[row], region_widths[c1], int(widths[row, c1]))
        areas2 = _available_areas(region_heights[row], region_widths[c2], int(widths[row, c2]))
        if is_tie:
            common = sorted(areas1 & areas2)
            if not common:
                return None
            area = random.choice(common)
            area_targets[row, c1] = area
            area_targets[row, c2] = area
        else:
            if larger_branch == 0:
                pairs = [(a1, a2) for a1 in areas1 for a2 in areas2 if a1 > a2]
            else:
                pairs = [(a1, a2) for a1 in areas1 for a2 in areas2 if a1 < a2]
            if not pairs:
                return None
            a1, a2 = random.choice(pairs)
            area_targets[row, c1] = a1
            area_targets[row, c2] = a2

    row_starts = []
    cursor = 0
    for height in region_heights:
        row_starts.append(cursor)
        cursor += height + 1
    col_starts = []
    cursor = 0
    for width in region_widths:
        col_starts.append(cursor)
        cursor += width + 1

    planned_row_dividers = {row_starts[r] + region_heights[r] for r in range(row_count - 1)}
    planned_col_dividers = {col_starts[c] + region_widths[c] for c in range(col_count - 1)}
    for _ in range(20):
        grid = np.zeros((grid_h, grid_w), dtype=int)
        actual_widths = np.zeros_like(widths)
        actual_areas = np.zeros_like(widths)

        ok = True
        for r in range(row_count):
            for c in range(col_count):
                target_area = None if area_targets[r, c] < 0 else int(area_targets[r, c])
                role, side = roles[r][c]
                if role == "row_cover":
                    placed = _make_row_cover(region_heights[r], region_widths[c])
                elif role == "col_cover":
                    placed = _place_col_cover(region_heights[r], region_widths[c], side, target_area)
                else:
                    placed = _place_shape(region_heights[r], region_widths[c], int(widths[r, c]), target_area)
                if placed is None:
                    ok = False
                    break
                base_r = row_starts[r]
                base_c = col_starts[c]
                color = int(colors[r, c])
                for rr, cc in placed:
                    grid[base_r + rr, base_c + cc] = color

                local = np.array(placed, dtype=int)
                actual_widths[r, c] = _bbox_width(local)
                actual_areas[r, c] = int(local.shape[0])

                region = grid[base_r : base_r + region_heights[r], base_c : base_c + region_widths[c]]
                if _count_components(region != 0) != 1:
                    ok = False
                    break
            if not ok:
                break

        if not ok:
            continue
        if len(set(int(v) for v in actual_widths.ravel())) < 2:
            continue
        if any(not np.all(grid[row_starts[r] + region_heights[r], :] == 0) for r in range(row_count - 1)):
            continue
        if any(not np.all(grid[:, col_starts[c] + region_widths[c]] == 0) for c in range(col_count - 1)):
            continue
        if set(_full_zero_dividers(grid, axis=0)) != planned_row_dividers:
            continue
        if set(_full_zero_dividers(grid, axis=1)) != planned_col_dividers:
            continue

        expected = _expected_output(colors, actual_widths, actual_areas)
        if np.array_equal(expected, colors):
            continue
        if not np.array_equal(main(grid), expected):
            continue

        has_wrap = any(c + int(actual_widths[r, c]) >= col_count for r in range(row_count) for c in range(col_count))
        if want_wrap and not has_wrap:
            continue

        has_collision = any(_row_has_collision(actual_widths[r]) for r in range(row_count))
        if want_collision and not has_collision:
            continue

        if collision_meta is not None:
            row, c1, c2, dst, larger_branch, is_tie = collision_meta
            dst1 = (c1 + int(actual_widths[row, c1])) % col_count
            dst2 = (c2 + int(actual_widths[row, c2])) % col_count
            if dst1 != dst or dst2 != dst:
                continue
            area1 = int(actual_areas[row, c1])
            area2 = int(actual_areas[row, c2])
            if is_tie:
                if area1 != area2 or int(expected[row, dst]) != int(colors[row, c1]):
                    continue
            elif larger_branch == 0:
                if not (area1 > area2 and int(expected[row, dst]) == int(colors[row, c1])):
                    continue
            else:
                if not (area2 > area1 and int(expected[row, dst]) == int(colors[row, c2])):
                    continue

        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
