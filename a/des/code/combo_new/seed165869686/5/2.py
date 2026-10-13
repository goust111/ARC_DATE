from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ARC_COLORS = tuple(range(10))

LEGEND_LEN_RANGE = (3, 6)
SHAPE_COUNT_RANGE = (2, 4)
TOP_MARGIN_RANGE = (1, 3)
ROW_GAP_RANGE = (2, 4)
LEGEND_GAP_RANGE = (1, 3)
ROW_COUNT_PROB = 0.55
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _analyze_shape(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Return (hole_count, outer_boundary_mask) for a single 4-connected shape mask.
    Only boundary cells adjacent to the exterior background are marked.
    """
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    bg_mask = ~padded
    seen = np.zeros_like(bg_mask, dtype=bool)
    exterior = np.zeros_like(bg_mask, dtype=bool)
    holes = 0

    for r0 in range(h + 2):
        for c0 in range(w + 2):
            if not bg_mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                if r == 0 or c == 0 or r == h + 1 or c == w + 1:
                    touches_border = True
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h + 2 and 0 <= cc < w + 2 and bg_mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            if touches_border:
                for r, c in cells:
                    exterior[r, c] = True
            else:
                holes += 1

    boundary = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            pr = r + 1
            pc = c + 1
            for rr, cc in _neighbors4(pr, pc):
                if exterior[rr, cc]:
                    boundary[r, c] = True
                    break
    return holes, boundary


def _find_legend(grid: np.ndarray) -> tuple[int, int, list[int]]:
    row = grid[-1]
    cols = np.flatnonzero(row != BG)
    assert cols.size >= 1, "expected a non-empty legend on the bottom row"
    assert np.all(np.diff(cols) == 1), "legend colors must form one contiguous strip"
    c0 = int(cols[0])
    c1 = int(cols[-1]) + 1
    colors = [int(v) for v in row[c0:c1]]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    assert np.all(row[:c0] == BG) and np.all(row[c1:] == BG), "bottom row must contain only the legend strip"
    return c0, c1, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey shape's outer contour according to the number of enclosed holes, using the bottom legend strip as a lookup table.

    Input structure:
    - The background is black.
    - A single horizontal legend strip sits on the bottom row as one contiguous run of distinct one-cell-wide colors.
    - Above the strip are one or more 4-connected grey closed shapes; a shape may contain zero or more enclosed black holes.

    Rule:
    - Read the bottom legend from left to right, so the first legend color corresponds to hole count 0, the second to hole count 1, and so on.
    - For each grey shape, count its enclosed black holes.
    - Recolor only the shape's outer-boundary grey pixels, meaning grey pixels adjacent by 4-neighborhood to the exterior black background, with the legend color for that hole count.
    - Keep the shape's interior grey pixels grey, keep holes black, and leave the legend unchanged.

    Key insight:
    - The legend is indexed by topology rather than geometry: count holes first, then recolor only the exterior contour instead of the whole object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _c0, _c1, legend = _find_legend(g)

    upper = g[:-1]
    shape_colors = set(map(int, np.unique(upper))) - {BG}
    assert len(shape_colors) == 1, "expected exactly one non-background shape color above the legend"
    shape_color = next(iter(shape_colors))

    out = g.copy()
    for comp in _components(upper == shape_color):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True

        hole_count, boundary = _analyze_shape(mask)
        assert hole_count < len(legend), "hole count must be representable by the legend"
        color = int(legend[hole_count])

        for rr, cc in np.argwhere(boundary):
            out[r0 + int(rr), c0 + int(cc)] = color

    return out


def _sample_hole_positions(core_h: int, core_w: int, holes: int) -> list[tuple[int, int]] | None:
    if holes == 0:
        return []
    candidates = [(r, c) for r in range(1, core_h - 1) for c in range(1, core_w - 1)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in chosen):
            continue
        chosen.append((int(r), int(c)))
        if len(chosen) == holes:
            return chosen
    return None


def _shape_symmetry(mask: np.ndarray) -> np.ndarray:
    out = mask
    if random.random() < 0.5:
        out = np.rot90(out, k=random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _build_shape(hole_count: int) -> np.ndarray | None:
    choices = [
        (h, w)
        for h in range(4, 10)
        for w in range(4, 10)
        if (h - 2) * (w - 2) >= hole_count + 1
        and ((h - 2) * (w - 2) + 1) // 2 >= hole_count
    ]
    if not choices:
        return None

    for _ in range(80):
        core_h, core_w = random.choice(choices)
        top = random.choice([0, 0, 1, 1, 2])
        bottom = random.choice([0, 0, 1, 1, 2])
        left = random.choice([0, 0, 1, 1, 2])
        right = random.choice([0, 0, 1, 1, 2])

        mask = np.zeros((core_h + top + bottom, core_w + left + right), dtype=bool)
        r0 = top
        c0 = left
        mask[r0 : r0 + core_h, c0 : c0 + core_w] = True

        if top:
            width = random.randint(max(2, core_w // 2), core_w)
            start = random.randint(0, core_w - width)
            mask[:top, c0 + start : c0 + start + width] = True
        if bottom:
            width = random.randint(max(2, core_w // 2), core_w)
            start = random.randint(0, core_w - width)
            mask[r0 + core_h :, c0 + start : c0 + start + width] = True
        if left:
            height = random.randint(max(2, core_h // 2), core_h)
            start = random.randint(0, core_h - height)
            mask[r0 + start : r0 + start + height, :left] = True
        if right:
            height = random.randint(max(2, core_h // 2), core_h)
            start = random.randint(0, core_h - height)
            mask[r0 + start : r0 + start + height, c0 + core_w :] = True

        holes = _sample_hole_positions(core_h, core_w, hole_count)
        if holes is None:
            continue
        for rr, cc in holes:
            mask[r0 + rr, c0 + cc] = False

        mask = _shape_symmetry(mask)
        comps = _components(mask)
        if len(comps) != 1:
            continue
        found_holes, boundary = _analyze_shape(mask)
        if found_holes != hole_count:
            continue
        if int(np.sum(mask & ~boundary)) == 0:
            continue
        return mask
    return None


def _choose_hole_counts(legend_len: int, shape_count: int) -> list[int]:
    counts: list[int] = [0]
    if shape_count >= 2 and legend_len >= 2:
        counts.append(random.randint(1, legend_len - 1))

    while len(counts) < shape_count:
        if random.random() < 0.5 and len(set(counts)) < legend_len:
            available = [v for v in range(legend_len) if v not in counts]
            counts.append(random.choice(available))
        else:
            counts.append(random.randint(0, legend_len - 1))

    random.shuffle(counts)
    return counts


def _layout_rows(shapes: list[np.ndarray]) -> list[list[np.ndarray]]:
    order = shapes[:]
    random.shuffle(order)
    if len(order) >= 3 and random.random() < ROW_COUNT_PROB:
        cut = random.randint(1, len(order) - 1)
        return [order[:cut], order[cut:]]
    return [order]


def _place_shapes(grid: np.ndarray, rows: list[list[np.ndarray]], top_margin: int, row_gap: int) -> None:
    row_heights = [max(shape.shape[0] for shape in row) for row in rows]
    row_widths: list[int] = []
    row_offsets: list[list[int]] = []

    for row in rows:
        gaps = [random.randint(2, 4) for _ in range(max(0, len(row) - 1))]
        offsets = [0]
        for i, shape in enumerate(row[:-1]):
            offsets.append(offsets[-1] + shape.shape[1] + gaps[i])
        row_offsets.append(offsets)
        row_widths.append(offsets[-1] + row[-1].shape[1])

    for row_w in row_widths:
        assert row_w <= grid.shape[1] - 2

    y = int(top_margin)
    for row, row_h, row_w, offsets in zip(rows, row_heights, row_widths, row_offsets, strict=False):
        start_x = random.randint(1, grid.shape[1] - row_w - 1)
        for shape, offset in zip(row, offsets, strict=False):
            jitter = random.randint(0, row_h - shape.shape[0])
            r0 = y + jitter
            c0 = start_x + offset
            region = grid[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]]
            region[shape] = GREY
        y += row_h + row_gap


def _make_sample() -> np.ndarray | None:
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    shape_count = random.randint(*SHAPE_COUNT_RANGE)
    hole_counts = _choose_hole_counts(legend_len, shape_count)
    shapes = []
    for count in hole_counts:
        shape = _build_shape(count)
        if shape is None:
            return None
        shapes.append(shape)

    rows = _layout_rows(shapes)
    row_heights = [max(shape.shape[0] for shape in row) for row in rows]
    row_widths = [sum(shape.shape[1] for shape in row) + max(0, len(row) - 1) * 4 for row in rows]

    top_margin = random.randint(*TOP_MARGIN_RANGE)
    row_gap = random.randint(*ROW_GAP_RANGE)
    legend_gap = random.randint(*LEGEND_GAP_RANGE)

    grid_h = top_margin + sum(row_heights) + max(0, len(rows) - 1) * row_gap + legend_gap + 1
    grid_w = max(max(row_widths) + 2, legend_len + 2) + random.randint(0, 4)
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _place_shapes(grid, rows, top_margin, row_gap)

    legend_colors = random.sample([c for c in ARC_COLORS if c not in (BG, GREY)], legend_len)
    start = random.randint(0, grid_w - legend_len)
    grid[-1, start : start + legend_len] = np.array(legend_colors, dtype=int)

    upper = grid[:-1]
    if np.any(upper[-legend_gap:, :] != BG):
        return None
    if len(_components(upper == GREY)) != shape_count:
        return None
    for comp in _components(upper == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        holes, boundary = _analyze_shape(mask)
        if holes >= legend_len:
            return None
        if int(np.sum(mask & ~boundary)) == 0:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        sample = _make_sample()
        if sample is None:
            continue
        out = main(sample)
        if out.shape != sample.shape:
            continue
        if np.array_equal(out, sample):
            continue
        return sample
    raise RuntimeError("failed to generate a valid sample")
