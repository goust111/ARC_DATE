import random

import numpy as np


BLACK = 0
GREY = 5
LEGEND_COUNTS = (4, 6, 8, 10)
LEGEND_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c), h, w):
            if not mask[nr, nc]:
                out[r, c] = True
                break
        else:
            if r in (0, h - 1) or c in (0, w - 1):
                out[r, c] = True
    return out


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    return mask & (~_boundary_mask(mask))


def _count_corners(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(int), 1)
    total = 0
    for r in range(padded.shape[0] - 1):
        for c in range(padded.shape[1] - 1):
            block = padded[r : r + 2, c : c + 2]
            s = int(block.sum())
            if s == 1 or s == 3:
                total += 1
            elif s == 2:
                if block[0, 0] and block[1, 1] and not block[0, 1] and not block[1, 0]:
                    total += 2
                elif block[0, 1] and block[1, 0] and not block[0, 0] and not block[1, 1]:
                    total += 2
    return total


def _legend_mapping(grid: np.ndarray) -> dict[int, int]:
    entries = []
    for color in sorted(int(x) for x in np.unique(grid)):
        if color in (BLACK, GREY):
            continue
        coords = np.argwhere(grid == int(color))
        assert coords.shape[0] == 2
        rows = sorted(int(x) for x in coords[:, 0])
        cols = sorted(int(x) for x in coords[:, 1])
        assert rows[1] == rows[0] + 1
        assert cols[0] == cols[1]
        entries.append((rows[0], color))
    entries.sort()
    assert len(entries) == len(LEGEND_COUNTS)
    return {count: color for count, (_row, color) in zip(LEGEND_COUNTS, entries)}


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey object's interior by matching its boundary-turn count to the right-side legend.

    Input structure:
    - The background is black.
    - Several 4-connected solid grey objects sit away from a vertical 2x1 color legend on the right.
    - The legend colors, read from top to bottom, encode the corner-count classes 4, 6, 8, and 10.

    Rule:
    - For each grey object, isolate its boundary pixels: object cells with at least one 4-neighbor outside the object.
    - Count the number of 90-degree turns on the object's orthogonal outline.
    - Use the legend entry for that count, recoloring only the object's non-boundary interior pixels; keep the boundary grey and leave the legend unchanged.

    Key insight:
    - The relevant feature is not the object's area or bbox but the complexity of its orthogonal silhouette, measured by its outline corner count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    corner_to_color = _legend_mapping(g)
    out = g.copy()
    for coords in _components_of_color(g, GREY):
        mask, (r0, c0) = _mask_from_coords(coords)
        corner_count = _count_corners(mask)
        assert corner_count in corner_to_color
        inside = _interior_mask(mask)
        rr, cc = np.where(inside)
        out[r0 + rr, c0 + cc] = int(corner_to_color[corner_count])
    return out


BASE_SHAPES = {
    4: (
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
    ),
    6: (
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    ),
    8: (
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
    ),
    10: (
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    ),
}


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    seen = set()
    variants = []
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            rot = np.rot90(base, k)
            key = rot.tobytes() + str(rot.shape).encode()
            if key not in seen:
                seen.add(key)
                variants.append(rot.copy())
    return variants


VARIANT_SHAPES = {
    count: tuple(
        variant
        for base in masks
        for variant in _unique_variants(base)
        if _count_corners(variant) == count
    )
    for count, masks in BASE_SHAPES.items()
}


def _expand_shape(base: np.ndarray) -> np.ndarray:
    for _ in range(80):
        row_reps = [random.randint(2, 3) for _ in range(base.shape[0])]
        col_reps = [random.randint(2, 3) for _ in range(base.shape[1])]
        mask = np.repeat(np.repeat(base, row_reps, axis=0), col_reps, axis=1)
        if _count_corners(mask) != _count_corners(base):
            continue
        if int(_interior_mask(mask).sum()) == 0:
            continue
        return mask
    raise RuntimeError("failed to expand shape")


def _sample_object(target_count: int) -> np.ndarray:
    base = random.choice(VARIANT_SHAPES[target_count])
    mask = _expand_shape(base)
    assert _count_corners(mask) == target_count
    return mask


def _pack_rows(objects: list[np.ndarray], width_budget: int) -> list[list[np.ndarray]] | None:
    rows: list[list[np.ndarray]] = []
    row_widths: list[int] = []
    min_gap = 2
    for obj in objects:
        ow = int(obj.shape[1])
        placed = False
        for idx in range(len(rows)):
            need = ow if not rows[idx] else min_gap + ow
            if row_widths[idx] + need <= width_budget:
                rows[idx].append(obj)
                row_widths[idx] += need
                placed = True
                break
        if not placed:
            if ow > width_budget:
                return None
            rows.append([obj])
            row_widths.append(ow)
    return rows if len(rows) <= 3 else None


def _draw_legend(grid: np.ndarray, color_map: dict[int, int], legend_c: int) -> None:
    h, _w = grid.shape
    legend_top = random.randint(0, h - 2 * len(LEGEND_COUNTS))
    for i, count in enumerate(LEGEND_COUNTS):
        color = int(color_map[count])
        r0 = legend_top + 2 * i
        grid[r0 : r0 + 2, legend_c] = color


def _gen_attempt() -> np.ndarray | None:
    counts = list(LEGEND_COUNTS)
    if random.random() < 0.6:
        counts.append(random.choice(LEGEND_COUNTS))
    random.shuffle(counts)

    try:
        objects = [_sample_object(count) for count in counts]
    except RuntimeError:
        return None

    random.shuffle(objects)
    width_budget = random.randint(16, 22)
    rows = _pack_rows(objects, width_budget)
    if rows is None:
        return None

    row_gaps = [random.randint(2, 3) for _ in range(max(0, len(rows) - 1))]
    row_heights = [max(int(obj.shape[0]) for obj in row) for row in rows]
    row_widths = []
    row_object_gaps = []
    for row in rows:
        gaps = [random.randint(2, 3) for _ in range(max(0, len(row) - 1))]
        row_object_gaps.append(gaps)
        width = sum(int(obj.shape[1]) for obj in row) + sum(gaps)
        row_widths.append(width)

    left_content_w = max(row_widths)
    left_margin = random.randint(1, 3)
    gap_to_legend = random.randint(2, 4)
    right_margin = random.randint(0, 2)
    min_width = left_margin + left_content_w + gap_to_legend + 1 + right_margin
    if min_width > 30:
        return None
    width = random.randint(min_width, 30)

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    content_h = sum(row_heights) + sum(row_gaps)
    min_height = max(2 * len(LEGEND_COUNTS), top_margin + content_h + bottom_margin)
    if min_height > 30:
        return None
    height = random.randint(min_height, 30)

    grid = np.full((height, width), BLACK, dtype=int)
    legend_colors = random.sample(LEGEND_COLOR_POOL, len(LEGEND_COUNTS))
    corner_to_color = {
        count: color for count, color in zip(LEGEND_COUNTS, legend_colors)
    }
    legend_c_min = left_margin + left_content_w + gap_to_legend
    legend_c_max = width - 1 - right_margin
    legend_c = random.randint(legend_c_min, legend_c_max)
    _draw_legend(grid, corner_to_color, legend_c)

    y = top_margin
    for row_idx, (row, row_h, row_w, gaps) in enumerate(
        zip(rows, row_heights, row_widths, row_object_gaps)
    ):
        max_start = left_margin + (left_content_w - row_w)
        x = random.randint(left_margin, max_start)
        for idx, obj in enumerate(row):
            oh, ow = obj.shape
            y_off = random.randint(0, row_h - oh)
            patch = np.full((oh, ow), BLACK, dtype=int)
            patch[obj] = GREY
            sub = grid[y + y_off : y + y_off + oh, x : x + ow]
            if np.any(sub[obj] != BLACK):
                return None
            sub[obj] = GREY
            x += ow
            if idx < len(gaps):
                x += gaps[idx]
        y += row_h
        if row_idx < len(row_gaps):
            y += row_gaps[row_idx]

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
