import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

# Connected 3- to 5-cell shapes, unique up to rotation.
SHAPE_LIBRARY = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
]

GEN_ATTEMPTS = 400


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert rows.any() and cols.any(), "shape mask must be non-empty"
    return mask[np.ix_(rows, cols)]


def _shape_signature(mask: np.ndarray) -> tuple:
    trimmed = _trim_mask(np.asarray(mask, dtype=bool))
    variants = []
    for k in range(4):
        rot = _trim_mask(np.rot90(trimmed, k))
        variants.append((rot.shape, tuple(int(x) for x in rot.astype(np.uint8).ravel())))
    return min(variants)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _intervals_from_lines(lines: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(int(x) for x in lines)
    spans = []
    for a, b in zip(ordered, ordered[1:]):
        if b - a >= 2:
            spans.append((a + 1, b - 1))
    return spans


def _extract_legend_mapping(band: np.ndarray) -> dict[tuple, int]:
    grey_comps = []
    for coords in _components(band == GREY):
        mask, bbox = _coords_to_mask(coords)
        grey_comps.append((bbox[1], bbox[0], _shape_signature(mask)))

    dots = []
    for r in range(band.shape[0]):
        for c in range(band.shape[1]):
            color = int(band[r, c])
            if color not in (BLACK, GREY):
                dots.append((c, r, color))

    assert grey_comps, "legend band must contain grey shape components"
    assert len(grey_comps) == len(dots), "legend shapes and color dots must pair one-to-one"

    grey_comps.sort()
    dots.sort()

    mapping: dict[tuple, int] = {}
    for (_, _, sig), (_, _, color) in zip(grey_comps, dots):
        mapping[sig] = int(color)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each region's key shape to the bottom legend up to rotation, then fill
    that region's black interior with the legend's color and remove the legend band.
    """
    g = np.asarray(grid, dtype=int)

    grey_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert grey_rows, "input must contain grey partition rows"
    split_row = max(grey_rows)
    assert split_row < g.shape[0] - 1, "legend band must exist below the partition grid"

    panel = g[: split_row + 1].copy()
    legend = g[split_row + 1 :]
    mapping = _extract_legend_mapping(legend)

    row_lines = [r for r in range(panel.shape[0]) if np.all(panel[r] == GREY)]
    col_lines = [c for c in range(panel.shape[1]) if np.all(panel[:, c] == GREY)]
    row_spans = _intervals_from_lines(row_lines)
    col_spans = _intervals_from_lines(col_lines)
    assert row_spans and col_spans, "panel must contain at least one rectangular region"

    out = panel.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = out[r0 : r1 + 1, c0 : c1 + 1]
            key_mask = (region != BLACK) & (region != GREY)
            comps = _components(key_mask)
            assert len(comps) == 1, "each region must contain exactly one connected key"
            key_shape, _ = _coords_to_mask(comps[0])
            fill_color = mapping[_shape_signature(key_shape)]
            region[region == BLACK] = fill_color
    return out


def _rotated(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, random.randrange(4))


def _place_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = canvas[top : top + h, left : left + w]
    view[mask] = int(color)


def _build_partition_grid(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    total_h = sum(row_heights) + len(row_heights) + 1
    total_w = sum(col_widths) + len(col_widths) + 1
    grid = np.full((total_h, total_w), BLACK, dtype=int)

    row_lines = [0]
    cur = 0
    for h in row_heights:
        cur += h + 1
        row_lines.append(cur)
    col_lines = [0]
    cur = 0
    for w in col_widths:
        cur += w + 1
        col_lines.append(cur)

    grid[row_lines, :] = GREY
    grid[:, col_lines] = GREY
    return grid, _intervals_from_lines(row_lines), _intervals_from_lines(col_lines)


def _legend_layout(width: int, shapes: list[np.ndarray], legend_h: int) -> tuple[list[int], list[int]] | None:
    slot_widths = [int(shape.shape[1]) + 2 for shape in shapes]
    gap_count = len(shapes) + 1
    needed = sum(slot_widths) + gap_count
    if needed > width:
        return None

    gaps = [1] * gap_count
    extra = width - needed
    for _ in range(extra):
        gaps[random.randrange(gap_count)] += 1

    starts = []
    x = gaps[0]
    for i, slot_w in enumerate(slot_widths):
        starts.append(x)
        x += slot_w + gaps[i + 1]

    for shape, start, slot_w in zip(shapes, starts, slot_widths):
        assert slot_w >= shape.shape[1] + 2
        assert legend_h >= shape.shape[0] + 2
        _ = start
    return starts, slot_widths


def _gen_attempt() -> np.ndarray | None:
    region_rows = random.randint(2, 4)
    region_cols = random.randint(2, 4)
    region_count = region_rows * region_cols

    legend_count = random.randint(2, min(5, len(SHAPE_LIBRARY), region_count))
    shape_ids = random.sample(range(len(SHAPE_LIBRARY)), legend_count)
    fill_colors = random.sample(COLOR_POOL, legend_count)
    fill_by_shape = {shape_id: color for shape_id, color in zip(shape_ids, fill_colors)}

    row_heights = [random.randint(5, 8) for _ in range(region_rows)]
    col_widths = [random.randint(5, 8) for _ in range(region_cols)]
    top, row_spans, col_spans = _build_partition_grid(row_heights, col_widths)

    legend_h = random.randint(5, 6)
    legend_shapes = [_rotated(SHAPE_LIBRARY[shape_id]) for shape_id in shape_ids]
    layout = _legend_layout(top.shape[1], legend_shapes, legend_h)
    if layout is None:
        return None
    legend_starts, slot_widths = layout

    assignments = shape_ids[:]
    while len(assignments) < region_count:
        assignments.append(random.choice(shape_ids))
    random.shuffle(assignments)

    expected = top.copy()
    assignment_idx = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region_h = r1 - r0 + 1
            region_w = c1 - c0 + 1
            shape_id = assignments[assignment_idx]
            assignment_idx += 1
            key_mask = _rotated(SHAPE_LIBRARY[shape_id])
            key_h, key_w = key_mask.shape
            if region_h < key_h + 2 or region_w < key_w + 2:
                return None

            top_margin = region_h - key_h - 1
            left_margin = region_w - key_w - 1
            key_top = r0 + random.randint(1, top_margin)
            key_left = c0 + random.randint(1, left_margin)
            key_color = random.choice(COLOR_POOL)

            _place_mask(top, key_mask, key_top, key_left, key_color)
            _place_mask(expected, key_mask, key_top, key_left, key_color)

            fill_color = fill_by_shape[shape_id]
            region_view = expected[r0 : r1 + 1, c0 : c1 + 1]
            region_view[region_view == BLACK] = fill_color

    legend = np.full((legend_h, top.shape[1]), BLACK, dtype=int)
    for shape_id, fill_color, shape_mask, start_x, slot_w in zip(
        shape_ids, fill_colors, legend_shapes, legend_starts, slot_widths
    ):
        shape_h, shape_w = shape_mask.shape
        row_max = legend_h - shape_h - 2
        if row_max < 0:
            return None
        shape_top = random.randint(0, row_max)
        shape_left = start_x + random.randint(0, slot_w - shape_w)
        dot_col = shape_left + shape_w // 2

        _place_mask(legend, shape_mask, shape_top, shape_left, GREY)
        legend[legend_h - 1, dot_col] = fill_color

        # Keep each legend item's dot vertically below its own slot.
        if not (start_x <= dot_col < start_x + slot_w):
            return None
        _ = shape_id

    grid = np.vstack([top, legend])
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
