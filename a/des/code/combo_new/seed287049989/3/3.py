from __future__ import annotations

import itertools
import random

import numpy as np


SEPARATOR_COLOR = 5
MARKER_COLOR = 2
ALL_COLORS = tuple(range(10))

MIN_REGION_SIZE = 7
MAX_REGION_SIZE = 10
RECT_COUNT_RANGE = (2, 4)
GRID_FACTORS = ((2, 2), (2, 3), (3, 2), (3, 3))

BOX_RETRIES = 200
GENERATION_RETRIES = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _render_rectangles(
    shape: tuple[int, int],
    bg: int,
    order: tuple[int, ...] | list[int],
    boxes: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in order:
        r0, c0, r1, c1 = boxes[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _constant_line_indices(grid: np.ndarray, axis: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    length = grid.shape[axis]
    for idx in range(length):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if np.all(line == line[0]):
            groups.setdefault(int(line[0]), []).append(idx)
    return groups


def _bands(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    bands: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        bands.append((start, prev))
        start = prev = idx
    bands.append((start, prev))
    return bands


def _segments_from_bands(length: int, bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    prev = 0
    for start, end in bands:
        if prev <= start - 1:
            segments.append((prev, start - 1))
        prev = end + 1
    if prev <= length - 1:
        segments.append((prev, length - 1))
    return segments


def _find_partition(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    h, w = grid.shape
    rows_by_color = _constant_line_indices(grid, axis=0)
    cols_by_color = _constant_line_indices(grid, axis=1)

    best = None
    for color in sorted(rows_by_color.keys() & cols_by_color.keys()):
        row_bands = _bands(rows_by_color[color])
        col_bands = _bands(cols_by_color[color])
        row_segments = _segments_from_bands(h, row_bands)
        col_segments = _segments_from_bands(w, col_bands)
        if len(row_segments) < 2 or len(col_segments) < 2:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_segments}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_segments}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        score = (
            len(row_segments) * len(col_segments),
            -sum(end - start + 1 for start, end in row_bands + col_bands),
        )
        candidate = (
            score,
            int(color),
            row_segments,
            col_segments,
            [idx for start, end in row_bands for idx in range(start, end + 1)],
            [idx for start, end in col_bands for idx in range(start, end + 1)],
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None, "expected a unique grid partition"
    _, color, row_segments, col_segments, row_lines, col_lines = best
    return color, row_segments, col_segments, row_lines, col_lines


def _background_color(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> int:
    mask = np.ones(grid.shape, dtype=bool)
    mask[row_lines, :] = False
    mask[:, col_lines] = False
    values = grid[mask]
    counts = np.bincount(values, minlength=10)
    return int(counts.argmax())


def _corner_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, 0] = True
    mask[0, -1] = True
    mask[-1, 0] = True
    mask[-1, -1] = True
    return mask


def _find_template_region(
    grid: np.ndarray,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    bg: int,
    separator: int,
) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for ridx, (r0, r1) in enumerate(row_segments):
        for cidx, (c0, c1) in enumerate(col_segments):
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            corners = {
                int(region[0, 0]),
                int(region[0, -1]),
                int(region[-1, 0]),
                int(region[-1, -1]),
            }
            if len(corners) != 1:
                continue
            marker = next(iter(corners))
            if marker in (bg, separator):
                continue
            matches.append((ridx, cidx))
    assert len(matches) == 1, "expected exactly one marked template region"
    return matches[0]


def _recover_boxes_and_reversed(
    template: np.ndarray,
    bg: int,
    marker: int,
) -> np.ndarray:
    observed = template.copy()
    observed[_corner_mask(template.shape)] = bg

    colors = [int(color) for color in np.unique(observed) if int(color) not in (bg, marker)]
    assert colors, "template must contain at least one rectangle"
    assert len(colors) <= 6, "too many permutations to search exhaustively"

    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        coords = np.argwhere(observed == color)
        assert coords.size > 0, "every rectangle color must remain visible"
        boxes[color] = _bbox(coords)

    reversed_renders: dict[bytes, np.ndarray] = {}
    for order in itertools.permutations(colors):
        rendered = _render_rectangles(observed.shape, bg, order, boxes)
        if not np.array_equal(rendered, observed):
            continue
        reversed_render = _render_rectangles(observed.shape, bg, order[::-1], boxes)
        reversed_renders[reversed_render.tobytes()] = reversed_render

    assert reversed_renders, "template rectangles must admit at least one valid layer order"
    assert len(reversed_renders) == 1, "reversed result must be unambiguous"
    return next(iter(reversed_renders.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the marked template cell and stamp its rectangles into every empty cell with the layer order reversed.

    Input structure:
    - The dominant background is split into equal rectangular subregions by full grey separator rows and columns.
    - Exactly one subregion is a template: it contains several differently colored axis-aligned filled rectangles with partial overlap, and its four corners are red marker pixels.
    - Every other subregion is initially blank background.

    Rule:
    - Detect the separator lattice and identify the unique subregion whose four corners share the red marker color.
    - Treat each visible non-background, non-marker color in that template as one rectangle; recover its full footprint from the visible color bbox, then keep only layer orders whose rectangle redraw reproduces the observed template.
    - Reverse that valid layer order and redraw the same rectangles into every blank stamp subregion, while leaving the template subregion and the separators unchanged.

    Key insight:
    - The corner markers identify where to read the hidden scene, and once each color's full rectangle bbox is recovered, the true stack order is exactly the order whose forward rendering matches the visible occlusion pattern.
    """
    g = np.asarray(grid, dtype=int)
    separator, row_segments, col_segments, row_lines, col_lines = _find_partition(g)
    bg = _background_color(g, row_lines, col_lines)

    template_ridx, template_cidx = _find_template_region(g, row_segments, col_segments, bg, separator)
    tr0, tr1 = row_segments[template_ridx]
    tc0, tc1 = col_segments[template_cidx]
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    marker = int(template[0, 0])
    reversed_stamp = _recover_boxes_and_reversed(template, bg, marker)

    out = g.copy()
    for ridx, (r0, r1) in enumerate(row_segments):
        for cidx, (c0, c1) in enumerate(col_segments):
            if ridx == template_ridx and cidx == template_cidx:
                continue
            out[r0 : r1 + 1, c0 : c1 + 1] = reversed_stamp
    return out


def _region_origin(region_row: int, region_col: int, region_h: int, region_w: int) -> tuple[int, int]:
    return region_row * (region_h + 1), region_col * (region_w + 1)


def _assemble_grid(
    region_rows: int,
    region_cols: int,
    region_h: int,
    region_w: int,
    bg: int,
    template_pos: tuple[int, int],
    template: np.ndarray,
    stamp: np.ndarray | None = None,
) -> np.ndarray:
    height = region_rows * region_h + (region_rows - 1)
    width = region_cols * region_w + (region_cols - 1)
    grid = np.full((height, width), SEPARATOR_COLOR, dtype=int)

    for ridx in range(region_rows):
        for cidx in range(region_cols):
            r0, c0 = _region_origin(ridx, cidx, region_h, region_w)
            grid[r0 : r0 + region_h, c0 : c0 + region_w] = bg
            if (ridx, cidx) == template_pos:
                grid[r0 : r0 + region_h, c0 : c0 + region_w] = template
            elif stamp is not None:
                grid[r0 : r0 + region_h, c0 : c0 + region_w] = stamp
    return grid


def _sample_boxes(region_h: int, region_w: int, colors: list[int]) -> dict[int, tuple[int, int, int, int]] | None:
    for _ in range(BOX_RETRIES):
        pivot_r = random.randint(2, region_h - 3)
        pivot_c = random.randint(2, region_w - 3)
        boxes: dict[int, tuple[int, int, int, int]] = {}
        seen: set[tuple[int, int, int, int]] = set()
        ok = True
        for color in colors:
            for _ in range(BOX_RETRIES):
                r0 = random.randint(1, pivot_r)
                c0 = random.randint(1, pivot_c)
                r1 = random.randint(pivot_r, region_h - 2)
                c1 = random.randint(pivot_c, region_w - 2)
                if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2:
                    continue
                box = (r0, c0, r1, c1)
                if box in seen:
                    continue
                seen.add(box)
                boxes[color] = box
                break
            else:
                ok = False
                break
        if ok:
            return boxes
    return None


def _visible_bbox_matches(body: np.ndarray, color: int, box: tuple[int, int, int, int]) -> bool:
    coords = np.argwhere(body == color)
    if coords.size == 0:
        return False
    if _bbox(coords) != box:
        return False
    r0, c0, r1, c1 = box
    return (
        np.any(body[r0, c0 : c1 + 1] == color)
        and np.any(body[r1, c0 : c1 + 1] == color)
        and np.any(body[r0 : r1 + 1, c0] == color)
        and np.any(body[r0 : r1 + 1, c1] == color)
    )


def _template_from_body(body: np.ndarray) -> np.ndarray:
    template = body.copy()
    template[0, 0] = MARKER_COLOR
    template[0, -1] = MARKER_COLOR
    template[-1, 0] = MARKER_COLOR
    template[-1, -1] = MARKER_COLOR
    return template


def _gen_attempt() -> np.ndarray | None:
    region_rows, region_cols = random.choice(GRID_FACTORS)
    max_h = min(MAX_REGION_SIZE, (30 - (region_rows - 1)) // region_rows)
    max_w = min(MAX_REGION_SIZE, (30 - (region_cols - 1)) // region_cols)
    if max_h < MIN_REGION_SIZE or max_w < MIN_REGION_SIZE:
        return None

    region_h = random.randint(MIN_REGION_SIZE, max_h)
    region_w = random.randint(MIN_REGION_SIZE, max_w)

    bg_choices = [color for color in ALL_COLORS if color not in (SEPARATOR_COLOR, MARKER_COLOR)]
    bg = random.choice(bg_choices)

    rect_count = random.randint(*RECT_COUNT_RANGE)
    rect_colors = random.sample([color for color in bg_choices if color != bg], rect_count)
    order = tuple(random.sample(rect_colors, len(rect_colors)))
    boxes = _sample_boxes(region_h, region_w, rect_colors)
    if boxes is None:
        return None

    body = _render_rectangles((region_h, region_w), bg, order, boxes)
    if not all(_visible_bbox_matches(body, color, boxes[color]) for color in rect_colors):
        return None

    template = _template_from_body(body)
    reversed_body = _render_rectangles((region_h, region_w), bg, order[::-1], boxes)
    if np.array_equal(body, reversed_body):
        return None

    try:
        inferred_reversed = _recover_boxes_and_reversed(template, bg, MARKER_COLOR)
    except AssertionError:
        return None
    if not np.array_equal(inferred_reversed, reversed_body):
        return None

    template_pos = divmod(random.randrange(region_rows * region_cols), region_cols)
    grid = _assemble_grid(region_rows, region_cols, region_h, region_w, bg, template_pos, template)
    expected = _assemble_grid(
        region_rows,
        region_cols,
        region_h,
        region_w,
        bg,
        template_pos,
        template,
        stamp=reversed_body,
    )

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid template-reversal puzzle")
