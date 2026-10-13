import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

PALETTE = [BLUE, RED, GREEN, MAGENTA, ORANGE, CYAN, MAROON]
GEN_RETRIES = 300

TEMPLATES: dict[int, tuple[tuple[tuple[int, ...], ...], ...]] = {
    2: (
        ((1, 2, 1), (2, 1)),
    ),
    3: (
        ((1, 2), (2, 1), (1, 3)),
        ((1, 2, 1), (2, 1), (1, 3)),
        ((1, 2, 1), (2, 1), (1, 3, 2)),
    ),
    4: (
        ((1, 2, 1), (2, 1), (1, 3, 2), (3, 4)),
        ((1, 2, 1), (2, 1), (1, 3, 2), (3, 4, 1)),
        ((1, 2, 1), (2, 1, 2), (1, 3, 1), (3, 4, 1)),
    ),
    5: (
        ((1, 2, 1), (2, 1, 3), (1, 3, 2), (3, 4, 2), (4, 5, 1)),
    ),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_frame_bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(comp) != 2 * (h + w) - 4:
        return None
    for r, c in comp:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    return r0, c0, r1, c1


def _extract_stripes(interior: np.ndarray) -> list[int]:
    h, w = interior.shape
    assert h >= 2 and w >= 1, "code boxes need room for at least two stripes"

    colors: list[int] = []
    r = 0
    while r < h:
        color = int(interior[r, 0])
        assert color not in (BLACK, GREY, YELLOW), "stripe colors must be non-background signal colors"
        assert np.all(interior[r] == color), "each stripe row must be solid"
        rr = r + 1
        while rr < h:
            next_color = int(interior[rr, 0])
            assert np.all(interior[rr] == next_color), "each stripe row must be solid"
            if next_color != color:
                break
            rr += 1
        assert np.all(interior[r:rr, :] == color), "each stripe must fill the box width"
        colors.append(color)
        r = rr

    assert 2 <= len(colors) <= 3, "each code box must contain 2 or 3 stripes"
    assert all(colors[i] != colors[i + 1] for i in range(len(colors) - 1))
    return colors


def _logical_marker_pos(rank: int, cell: int) -> int:
    return 1 + (rank - 1) * cell + cell // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode framed stripe codes into canvas coordinates and mark the cropped board.

    Input structure:
    - The background is Black.
    - The left side contains several small Grey 1-pixel rectangular frames whose
      interiors are filled with 2 or 3 full-width horizontal color stripes.
    - The right side contains one Grey-framed square canvas with an empty Black
      interior that is logically divided into as many rows and columns as there
      are code boxes.

    Rule:
    - Read each code box from top to bottom and treat every stripe band as one
      color occurrence, regardless of stripe height.
    - Count stripe-band occurrences across all code boxes; sorting colors by
      frequency from high to low assigns coordinate ranks 1..N.
    - In each box, the first stripe gives the row rank, the second stripe gives
      the column rank, and any third stripe only contributes to the global tally.
    - Output the cropped canvas and place a Yellow pixel at the center of every
      addressed logical cell.

    Key insight:
    - The stripe colors do not have numeric meaning by themselves; their row and
      column values are inferred from the visible global frequency ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    assert bg == BLACK, "background should be black"

    frames = []
    for comp in _components_of_color(g, GREY):
        bbox = _rect_frame_bbox(comp)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        frames.append({"bbox": bbox, "interior": interior})

    assert frames, "expected grey frames"

    canvas_frames = []
    code_frames = []
    for frame in frames:
        interior = frame["interior"]
        ih, iw = interior.shape
        if ih == iw and bool((interior == bg).all()):
            canvas_frames.append(frame)
        else:
            code_frames.append(frame)

    assert len(canvas_frames) == 1, "expected exactly one blank square canvas"
    assert len(code_frames) >= 2, "expected at least two code boxes"

    stripe_sequences = [_extract_stripes(frame["interior"]) for frame in code_frames]
    box_count = len(stripe_sequences)

    freq = Counter(color for seq in stripe_sequences for color in seq)
    assert len(freq) == box_count, "the number of code colors should match the number of boxes"

    counts = list(freq.values())
    assert len(set(counts)) == len(counts), "frequency ranking must be unambiguous"

    ordered_colors = [color for color, _ in sorted(freq.items(), key=lambda item: -item[1])]
    rank_by_color = {color: idx + 1 for idx, color in enumerate(ordered_colors)}

    canvas_info = canvas_frames[0]
    r0, c0, r1, c1 = canvas_info["bbox"]
    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    inner = out[1:-1, 1:-1]
    assert inner.shape[0] == inner.shape[1], "canvas must be square"
    assert inner.shape[0] % box_count == 0, "canvas side must be divisible by the number of boxes"
    cell = inner.shape[0] // box_count

    for seq in stripe_sequences:
        row_rank = rank_by_color[int(seq[0])]
        col_rank = rank_by_color[int(seq[1])]
        rr = _logical_marker_pos(row_rank, cell)
        cc = _logical_marker_pos(col_rank, cell)
        out[rr, cc] = YELLOW

    return out


def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    dst[top : top + h, left : left + w] = src


def _make_canvas(box_count: int, cell: int) -> np.ndarray:
    side = box_count * cell + 2
    canvas = np.full((side, side), GREY, dtype=int)
    canvas[1:-1, 1:-1] = BLACK
    return canvas


def _render_box(seq: tuple[int, ...], rank_colors: list[int], rng: random.Random) -> np.ndarray:
    stripe_heights = [rng.randint(1, 2) for _ in seq]
    interior_h = sum(stripe_heights)
    width_choices = [w for w in range(3, 6) if w != interior_h]
    if not width_choices:
        width_choices = [3, 4, 5]
    interior_w = rng.choice(width_choices)

    box = np.full((interior_h + 2, interior_w + 2), GREY, dtype=int)
    box[1:-1, 1:-1] = BLACK
    r = 1
    for rank, stripe_h in zip(seq, stripe_heights):
        box[r : r + stripe_h, 1:-1] = rank_colors[rank - 1]
        r += stripe_h
    return box


def _template_counts(template: tuple[tuple[int, ...], ...]) -> Counter[int]:
    return Counter(rank for seq in template for rank in seq)


def _expected_output(template: tuple[tuple[int, ...], ...], cell: int) -> np.ndarray:
    box_count = len(template)
    out = _make_canvas(box_count, cell)
    for seq in template:
        rr = _logical_marker_pos(seq[0], cell)
        cc = _logical_marker_pos(seq[1], cell)
        out[rr, cc] = YELLOW
    return out


def _pack_boxes(boxes: list[np.ndarray], columns: int, rng: random.Random) -> tuple[dict[int, tuple[int, int]], int, int]:
    order = list(range(len(boxes)))
    rng.shuffle(order)
    v_gap = rng.randint(1, 2)
    h_gap = rng.randint(1, 2)

    column_items: list[list[int]] = [[] for _ in range(columns)]
    column_heights = [0] * columns
    column_widths = [0] * columns

    for idx in order:
        col = min(range(columns), key=lambda col_idx: (column_heights[col_idx], len(column_items[col_idx])))
        if column_items[col]:
            column_heights[col] += v_gap
        column_items[col].append(idx)
        column_heights[col] += int(boxes[idx].shape[0])
        column_widths[col] = max(column_widths[col], int(boxes[idx].shape[1]))

    positions: dict[int, tuple[int, int]] = {}
    x = 0
    for col_idx, items in enumerate(column_items):
        y = 0
        for item_idx, box_idx in enumerate(items):
            if item_idx:
                y += v_gap
            positions[box_idx] = (y, x)
            y += int(boxes[box_idx].shape[0])
        x += column_widths[col_idx]
        if col_idx + 1 < columns:
            x += h_gap

    block_h = max(column_heights)
    block_w = sum(column_widths) + h_gap * (columns - 1)
    return positions, block_h, block_w


def _build_grid(boxes: list[np.ndarray], canvas: np.ndarray, rng: random.Random) -> np.ndarray | None:
    column_options = [1, 2]
    rng.shuffle(column_options)
    for _ in range(16):
        columns = next((c for c in column_options if c <= len(boxes)), None)
        if columns is None:
            return None
        if len(column_options) == 2:
            columns = rng.choice([c for c in column_options if c <= len(boxes)])
        positions, block_h, block_w = _pack_boxes(boxes, columns, rng)

        left_margin = rng.randint(1, 2)
        right_margin = rng.randint(1, 2)
        top_margin = rng.randint(1, 2)
        bottom_margin = rng.randint(1, 2)
        mid_gap = rng.randint(2, 3)

        content_h = max(block_h, int(canvas.shape[0]))
        boxes_top = rng.randint(0, content_h - block_h)
        canvas_top = rng.randint(0, content_h - int(canvas.shape[0]))

        total_h = top_margin + content_h + bottom_margin
        total_w = left_margin + block_w + mid_gap + int(canvas.shape[1]) + right_margin
        if total_h > 30 or total_w > 30:
            continue

        canvas_left = left_margin + block_w + mid_gap
        if canvas_left < total_w // 2:
            continue

        grid = np.zeros((total_h, total_w), dtype=int)
        for idx, box in enumerate(boxes):
            box_top, box_left = positions[idx]
            _paste(grid, box, top_margin + boxes_top + box_top, left_margin + box_left)
        _paste(grid, canvas, top_margin + canvas_top, canvas_left)
        return grid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    box_count = rng.randint(2, 5)
    template = rng.choice(TEMPLATES[box_count])
    rank_colors = rng.sample(PALETTE, box_count)

    cell_choices = [1, 3] if box_count <= 4 else [1]
    cell = rng.choice(cell_choices)

    boxes = [_render_box(seq, rank_colors, rng) for seq in template]
    canvas = _make_canvas(box_count, cell)
    grid = _build_grid(boxes, canvas, rng)
    if grid is None:
        return None

    expected = _expected_output(template, cell)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for templates in TEMPLATES.values():
        for template in templates:
            counts = _template_counts(template)
            box_count = len(template)
            assert len(counts) == box_count
            assert len(set(counts.values())) == box_count
            assert all(seq[0] != seq[1] for seq in template)

    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-code canvas input")
