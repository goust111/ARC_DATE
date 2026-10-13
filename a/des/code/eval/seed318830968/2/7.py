import random

import numpy as np


BG = 0
GREY = 5

LAYER_CHOICES = (2, 3, 4)
GRID_H_RANGE = (12, 26)
RIGHT_WIDTH_RANGE = (2, 5)
TEMPLATE_MARGIN_RANGE = (1, 4)
SEPARATOR_GAP_RANGE = (1, 4)
EXTRA_SLACK_RANGE = (0, 4)
OUTER_MAX_SIZE = 18
INNER_H_RANGE = (3, 8)
INNER_W_RANGE = (3, 10)
INSET_RANGE = (2, 4)
GEN_RETRIES = 300


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    sub = np.zeros((h, w), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(sub, border))


def _sort_nested_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    ordered = sorted(
        boxes,
        key=lambda box: -((box[2] - box[0] + 1) * (box[3] - box[1] + 1)),
    )
    for outer, inner in zip(ordered, ordered[1:]):
        assert outer[0] < inner[0] and outer[1] < inner[1]
        assert outer[2] > inner[2] and outer[3] > inner[3]
    return ordered


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color == BG or not np.all(grid[:, c] == color):
            continue
        left_same = int((grid[:, :c] == color).sum())
        right_non_bg = int((grid[:, c + 1 :] != BG).sum())
        right_other = int(((grid[:, c + 1 :] != BG) & (grid[:, c + 1 :] != color)).sum())
        if left_same > 0 and right_non_bg > 0 and right_other > 0:
            candidates.append((left_same, c, color))
    assert candidates, "separator not found"
    left_same, sep_c, grey = max(candidates, key=lambda item: (item[0], -item[1]))
    assert left_same > 0
    return int(sep_c), int(grey)


def _read_palette(right: np.ndarray, grey: int) -> list[int]:
    pts = np.argwhere((right != BG) & (right != grey))
    assert len(pts) > 0, "palette not found"
    cols = {int(c) for _, c in pts}
    assert len(cols) == 1, "palette pixels must share one column"
    ordered = sorted((int(r), int(right[r, c])) for r, c in pts)
    colors = [color for _, color in ordered]
    assert len(colors) == len(set(colors)), "palette colors must be distinct"
    return colors


def _fill_layers(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    sep_c: int,
) -> np.ndarray:
    out = grid.copy()
    out[:, sep_c + 1 :] = BG
    for idx, (r0, c0, r1, c1) in enumerate(boxes):
        fill = int(colors[idx])
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
        if idx + 1 < len(boxes):
            nr0, nc0, nr1, nc1 = boxes[idx + 1]
            out[nr0 : nr1 + 1, nc0 : nc1 + 1] = grid[nr0 : nr1 + 1, nc0 : nc1 + 1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular grey rings using the top-to-bottom palette on the right.
    """
    g = np.asarray(grid, dtype=int)
    sep_c, grey = _find_separator(g)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]

    ring_boxes = []
    for comp in _components4(left == grey):
        if _is_rectangular_ring(comp):
            ring_boxes.append(_bbox(comp))
    assert 2 <= len(ring_boxes) <= 4, "expected 2 to 4 nested rings"
    ring_boxes = _sort_nested_boxes(ring_boxes)

    palette = _read_palette(right, grey)
    assert len(palette) == len(ring_boxes), "palette size must match ring depth"

    return _fill_layers(g, ring_boxes, palette, sep_c)


def _sample_nested_boxes(layer_count: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(200):
        insets = [
            (
                random.randint(*INSET_RANGE),
                random.randint(*INSET_RANGE),
                random.randint(*INSET_RANGE),
                random.randint(*INSET_RANGE),
            )
            for _ in range(layer_count - 1)
        ]
        extra_h = sum(top + bottom for top, bottom, _, _ in insets)
        extra_w = sum(left + right for _, _, left, right in insets)
        inner_h_max = min(INNER_H_RANGE[1], OUTER_MAX_SIZE - extra_h)
        inner_w_max = min(INNER_W_RANGE[1], OUTER_MAX_SIZE - extra_w)
        if inner_h_max < INNER_H_RANGE[0] or inner_w_max < INNER_W_RANGE[0]:
            continue

        r0 = c0 = 0
        r1 = random.randint(INNER_H_RANGE[0], inner_h_max) - 1
        c1 = random.randint(INNER_W_RANGE[0], inner_w_max) - 1
        boxes = [(r0, c0, r1, c1)]
        for top, bottom, left, right in insets:
            r0 -= top
            c0 -= left
            r1 += bottom
            c1 += right
            boxes.append((r0, c0, r1, c1))

        min_r = min(box[0] for box in boxes)
        min_c = min(box[1] for box in boxes)
        shifted = [
            (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c)
            for r0, c0, r1, c1 in boxes
        ]
        outer = shifted[-1]
        if outer[2] - outer[0] + 1 > OUTER_MAX_SIZE or outer[3] - outer[1] + 1 > OUTER_MAX_SIZE:
            continue
        return shifted[::-1]
    return None


def _sample_palette_rows(height: int, layer_count: int) -> list[int] | None:
    min_gap = random.choice((1, 2))
    valid_rows = list(range(1, height - 1))
    for _ in range(100):
        rows = sorted(random.sample(valid_rows, layer_count))
        if all(rows[i + 1] - rows[i] >= min_gap for i in range(layer_count - 1)):
            return rows
    return None


def _draw_ring(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_input(
    height: int,
    width: int,
    sep_c: int,
    boxes: list[tuple[int, int, int, int]],
    palette_rows: list[int],
    palette_col: int,
    palette_colors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep_c] = GREY
    for box in boxes:
        _draw_ring(grid, box, GREY)
    for row, color in zip(palette_rows, palette_colors):
        grid[int(row), int(palette_col)] = int(color)
    return grid


def _background_is_clear(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.reshape(-1), minlength=10)
    return int(np.argmax(counts)) == BG


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.choice(LAYER_CHOICES)
    local_boxes = _sample_nested_boxes(layer_count)
    if local_boxes is None:
        return None

    outer_h = local_boxes[0][2] - local_boxes[0][0] + 1
    outer_w = local_boxes[0][3] - local_boxes[0][1] + 1

    top_margin = random.randint(*TEMPLATE_MARGIN_RANGE)
    bottom_margin = random.randint(*TEMPLATE_MARGIN_RANGE)
    extra_h = random.randint(*EXTRA_SLACK_RANGE)
    height = outer_h + top_margin + bottom_margin + extra_h
    if not (GRID_H_RANGE[0] <= height <= GRID_H_RANGE[1]):
        return None
    top_shift = top_margin + random.randint(0, extra_h)

    left_margin = random.randint(*TEMPLATE_MARGIN_RANGE)
    gap_to_separator = random.randint(*SEPARATOR_GAP_RANGE)
    right_width = random.randint(*RIGHT_WIDTH_RANGE)
    extra_w = random.randint(*EXTRA_SLACK_RANGE)
    width = left_margin + outer_w + gap_to_separator + 1 + right_width + extra_w
    if width > 30:
        return None
    left_shift = left_margin + random.randint(0, extra_w)
    sep_c = left_shift + outer_w + gap_to_separator
    if sep_c >= width - 1:
        return None

    boxes = [
        (r0 + top_shift, c0 + left_shift, r1 + top_shift, c1 + left_shift)
        for r0, c0, r1, c1 in local_boxes
    ]
    palette_rows = _sample_palette_rows(height, layer_count)
    if palette_rows is None:
        return None
    palette_col = random.randint(sep_c + 1, width - 1)
    palette_colors = random.sample([c for c in range(1, 10) if c != GREY], layer_count)

    inp = _render_input(height, width, sep_c, boxes, palette_rows, palette_col, palette_colors)
    if not _background_is_clear(inp):
        return None

    expected = _fill_layers(inp, boxes, palette_colors, sep_c)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
