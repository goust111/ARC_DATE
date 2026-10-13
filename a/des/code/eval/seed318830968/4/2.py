import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ORTHOGONALS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cell_count(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return 2 * (h + w) - 4


def _is_solid_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_frame_component(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if len(coords) != _border_cell_count((r0, c0, r1, c1)):
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    return True


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _paint_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _ring_count(height: int, width: int) -> int:
    return (min(height, width) + 1) // 2


def _fill_grey_region(
    grid: np.ndarray,
    grey_bbox: tuple[int, int, int, int],
    border_colors: list[int],
    sample_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    recipe = border_colors + sample_colors
    assert recipe and sample_colors
    start_idx = len(border_colors) % len(recipe)
    r0, c0, r1, c1 = grey_bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            layer = min(r - r0, r1 - r, c - c0, c1 - c)
            out[r, c] = recipe[(start_idx + layer) % len(recipe)]
    return out


def _detect_structure(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_components = [
        coords
        for coords in _components(g == GREY)
        if _is_solid_rectangle(coords)
    ]
    assert len(grey_components) == 1, "expected one solid grey rectangle"
    grey_coords = grey_components[0]
    grey_bbox = _bbox(grey_coords)

    frame_infos: list[tuple[tuple[int, int, int, int], int, list[tuple[int, int]]]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        for coords in _components(g == color):
            if not _is_frame_component(g, coords, color):
                continue
            box = _bbox(coords)
            if _strictly_contains(box, grey_bbox):
                frame_infos.append((box, color, coords))

    assert frame_infos, "expected nested rectangular frames around the grey rectangle"
    frame_infos.sort(key=lambda item: (item[0][0], item[0][1], -(item[0][2] - item[0][0]), -(item[0][3] - item[0][1])))

    border_boxes = [info[0] for info in frame_infos]
    border_colors = [info[1] for info in frame_infos]
    for outer, inner in zip(border_boxes, border_boxes[1:]):
        assert _strictly_contains(outer, inner), "frame boxes must be properly nested"
    assert _strictly_contains(border_boxes[-1], grey_bbox), "innermost frame must enclose the grey rectangle"

    used = np.zeros_like(g, dtype=bool)
    for r, c in grey_coords:
        used[r, c] = True
    for _, _, coords in frame_infos:
        for r, c in coords:
            used[r, c] = True

    sample_positions = [(r, c) for r, c in zip(*np.where((g != BLACK) & ~used))]
    assert sample_positions, "expected a separate sample column"
    sample_cols = {c for _, c in sample_positions}
    assert len(sample_cols) == 1, "sample pixels must form one vertical column"
    sample_col = next(iter(sample_cols))
    sample_rows = sorted(r for r, _ in sample_positions)
    expected_rows = list(range(sample_rows[0], sample_rows[0] + len(sample_rows)))
    assert sample_rows == expected_rows, "sample pixels must be vertically contiguous"
    sample_colors = [int(g[r, sample_col]) for r in sample_rows]

    return grey_bbox, border_colors, sample_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the nested frame colors plus the separate sample column as one cyclic recipe,
    then repaint the solid grey rectangle as concentric colored rectangular rings.
    """
    g = np.asarray(grid, dtype=int)
    grey_bbox, border_colors, sample_colors = _detect_structure(g)
    return _fill_grey_region(g, grey_bbox, border_colors, sample_colors)


MIN_BORDERS = 2
MAX_BORDERS = 4
MIN_SAMPLE = 1
MAX_SAMPLE = 4
MAX_GREY_DIM = 11
MIN_MARGIN = 1
MAX_MARGIN = 4
MIN_SAMPLE_GAP = 2
MAX_SAMPLE_GAP = 4
MAX_GENERATION_ATTEMPTS = 400


def _choose_grey_dims(sample_len: int, want_wrap: bool) -> tuple[int, int] | None:
    max_rings = _ring_count(MAX_GREY_DIM, MAX_GREY_DIM)
    min_rings = sample_len + 1 if want_wrap else 1
    max_allowed = max_rings if want_wrap else sample_len
    feasible = [rings for rings in range(min_rings, max_allowed + 1) if 2 * rings - 1 <= MAX_GREY_DIM]
    if not feasible:
        return None

    rings = random.choice(feasible)
    if rings == 1:
        min_side = 2
    else:
        choices = [2 * rings - 1]
        if 2 * rings <= MAX_GREY_DIM:
            choices.append(2 * rings)
        min_side = random.choice(choices)

    other_side = random.randint(min_side, MAX_GREY_DIM)
    if random.choice([False, True]):
        return min_side, other_side
    return other_side, min_side


def _build_input_and_output() -> tuple[np.ndarray, np.ndarray] | None:
    border_count = random.randint(MIN_BORDERS, MAX_BORDERS)
    sample_len = random.randint(MIN_SAMPLE, MAX_SAMPLE)
    want_wrap = random.choice([False, True])
    grey_dims = _choose_grey_dims(sample_len, want_wrap)
    if grey_dims is None:
        return None

    grey_h, grey_w = grey_dims
    outer_h = grey_h + 4 * border_count - 2
    outer_w = grey_w + 4 * border_count - 2

    top_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    bottom_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    left_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    right_margin = random.randint(MIN_MARGIN, MAX_MARGIN)
    sample_gap = random.randint(MIN_SAMPLE_GAP, MAX_SAMPLE_GAP)
    extra_h = random.randint(0, 4)
    extra_w = random.randint(0, 4)
    sample_on_left = random.choice([False, True])

    height = max(outer_h, sample_len) + top_margin + bottom_margin + extra_h
    width = outer_w + 1 + sample_gap + left_margin + right_margin + extra_w
    if height <= outer_h or width <= outer_w + 1 + sample_gap:
        return None

    shift_x = random.randint(0, extra_w)
    if sample_on_left:
        sample_col = left_margin + shift_x
        outer_c0 = sample_col + 1 + sample_gap
    else:
        outer_c0 = left_margin + shift_x
        sample_col = outer_c0 + outer_w + sample_gap
    outer_c1 = outer_c0 + outer_w - 1
    if outer_c0 < 0 or outer_c1 >= width or sample_col < 0 or sample_col >= width:
        return None

    outer_top_max = height - bottom_margin - outer_h
    sample_top_max = height - bottom_margin - sample_len
    if outer_top_max < top_margin or sample_top_max < top_margin:
        return None
    outer_r0 = random.randint(top_margin, outer_top_max)
    sample_r0 = random.randint(top_margin, sample_top_max)

    grey_r0 = outer_r0 + (2 * border_count - 1)
    grey_c0 = outer_c0 + (2 * border_count - 1)
    grey_r1 = grey_r0 + grey_h - 1
    grey_c1 = grey_c0 + grey_w - 1
    if grey_r1 >= height or grey_c1 >= width:
        return None

    colors = random.sample(PALETTE_COLORS, border_count + sample_len)
    border_colors = colors[:border_count]
    sample_colors = colors[border_count:]

    inp = np.zeros((height, width), dtype=int)
    inp[grey_r0 : grey_r1 + 1, grey_c0 : grey_c1 + 1] = GREY

    for idx, color in enumerate(border_colors):
        pad = 1 + 2 * (border_count - 1 - idx)
        bbox = (grey_r0 - pad, grey_c0 - pad, grey_r1 + pad, grey_c1 + pad)
        _paint_border(inp, bbox, color)

    for offset, color in enumerate(sample_colors):
        inp[sample_r0 + offset, sample_col] = color

    out = _fill_grey_region(inp, (grey_r0, grey_c0, grey_r1, grey_c1), border_colors, sample_colors)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_input_and_output()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
