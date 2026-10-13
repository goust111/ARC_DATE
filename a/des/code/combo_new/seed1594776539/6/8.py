import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BLACK = 0
SEPARATOR = 5
TILE_SIZE = 5
LEGEND_STRIP = 7
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_RETRIES = 300


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_coords(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_solid_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_separator(grid: np.ndarray, bg: int) -> Tuple[str, int, int]:
    h, w = grid.shape
    row_candidates = [
        r
        for r in range(1, h - 1)
        if grid[r, 0] != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        c
        for c in range(1, w - 1)
        if grid[0, c] != bg and np.all(grid[:, c] == grid[0, c])
    ]
    if len(row_candidates) == 1 and not col_candidates:
        r = row_candidates[0]
        return "horizontal", r, int(grid[r, 0])
    if len(col_candidates) == 1 and not row_candidates:
        c = col_candidates[0]
        return "vertical", c, int(grid[0, c])
    raise AssertionError("expected exactly one full separator row or column")


def _extract_templates(legend: np.ndarray, bg: int, sep_color: int) -> Dict[int, np.ndarray]:
    templates: Dict[int, np.ndarray] = {}
    for color in np.unique(legend):
        color = int(color)
        if color in (bg, sep_color):
            continue
        coords = np.argwhere(legend == color)
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        templates[color] = mask
    assert templates, "legend must define at least one color-to-shape mapping"
    return templates


def _scale_index(idx: int, src_len: int, dst_len: int) -> int:
    if dst_len <= 1:
        return 0
    if src_len <= 1:
        return dst_len // 2
    return int(round(idx * (dst_len - 1) / (src_len - 1)))


def _draw_line(mask: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    steps = max(abs(r1 - r0), abs(c1 - c0))
    if steps == 0:
        mask[r0, c0] = True
        return
    for i in range(steps + 1):
        rr = int(round(r0 + (r1 - r0) * i / steps))
        cc = int(round(c0 + (c1 - c0) * i / steps))
        mask[rr, cc] = True


def _scale_template(template: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    src_h, src_w = template.shape
    dst_h, dst_w = out_shape
    out = np.zeros((dst_h, dst_w), dtype=bool)
    points = np.argwhere(template)
    assert points.size > 0

    mapped: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for r, c in points:
        mr = _scale_index(int(r), src_h, dst_h)
        mc = _scale_index(int(c), src_w, dst_w)
        mapped[(int(r), int(c))] = (mr, mc)
        out[mr, mc] = True

    point_set = {tuple(map(int, p)) for p in points}
    for r, c in point_set:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in point_set and (nr, nc) >= (r, c):
                _draw_line(out, mapped[(r, c)], mapped[(nr, nc)])
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in point_set or (nr, nc) < (r, c):
                continue
            if template[r, nc] or template[nr, c]:
                continue
            _draw_line(out, mapped[(r, c)], mapped[(nr, nc)])
    return out


def _transform_canvas(canvas: np.ndarray, templates: Dict[int, np.ndarray], bg: int) -> np.ndarray:
    out = canvas.copy()
    for color, template in templates.items():
        comps = _component_coords(canvas == color)
        for coords in comps:
            if not _is_solid_rectangle(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            out[r0 : r1 + 1, c0 : c1 + 1] = bg
            scaled = _scale_template(template, (r1 - r0 + 1, c1 - c0 + 1))
            out[r0 : r1 + 1, c0 : c1 + 1][scaled] = color
    return out


def _plus_mask() -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    mid = TILE_SIZE // 2
    mask[mid, :] = True
    mask[:, mid] = True
    return mask


def _x_mask() -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    idx = np.arange(TILE_SIZE)
    mask[idx, idx] = True
    mask[idx, TILE_SIZE - 1 - idx] = True
    return mask


def _diamond_mask() -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    mid = TILE_SIZE // 2
    for r in range(TILE_SIZE):
        for c in range(TILE_SIZE):
            if abs(r - mid) + abs(c - mid) == mid:
                mask[r, c] = True
    return mask


def _frame_mask() -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _double_frame_mask() -> np.ndarray:
    mask = _frame_mask()
    mask[1, 1:-1] = True
    mask[-2, 1:-1] = True
    mask[1:-1, 1] = True
    mask[1:-1, -2] = True
    return mask


def _h_mask() -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[TILE_SIZE // 2, :] = True
    return mask


def _t_mask() -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    mask[0, :] = True
    mask[:, TILE_SIZE // 2] = True
    return mask


SHAPE_LIBRARY = {
    "plus": _plus_mask,
    "x": _x_mask,
    "diamond": _diamond_mask,
    "frame": _frame_mask,
    "double_frame": _double_frame_mask,
    "h": _h_mask,
    "t": _t_mask,
}


def _legend_content_length(count: int) -> int:
    return count * TILE_SIZE + (count - 1)


def _place_legend_top(
    grid: np.ndarray, colors: Sequence[int], shapes: Sequence[np.ndarray], rng: random.Random
) -> None:
    content_w = _legend_content_length(len(colors))
    row0 = (LEGEND_STRIP - TILE_SIZE) // 2
    start_col = rng.randint(0, grid.shape[1] - content_w)
    col = start_col
    for color, shape in zip(colors, shapes):
        tile = grid[row0 : row0 + TILE_SIZE, col : col + TILE_SIZE]
        tile[shape] = color
        col += TILE_SIZE + 1


def _place_legend_left(
    grid: np.ndarray, colors: Sequence[int], shapes: Sequence[np.ndarray], rng: random.Random
) -> None:
    content_h = _legend_content_length(len(colors))
    col0 = (LEGEND_STRIP - TILE_SIZE) // 2
    start_row = rng.randint(0, grid.shape[0] - content_h)
    row = start_row
    for color, shape in zip(colors, shapes):
        tile = grid[row : row + TILE_SIZE, col0 : col0 + TILE_SIZE]
        tile[shape] = color
        row += TILE_SIZE + 1


def _sample_rect_sizes(
    rng: random.Random, canvas_h: int, canvas_w: int, count: int
) -> List[Tuple[int, int]]:
    min_h = 5
    min_w = 5
    max_h = min(9, canvas_h - 2)
    max_w = min(10, canvas_w - 2)
    if max_h < min_h or max_w < min_w:
        raise ValueError("canvas too small for rectangles")
    sizes = []
    for _ in range(count):
        rh = rng.randint(min_h, max_h)
        rw = rng.randint(min_w, max_w)
        sizes.append((rh, rw))
    if count >= 2 and len(set(sizes)) == 1:
        rh = max(min_h, sizes[-1][0] - 1)
        rw = min(max_w, sizes[-1][1] + 1)
        sizes[-1] = (rh, rw)
    return sizes


def _margin_positions(limit: int, size: int) -> Iterable[int]:
    if limit - size >= 2:
        return range(1, limit - size)
    return range(0, limit - size + 1)


def _place_rectangles(
    grid: np.ndarray,
    rect_origin: Tuple[int, int],
    canvas_shape: Tuple[int, int],
    colors: Sequence[int],
    rng: random.Random,
) -> bool:
    canvas_h, canvas_w = canvas_shape
    sizes = _sample_rect_sizes(rng, canvas_h, canvas_w, len(colors))
    occupancy = np.zeros(canvas_shape, dtype=bool)
    placements: List[Tuple[int, int, int, int, int]] = []

    for color, (rh, rw) in zip(colors, sizes):
        candidates = []
        for r0 in _margin_positions(canvas_h, rh):
            for c0 in _margin_positions(canvas_w, rw):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(canvas_h, r0 + rh + 1)
                cc1 = min(canvas_w, c0 + rw + 1)
                if occupancy[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        occupancy[r0 : r0 + rh, c0 : c0 + rw] = True
        placements.append((color, r0, c0, rh, rw))

    base_r, base_c = rect_origin
    for color, r0, c0, rh, rw in placements:
        grid[base_r + r0 : base_r + r0 + rh, base_c + c0 : base_c + c0 + rw] = color
    return True


def _choose_shapes(rng: random.Random, count: int) -> List[np.ndarray]:
    names = rng.sample(list(SHAPE_LIBRARY), count)
    return [SHAPE_LIBRARY[name]().copy() for name in names]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["top", "left"])
    color_count = rng.choice([2, 3, 4] if orientation == "top" else [2, 3])
    colors = rng.sample(COLOR_POOL, color_count)
    shapes = _choose_shapes(rng, color_count)

    if orientation == "top":
        min_w = max(16, _legend_content_length(color_count))
        width = rng.randint(min_w, 30)
        canvas_h = rng.randint(8, 30 - LEGEND_STRIP - 1)
        height = LEGEND_STRIP + 1 + canvas_h
        grid = np.full((height, width), BLACK, dtype=int)
        grid[LEGEND_STRIP, :] = SEPARATOR
        _place_legend_top(grid[:LEGEND_STRIP, :], colors, shapes, rng)
        if not _place_rectangles(grid, (LEGEND_STRIP + 1, 0), (canvas_h, width), colors, rng):
            return None
    else:
        min_h = max(16, _legend_content_length(color_count))
        height = rng.randint(min_h, 30)
        canvas_w = rng.randint(8, 30 - LEGEND_STRIP - 1)
        width = LEGEND_STRIP + 1 + canvas_w
        grid = np.full((height, width), BLACK, dtype=int)
        grid[:, LEGEND_STRIP] = SEPARATOR
        _place_legend_left(grid[:, :LEGEND_STRIP], colors, shapes, rng)
        if not _place_rectangles(grid, (0, LEGEND_STRIP + 1), (height, canvas_w), colors, rng):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid canvas rectangle with the legend shape drawn in the same color and scaled to that rectangle.

    Input structure:
    - A single full grey row or column splits the grid into a legend region and a canvas region.
    - The legend region has black background and several small example shapes, each drawn in one unique color.
    - The canvas region also has black background and contains solid monochrome rectangles whose colors match the legend colors.

    Rule:
    - Find the separator line and keep it unchanged.
    - In the legend, build a mapping from each color to that color's shape mask, normalized to the shape's own bounding box.
    - In the canvas, for every solid rectangle of a mapped color, erase the rectangle to black and redraw the corresponding shape by stretching the template bbox to the rectangle bbox and rasterizing the stretched outline.

    Key insight:
    - The legend colors are not paints for arbitrary fill; they index reusable shape templates that must be detected, normalized, and geometrically scaled onto larger solid rectangles.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    orient, axis, sep_color = _find_separator(g, bg)

    if orient == "horizontal":
        legend = g[:axis, :]
        canvas = g[axis + 1 :, :]
        transformed = _transform_canvas(canvas, _extract_templates(legend, bg, sep_color), bg)
        out = g.copy()
        out[axis + 1 :, :] = transformed
        return out

    legend = g[:, :axis]
    canvas = g[:, axis + 1 :]
    transformed = _transform_canvas(canvas, _extract_templates(legend, bg, sep_color), bg)
    out = g.copy()
    out[:, axis + 1 :] = transformed
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 8")
