from __future__ import annotations

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 500
GEN_SIDES = ("top", "bottom", "left", "right")
GEN_WIDTHS = (1, 2, 3, 4)
GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_REGION_COUNT = (2, 4)
GEN_REGION_INNER_W = (3, 8)
GEN_LEGEND_INNER_W = (3, 8)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    pt_set = set(points)
    if len(pt_set) != 2 * h + 2 * w - 4:
        return None

    for r in range(r0, r1 + 1):
        if (r, c0) not in pt_set or (r, c1) not in pt_set:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in pt_set or (r1, c) not in pt_set:
            return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_border_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[tuple[int, int], int, list[tuple[int, int, int, int]]]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue

        frames: list[tuple[int, int, int, int]] = []
        for comp in _components(grid == color):
            box = _frame_bbox(comp)
            if box is not None:
                frames.append(box)

        if not frames:
            continue

        score = (
            len(frames),
            sum(2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4 for r0, c0, r1, c1 in frames),
        )
        candidates.append((score, color, frames))

    assert candidates, "expected at least one frame color"
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1:
        assert candidates[0][0] > candidates[1][0], "frame color must be unique"
    score, border_color, frames = candidates[0]
    assert score[0] >= 2, "expected legend plus at least one target frame"
    return int(border_color), frames


def _legend_mapping(interior: np.ndarray, bg: int, border: int) -> dict[int, int] | None:
    if interior.size == 0 or interior.shape[0] < 2:
        return None

    row_colors: list[int] = []
    for r in range(interior.shape[0]):
        values = np.unique(interior[r])
        if len(values) != 1:
            return None
        color = int(values[0])
        if color in (bg, border):
            return None
        row_colors.append(color)

    mapping: dict[int, int] = {}
    used: set[int] = set()
    start = 0
    while start < len(row_colors):
        color = row_colors[start]
        end = start + 1
        while end < len(row_colors) and row_colors[end] == color:
            end += 1
        if color in used:
            return None
        used.add(color)
        mapping[color] = end - start
        start = end

    if len(mapping) < 2:
        return None
    return mapping


def _marker_color(interior: np.ndarray, bg: int, border: int) -> int:
    marker_mask = (interior != bg) & (interior != border)
    coords = np.argwhere(marker_mask)
    assert len(coords) == 1, "target frame must contain exactly one marker"
    rr, cc = map(int, coords[0])
    return int(interior[rr, cc])


def _paint_region(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, bg: int, width: int) -> None:
    r0, c0, r1, c1 = bbox
    for offset in range(r1 - r0 - 1):
        fill = color if (offset // width) % 2 == 0 else bg
        out[r0 + 1 + offset, c0 + 1 : c1] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read stripe widths from the legend frame, then fill each marked frame with
    alternating horizontal bands of the marker color and the background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    border, frames = _find_border_frames(g, bg)

    legend: dict[int, int] | None = None
    region_jobs: list[tuple[tuple[int, int, int, int], int]] = []

    for bbox in frames:
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        mapping = _legend_mapping(interior, bg, border)
        if mapping is not None:
            assert legend is None, "expected exactly one legend frame"
            legend = mapping
            continue
        region_jobs.append((bbox, _marker_color(interior, bg, border)))

    assert legend is not None, "missing legend frame"
    assert region_jobs, "expected at least one marked target frame"

    out = g.copy()
    for bbox, color in region_jobs:
        assert color in legend, "marker color missing from legend"
        _paint_region(out, bbox, color, bg, legend[color])
    return out


def _unique_mode_is(grid: np.ndarray, color: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    if int(values[order[0]]) != int(color):
        return False
    if len(order) > 1 and int(counts[order[0]]) == int(counts[order[1]]):
        return False
    return True


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _positions(
    reserved: np.ndarray,
    box_h: int,
    box_w: int,
    edge_margin: int,
    side: str | None,
) -> list[tuple[int, int]]:
    hh, ww = reserved.shape
    positions: list[tuple[int, int]] = []

    if side is None:
        r_start = edge_margin
        r_stop = hh - box_h - edge_margin + 1
        c_start = edge_margin
        c_stop = ww - box_w - edge_margin + 1
        if r_start >= r_stop or c_start >= c_stop:
            return []
        for r0 in range(r_start, r_stop):
            for c0 in range(c_start, c_stop):
                if _can_place(reserved, r0, c0, box_h, box_w):
                    positions.append((r0, c0))
        return positions

    if side == "top":
        r_values = [0]
        c_values = range(0, ww - box_w + 1)
    elif side == "bottom":
        r_values = [hh - box_h]
        c_values = range(0, ww - box_w + 1)
    elif side == "left":
        r_values = range(0, hh - box_h + 1)
        c_values = [0]
    else:
        r_values = range(0, hh - box_h + 1)
        c_values = [ww - box_w]

    for r0 in r_values:
        for c0 in c_values:
            if _can_place(reserved, r0, c0, box_h, box_w):
                positions.append((int(r0), int(c0)))
    return positions


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, h, w = box
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_legend(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    colors: list[int],
    widths: list[int],
) -> None:
    r0, c0, h, w = box
    row = r0 + 1
    for color, height in zip(colors, widths, strict=True):
        grid[row : row + height, c0 + 1 : c0 + w - 1] = color
        row += height


def _draw_region_marker(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    color: int,
    rng: np.random.Generator,
) -> None:
    r0, c0, h, w = box
    rr = int(rng.integers(r0 + 1, r0 + h - 1))
    cc = int(rng.integers(c0 + 1, c0 + w - 1))
    grid[rr, cc] = color


def _draw_region_output(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    color: int,
    bg: int,
    width: int,
) -> None:
    r0, c0, h, w = box
    for offset in range(h - 2):
        fill = color if (offset // width) % 2 == 0 else bg
        grid[r0 + 1 + offset, c0 + 1 : c0 + w - 1] = fill


def _attempt_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    palette = list(range(10))
    bg, border = map(int, rng.choice(palette, size=2, replace=False))

    region_count = int(rng.integers(GEN_REGION_COUNT[0], GEN_REGION_COUNT[1] + 1))
    marker_pool = [c for c in palette if c not in (bg, border)]
    marker_colors = [int(x) for x in rng.choice(marker_pool, size=region_count, replace=False)]
    stripe_widths = [int(x) for x in rng.choice(GEN_WIDTHS, size=region_count, replace=False)]

    region_specs: list[tuple[int, int, int]] = []
    for width in stripe_widths:
        inner_h_min = max(3, 2 * width + 1)
        inner_h_max = min(12, 4 * width + 1)
        if inner_h_min > inner_h_max:
            return None
        inner_h = int(rng.integers(inner_h_min, inner_h_max + 1))
        inner_w = int(rng.integers(GEN_REGION_INNER_W[0], GEN_REGION_INNER_W[1] + 1))
        region_specs.append((inner_h + 2, inner_w + 2, width))

    legend_inner_h = sum(stripe_widths)
    legend_inner_w = int(rng.integers(GEN_LEGEND_INNER_W[0], GEN_LEGEND_INNER_W[1] + 1))
    legend_box_h = legend_inner_h + 2
    legend_box_w = legend_inner_w + 2
    side = GEN_SIDES[int(rng.integers(len(GEN_SIDES)))]

    min_h = max(GEN_GRID_MIN, legend_box_h + 2)
    min_w = max(GEN_GRID_MIN, legend_box_w + 2)
    if min_h > GEN_GRID_MAX or min_w > GEN_GRID_MAX:
        return None
    grid_h = int(rng.integers(min_h, GEN_GRID_MAX + 1))
    grid_w = int(rng.integers(min_w, GEN_GRID_MAX + 1))

    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    legend_positions = _positions(reserved, legend_box_h, legend_box_w, edge_margin=0, side=side)
    if not legend_positions:
        return None
    legend_box = (*legend_positions[int(rng.integers(len(legend_positions)))], legend_box_h, legend_box_w)
    _reserve(reserved, legend_box[0], legend_box[1], legend_box[2], legend_box[3])

    region_boxes: list[tuple[int, int, int, int] | None] = [None] * region_count
    order = sorted(range(region_count), key=lambda idx: region_specs[idx][0] * region_specs[idx][1], reverse=True)
    for idx in order:
        box_h, box_w, _width = region_specs[idx]
        positions = _positions(reserved, box_h, box_w, edge_margin=1, side=None)
        if not positions:
            return None
        r0, c0 = positions[int(rng.integers(len(positions)))]
        region_boxes[idx] = (r0, c0, box_h, box_w)
        _reserve(reserved, r0, c0, box_h, box_w)

    assert all(box is not None for box in region_boxes)
    final_region_boxes = [box for box in region_boxes if box is not None]

    legend_order = list(range(region_count))
    rng.shuffle(legend_order)
    legend_colors = [marker_colors[idx] for idx in legend_order]
    legend_widths = [stripe_widths[idx] for idx in legend_order]

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = inp.copy()

    _draw_frame(inp, legend_box, border)
    _draw_frame(out, legend_box, border)
    _draw_legend(inp, legend_box, legend_colors, legend_widths)
    _draw_legend(out, legend_box, legend_colors, legend_widths)

    for box, color, width in zip(final_region_boxes, marker_colors, stripe_widths, strict=True):
        _draw_frame(inp, box, border)
        _draw_frame(out, box, border)
        _draw_region_marker(inp, box, color, rng)
        _draw_region_output(out, box, color, bg, width)

    if not _unique_mode_is(inp, bg):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        case = _attempt_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
