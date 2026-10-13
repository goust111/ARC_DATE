from __future__ import annotations

from collections import Counter

import numpy as np


RNG = np.random.default_rng()
PALETTE = np.arange(10, dtype=int)
MAX_GENERATION_ATTEMPTS = 400

# Feasible generation modes under the ARC 30x30 limit.
GENERATION_MODES = (
    {"layers": 3, "step": 1, "inner_h": (3, 10), "inner_w": (3, 7)},
    {"layers": 3, "step": 2, "inner_h": (3, 8), "inner_w": (3, 5)},
    {"layers": 4, "step": 1, "inner_h": (3, 10), "inner_w": (3, 3)},
)


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    bg, top_count = counts.most_common(1)[0]
    assert sum(count == top_count for count in counts.values()) == 1
    return int(bg)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _perimeter_mask(height: int, width: int, offset: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    inner_h = int(height - 2 * offset)
    inner_w = int(width - 2 * offset)
    if inner_h < 3 or inner_w < 3:
        return mask

    r0 = int(offset)
    c0 = int(offset)
    r1 = int(height - 1 - offset)
    c1 = int(width - 1 - offset)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = int(top + height - 1)
    right = int(left + width - 1)
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)


def _nested_frame_candidate(
    grid: np.ndarray, color: int
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]] | None:
    coords = np.argwhere(grid == int(color))
    if len(coords) == 0:
        return None

    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    local = grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)
    height, width = map(int, local.shape)

    valid_offsets: list[int] = []
    expected = np.zeros_like(local, dtype=bool)
    max_offset = int((min(height, width) - 1) // 2)
    for offset in range(max_offset):
        ring = _perimeter_mask(height, width, offset)
        if ring.any() and bool(np.all(local[ring])):
            valid_offsets.append(int(offset))
            expected |= ring

    if len(valid_offsets) < 3 or valid_offsets[0] != 0:
        return None
    if not np.array_equal(local, expected):
        return None

    boxes = [
        (int(r0 + offset), int(c0 + offset), int(r1 - offset), int(c1 - offset))
        for offset in valid_offsets
    ]
    return boxes, bbox


def _detect_frame_layers(
    grid: np.ndarray, bg: int
) -> tuple[int, list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        candidate = _nested_frame_candidate(grid, color)
        if candidate is None:
            continue
        boxes, bbox = candidate
        inner = boxes[-1]
        ir0, ic0, ir1, ic1 = inner
        inside = grid[ir0 + 1 : ir1, ic0 + 1 : ic1]
        if inside.size == 0:
            continue
        if not np.any((inside != int(bg)) & (inside != int(color))):
            continue
        candidates.append((color, boxes, bbox, int(np.sum(grid == color))))

    assert len(candidates) == 1, "expected one nested-frame color"
    color, boxes, bbox, _count = candidates[0]
    return int(color), boxes, bbox


def _read_sample_column(
    grid: np.ndarray, bg: int, outer_bbox: tuple[int, int, int, int], n_layers: int
) -> list[int]:
    r0, c0, r1, c1 = outer_bbox
    outside = np.ones_like(grid, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    sample_coords = np.argwhere(outside & (grid != int(bg)))
    assert len(sample_coords) == n_layers, "sample count must match frame depth"
    sample_coords = sample_coords[np.argsort(sample_coords[:, 0])]
    cols = sample_coords[:, 1]
    assert np.all(cols == cols[0]), "samples must form one vertical column"
    colors = [int(grid[int(r), int(c)]) for r, c in sample_coords]
    assert len(set(colors)) == n_layers, "sample colors must be distinct"
    return colors


def _extract_pattern(
    grid: np.ndarray, bg: int, frame_color: int, inner_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = inner_bbox
    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere((inside != int(bg)) & (inside != int(frame_color)))
    assert len(coords) > 0, "innermost frame must contain a pattern"
    return [(int(r + 1), int(c + 1), int(inside[int(r), int(c)])) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unpack concentric rectangular borders into a horizontal strip recolored by a sample column.

    Input structure:
    - A dominant background fills the canvas.
    - One foreground color forms several concentric axis-aligned rectangular borders with a shared center.
    - The innermost border encloses a small non-background pattern.
    - A separate vertical column of distinct single-pixel color samples sits outside the frame group near the lower-right.

    Rule:
    - Detect the nested borders and order them from innermost to outermost.
    - Read the sample column from top to bottom.
    - Create an output grid by placing the extracted borders left to right, top-aligned, with one background column between neighbors.
    - Recolor the k-th extracted border with the k-th sample color.
    - Preserve the interior pattern only inside the first extracted border; leave every other interior empty.

    Key insight:
    - The thick same-color frame stack is really a depth-ordered sequence of nested perimeters, and the external sample column provides the visible order for recoloring those depths after the nesting is serialized.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color, outer_to_inner, outer_bbox = _detect_frame_layers(g, bg)
    sample_colors = _read_sample_column(g, bg, outer_bbox, len(outer_to_inner))
    inner_bbox = outer_to_inner[-1]
    pattern = _extract_pattern(g, bg, frame_color, inner_bbox)

    layers = list(reversed(outer_to_inner))
    heights = [int(r1 - r0 + 1) for r0, _c0, r1, _c1 in layers]
    widths = [int(c1 - c0 + 1) for _r0, c0, _r1, c1 in layers]
    out_height = max(heights)
    out_width = int(sum(widths) + len(widths) - 1)
    out = np.full((out_height, out_width), int(bg), dtype=int)

    left = 0
    for index, ((r0, c0, r1, c1), color) in enumerate(zip(layers, sample_colors)):
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        _draw_frame(out, 0, left, height, width, int(color))
        if index == 0:
            for pr, pc, pcolor in pattern:
                out[int(pr), int(left + pc)] = int(pcolor)
        left += width + 1

    return out


def _sample_rows(height: int, count: int, bottom_margin: int) -> list[int] | None:
    gaps = RNG.integers(0, 2, size=max(0, count - 1))
    span = int(count + gaps.sum())
    start = int(height - bottom_margin - span)
    if start < 0:
        return None
    rows = [start]
    for gap in gaps:
        rows.append(int(rows[-1] + 1 + gap))
    return rows


def _pattern_pixels(inner_h: int, inner_w: int, palette: np.ndarray) -> list[tuple[int, int, int]]:
    hole_h = int(inner_h - 2)
    hole_w = int(inner_w - 2)
    area = int(hole_h * hole_w)
    count = int(RNG.integers(1, min(area, 5) + 1))
    picks = RNG.choice(area, size=count, replace=False)
    pixels = []
    for flat in picks:
        rr = int(flat // hole_w)
        cc = int(flat % hole_w)
        color = int(palette[int(RNG.integers(len(palette)))])
        pixels.append((rr, cc, color))
    return pixels


def _gen_attempt() -> np.ndarray | None:
    mode = GENERATION_MODES[int(RNG.integers(len(GENERATION_MODES)))]
    n_layers = int(mode["layers"])
    step = int(mode["step"])

    inner_h = int(RNG.integers(mode["inner_h"][0], mode["inner_h"][1] + 1))
    inner_w = int(RNG.integers(mode["inner_w"][0], mode["inner_w"][1] + 1))
    outer_h = int(inner_h + 2 * step * (n_layers - 1))
    outer_w = int(inner_w + 2 * step * (n_layers - 1))

    bottom_margin = int(RNG.integers(1, 4))
    right_margin = int(RNG.integers(1, 3))
    sample_gap_x = int(RNG.integers(2, 6))

    min_h = max(outer_h + 2, n_layers + bottom_margin + 1)
    max_h = min(30, min_h + 8)
    if min_h > max_h:
        return None
    height = int(RNG.integers(min_h, max_h + 1))
    sample_rows = _sample_rows(height, n_layers, bottom_margin)
    if sample_rows is None:
        return None

    min_w = int(outer_w + sample_gap_x + right_margin + 2)
    max_w = min(30, min_w + 8)
    if min_w > max_w:
        return None
    width = int(RNG.integers(min_w, max_w + 1))

    sample_col = int(width - 1 - right_margin)
    c0_max = int(sample_col - sample_gap_x - outer_w)
    if c0_max < 1:
        return None
    r0_max = int(height - outer_h - 1)
    if r0_max < 1:
        return None

    r0 = int(RNG.integers(1, r0_max + 1))
    c0 = int(RNG.integers(1, c0_max + 1))

    needed_colors = int(2 + n_layers + 1)
    if needed_colors > len(PALETTE):
        return None
    chosen = RNG.choice(PALETTE, size=needed_colors, replace=False)
    bg = int(chosen[0])
    frame_color = int(chosen[1])
    sample_colors = [int(x) for x in chosen[2 : 2 + n_layers]]
    pattern_palette = chosen[2 + n_layers :]

    grid = np.full((height, width), int(bg), dtype=int)
    for layer_index in range(n_layers):
        inset = int(layer_index * step)
        top = int(r0 + inset)
        left = int(c0 + inset)
        frame_h = int(outer_h - 2 * inset)
        frame_w = int(outer_w - 2 * inset)
        _draw_frame(grid, top, left, frame_h, frame_w, frame_color)

    pattern = _pattern_pixels(inner_h, inner_w, pattern_palette)
    inner_inset = int(step * (n_layers - 1))
    inner_top = int(r0 + inner_inset)
    inner_left = int(c0 + inner_inset)
    for pr, pc, color in pattern:
        grid[int(inner_top + 1 + pr), int(inner_left + 1 + pc)] = int(color)

    for row, color in zip(sample_rows, sample_colors):
        grid[int(row), int(sample_col)] = int(color)

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
