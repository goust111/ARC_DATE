from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
LAYER_COUNTS = (1, 2, 3)
OUTER_PADDING_RANGE = (1, 3)

SMALL_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if int(coords.shape[0]) != 2 * (height + width) - 4:
        return False
    cells = {(int(r), int(c)) for r, c in coords}
    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    return True


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components_of_color(grid, color):
            if _is_frame_component(comp):
                frames.append({"color": int(color), "bbox": _bbox(comp)})

    assert frames, "expected at least one rectangular frame"

    levels = []
    for frame in frames:
        bbox = frame["bbox"]
        level = sum(_strictly_contains(other["bbox"], bbox) for other in frames if other is not frame)
        levels.append(int(level))

    order = sorted(range(len(frames)), key=lambda i: (levels[i], -_area(frames[i]["bbox"])))
    return [frames[i] for i in order]


def _area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _layer_canvas(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    next_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    layer = np.zeros((inner_h, inner_w), dtype=int)

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if next_bbox is not None:
                nr0, nc0, nr1, nc1 = next_bbox
                if nr0 <= r <= nr1 and nc0 <= c <= nc1:
                    continue
            value = int(grid[r, c])
            if value != BG:
                layer[r - (r0 + 1), c - (c0 + 1)] = value
    return layer


def _stack_layers(layers: list[np.ndarray], frame_colors: list[int]) -> np.ndarray:
    max_width = max(int(layer.shape[1]) for layer in layers)
    total_height = sum(int(layer.shape[0]) for layer in layers) + len(layers) - 1
    out = np.zeros((total_height, max_width), dtype=int)

    row = 0
    for idx, layer in enumerate(layers):
        height, width = map(int, layer.shape)
        out[row : row + height, :width] = layer
        row += height
        if idx + 1 < len(layers):
            out[row, :] = int(frame_colors[idx])
            row += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unwrap nested colored rectangular frames and stack each layer's interior markers from outermost to innermost.

    Input structure:
    - The background is black.
    - Several hollow axis-aligned rectangular frames are nested strictly inside one another, with black gaps between consecutive frames.
    - Each frame has its own border color, and the ring between one frame and the next deeper frame contains one or more small same-colored feature blocks.
    - The innermost frame contains only its own feature blocks and no deeper frame.

    Rule:
    - Detect every hollow rectangular frame and order them by containment from the outside inward.
    - For each frame, take the pixels inside that frame but outside the next deeper frame's full bounding box; preserve those feature pixels at the same offsets from the current frame interior's top-left corner.
    - Build an output whose width is the maximum frame-interior width and whose height is the sum of all frame-interior heights plus one separator row between consecutive layers.
    - Stack the extracted layer contents from top to bottom, and insert a full-width separator row in the current frame's border color after each layer except the last.

    Key insight:
    - The frames define a single containment chain, so each layer can be isolated by subtracting the next inner frame's whole box and then linearly reassembled in nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    layers: list[np.ndarray] = []
    frame_colors: list[int] = []

    for idx, frame in enumerate(frames):
        bbox = frame["bbox"]
        next_bbox = frames[idx + 1]["bbox"] if idx + 1 < len(frames) else None
        layers.append(_layer_canvas(g, bbox, next_bbox))
        frame_colors.append(int(frame["color"]))

    return _stack_layers(layers, frame_colors)


def _choose_layout() -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]], tuple[int, int, int, int]] | None:
    layer_count = random.choice(LAYER_COUNTS)
    if layer_count == 3:
        inner_h = random.randint(3, 4)
        inner_w = random.randint(3, 5)
        gap_hi = 2
    elif layer_count == 2:
        inner_h = random.randint(3, 7)
        inner_w = random.randint(3, 8)
        gap_hi = 3
    else:
        inner_h = random.randint(3, 12)
        inner_w = random.randint(3, 12)
        gap_hi = 0

    sizes: list[tuple[int, int] | None] = [None] * layer_count
    gaps: list[tuple[int, int, int, int] | None] = [None] * max(0, layer_count - 1)
    sizes[-1] = (inner_h, inner_w)

    cur_h, cur_w = inner_h, inner_w
    for idx in range(layer_count - 2, -1, -1):
        top = random.randint(1, gap_hi)
        bottom = random.randint(1, gap_hi)
        left = random.randint(1, gap_hi)
        right = random.randint(1, gap_hi)
        cur_h = cur_h + 2 + top + bottom
        cur_w = cur_w + 2 + left + right
        sizes[idx] = (cur_h, cur_w)
        gaps[idx] = (top, bottom, left, right)

    layer_sizes = [size for size in sizes if size is not None]
    layer_gaps = [gap for gap in gaps if gap is not None]
    output_h = sum(h for h, _ in layer_sizes) + layer_count - 1
    output_w = max(w for _, w in layer_sizes)
    outer_h = layer_sizes[0][0] + 2
    outer_w = layer_sizes[0][1] + 2
    if output_h > 30 or output_w > 30 or outer_h > 30 or outer_w > 30:
        return None

    max_top = min(OUTER_PADDING_RANGE[1], 30 - outer_h)
    max_left = min(OUTER_PADDING_RANGE[1], 30 - outer_w)
    if max_top < OUTER_PADDING_RANGE[0] or max_left < OUTER_PADDING_RANGE[0]:
        return None

    pad_top = random.randint(OUTER_PADDING_RANGE[0], max_top)
    pad_bottom = random.randint(OUTER_PADDING_RANGE[0], min(OUTER_PADDING_RANGE[1], 30 - outer_h - pad_top))
    pad_left = random.randint(OUTER_PADDING_RANGE[0], max_left)
    pad_right = random.randint(OUTER_PADDING_RANGE[0], min(OUTER_PADDING_RANGE[1], 30 - outer_w - pad_left))
    return layer_sizes, layer_gaps, (pad_top, pad_bottom, pad_left, pad_right)


def _allowed_mask(
    size: tuple[int, int],
    next_size: tuple[int, int] | None,
    gap: tuple[int, int, int, int] | None,
) -> np.ndarray:
    height, width = size
    mask = np.ones((height, width), dtype=bool)
    if next_size is None or gap is None:
        return mask
    top, _bottom, left, _right = gap
    next_h, next_w = next_size
    mask[top : top + next_h + 2, left : left + next_w + 2] = False
    return mask


def _fits_shape(
    occupied: np.ndarray,
    allowed: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    r0: int,
    c0: int,
) -> bool:
    height, width = map(int, allowed.shape)
    cells = []
    for dr, dc in shape:
        rr = r0 + dr
        cc = c0 + dc
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        if not allowed[rr, cc]:
            return False
        cells.append((rr, cc))
    for rr, cc in cells:
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < height and 0 <= nc < width and occupied[nr, nc]:
                    return False
    return True


def _place_features(
    size: tuple[int, int],
    next_size: tuple[int, int] | None,
    gap: tuple[int, int, int, int] | None,
    color: int,
) -> np.ndarray | None:
    allowed = _allowed_mask(size, next_size, gap)
    allowed_count = int(allowed.sum())
    if allowed_count == 0:
        return None

    canvas = np.zeros(size, dtype=int)
    occupied = np.zeros(size, dtype=bool)
    max_components = min(4, max(1, allowed_count // 3))
    target_components = random.randint(1, max_components)

    placed = 0
    while placed < target_components:
        placements: list[tuple[tuple[tuple[int, int], ...], int, int]] = []
        shapes = list(SMALL_SHAPES)
        random.shuffle(shapes)
        for shape in shapes:
            for r in range(size[0]):
                for c in range(size[1]):
                    if _fits_shape(occupied, allowed, shape, r, c):
                        placements.append((shape, r, c))
        if not placements:
            if placed == 0:
                return None
            break
        shape, r0, c0 = random.choice(placements)
        for dr, dc in shape:
            rr = r0 + dr
            cc = c0 + dc
            canvas[rr, cc] = int(color)
            occupied[rr, cc] = True
        placed += 1

    return canvas if placed > 0 else None


def _build_expected(layers: list[np.ndarray], frame_colors: list[int]) -> np.ndarray:
    return _stack_layers(layers, frame_colors)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _render_input(
    layer_sizes: list[tuple[int, int]],
    layer_gaps: list[tuple[int, int, int, int]],
    padding: tuple[int, int, int, int],
    border_colors: list[int],
    layers: list[np.ndarray],
) -> np.ndarray:
    pad_top, pad_bottom, pad_left, pad_right = padding
    outer_h = layer_sizes[0][0] + 2
    outer_w = layer_sizes[0][1] + 2
    grid_h = pad_top + outer_h + pad_bottom
    grid_w = pad_left + outer_w + pad_right
    grid = np.zeros((grid_h, grid_w), dtype=int)

    r0 = pad_top
    c0 = pad_left
    for idx, size in enumerate(layer_sizes):
        inner_h, inner_w = size
        bbox = (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1)
        _draw_frame(grid, bbox, border_colors[idx])
        layer = layers[idx]
        mask = layer != BG
        grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w][mask] = layer[mask]

        if idx + 1 < len(layer_sizes):
            top, _bottom, left, _right = layer_gaps[idx]
            r0 = r0 + 1 + top
            c0 = c0 + 1 + left

    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None
    layer_sizes, layer_gaps, padding = layout
    layer_count = len(layer_sizes)

    palette = random.sample(COLORS, 2 * layer_count)
    border_colors = palette[:layer_count]
    feature_colors = palette[layer_count:]

    layers: list[np.ndarray] = []
    for idx, size in enumerate(layer_sizes):
        next_size = layer_sizes[idx + 1] if idx + 1 < layer_count else None
        gap = layer_gaps[idx] if idx < len(layer_gaps) else None
        canvas = _place_features(size, next_size, gap, feature_colors[idx])
        if canvas is None:
            return None
        layers.append(canvas)

    expected = _build_expected(layers, border_colors)
    if expected.ndim != 2 or max(expected.shape) > 30:
        return None

    grid = _render_input(layer_sizes, layer_gaps, padding, border_colors, layers)
    if grid.ndim != 2 or max(grid.shape) > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
