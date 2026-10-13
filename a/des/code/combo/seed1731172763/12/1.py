from __future__ import annotations

import math

import numpy as np


RNG = np.random.default_rng()
BG = 0
DIRS = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 300
OBJECT_COUNT_CHOICES = (4, 5)
MIN_SHAPE_DIM = 3
MAX_STANDARD_HEIGHT = 15
MAX_STANDARD_WIDTH = 17
PLACEMENT_MARGIN = 1
PLACEMENT_TRIES = 250


def _most_common_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=max(10, int(values.max()) + 1))
    return int(np.argmax(counts))


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
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))

    return comps


def _component_to_shape(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    color_values = np.unique(grid[rows, cols])
    assert color_values.size == 1, "each object must be monochromatic"
    color = int(color_values[0])

    shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    shape[rows - r0, cols - c0] = color
    return shape


def _opening_direction(shape: np.ndarray) -> str:
    mask = shape != BG
    assert mask.ndim == 2 and mask.any(), "object must be non-empty"
    top_closed = bool(mask[0, :].all())
    bottom_closed = bool(mask[-1, :].all())
    left_closed = bool(mask[:, 0].all())
    right_closed = bool(mask[:, -1].all())
    openings = [
        direction
        for direction, closed in (
            ("up", top_closed),
            ("down", bottom_closed),
            ("left", left_closed),
            ("right", right_closed),
        )
        if not closed
    ]
    assert len(openings) == 1, "expected exactly one opening side"
    return openings[0]


def _make_up_u(height: int, width: int, color: int) -> np.ndarray:
    assert height >= MIN_SHAPE_DIM and width >= MIN_SHAPE_DIM
    shape = np.zeros((height, width), dtype=int)
    shape[:, 0] = color
    shape[:, -1] = color
    shape[-1, :] = color
    return shape


def _rotate_up_shape_to_direction(shape: np.ndarray, direction: str) -> np.ndarray:
    k_by_direction = {"up": 0, "left": 1, "down": 2, "right": 3}
    return np.rot90(shape, k_by_direction[direction])


def _normalize_to_up(shape: np.ndarray) -> np.ndarray:
    open_dir = _opening_direction(shape)
    k_to_up = {"up": 0, "right": 1, "down": 2, "left": 3}
    rotated = np.rot90(shape, k_to_up[open_dir])
    color_values = np.unique(rotated[rotated != BG])
    assert color_values.size == 1, "rotated object must stay monochromatic"
    color = int(color_values[0])
    expected = _make_up_u(rotated.shape[0], rotated.shape[1], color)
    assert np.array_equal(rotated, expected), "object must be a one-cell-thick three-sided frame"
    return rotated


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    sub = dst[top : top + h, left : left + w]
    mask = src != BG
    assert not np.any(sub[mask] != BG), "nested objects must not overlap"
    sub[mask] = src[mask]


def _assemble_nested(up_shapes: list[np.ndarray]) -> np.ndarray:
    ordered = sorted(
        up_shapes,
        key=lambda shape: (
            -int(np.count_nonzero(shape)),
            -int(shape.shape[0]),
            -int(shape.shape[1]),
        ),
    )
    outer_h, outer_w = map(int, ordered[0].shape)
    out = np.zeros((outer_h, outer_w), dtype=int)

    top = 0
    left = 0
    prev_h = outer_h
    prev_w = outer_w
    for shape in ordered:
        h, w = map(int, shape.shape)
        if shape is not ordered[0]:
            top = top + prev_h - h - 2
            left = left + (prev_w - w) // 2
        assert 0 <= top <= outer_h - h
        assert 0 <= left <= outer_w - w
        _paste_nonzero(out, shape, top, left)
        prev_h = h
        prev_w = w

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate open-sided rectangular frames to face up, then bottom-center nest them by size."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    shapes = [_component_to_shape(g, comp) for comp in _components(g != bg)]
    assert shapes, "expected at least one foreground object"

    normalized = [_normalize_to_up(shape) for shape in shapes]
    return _assemble_nested(normalized)


def _sample_nested_dimensions(count: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        smallest_w = int(RNG.choice(np.array([3, 4, 5, 6])))
        smallest_h = int(RNG.integers(3, 7))
        widths = [smallest_w]
        heights = [smallest_h]
        for _ in range(count - 1):
            widths.append(int(widths[-1] + 2 * RNG.integers(1, 3)))
            heights.append(int(heights[-1] + RNG.integers(2, 5)))
        if widths[-1] <= MAX_STANDARD_WIDTH and heights[-1] <= MAX_STANDARD_HEIGHT:
            dims = list(reversed(list(zip(heights, widths))))
            return [(int(h), int(w)) for h, w in dims]
    return None


def _sample_directions(count: int) -> list[str]:
    directions = list(DIRS)
    RNG.shuffle(directions)
    if count == 4:
        return directions
    extra = directions[int(RNG.integers(0, len(directions)))]
    directions.append(extra)
    RNG.shuffle(directions)
    return directions


def _choose_canvas(shapes: list[np.ndarray]) -> tuple[int, int] | None:
    dims = np.array([shape.shape for shape in shapes], dtype=int)
    max_h = int(dims[:, 0].max())
    max_w = int(dims[:, 1].max())
    padded_area = int(sum((h + 2) * (w + 2) for h, w in dims))
    base = max(max_h + 4, max_w + 4, int(math.ceil(math.sqrt(padded_area * 1.2))))
    h = int(base + RNG.integers(0, 4))
    w = int(base + RNG.integers(0, 4))
    if h > 30 or w > 30:
        return None
    return h, w


def _can_place_bbox(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - PLACEMENT_MARGIN)
    c0 = max(0, left - PLACEMENT_MARGIN)
    r1 = min(h, top + height + PLACEMENT_MARGIN)
    c1 = min(w, left + width + PLACEMENT_MARGIN)
    return not bool(occupied[r0:r1, c0:c1].any())


def _render_scattered_input(shapes: list[np.ndarray]) -> np.ndarray | None:
    canvas = _choose_canvas(shapes)
    if canvas is None:
        return None
    grid_h, grid_w = canvas
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(range(len(shapes)), key=lambda i: -int(np.prod(shapes[i].shape)))
    for idx in order:
        shape = shapes[idx]
        h, w = map(int, shape.shape)
        if grid_h - h < 2 or grid_w - w < 2:
            return None
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = int(RNG.integers(1, grid_h - h))
            left = int(RNG.integers(1, grid_w - w))
            if not _can_place_bbox(occupied, top, left, h, w):
                continue
            sub = grid[top : top + h, left : left + w]
            mask = shape != BG
            sub[mask] = shape[mask]
            occupied[top : top + h, left : left + w] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.choice(np.array(OBJECT_COUNT_CHOICES)))
    dims = _sample_nested_dimensions(count)
    if dims is None:
        return None

    colors = RNG.choice(np.arange(1, 10), size=count, replace=False)
    standard_shapes = [
        _make_up_u(height, width, int(color))
        for (height, width), color in zip(dims, colors, strict=True)
    ]
    directions = _sample_directions(count)
    input_shapes = [
        _rotate_up_shape_to_direction(shape, direction)
        for shape, direction in zip(standard_shapes, directions, strict=True)
    ]

    grid = _render_scattered_input(input_shapes)
    if grid is None:
        return None

    components = _components(grid != BG)
    if len(components) != count:
        return None

    expected = _assemble_nested(standard_shapes)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
