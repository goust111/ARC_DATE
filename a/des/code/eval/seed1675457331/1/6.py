import numpy as np


BG = 0
_RNG = np.random.default_rng()

# Generator configuration.
_COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_LAYER_COUNT_RANGE = (2, 4)
_INNER_SIZE_RANGE = (3, 9)
_MARGIN_RANGE = (1, 5)
_FILL_PROBABILITY = 0.5
_GEN_RETRIES = 400


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_color(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    color = int(grid[r0, c0])
    if color == BG:
        return None
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return None
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return None
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return None
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return None

    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
        return None
    return color


def _extract_fill(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[int, int, int, int, int] | None:
    r0, c0, r1, c1 = bbox
    hole = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = hole != BG
    fill_bbox = _bbox_of_mask(mask)
    if fill_bbox is None:
        return None

    fr0, fc0, fr1, fc1 = fill_bbox
    block = hole[fr0 : fr1 + 1, fc0 : fc1 + 1]
    color = int(block[0, 0])
    assert color != BG, "fill rectangle must be non-black"
    assert np.all(block == color), "fill must be a solid rectangle"
    assert int(mask.sum()) == block.size, "innermost content must be a single rectangle"
    return int(fr0 + 1), int(fc0 + 1), int(fr1 + 1), int(fc1 + 1), color


def _extract_layers(
    grid: np.ndarray,
) -> tuple[list[tuple[tuple[int, int, int, int], int]], tuple[int, int, int, int, int] | None]:
    bbox = _bbox_of_mask(grid != BG)
    assert bbox is not None, "expected a non-empty nested structure"

    layers: list[tuple[tuple[int, int, int, int], int]] = []
    seen_colors: set[int] = set()

    while True:
        color = _frame_color(grid, bbox)
        assert color is not None, f"expected a hollow frame at {bbox}"
        assert color not in seen_colors, "frame colors must be distinct"
        layers.append((bbox, color))
        seen_colors.add(color)

        r0, c0, r1, c1 = bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_bbox_rel = _bbox_of_mask(inner != BG)
        if inner_bbox_rel is None:
            return layers, None

        ir0, ic0, ir1, ic1 = inner_bbox_rel
        inner_bbox = (r0 + 1 + ir0, c0 + 1 + ic0, r0 + 1 + ir1, c0 + 1 + ic1)
        expected_next = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)

        if inner_bbox == expected_next:
            next_color = _frame_color(grid, inner_bbox)
            if next_color is not None:
                bbox = inner_bbox
                continue

        fill = _extract_fill(grid, bbox)
        assert fill is not None, "expected a single fill rectangle inside the innermost frame"
        return layers, fill


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_frame_piece(
    height: int, width: int, color: int, fill: tuple[int, int, int, int, int] | None = None
) -> np.ndarray:
    piece = np.zeros((height, width), dtype=int)
    piece[0, :] = color
    piece[-1, :] = color
    piece[:, 0] = color
    piece[:, -1] = color
    if fill is not None:
        fr0, fc0, fr1, fc1, fill_color = fill
        piece[fr0 : fr1 + 1, fc0 : fc1 + 1] = fill_color
    return piece


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unpack a nested frame stack into side-by-side frames ordered from inner to outer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    layers, fill = _extract_layers(g)
    assert 2 <= len(layers) <= 4, "expected between 2 and 4 nested frames"

    pieces: list[np.ndarray] = []
    innermost_bbox, _innermost_color = layers[-1]
    inner_height = int(innermost_bbox[2] - innermost_bbox[0] + 1)
    inner_width = int(innermost_bbox[3] - innermost_bbox[1] + 1)

    for index, (bbox, color) in enumerate(reversed(layers)):
        height = int(bbox[2] - bbox[0] + 1)
        width = int(bbox[3] - bbox[1] + 1)
        piece_fill = fill if index == 0 and height == inner_height and width == inner_width else None
        pieces.append(_build_frame_piece(height, width, color, piece_fill))

    out_height = max(int(piece.shape[0]) for piece in pieces)
    out_width = sum(int(piece.shape[1]) for piece in pieces) + len(pieces) - 1
    out = np.zeros((out_height, out_width), dtype=int)

    c = 0
    for piece in pieces:
        h, w = map(int, piece.shape)
        out[:h, c : c + w] = piece
        c += w + 1

    return out


def _gen_attempt() -> np.ndarray | None:
    layer_count = int(_RNG.integers(_LAYER_COUNT_RANGE[0], _LAYER_COUNT_RANGE[1] + 1))
    inner_h = int(_RNG.integers(_INNER_SIZE_RANGE[0], _INNER_SIZE_RANGE[1] + 1))
    inner_w = int(_RNG.integers(_INNER_SIZE_RANGE[0], _INNER_SIZE_RANGE[1] + 1))

    total_h = inner_h + 4 * (layer_count - 1)
    total_w = inner_w + 4 * (layer_count - 1)

    top = int(_RNG.integers(_MARGIN_RANGE[0], _MARGIN_RANGE[1] + 1))
    bottom = int(_RNG.integers(_MARGIN_RANGE[0], _MARGIN_RANGE[1] + 1))
    left = int(_RNG.integers(_MARGIN_RANGE[0], _MARGIN_RANGE[1] + 1))
    right = int(_RNG.integers(_MARGIN_RANGE[0], _MARGIN_RANGE[1] + 1))

    grid = np.zeros((top + total_h + bottom, left + total_w + right), dtype=int)

    frame_colors = _RNG.choice(_COLOR_POOL, size=layer_count, replace=False)
    for depth, color in enumerate(frame_colors):
        inset = 2 * depth
        bbox = (
            top + inset,
            left + inset,
            top + total_h - 1 - inset,
            left + total_w - 1 - inset,
        )
        _draw_frame(grid, bbox, int(color))

    if float(_RNG.random()) < _FILL_PROBABILITY:
        inner_bbox = (
            top + 2 * (layer_count - 1),
            left + 2 * (layer_count - 1),
            top + total_h - 1 - 2 * (layer_count - 1),
            left + total_w - 1 - 2 * (layer_count - 1),
        )
        hole_h = int(inner_bbox[2] - inner_bbox[0] - 1)
        hole_w = int(inner_bbox[3] - inner_bbox[1] - 1)
        assert hole_h >= 1 and hole_w >= 1

        fill_h = int(_RNG.integers(1, hole_h + 1))
        fill_w = int(_RNG.integers(1, hole_w + 1))
        fill_r0 = int(inner_bbox[0] + 1 + _RNG.integers(0, hole_h - fill_h + 1))
        fill_c0 = int(inner_bbox[1] + 1 + _RNG.integers(0, hole_w - fill_w + 1))

        remaining_colors = [int(color) for color in _COLOR_POOL if int(color) not in set(map(int, frame_colors))]
        fill_color = int(_RNG.choice(remaining_colors))
        grid[fill_r0 : fill_r0 + fill_h, fill_c0 : fill_c0 + fill_w] = fill_color

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
