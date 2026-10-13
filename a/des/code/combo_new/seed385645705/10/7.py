from __future__ import annotations

import random

import numpy as np


BLACK = 0
AXIS_COLOR = 9
MARKER_COLORS = (1, 2)
OBJECT_COLORS = (3, 4, 5, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400

_BASE_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _axis_rows(g: np.ndarray) -> list[tuple[int, int, int]]:
    hits = []
    width = g.shape[1]
    for r in range(g.shape[0]):
        line = g[r, :]
        non_axis = np.flatnonzero(line != AXIS_COLOR)
        if non_axis.size != 1:
            continue
        marker_col = int(non_axis[0])
        marker_color = int(line[marker_col])
        if marker_color != BLACK and int(np.count_nonzero(line == AXIS_COLOR)) == width - 1:
            hits.append((r, marker_col, marker_color))
    return hits


def _axis_cols(g: np.ndarray) -> list[tuple[int, int, int]]:
    hits = []
    height = g.shape[0]
    for c in range(g.shape[1]):
        line = g[:, c]
        non_axis = np.flatnonzero(line != AXIS_COLOR)
        if non_axis.size != 1:
            continue
        marker_row = int(non_axis[0])
        marker_color = int(line[marker_row])
        if marker_color != BLACK and int(np.count_nonzero(line == AXIS_COLOR)) == height - 1:
            hits.append((c, marker_row, marker_color))
    return hits


def _merge_layers(stationary: np.ndarray, folded: np.ndarray, marker_color: int) -> np.ndarray:
    stationary = np.asarray(stationary, dtype=int)
    folded = np.asarray(folded, dtype=int)
    assert stationary.shape == folded.shape

    out = stationary.copy()
    stationary_fg = stationary != BLACK
    folded_fg = folded != BLACK

    take_folded = ~stationary_fg & folded_fg
    out[take_folded] = folded[take_folded]

    marker_priority = stationary_fg & folded_fg & (
        (stationary == marker_color) | (folded == marker_color)
    )
    out[marker_priority] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the marked half of the grid across the maroon axis and merge the overlap with marker-color priority.

    Input structure:
    - Black is the background.
    - Exactly one full maroon line spans the whole grid as a horizontal row or vertical column; one cell on that line is replaced by a non-black direction marker.
    - The axis splits the grid into two equal halves, each containing colored pattern pixels away from the axis.

    Rule:
    - Identify the unique maroon axis and its off-color marker.
    - For a horizontal axis, a marker nearer the left end folds the top half downward, while a marker nearer the right end folds the bottom half upward.
    - For a vertical axis, a marker nearer the top end folds the left half rightward, while a marker nearer the bottom end folds the right half leftward.
    - Mirror the folding half across the axis, discard the axis itself, and merge onto the stationary half: marker color beats any non-black conflict, otherwise the stationary-half pixel wins when both sides are non-black, and lone non-black pixels are kept.

    Key insight:
    - The odd-colored cell on the maroon axis does two jobs at once: its position chooses which half folds, and its color is also the top-priority color in overlap conflicts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape

    row_axes = _axis_rows(g)
    col_axes = _axis_cols(g)
    assert len(row_axes) + len(col_axes) == 1, "expected exactly one marked fold axis"

    if row_axes:
        axis_row, marker_col, marker_color = row_axes[0]
        top = g[:axis_row, :]
        bottom = g[axis_row + 1 :, :]
        assert top.shape == bottom.shape and top.size > 0, "horizontal axis must split equal halves"
        assert marker_col != width - 1 - marker_col, "direction marker must favor one axis end"
        if marker_col < width - 1 - marker_col:
            stationary = bottom
            folded = np.flipud(top)
        else:
            stationary = top
            folded = np.flipud(bottom)
        return _merge_layers(stationary, folded, marker_color)

    axis_col, marker_row, marker_color = col_axes[0]
    left = g[:, :axis_col]
    right = g[:, axis_col + 1 :]
    assert left.shape == right.shape and left.size > 0, "vertical axis must split equal halves"
    assert marker_row != height - 1 - marker_row, "direction marker must favor one axis end"
    if marker_row < height - 1 - marker_row:
        stationary = right
        folded = np.fliplr(left)
    else:
        stationary = left
        folded = np.fliplr(right)
    return _merge_layers(stationary, folded, marker_color)


def _shape_variants(shape: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(shape, k)
        for candidate in (rotated, np.fliplr(rotated)):
            key = (candidate.shape, candidate.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(candidate.copy())
    return variants


_SHAPES = []
for _base in _BASE_SHAPES:
    _SHAPES.extend(_shape_variants(_base))


def _random_shape(max_h: int, max_w: int) -> np.ndarray:
    candidates = [shape for shape in _SHAPES if shape.shape[0] <= max_h and shape.shape[1] <= max_w]
    assert candidates, "no shape fits the target region"
    return random.choice(candidates).copy()


def _split_regions(height: int, width: int) -> list[tuple[int, int, int, int]]:
    row_mid = height // 2
    col_mid = width // 2
    return [
        (0, row_mid, 0, col_mid),
        (0, row_mid, col_mid, width),
        (row_mid, height, 0, col_mid),
        (row_mid, height, col_mid, width),
    ]


def _place_shape(
    layer: np.ndarray,
    region: tuple[int, int, int, int],
    shape: np.ndarray,
    color: int,
) -> np.ndarray:
    r0, r1, c0, c1 = region
    sh, sw = shape.shape
    assert r1 - r0 >= sh and c1 - c0 >= sw
    anchor_r = random.randint(r0, r1 - sh)
    anchor_c = random.randint(c0, c1 - sw)
    view = layer[anchor_r : anchor_r + sh, anchor_c : anchor_c + sw]
    view[shape] = color
    return np.argwhere(shape) + np.array([anchor_r, anchor_c])


def _add_extra_shape(
    stationary: np.ndarray,
    folded: np.ndarray,
    color: int,
) -> bool:
    height, width = stationary.shape
    tiny_shapes = [shape for shape in _SHAPES if shape.shape[0] <= 3 and shape.shape[1] <= 3]
    target = stationary if random.random() < 0.5 else folded
    other = folded if target is stationary else stationary

    for _ in range(50):
        shape = random.choice(tiny_shapes)
        sh, sw = shape.shape
        anchor_r = random.randint(0, height - sh)
        anchor_c = random.randint(0, width - sw)
        target_view = target[anchor_r : anchor_r + sh, anchor_c : anchor_c + sw]
        other_view = other[anchor_r : anchor_r + sh, anchor_c : anchor_c + sw]
        if np.any(target_view[shape] != BLACK):
            continue
        if np.any(other_view[shape] != BLACK):
            continue
        target_view[shape] = color
        return True
    return False


def _biased_axis_marker(length: int, toward_start: bool) -> int:
    band = max(2, length // 3)
    if toward_start:
        hi = min(length - 1, band - 1)
        return random.randint(0, hi)
    lo = max(0, length - band)
    return random.randint(lo, length - 1)


def _assemble_input(
    stationary: np.ndarray,
    folded: np.ndarray,
    orientation: str,
    fold_first_half: bool,
    marker_color: int,
) -> np.ndarray:
    out_h, out_w = stationary.shape
    if orientation == "horizontal":
        grid = np.full((2 * out_h + 1, out_w), BLACK, dtype=int)
        axis_row = out_h
        if fold_first_half:
            grid[:axis_row, :] = np.flipud(folded)
            grid[axis_row + 1 :, :] = stationary
            marker_col = _biased_axis_marker(out_w, toward_start=True)
        else:
            grid[:axis_row, :] = stationary
            grid[axis_row + 1 :, :] = np.flipud(folded)
            marker_col = _biased_axis_marker(out_w, toward_start=False)
        grid[axis_row, :] = AXIS_COLOR
        grid[axis_row, marker_col] = marker_color
        return grid

    grid = np.full((out_h, 2 * out_w + 1), BLACK, dtype=int)
    axis_col = out_w
    if fold_first_half:
        grid[:, :axis_col] = np.fliplr(folded)
        grid[:, axis_col + 1 :] = stationary
        marker_row = _biased_axis_marker(out_h, toward_start=True)
    else:
        grid[:, :axis_col] = stationary
        grid[:, axis_col + 1 :] = np.fliplr(folded)
        marker_row = _biased_axis_marker(out_h, toward_start=False)
    grid[:, axis_col] = AXIS_COLOR
    grid[marker_row, axis_col] = marker_color
    return grid


def _sample_dimensions(orientation: str) -> tuple[int, int]:
    if orientation == "horizontal":
        return random.randint(8, 14), random.randint(8, 20)
    return random.randint(8, 20), random.randint(8, 14)


def _build_layers(shape: tuple[int, int], marker_color: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    stationary = np.full((height, width), BLACK, dtype=int)
    folded = np.full((height, width), BLACK, dtype=int)

    regions = _split_regions(height, width)
    random.shuffle(regions)
    color_pool = random.sample(OBJECT_COLORS, 5)
    marker_on_stationary = random.choice((True, False))

    feature_specs = [
        ("stationary_only", color_pool[0], None),
        ("folded_only", None, color_pool[1]),
        ("stationary_wins", color_pool[2], color_pool[3]),
        (
            "marker_wins",
            marker_color if marker_on_stationary else color_pool[4],
            color_pool[4] if marker_on_stationary else marker_color,
        ),
    ]

    for region, (kind, stationary_color, folded_color) in zip(regions, feature_specs):
        r0, r1, c0, c1 = region
        shape_mask = _random_shape(r1 - r0, c1 - c0)
        coords = None
        if kind == "stationary_only":
            _place_shape(stationary, region, shape_mask, int(stationary_color))
        elif kind == "folded_only":
            _place_shape(folded, region, shape_mask, int(folded_color))
        else:
            coords = _place_shape(stationary, region, shape_mask, int(stationary_color))
            for rr, cc in coords:
                folded[int(rr), int(cc)] = int(folded_color)

    extra_colors = [color for color in OBJECT_COLORS if color not in color_pool]
    for _ in range(random.randint(0, 2)):
        _add_extra_shape(stationary, folded, random.choice(extra_colors))

    assert np.any(stationary != BLACK)
    assert np.any(folded != BLACK)
    assert not np.array_equal(stationary, folded)
    return stationary, folded


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("horizontal", "vertical"))
    fold_first_half = random.choice((True, False))
    marker_color = random.choice(MARKER_COLORS)
    shape = _sample_dimensions(orientation)

    stationary, folded = _build_layers(shape, marker_color)
    inp = _assemble_input(stationary, folded, orientation, fold_first_half, marker_color)
    expected = _merge_layers(stationary, folded, marker_color)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-axis puzzle input")
