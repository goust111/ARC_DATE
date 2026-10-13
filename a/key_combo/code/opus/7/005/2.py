from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

RING_COLORS = (BLUE, RED, GREEN, GREY)


def _depth_map(shape: tuple[int, int]) -> np.ndarray:
    """Return the per-cell distance to the nearest border of a rectangle."""
    height, width = map(int, shape)
    rr, cc = np.indices((height, width))
    return np.minimum.reduce((rr, cc, height - 1 - rr, width - 1 - cc))


def _render_layers(shape: tuple[int, int], layer_colors: list[int]) -> np.ndarray:
    """Paint one color per border-distance layer."""
    depths = _depth_map(shape)
    out = np.zeros(shape, dtype=int)
    for depth, color in enumerate(layer_colors):
        out[depths == depth] = int(color)
    return out


def _extract_layer_colors(grid: np.ndarray) -> list[int]:
    """Read the unique color of every concentric layer in a half-grid."""
    depths = _depth_map(grid.shape)
    layer_count = int(depths.max()) + 1
    colors: list[int] = []
    for depth in range(layer_count):
        vals = np.unique(grid[depths == depth])
        assert len(vals) == 1, f"layer {depth} must be monochromatic"
        colors.append(int(vals[0]))
    return colors


def _split_halves(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep_cols = np.where(np.all(grid == YELLOW, axis=0))[0]
    assert len(sep_cols) == 1, "expected one full-height yellow separator column"
    sep = int(sep_cols[0])
    assert sep == grid.shape[1] // 2, "separator must be centered"
    assert int(np.count_nonzero(grid == YELLOW)) == grid.shape[0], "yellow must appear only in the separator"

    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    assert left.shape == right.shape and left.size > 0, "halves must have equal non-empty shapes"
    return left, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the left and right concentric ring stacks layer by layer and repaint the result rings.

    Input structure:
    - The grid has odd width and contains one full-height yellow separator column at the center.
    - The separator splits the grid into equal left and right halves, each filled by 2-4 one-cell-thick concentric rectangular rings.
    - Every ring layer in each half is monochromatic, and ring colors come from blue, red, green, or grey.

    Rule:
    - Remove the yellow separator and treat each half in its own local coordinate frame.
    - Group cells by their minimum distance to that half's top, bottom, left, or right border, so distance 0 is the outer ring, distance 1 the next ring, and so on.
    - Read the color of each depth layer on the left and right, compare equal depths, and mark that depth green when the two colors differ or black when they match.
    - Output a grid with the same size as one half and paint its concentric layers with those comparison colors.

    Key insight:
    - The comparison unit is an entire depth layer, not an arbitrary local patch: first align both halves by border depth, then convert each layer to a single match/mismatch color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    left, right = _split_halves(g)
    assert not np.any(left == YELLOW) and not np.any(right == YELLOW), "yellow is reserved for the separator"

    left_layers = _extract_layer_colors(left)
    right_layers = _extract_layer_colors(right)
    assert len(left_layers) == len(right_layers), "both halves must have the same number of layers"
    assert 2 <= len(left_layers) <= 4, "expected 2-4 concentric layers"

    out_layers = [
        GREEN if left_color != right_color else BLACK
        for left_color, right_color in zip(left_layers, right_layers)
    ]
    return _render_layers(left.shape, out_layers)


def _sample_half_shape(layer_count: int) -> tuple[int, int]:
    """Choose a half-grid shape whose border-depth count is exactly layer_count."""
    narrow = random.choice((2 * layer_count - 1, 2 * layer_count))
    if random.random() < 0.5:
        height = narrow
        width = random.randint(2 * layer_count - 1, 14)
    else:
        width = narrow
        height = random.randint(2 * layer_count - 1, 16)
    return int(height), int(width)


def _sample_compare_pattern(layer_count: int) -> list[bool]:
    """
    Sample an alternating same/different pattern.

    This guarantees both branches appear and keeps output ring boundaries visible.
    """
    start_same = bool(random.getrandbits(1))
    return [bool((depth + int(start_same)) % 2 == 0) for depth in range(layer_count)]


def _sample_right_colors(left_colors: list[int], same_mask: list[bool]) -> list[int]:
    """Choose right-half colors while preserving visible boundaries between adjacent layers."""
    def backtrack(depth: int, built: list[int]) -> list[int] | None:
        if depth == len(left_colors):
            return built[:]

        left_color = int(left_colors[depth])
        if same_mask[depth]:
            choices = [left_color]
        else:
            choices = [int(color) for color in RING_COLORS if color != left_color]
            random.shuffle(choices)

        for color in choices:
            if depth > 0 and color == built[-1]:
                continue
            built.append(color)
            solved = backtrack(depth + 1, built)
            if solved is not None:
                return solved
            built.pop()
        return None

    right_colors = backtrack(0, [])
    assert right_colors is not None, "failed to assign right-half colors"
    return right_colors


def _assemble_input(
    shape: tuple[int, int],
    left_colors: list[int],
    right_colors: list[int],
) -> np.ndarray:
    height, width = map(int, shape)
    left = _render_layers((height, width), left_colors)
    right = _render_layers((height, width), right_colors)
    separator = np.full((height, 1), YELLOW, dtype=int)
    return np.hstack((left, separator, right))


def _validate_semantics(
    inp: np.ndarray,
    planned_left: list[int] | None = None,
    planned_right: list[int] | None = None,
) -> None:
    """Check puzzle semantics, not just closure through main()."""
    assert isinstance(inp, np.ndarray)
    assert inp.ndim == 2 and max(inp.shape) <= 30

    left, right = _split_halves(inp)
    assert set(np.unique(left)).issubset(set(RING_COLORS))
    assert set(np.unique(right)).issubset(set(RING_COLORS))

    left_layers = _extract_layer_colors(left)
    right_layers = _extract_layer_colors(right)
    layer_count = len(left_layers)
    assert 2 <= layer_count <= 4
    assert all(left_layers[i] != left_layers[i - 1] for i in range(1, layer_count))
    assert all(right_layers[i] != right_layers[i - 1] for i in range(1, layer_count))

    if planned_left is not None:
        assert left_layers == planned_left
    if planned_right is not None:
        assert right_layers == planned_right

    same_mask = [left_color == right_color for left_color, right_color in zip(left_layers, right_layers)]
    assert any(same_mask), "at least one layer must match"
    assert not all(same_mask), "at least one layer must differ"
    assert any(same_mask[i] != same_mask[i - 1] for i in range(1, layer_count)), "comparison branches should both be visible"

    expected = _render_layers(
        left.shape,
        [BLACK if matched else GREEN for matched in same_mask],
    )
    actual = main(inp)
    assert np.array_equal(actual, expected)
    assert set(np.unique(actual)) == {BLACK, GREEN}, "output must contain both black and green"


def generate_input() -> np.ndarray:
    for _ in range(300):
        layer_count = random.randint(2, 4)
        shape = _sample_half_shape(layer_count)
        left_colors = random.sample(RING_COLORS, layer_count)
        same_mask = _sample_compare_pattern(layer_count)
        right_colors = _sample_right_colors(left_colors, same_mask)

        inp = _assemble_input(shape, left_colors, right_colors)
        _validate_semantics(inp, left_colors, right_colors)
        return inp
    raise RuntimeError("failed to generate a valid concentric-ring comparison input")
