from __future__ import annotations

import random

import numpy as np


SEPARATOR = 4
BLACK = 0
PALETTE = (1, 2, 3, 5, 6, 7, 8)
GEN_ATTEMPTS = 400


def _depth_map(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    return np.minimum.reduce([rr, cc, h - 1 - rr, w - 1 - cc])


def _render_panel(layers: list[int], shape: tuple[int, int]) -> np.ndarray:
    depth = _depth_map(shape)
    panel = np.empty(shape, dtype=int)
    for d, color in enumerate(layers):
        panel[depth == d] = int(color)
    return panel


def _extract_layers(panel: np.ndarray) -> list[int]:
    g = np.asarray(panel, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty panel"
    depth = _depth_map(g.shape)
    n_layers = int(depth.max()) + 1
    layers: list[int] = []
    for d in range(n_layers):
        colors = np.unique(g[depth == d])
        assert len(colors) == 1, f"depth {d} must be monochrome"
        layers.append(int(colors[0]))
    return layers


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_cols = np.flatnonzero(np.all(g == SEPARATOR, axis=0))
    assert len(sep_cols) == 1, "expected exactly one full separator column"
    sep = int(sep_cols[0])
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0, "separator must split two non-empty panels"
    assert left.shape == right.shape, "expected equal-size left and right panels"
    assert not np.any(left == SEPARATOR) and not np.any(right == SEPARATOR), "separator color must be unique"
    return left, right


def _expected_from_layers(left_layers: list[int], right_layers: list[int], shape: tuple[int, int]) -> np.ndarray:
    n_layers = len(left_layers)
    assert n_layers == len(right_layers) and n_layers > 0
    out_layers = []
    for d in range(n_layers):
        if left_layers[d] == right_layers[d]:
            out_layers.append(BLACK)
        else:
            out_layers.append(left_layers[n_layers - 1 - d])
    return _render_panel(out_layers, shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two concentric-ring panels depth by depth and recolor only the differing layers with reversed left-panel depth colors.

    Input structure:
    - A single full yellow separator column splits the grid into equal-size left and right panels.
    - Each panel is completely filled by concentric rectangular rings, where every cell with the same
      minimum distance to that panel's border shares one color.
    - The separator color appears only in the middle column, and each panel has the same number of layers.

    Rule:
    - Locate the separator column and split the grid into the two panels.
    - Read each panel's layer colors from outermost depth to innermost depth.
    - For every depth d, output black if the two panels have the same color at depth d.
    - Otherwise output the color from the mirrored left-panel depth, namely left[N-1-d], on that whole ring.

    Key insight:
    The depthwise comparison only decides which rings survive; the surviving rings are not colored by
    their own depth, but by the left panel read from the opposite depth order.
    """
    left, right = _split_panels(np.asarray(grid, dtype=int))
    left_layers = _extract_layers(left)
    right_layers = _extract_layers(right)
    assert len(left_layers) == len(right_layers), "both panels must have the same layer count"
    return _expected_from_layers(left_layers, right_layers, left.shape)


def _panel_shape_for_layers(n_layers: int) -> tuple[int, int]:
    min_side = random.choice((2 * n_layers - 1, 2 * n_layers))
    max_side = min(14, min_side + 4)
    other_side = random.randint(min_side, max_side)
    if random.random() < 0.5:
        return min_side, other_side
    return other_side, min_side


def _sample_left_layers(n_layers: int) -> list[int]:
    return random.sample(PALETTE, n_layers)


def _sample_same_mask(n_layers: int) -> list[bool]:
    same_count = random.randint(1, n_layers - 2)
    same_depths = set(random.sample(range(n_layers), same_count))
    return [d in same_depths for d in range(n_layers)]


def _build_right_layers(left_layers: list[int], same_mask: list[bool]) -> list[int] | None:
    n_layers = len(left_layers)
    right_layers: list[int] = []

    def search(depth: int) -> bool:
        if depth == n_layers:
            return True

        if same_mask[depth]:
            color = left_layers[depth]
            if depth > 0 and right_layers[-1] == color:
                return False
            right_layers.append(color)
            ok = search(depth + 1)
            if ok:
                return True
            right_layers.pop()
            return False

        choices = [
            color
            for color in PALETTE
            if color != left_layers[depth] and (depth == 0 or color != right_layers[-1])
        ]
        random.shuffle(choices)
        for color in choices:
            right_layers.append(color)
            if search(depth + 1):
                return True
            right_layers.pop()
        return False

    return right_layers if search(0) else None


def _wrong_direct_output(left_layers: list[int], right_layers: list[int], shape: tuple[int, int]) -> np.ndarray:
    out_layers = []
    for l_color, r_color in zip(left_layers, right_layers):
        out_layers.append(BLACK if l_color == r_color else l_color)
    return _render_panel(out_layers, shape)


def _wrong_right_reversed_output(left_layers: list[int], right_layers: list[int], shape: tuple[int, int]) -> np.ndarray:
    n_layers = len(left_layers)
    out_layers = []
    for d in range(n_layers):
        if left_layers[d] == right_layers[d]:
            out_layers.append(BLACK)
        else:
            out_layers.append(right_layers[n_layers - 1 - d])
    return _render_panel(out_layers, shape)


def _validate_instance(grid: np.ndarray, left_layers: list[int], right_layers: list[int]) -> bool:
    left, right = _split_panels(grid)
    if _extract_layers(left) != left_layers or _extract_layers(right) != right_layers:
        return False

    same_count = sum(l == r for l, r in zip(left_layers, right_layers))
    diff_count = len(left_layers) - same_count
    if same_count < 1 or diff_count < 2:
        return False

    expected = _expected_from_layers(left_layers, right_layers, left.shape)
    if not np.any(expected == BLACK) or not np.any(expected != BLACK):
        return False

    if np.array_equal(expected, _wrong_direct_output(left_layers, right_layers, left.shape)):
        return False
    if np.array_equal(expected, _wrong_right_reversed_output(left_layers, right_layers, left.shape)):
        return False

    noncentral_diff = any(
        left_layers[d] != right_layers[d] and left_layers[d] != left_layers[len(left_layers) - 1 - d]
        for d in range(len(left_layers))
    )
    if not noncentral_diff:
        return False

    actual = main(grid)
    return np.array_equal(actual, expected)


def _generate_candidate() -> np.ndarray | None:
    n_layers = random.randint(4, 6)
    shape = _panel_shape_for_layers(n_layers)
    left_layers = _sample_left_layers(n_layers)
    same_mask = _sample_same_mask(n_layers)
    right_layers = _build_right_layers(left_layers, same_mask)
    if right_layers is None:
        return None

    left = _render_panel(left_layers, shape)
    right = _render_panel(right_layers, shape)
    grid = np.concatenate([left, np.full((shape[0], 1), SEPARATOR, dtype=int), right], axis=1)

    if not _validate_instance(grid, left_layers, right_layers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring xor task")
