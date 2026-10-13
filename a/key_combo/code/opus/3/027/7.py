from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, colors_present, scale_sprite
except Exception:
    class Color:
        BLACK = 0
        GREY = 5

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        colors = set(map(int, np.asarray(grid).ravel()))
        if exclude is not None:
            colors -= set(map(int, exclude))
        return sorted(colors)

    def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
        return np.kron(np.asarray(sprite, dtype=int), np.ones((factor, factor), dtype=int))


GRID_SHAPE = (5, 11)
LEFT_SHAPE = (5, 5)
RIGHT_PANEL_SHAPE = (5, 5)
RIGHT_PATTERN_SLICE = (slice(1, 4), slice(1, 4))
SEPARATOR_COL = 5
BACKGROUND = Color.BLACK
SEPARATOR = Color.GREY
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

X_COORDS = (
    (0, 0),
    (0, 4),
    (1, 1),
    (1, 3),
    (2, 2),
    (3, 1),
    (3, 3),
    (4, 0),
    (4, 4),
)
FOLD_MAP = {
    (0, 0): (0, 0),
    (1, 1): (0, 1),
    (0, 4): (0, 2),
    (3, 1): (1, 0),
    (2, 2): (1, 1),
    (1, 3): (1, 2),
    (4, 0): (2, 0),
    (3, 3): (2, 1),
    (4, 4): (2, 2),
}

_OUTPUT_SIGNATURES: set[str] = set()
_SUCCESSFUL_GENERATIONS = 0
_SCALE_CYCLE = [2, 3, 4]
_RIGHT_COLOR_COUNT_CYCLE = {2: 3, 3: 1, 4: 2}


def _x_mask() -> np.ndarray:
    mask = np.zeros(LEFT_SHAPE, dtype=bool)
    for r, c in X_COORDS:
        mask[r, c] = True
    return mask


def _fold_x(left: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=int)
    assert left.shape == LEFT_SHAPE, "left subgrid must be 5x5"
    mask = _x_mask()
    assert np.all(left[~mask] == BACKGROUND), "only the X pixels may be foreground"
    assert np.all(left[mask] != BACKGROUND), "all X pixels must be colored"

    out = np.full((3, 3), BACKGROUND, dtype=int)
    for src, dst in FOLD_MAP.items():
        out[dst] = int(left[src])
    return out


def _count_folded_colors(left: np.ndarray) -> int:
    folded = _fold_x(left)
    colors = colors_present(folded, exclude={BACKGROUND})
    assert 2 <= len(colors) <= 4, "folded key must contribute a scale in [2, 4]"
    return len(colors)


def _extract_right_pattern(grid: np.ndarray) -> np.ndarray:
    right_panel = np.asarray(grid[:, 6:11], dtype=int)
    assert right_panel.shape == RIGHT_PANEL_SHAPE, "right panel must be 5x5"
    pattern = right_panel[RIGHT_PATTERN_SLICE]
    ring = right_panel.copy()
    ring[RIGHT_PATTERN_SLICE] = BACKGROUND
    assert np.all(ring == BACKGROUND), "the 3x3 right pattern must be centered in a black 5x5 panel"
    assert np.all(pattern != BACKGROUND), "the 3x3 right pattern must have no background pixels"
    return pattern.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the left X-key to count its colors, then upscale the separate right sprite by that count.

    Input structure:
    - The input is a 5x11 grid split into a left 5x5 panel, a full-height grey separator column, and a right 5x5 panel.
    - The left panel is black except for the 9 diagonal cells of a colored 5x5 X.
    - The right panel is black except for a centered 3x3 non-black sprite whose 9 cells are all foreground.

    Rule:
    - Fold the left 5x5 X into a 3x3 square-outline layout using the fixed diagonal-to-edge mapping.
    - Count the distinct non-black colors in that folded 3x3 result; call the count C.
    - Extract the centered 3x3 sprite from the right panel and replace each of its pixels with a CxC same-color block.
    - Output only the enlarged right sprite, discarding the left X panel and the separator.

    Key insight:
    - The left X is a key panel rather than output content: it determines only the integer upscale factor, while the visible answer is the right panel alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == GRID_SHAPE, "expected a 5x11 grid"
    assert np.all(g[:, SEPARATOR_COL] == SEPARATOR), "expected a full grey separator column"

    left = g[:, :5]
    scale = _count_folded_colors(left)
    right = _extract_right_pattern(g)
    return scale_sprite(right, scale)


def _output_signature(grid: np.ndarray) -> str:
    grid = np.asarray(grid, dtype=int)
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _assign_all_colors(cells: int, palette: list[int]) -> np.ndarray:
    values = list(palette)
    while len(values) < cells:
        values.append(random.choice(palette))
    random.shuffle(values)
    return np.asarray(values, dtype=int)


def _sample_left_panel(scale: int) -> tuple[np.ndarray, set[int]]:
    palette = random.sample(PALETTE, scale)
    values = _assign_all_colors(len(X_COORDS), palette)

    left = np.zeros(LEFT_SHAPE, dtype=int)
    for (r, c), color in zip(X_COORDS, values):
        left[r, c] = int(color)

    # Keep the X visibly heterogeneous so the key is readable.
    arm_triplets = (
        {(0, 0), (1, 1), (2, 2)},
        {(0, 4), (1, 3), (2, 2)},
        {(4, 0), (3, 1), (2, 2)},
        {(4, 4), (3, 3), (2, 2)},
    )
    if all(len({int(left[r, c]) for r, c in arm}) == 1 for arm in arm_triplets):
        r, c = random.choice(X_COORDS)
        alternatives = [color for color in palette if color != int(left[r, c])]
        if alternatives:
            left[r, c] = random.choice(alternatives)

    assert _count_folded_colors(left) == scale
    return left, set(map(int, palette))


def _sample_right_pattern(scale: int, left_colors: set[int], target_right_color_count: int | None) -> np.ndarray:
    allowed_counts = [k for k in (1, 2, 3) if k != scale]
    if target_right_color_count is None:
        right_color_count = random.choice(allowed_counts)
    else:
        right_color_count = int(target_right_color_count)
        assert right_color_count in allowed_counts

    exclusive = [c for c in PALETTE if c not in left_colors]
    assert exclusive, "right panel needs at least one color outside the left palette"

    palette = [random.choice(exclusive)]
    if right_color_count > 1:
        pool = [c for c in PALETTE if c not in palette]
        palette.extend(random.sample(pool, right_color_count - 1))
    random.shuffle(palette)

    values = _assign_all_colors(9, palette)
    pattern = values.reshape(3, 3)

    assert len(colors_present(pattern, exclude={BACKGROUND})) == right_color_count
    assert any(int(v) not in left_colors for v in pattern.ravel()), "whole-input color counting must overcount"
    return pattern


def _build_input(left: np.ndarray, right_pattern: np.ndarray) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[:, :5] = left
    grid[:, SEPARATOR_COL] = SEPARATOR
    grid[RIGHT_PATTERN_SLICE[0], 7:10] = right_pattern
    return grid


def _choose_scale_and_right_count() -> tuple[int, int | None]:
    planned_index = _SUCCESSFUL_GENERATIONS
    if planned_index < len(_SCALE_CYCLE):
        scale = _SCALE_CYCLE[planned_index]
        return scale, _RIGHT_COLOR_COUNT_CYCLE[scale]
    return random.choice([2, 3, 4]), None


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.shape == GRID_SHAPE
    assert np.all(inp[:, SEPARATOR_COL] == SEPARATOR)

    left = inp[:, :5]
    right = _extract_right_pattern(inp)
    scale = _count_folded_colors(left)
    right_color_count = len(colors_present(right, exclude={BACKGROUND}))
    total_visible_without_separator = len(colors_present(inp, exclude={BACKGROUND, SEPARATOR}))

    assert right_color_count != scale, "counting right-panel colors must be a wrong shortcut"
    assert total_visible_without_separator > scale, "counting all visible foreground colors must be a wrong shortcut"
    assert out.shape == (3 * scale, 3 * scale)
    assert np.array_equal(out, scale_sprite(right, scale))


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    scale, target_right_color_count = _choose_scale_and_right_count()
    left, left_colors = _sample_left_panel(scale)
    right = _sample_right_pattern(scale, left_colors, target_right_color_count)
    inp = _build_input(left, right)
    out = main(inp)
    _semantic_checks(inp, out)

    out_sig = _output_signature(out)
    if out_sig in _OUTPUT_SIGNATURES:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    global _SUCCESSFUL_GENERATIONS

    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        _OUTPUT_SIGNATURES.add(_output_signature(out))
        _SUCCESSFUL_GENERATIONS += 1
        return inp
    raise RuntimeError("failed to generate a valid input")
