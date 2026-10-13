from collections import Counter

import numpy as np
import random as random
from arc_utils import find_components as find_components


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the dominant-color object by adding its left-right mirror image.

    Input structure
    A rectangular grid contains one uniquely most frequent color forming the main
    foreground object, with other colors acting as clutter/noise.

    Rule
    Reflect every pixel of the dominant color across the vertical centerline of
    the grid and recolor the reflected cells with that same dominant color.

    Key insight
    The noisy cells are exactly the missing counterpart pixels needed to make the
    dominant-color mask horizontally symmetric.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = Counter(map(int, g.flat))
    ranked = counts.most_common()
    assert len(ranked) >= 2
    main_color, main_count = ranked[0]
    assert main_count > ranked[1][1]

    out = g.copy()
    rows, cols = np.where(g == main_color)
    out[rows, g.shape[1] - 1 - cols] = main_color
    return out

# --------------- generator (o5x) ---------------

# Canvas/mirror geometry.
GRID_SIZE = 10
PAIR_COLS = GRID_SIZE // 2
LEFT_CENTER_COL, RIGHT_CENTER_COL = PAIR_COLS - 1, PAIR_COLS
BACKGROUND_SENTINEL = -1
# Palette bounds.
DOMINANT_COLORS = range(1, 10)
BACKGROUND_COLOR, NONDOMINANT_NONZERO_COUNT, COLOR_COUNT = 0, 2, 10
# Blob/noise validation bounds.
INPUT_DOMINANT_COUNT_RANGE = (30, 34)
BLOB_SIZE_RANGE, NOISE_COUNT_RANGE, ROW_SPAN_RANGE = (36, 48), (6, 14), (6, 8)
MAX_RIGHT_NOISE_FRACTION, MIN_DOMINANCE_GAP = 0.58, 3
# Retry and adjacency controls.
GENERATION_RETRIES = 300
ORTHOGONAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_choose_palette():
    _gen_dominant = random.choice(DOMINANT_COLORS)
    _gen_other_pool = [c for c in DOMINANT_COLORS if c != _gen_dominant]
    _gen_nondom = [BACKGROUND_COLOR] + random.sample(_gen_other_pool, NONDOMINANT_NONZERO_COUNT)
    random.shuffle(_gen_nondom)
    return _gen_dominant, _gen_nondom


def _gen_choose_blob_parameters():
    _gen_input_d_count = random.randint(*INPUT_DOMINANT_COUNT_RANGE)
    _gen_min_blob = max(BLOB_SIZE_RANGE[0], _gen_input_d_count + NOISE_COUNT_RANGE[0])
    _gen_max_blob = min(BLOB_SIZE_RANGE[1], _gen_input_d_count + NOISE_COUNT_RANGE[1])
    _gen_valid_blob_sizes = [
        n
        for n in range(_gen_min_blob, _gen_max_blob + 1)
        if n % 2 == 0
        and (n - _gen_input_d_count) / (n // 2) <= MAX_RIGHT_NOISE_FRACTION
    ]
    _gen_blob_size = random.choice(_gen_valid_blob_sizes)
    _gen_span = random.randint(*ROW_SPAN_RANGE)
    _gen_top = random.randint(0, GRID_SIZE - _gen_span)
    return _gen_input_d_count, _gen_blob_size // 2, _gen_blob_size - _gen_input_d_count, _gen_span, _gen_top


def _gen_connected_pair_mask(_gen_rows, _gen_pair_count):
    """Grow a connected mask in half-grid pair coordinates."""
    _gen_mask = np.zeros((_gen_rows, PAIR_COLS), dtype=bool)
    _gen_mask[:, 0] = True
    _gen_active = {(r, 0) for r in range(_gen_rows)}
    _gen_frontier = {(r, 1) for r in range(_gen_rows)}

    while len(_gen_active) < _gen_pair_count:
        _gen_choice = random.choice(tuple(_gen_frontier))
        _gen_frontier.remove(_gen_choice)
        _gen_r, _gen_p = _gen_choice
        _gen_mask[_gen_r, _gen_p] = True
        _gen_active.add(_gen_choice)
        for _gen_dr, _gen_dp in ORTHOGONAL_STEPS:
            _gen_nr = _gen_r + _gen_dr
            _gen_np = _gen_p + _gen_dp
            if 0 <= _gen_nr < _gen_rows and 0 <= _gen_np < PAIR_COLS:
                if not _gen_mask[_gen_nr, _gen_np]:
                    _gen_frontier.add((_gen_nr, _gen_np))
    return _gen_mask


def _gen_reflect_pair_mask(_gen_pair_mask, _gen_top):
    """Expand half-grid pair coordinates into a full LR-symmetric blob mask."""
    _gen_blob = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    _gen_right_cells = []
    for _gen_rr in range(_gen_pair_mask.shape[0]):
        _gen_r = _gen_top + _gen_rr
        for _gen_p in range(PAIR_COLS):
            if _gen_pair_mask[_gen_rr, _gen_p]:
                _gen_left = LEFT_CENTER_COL - _gen_p
                _gen_right = RIGHT_CENTER_COL + _gen_p
                _gen_blob[_gen_r, _gen_left] = True
                _gen_blob[_gen_r, _gen_right] = True
                _gen_right_cells.append((_gen_r, _gen_right))
    return _gen_blob, _gen_right_cells


def _gen_balanced_values(_gen_colors, _gen_count):
    _gen_order = list(_gen_colors)
    random.shuffle(_gen_order)
    _gen_base = _gen_count // len(_gen_order)
    _gen_extra = _gen_count % len(_gen_order)
    _gen_values = []
    for _gen_i, _gen_color in enumerate(_gen_order):
        _gen_values.extend([_gen_color] * (_gen_base + (_gen_i < _gen_extra)))
    random.shuffle(_gen_values)
    return _gen_values


def _gen_build_grids(_gen_blob, _gen_right_cells, _gen_noise_count, _gen_dominant, _gen_nondom):
    """Render output and input grids while preserving background/noise value order."""
    _gen_noise_cells = set(random.sample(_gen_right_cells, _gen_noise_count))
    _gen_out = np.full((GRID_SIZE, GRID_SIZE), _gen_dominant, dtype=int)
    _gen_inp = _gen_out.copy()

    _gen_out[~_gen_blob] = BACKGROUND_SENTINEL
    _gen_inp[~_gen_blob] = BACKGROUND_SENTINEL
    _gen_positions = [(int(_gen_r), int(_gen_c), True) for _gen_r, _gen_c in np.argwhere(~_gen_blob)]
    for _gen_r, _gen_c in _gen_noise_cells:
        _gen_positions.append((_gen_r, _gen_c, False))

    _gen_values = _gen_balanced_values(_gen_nondom, len(_gen_positions))
    for (_gen_r, _gen_c, _gen_is_background), _gen_value in zip(_gen_positions, _gen_values):
        _gen_inp[_gen_r, _gen_c] = _gen_value
        if _gen_is_background:
            _gen_out[_gen_r, _gen_c] = _gen_value

    return _gen_inp, _gen_out


def _gen_is_valid_instance(_gen_inp, _gen_out, _gen_dominant, _gen_input_d_count):
    _gen_counts = np.bincount(_gen_inp.ravel(), minlength=COLOR_COUNT)
    _gen_other_max = max(_gen_counts[c] for c in range(COLOR_COUNT) if c != _gen_dominant)
    if _gen_counts[_gen_dominant] != _gen_input_d_count:
        return False
    if _gen_counts[_gen_dominant] - _gen_other_max < MIN_DOMINANCE_GAP:
        return False
    if np.array_equal(_gen_inp == _gen_dominant, _gen_out == _gen_dominant):
        return False
    _gen_components = [
        comp
        for comp in find_components(
            _gen_out, bg=BACKGROUND_SENTINEL, connectivity=4, monochromatic=True,
            colors=[_gen_dominant], return_format="dict",
        )
        if comp["color"] == _gen_dominant
    ]
    return len(_gen_components) == 1 and np.array_equal(main(_gen_inp), _gen_out)


def _gen_attempt() -> np.ndarray | None:
    _gen_dominant, _gen_nondom = _gen_choose_palette()
    _gen_input_d_count, _gen_pair_count, _gen_noise_count, _gen_span, _gen_top = _gen_choose_blob_parameters()
    _gen_pair_mask = _gen_connected_pair_mask(_gen_span, _gen_pair_count)
    _gen_blob, _gen_right_cells = _gen_reflect_pair_mask(_gen_pair_mask, _gen_top)
    _gen_inp, _gen_out = _gen_build_grids(_gen_blob, _gen_right_cells, _gen_noise_count, _gen_dominant, _gen_nondom)
    if not _gen_is_valid_instance(_gen_inp, _gen_out, _gen_dominant, _gen_input_d_count):
        return None
    return _gen_inp


def generate_input() -> np.ndarray:
    for _gen_attempt_i in range(GENERATION_RETRIES):
        _gen_grid = _gen_attempt()
        if _gen_grid is not None:
            return _gen_grid
    raise RuntimeError("failed to generate c35c1b4c input")
