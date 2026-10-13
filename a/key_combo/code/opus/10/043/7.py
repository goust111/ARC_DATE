from __future__ import annotations

import numpy as np


BACKGROUND = 0
ACTIVE = 2
MARKER = 4
MACRO_DIM = 4
BLOCK_SIZE = 5
GRID_SIZE = MACRO_DIM * BLOCK_SIZE
AXIS_HORIZONTAL = "horizontal"
AXIS_VERTICAL = "vertical"
MAX_GENERATION_ATTEMPTS = 400


def _center_indices(length: int) -> tuple[int, ...]:
    mid = length // 2
    if length % 2 == 1:
        return (mid,)
    return (mid - 1, mid)


def _detect_axis(grid: np.ndarray) -> str:
    yellow = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    assert len(yellow) == 2, "expected exactly two yellow edge markers"

    (r0, c0), (r1, c1) = yellow
    h, w = grid.shape
    central_rows = set(_center_indices(h))
    central_cols = set(_center_indices(w))

    if {r0, r1} == {0, h - 1} and c0 == c1 and c0 in central_cols:
        return AXIS_HORIZONTAL
    if {c0, c1} == {0, w - 1} and r0 == r1 and r0 in central_rows:
        return AXIS_VERTICAL
    raise AssertionError("yellow markers must form one centered opposite-edge pair")


def _compress(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    assert h % MACRO_DIM == 0 and w % MACRO_DIM == 0, "grid must split into a 4x4 macro array"
    block_h = h // MACRO_DIM
    block_w = w // MACRO_DIM

    occ = np.zeros((MACRO_DIM, MACRO_DIM), dtype=bool)
    for br in range(MACRO_DIM):
        for bc in range(MACRO_DIM):
            block = grid[br * block_h : (br + 1) * block_h, bc * block_w : (bc + 1) * block_w]
            occ[br, bc] = bool(np.any(block == ACTIVE))
    return occ


def _symmetrize(occ: np.ndarray, axis: str) -> np.ndarray:
    if axis == AXIS_HORIZONTAL:
        return occ | occ[::-1, :]
    if axis == AXIS_VERTICAL:
        return occ | occ[:, ::-1]
    raise ValueError(f"unknown axis: {axis}")


def _render_output(occ: np.ndarray) -> np.ndarray:
    return np.where(occ, ACTIVE, BACKGROUND).astype(int)


def _block_origin(block_r: int, block_c: int) -> tuple[int, int]:
    return block_r * BLOCK_SIZE, block_c * BLOCK_SIZE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a macro-block grid, then enforce the symmetry axis indicated by centered opposite-edge yellow markers.

    Input structure:
    - The background is black, with sparse red pixels scattered inside some cells of a 4x4 array of equal macro-blocks.
    - Each macro-block is treated only by whether it contains at least one red pixel.
    - Exactly two yellow pixels lie at the center of one pair of opposite edges, indicating either a horizontal or vertical symmetry axis.

    Rule:
    - Compress the large grid into a 4x4 occupancy map where a cell is active iff its macro-block contains any red pixel.
    - Read the yellow marker orientation: top+bottom markers mean a horizontal axis, while left+right markers mean a vertical axis.
    - For every mirrored pair across that axis, activate both cells whenever either member of the pair is active.
    - Output the resulting 4x4 grid with active cells in red and inactive cells in black.

    Key insight:
    - The yellow edge markers do not act on the raw red pixels directly; they specify how to symmetrize the compressed occupancy pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE), "expected a 20x20 input grid"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, ACTIVE, MARKER}), "unexpected colors present"

    axis = _detect_axis(g)
    compressed = _compress(g)
    return _render_output(_symmetrize(compressed, axis))


def _sample_compressed_map(rng: np.random.Generator, axis: str) -> np.ndarray | None:
    other_axis = AXIS_VERTICAL if axis == AXIS_HORIZONTAL else AXIS_HORIZONTAL
    force_other_symmetry = bool(rng.integers(0, 2))

    for _ in range(200):
        active_prob = float(rng.uniform(0.28, 0.62))
        if axis == AXIS_HORIZONTAL and force_other_symmetry:
            left_half = rng.random((MACRO_DIM, MACRO_DIM // 2)) < active_prob
            occ = np.concatenate([left_half, left_half[:, ::-1]], axis=1)
        elif axis == AXIS_VERTICAL and force_other_symmetry:
            top_half = rng.random((MACRO_DIM // 2, MACRO_DIM)) < active_prob
            occ = np.concatenate([top_half, top_half[::-1, :]], axis=0)
        else:
            occ = rng.random((MACRO_DIM, MACRO_DIM)) < active_prob

        active_count = int(occ.sum())
        if active_count < 2 or active_count > 10:
            continue
        if np.array_equal(occ, _symmetrize(occ, axis)):
            continue

        chosen = _symmetrize(occ, axis)
        other = _symmetrize(occ, other_axis)
        if np.array_equal(chosen, other):
            continue
        if np.array_equal(chosen, occ):
            continue
        if int(chosen.sum()) > 12:
            continue
        return occ
    return None


def _place_marker_pair(grid: np.ndarray, axis: str, center_index: int) -> None:
    if axis == AXIS_HORIZONTAL:
        grid[0, center_index] = MARKER
        grid[GRID_SIZE - 1, center_index] = MARKER
    else:
        grid[center_index, 0] = MARKER
        grid[center_index, GRID_SIZE - 1] = MARKER


def _expand_to_input(
    occ: np.ndarray,
    axis: str,
    center_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    _place_marker_pair(grid, axis, center_index)

    reserved = {(int(r), int(c)) for r, c in np.argwhere(grid == MARKER)}
    for br in range(MACRO_DIM):
        for bc in range(MACRO_DIM):
            if not occ[br, bc]:
                continue
            r0, c0 = _block_origin(br, bc)
            cells = [
                (r0 + dr, c0 + dc)
                for dr in range(BLOCK_SIZE)
                for dc in range(BLOCK_SIZE)
                if (r0 + dr, c0 + dc) not in reserved
            ]
            red_count = int(rng.integers(1, 4))
            chosen = rng.choice(len(cells), size=red_count, replace=False)
            for idx in np.atleast_1d(chosen):
                r, c = cells[int(idx)]
                grid[r, c] = ACTIVE
    return grid


def _validate_generated(grid: np.ndarray, occ: np.ndarray, axis: str) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if not set(map(int, np.unique(grid))).issubset({BACKGROUND, ACTIVE, MARKER}):
        return False

    center_set = set(_center_indices(GRID_SIZE))
    yellow = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    if len(yellow) != 2:
        return False
    if axis == AXIS_HORIZONTAL:
        if {yellow[0][0], yellow[1][0]} != {0, GRID_SIZE - 1} or yellow[0][1] != yellow[1][1]:
            return False
        if yellow[0][1] not in center_set:
            return False
    else:
        if {yellow[0][1], yellow[1][1]} != {0, GRID_SIZE - 1} or yellow[0][0] != yellow[1][0]:
            return False
        if yellow[0][0] not in center_set:
            return False

    compressed = _compress(grid)
    if not np.array_equal(compressed, occ):
        return False
    if np.array_equal(compressed, _symmetrize(compressed, axis)):
        return False

    other_axis = AXIS_VERTICAL if axis == AXIS_HORIZONTAL else AXIS_HORIZONTAL
    expected_occ = _symmetrize(compressed, axis)
    other_occ = _symmetrize(compressed, other_axis)
    if np.array_equal(expected_occ, other_occ):
        return False

    for br in range(MACRO_DIM):
        for bc in range(MACRO_DIM):
            r0, c0 = _block_origin(br, bc)
            block = grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
            red_count = int(np.count_nonzero(block == ACTIVE))
            if occ[br, bc]:
                if not (1 <= red_count <= 3):
                    return False
            elif red_count != 0:
                return False

    expected = _render_output(expected_occ)
    other = _render_output(other_occ)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected) and not np.array_equal(actual, other)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    axes = (AXIS_HORIZONTAL, AXIS_VERTICAL)
    centers = _center_indices(GRID_SIZE)

    for _ in range(MAX_GENERATION_ATTEMPTS):
        axis = axes[int(rng.integers(0, len(axes)))]
        occ = _sample_compressed_map(rng, axis)
        if occ is None:
            continue

        center_index = int(centers[int(rng.integers(0, len(centers)))])
        grid = _expand_to_input(occ, axis, center_index, rng)
        if _validate_generated(grid, occ, axis):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 7")
