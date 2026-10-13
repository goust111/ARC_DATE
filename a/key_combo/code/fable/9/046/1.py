from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
DIVIDER = 5
PANEL_SIDE = 9
MASK_SIDE = 3
BLOCK_SIDE = PANEL_SIDE // MASK_SIDE
GRID_SHAPE = (PANEL_SIDE, PANEL_SIDE * 2 + 1)
LEFT_SLICE = slice(0, PANEL_SIDE)
DIVIDER_COL = PANEL_SIDE
RIGHT_SLICE = slice(PANEL_SIDE + 1, PANEL_SIDE * 2 + 1)
PANEL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _mask_components(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                        if mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
    return count


def _all_candidate_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (MASK_SIDE * MASK_SIDE)):
        mask = np.array(
            [(bits >> i) & 1 for i in range(MASK_SIDE * MASK_SIDE)],
            dtype=bool,
        ).reshape(MASK_SIDE, MASK_SIDE)
        on_count = int(mask.sum())
        if not (2 <= on_count <= 6):
            continue
        if np.count_nonzero(mask.any(axis=0)) < 2:
            continue
        if np.count_nonzero(mask.any(axis=1)) < 2:
            continue
        masks.append(mask)
    return masks


CANDIDATE_MASKS = _all_candidate_masks()
CONNECTED_MASKS = [mask for mask in CANDIDATE_MASKS if _mask_components(mask) == 1]
DISCONNECTED_MASKS = [mask for mask in CANDIDATE_MASKS if _mask_components(mask) >= 2]


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _compose_panel(macro: np.ndarray, micro: np.ndarray, color: int) -> np.ndarray:
    return np.where(np.kron(macro.astype(bool), micro.astype(bool)), int(color), BG).astype(int)


def _panel_color(panel: np.ndarray) -> int:
    colors = set(map(int, np.unique(panel))) - {BG}
    assert len(colors) == 1, "each panel must use exactly one foreground color"
    color = colors.pop()
    assert color != DIVIDER, "panel foreground color must differ from the divider"
    return color


def _decompose_panel(panel: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    assert panel.shape == (PANEL_SIDE, PANEL_SIDE), "each panel must be 9x9"
    assert set(map(int, np.unique(panel))) <= set(PANEL_COLORS) | {BG}, "invalid panel colors"

    color = _panel_color(panel)
    macro = np.zeros((MASK_SIDE, MASK_SIDE), dtype=bool)
    micro: np.ndarray | None = None

    for br in range(MASK_SIDE):
        for bc in range(MASK_SIDE):
            tile = panel[
                br * BLOCK_SIDE : (br + 1) * BLOCK_SIDE,
                bc * BLOCK_SIDE : (bc + 1) * BLOCK_SIDE,
            ]
            assert set(map(int, np.unique(tile))) <= {BG, color}, "tiles may only contain bg and panel fg"
            tile_mask = tile == color
            if tile_mask.any():
                macro[br, bc] = True
                if micro is None:
                    micro = tile_mask.copy()
                else:
                    assert np.array_equal(tile_mask, micro), "all active macro blocks must share one micro mask"
            else:
                assert np.all(tile == BG), "inactive macro blocks must be fully background"

    assert micro is not None, "panel must contain at least one active macro block"
    return macro, micro, color


def _pairwise_distinct(*masks: np.ndarray) -> bool:
    keys = {_mask_key(mask) for mask in masks}
    return len(keys) == len(masks)


def _wrong_panel_swap(a_macro: np.ndarray, b_micro: np.ndarray, a_color: int, c_macro: np.ndarray, d_micro: np.ndarray, c_color: int) -> np.ndarray:
    out = np.full(GRID_SHAPE, BG, dtype=int)
    out[:, DIVIDER_COL] = DIVIDER
    out[:, LEFT_SLICE] = _compose_panel(c_macro, d_micro, c_color)
    out[:, RIGHT_SLICE] = _compose_panel(a_macro, b_micro, a_color)
    return out


def _wrong_color_swap(a_macro: np.ndarray, b_micro: np.ndarray, c_macro: np.ndarray, d_micro: np.ndarray, a_color: int, c_color: int) -> np.ndarray:
    out = np.full(GRID_SHAPE, BG, dtype=int)
    out[:, DIVIDER_COL] = DIVIDER
    out[:, LEFT_SLICE] = _compose_panel(a_macro, b_micro, c_color)
    out[:, RIGHT_SLICE] = _compose_panel(c_macro, d_micro, a_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep each panel's macro skeleton in place while swapping the micro mask and its color across the divider.

    Input structure:
    - The grid is a black 9x19 canvas with a full-height gray divider column in the middle.
    - The left and right 9x9 panels each use one non-black foreground color.
    - Within each panel, the 9x9 image is partitioned into a 3x3 grid of 3x3 blocks: non-empty block positions form a macro mask, and every non-empty block contains the same 3x3 micro mask.

    Rule:
    - Decompose the left panel into macro mask A and micro mask B, and the right panel into macro mask C and micro mask D.
    - Rebuild the left panel as A⊗D using the right panel's foreground color, and rebuild the right panel as C⊗B using the left panel's foreground color.
    - Keep the divider column, panel positions, and black background unchanged.

    Key insight:
    - The visible pattern is a Kronecker-style macro×micro product on each side, so the correct transform swaps only the repeated 3x3 micro symbol and its color while leaving each panel's 3x3 macro occupancy layout in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == GRID_SHAPE, "expected a 9x19 ARC grid"
    assert _background(g) == BG, "background must be black"
    assert np.all(g[:, DIVIDER_COL] == DIVIDER), "expected one full-height gray divider"

    left = g[:, LEFT_SLICE]
    right = g[:, RIGHT_SLICE]
    a_macro, b_micro, a_color = _decompose_panel(left)
    c_macro, d_micro, c_color = _decompose_panel(right)
    assert a_color != c_color, "the two panels must use distinct foreground colors"
    assert _pairwise_distinct(a_macro, b_micro, c_macro, d_micro), "A, B, C, D must be pairwise distinct"

    out = np.full_like(g, BG)
    out[:, DIVIDER_COL] = DIVIDER
    out[:, LEFT_SLICE] = _compose_panel(a_macro, d_micro, c_color)
    out[:, RIGHT_SLICE] = _compose_panel(c_macro, b_micro, a_color)
    return out


def _sample_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pools = [CONNECTED_MASKS, DISCONNECTED_MASKS]
    while True:
        chosen = [random.choice(random.choice(pools)).copy() for _ in range(4)]
        if _pairwise_distinct(*chosen):
            return chosen[0], chosen[1], chosen[2], chosen[3]


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    a_macro: np.ndarray,
    b_micro: np.ndarray,
    c_macro: np.ndarray,
    d_micro: np.ndarray,
    a_color: int,
    c_color: int,
) -> bool:
    if np.array_equal(grid, expected):
        return False
    if np.array_equal(expected, _wrong_panel_swap(a_macro, b_micro, a_color, c_macro, d_micro, c_color)):
        return False
    if np.array_equal(expected, _wrong_color_swap(a_macro, b_micro, c_macro, d_micro, a_color, c_color)):
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(main(actual), grid):
        return False

    left_out = actual[:, LEFT_SLICE]
    right_out = actual[:, RIGHT_SLICE]
    out_a_macro, out_d_micro, out_c_color = _decompose_panel(left_out)
    out_c_macro, out_b_micro, out_a_color = _decompose_panel(right_out)
    if out_c_color != c_color or out_a_color != a_color:
        return False
    if not np.array_equal(out_a_macro, a_macro):
        return False
    if not np.array_equal(out_c_macro, c_macro):
        return False
    if not np.array_equal(out_d_micro, d_micro):
        return False
    if not np.array_equal(out_b_micro, b_micro):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    a_macro, b_micro, c_macro, d_micro = _sample_masks()
    a_color, c_color = random.sample(PANEL_COLORS, 2)

    grid = np.full(GRID_SHAPE, BG, dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER
    grid[:, LEFT_SLICE] = _compose_panel(a_macro, b_micro, a_color)
    grid[:, RIGHT_SLICE] = _compose_panel(c_macro, d_micro, c_color)

    expected = np.full(GRID_SHAPE, BG, dtype=int)
    expected[:, DIVIDER_COL] = DIVIDER
    expected[:, LEFT_SLICE] = _compose_panel(a_macro, d_micro, c_color)
    expected[:, RIGHT_SLICE] = _compose_panel(c_macro, b_micro, a_color)

    if _semantic_checks(grid, expected, a_macro, b_micro, c_macro, d_micro, a_color, c_color):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid macro-micro cross-panel sample")
