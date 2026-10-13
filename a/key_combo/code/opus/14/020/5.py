from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
TEAL = 8
MAROON = 9

MAX_GENERATION_ATTEMPTS = 400
STATE_MIN_COUNTS = np.array([3, 1, 1, 1], dtype=int)
STATE_WEIGHT_PRESETS = (
    np.array([0.30, 0.25, 0.25, 0.20], dtype=float),
    np.array([0.42, 0.20, 0.20, 0.18], dtype=float),
    np.array([0.22, 0.30, 0.20, 0.28], dtype=float),
    np.array([0.22, 0.20, 0.30, 0.28], dtype=float),
)


def _separator_column(grid: np.ndarray) -> int:
    matches = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLUE)]
    assert len(matches) == 1, f"expected one full-height blue separator, got {len(matches)}"
    return int(matches[0])


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bar = _separator_column(g)
    left = g[:, :bar]
    right = g[:, bar + 1 :]
    assert left.shape == right.shape and left.size > 0, "expected two equal nonempty panels"
    assert set(map(int, np.unique(left))) <= {BLACK, MAROON}, "left panel must be maroon/black"
    assert set(map(int, np.unique(right))) <= {BLACK, RED}, "right panel must be red/black"
    return left, right


def _nor_seed(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left == BLACK) & (right == BLACK)


def _diagonal_spread(seed_mask: np.ndarray) -> np.ndarray:
    h, w = map(int, seed_mask.shape)
    out = np.full((2 * h, 2 * w), BLACK, dtype=int)
    for r, c in np.argwhere(seed_mask):
        rr = int(r)
        cc = int(c)
        while rr < 2 * h and cc < 2 * w:
            out[rr, cc] = TEAL
            rr += 1
            cc += 1
    return out


def _panel_states(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_on = left == MAROON
    right_on = right == RED
    return left_on.astype(int) + 2 * right_on.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compute a NOR seed from two separator-split masks, then spread that seed diagonally on a doubled canvas.

    Input structure:
    - The grid contains a full-height blue separator column between two equal-width panels.
    - The left panel is a maroon/black binary mask and the right panel is a red/black binary mask.
    - Black means inactive, while maroon and red mark active cells in their own panels.

    Rule:
    - Locate the blue separator and extract the aligned left and right panels.
    - Compute the pixelwise logical NOR: a panel cell becomes teal in the seed iff both aligned input cells are black.
    - Create a black canvas with twice the seed height and twice the seed width.
    - For every teal seed cell at (r, c), paint teal at every doubled-canvas position (r+k, c+k) that stays in bounds.

    Key insight:
    - The visible diagonal pattern is not derived from either input mask directly; it comes from the emergent NOR mask, which is then reused as the transparent seed for down-right offset copying.
    """
    left, right = _extract_panels(grid)
    return _diagonal_spread(_nor_seed(left, right))


def _choose_panel_shape() -> tuple[int, int]:
    mode = random.randrange(3)
    if mode == 0:
        return random.randint(6, 9), random.randint(4, 6)
    if mode == 1:
        return random.randint(4, 6), random.randint(6, 9)
    return random.randint(4, 8), random.randint(4, 8)


def _sample_state_grid(h: int, w: int) -> np.ndarray:
    total = h * w
    weights = random.choice(STATE_WEIGHT_PRESETS)
    counts = STATE_MIN_COUNTS + np.random.multinomial(total - int(STATE_MIN_COUNTS.sum()), weights)

    flat = np.empty(total, dtype=int)
    offset = 0
    for state, count in enumerate(counts):
        flat[offset : offset + int(count)] = state
        offset += int(count)
    np.random.shuffle(flat)
    return flat.reshape(h, w)


def _states_to_input(states: np.ndarray) -> np.ndarray:
    h, w = map(int, states.shape)
    left = np.full((h, w), BLACK, dtype=int)
    right = np.full((h, w), BLACK, dtype=int)
    left[np.isin(states, (1, 3))] = MAROON
    right[np.isin(states, (2, 3))] = RED
    bar = np.full((h, 1), BLUE, dtype=int)
    return np.concatenate((left, bar, right), axis=1)


def _depends_on_both_masks(grid: np.ndarray) -> bool:
    left, right = _extract_panels(grid)
    states = _panel_states(left, right)
    baseline = main(grid)

    left_only = np.argwhere(states == 1)
    right_only = np.argwhere(states == 2)
    if left_only.size == 0 or right_only.size == 0:
        return False

    left_matters = False
    for r, c in left_only:
        probe = grid.copy()
        probe[int(r), int(c)] = BLACK
        if not np.array_equal(main(probe), baseline):
            left_matters = True
            break
    if not left_matters:
        return False

    for r, c in right_only:
        probe = grid.copy()
        probe[int(r), left.shape[1] + 1 + int(c)] = BLACK
        if not np.array_equal(main(probe), baseline):
            return True
    return False


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        left, right = _extract_panels(grid)
        out = main(grid)
    except Exception:
        return False

    h, w = map(int, left.shape)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if out.shape != (2 * h, 2 * w):
        return False

    states = _panel_states(left, right)
    counts = np.bincount(states.ravel(), minlength=4)
    if counts[0] < 3 or np.any(counts[1:] < 1):
        return False

    left_on = left == MAROON
    right_on = right == RED
    if left_on.sum() < 2 or right_on.sum() < 2:
        return False
    if left_on.sum() >= left.size - 2 or right_on.sum() >= right.size - 2:
        return False

    seed = states == 0
    seed_coords = np.argwhere(seed)
    if len(seed_coords) < 3 or len(seed_coords) == h * w:
        return False
    if len({int(r) for r, _c in seed_coords}) < 2:
        return False
    if len({int(c) for _r, c in seed_coords}) < 2:
        return False
    if len({int(r - c) for r, c in seed_coords}) < 2:
        return False
    if not np.any((seed_coords[:, 0] < h - 1) & (seed_coords[:, 1] < w - 1)):
        return False

    teal_cells = int(np.count_nonzero(out == TEAL))
    if teal_cells <= int(seed.sum()):
        return False
    teal_fraction = teal_cells / out.size
    if not (0.12 <= teal_fraction <= 0.72):
        return False

    expected = _diagonal_spread(seed)
    if not np.array_equal(out, expected):
        return False

    return _depends_on_both_masks(grid)


def _generate_attempt() -> np.ndarray:
    h, w = _choose_panel_shape()
    states = _sample_state_grid(h, w)
    return _states_to_input(states)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid diagonal NOR tiling input")
