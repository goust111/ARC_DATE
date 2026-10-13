import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
MARKER = 6
PALETTE = (BG, BLUE, RED, GREEN)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_marker_rect(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.asarray(grid == MARKER)
    rect = _bbox(mask)
    r0, c0, r1, c1 = rect
    rect_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert int(mask.sum()) == int(rect_area), "marker cells must form one solid rectangle"
    return rect


def _restore_hidden_patch(grid: np.ndarray) -> np.ndarray:
    restored = np.asarray(grid, dtype=int).copy()
    _, width = restored.shape
    r0, c0, r1, c1 = _find_marker_rect(restored)

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            mirror_c = width - 1 - c
            assert int(restored[r, mirror_c]) != MARKER, "mirror counterpart must stay visible"
            restored[r, c] = int(restored[r, mirror_c])

    assert np.array_equal(restored, np.fliplr(restored)), "restored grid must be left-right symmetric"
    return restored


def _minimal_horizontal_period(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    _, width = g.shape
    for period in range(1, width + 1):
        if np.array_equal(g[:, : width - period], g[:, period:]):
            return period
    raise AssertionError("expected at least the trivial full-width period")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the masked symmetric pattern, then crop the smallest horizontal period tile from the top-left.

    Input structure:
    - A black background grid contains a red/blue/green periodic pattern whose full image is left-right mirror symmetric.
    - Exactly one solid magenta rectangle masks part of the pattern, and the rectangle stays entirely on one side of the vertical midline.
    - The mirror position of every masked cell remains visible on the opposite side of the grid.

    Rule:
    - Find the unique solid magenta rectangle and replace each masked cell with the color at its left-right mirror coordinate.
    - On the restored full grid, detect the smallest positive horizontal period P.
    - Return the crop consisting of all rows and the first P columns.

    Key insight:
    - The marker breaks the periodicity, so the correct tile can only be extracted after the hidden cells are restored from the independent mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    restored = _restore_hidden_patch(g)
    period = _minimal_horizontal_period(restored)
    return restored[:, :period].copy()


def _palindrome_from_seed(seed: list[int], period: int) -> np.ndarray:
    if period % 2 == 0:
        row = seed + seed[::-1]
    else:
        row = seed + seed[-2::-1]
    return np.array(row, dtype=int)


def _sample_palindromic_row(period: int, *, exact_period: bool, repeats: int) -> np.ndarray:
    half = (period + 1) // 2
    weights = [4, 2, 2, 2]
    for _ in range(300):
        seed = random.choices(PALETTE, weights=weights, k=half)
        row = _palindrome_from_seed(seed, period)
        full_row = np.tile(row, repeats)[None, :]
        row_period = _minimal_horizontal_period(full_row)
        has_signal = np.any(row != BG)
        if exact_period:
            if row_period == period and has_signal and len(set(map(int, row))) >= 2:
                return row
        else:
            if has_signal or random.random() < 0.25:
                return row
    raise RuntimeError("failed to sample a valid palindromic row")


def _sample_base_tile(height: int, period: int, repeats: int) -> tuple[np.ndarray, list[int]]:
    signature_count = random.randint(1, min(3, height))
    signature_rows = sorted(random.sample(range(height), signature_count))
    tile = np.zeros((height, period), dtype=int)

    for r in range(height):
        tile[r] = _sample_palindromic_row(
            period,
            exact_period=r in signature_rows,
            repeats=repeats,
        )

    color_set = set(map(int, np.unique(tile)))
    active_rows = int(np.sum(np.any(tile != BG, axis=1)))
    if not {BLUE, RED, GREEN}.issubset(color_set):
        raise RuntimeError("tile missed one of the foreground colors")
    if active_rows < 3:
        raise RuntimeError("tile is too visually sparse")
    return tile, signature_rows


def _sample_marker_rect(
    full_grid: np.ndarray,
    period: int,
    signature_rows: list[int],
) -> tuple[int, int, int, int]:
    height, width = full_grid.shape
    axis = width // 2
    rect_h = random.randint(2, min(4, height))
    rect_w = random.randint(1, min(period - 1, 3))

    row_starts = [
        r0
        for r0 in range(height - rect_h + 1)
        if any(r0 <= sr < r0 + rect_h for sr in signature_rows)
    ]
    if not row_starts:
        row_starts = list(range(height - rect_h + 1))

    left_first_tile = [
        c0
        for c0 in range(0, max(0, period - rect_w + 1))
        if c0 + rect_w <= axis
    ]
    left_other = [
        c0
        for c0 in range(period, axis - rect_w + 1)
    ]
    right_side = list(range(axis, width - rect_w + 1))

    branch_candidates = []
    if left_first_tile:
        branch_candidates.append(left_first_tile)
    if left_other:
        branch_candidates.append(left_other)
    if right_side:
        branch_candidates.append(right_side)
    if not branch_candidates:
        raise RuntimeError("no valid marker placement branch")

    for _ in range(300):
        r0 = random.choice(row_starts)
        c0 = random.choice(random.choice(branch_candidates))
        patch = full_grid[r0 : r0 + rect_h, c0 : c0 + rect_w]
        if np.all(patch == BG):
            continue
        if len(set(map(int, np.unique(patch)))) < 2:
            continue
        mirror_c0 = width - (c0 + rect_w)
        mirror_c1 = mirror_c0 + rect_w - 1
        if not (c0 + rect_w - 1 < mirror_c0 or mirror_c1 < c0):
            continue
        return r0, c0, rect_h, rect_w

    raise RuntimeError("failed to place the marker rectangle")


def _oracle_valid(
    masked: np.ndarray,
    expected: np.ndarray,
    restored: np.ndarray,
    period: int,
) -> bool:
    try:
        actual = main(masked)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(_restore_hidden_patch(masked), restored):
        return False
    if _minimal_horizontal_period(restored) != period:
        return False

    masked_period = _minimal_horizontal_period(masked)
    direct_crop = masked[:, :masked_period]
    if masked_period == period:
        return False
    if np.array_equal(direct_crop, expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    period = random.randint(3, 5)
    repeats = random.randint(4, 30 // period)
    width = period * repeats
    if width > 30 or repeats % 2 != 0:
        return None

    height_max = min(14, width - 1)
    if height_max < 8:
        return None
    height = random.randint(8, height_max)

    try:
        tile, signature_rows = _sample_base_tile(height, period, repeats)
    except RuntimeError:
        return None

    restored = np.tile(tile, (1, repeats))
    if not np.array_equal(restored, np.fliplr(restored)):
        return None
    if _minimal_horizontal_period(restored) != period:
        return None

    try:
        r0, c0, rect_h, rect_w = _sample_marker_rect(restored, period, signature_rows)
    except RuntimeError:
        return None

    masked = restored.copy()
    masked[r0 : r0 + rect_h, c0 : c0 + rect_w] = MARKER
    expected = restored[:, :period].copy()

    if not _oracle_valid(masked, expected, restored, period):
        return None
    return masked


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-and-period sample")
