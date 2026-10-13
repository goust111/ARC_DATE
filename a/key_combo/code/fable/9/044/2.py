from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, is_contiguous, paste


def _run_start_indices(lines: np.ndarray) -> list[int]:
    keep = [0]
    for idx in range(1, lines.shape[0]):
        if not np.array_equal(lines[idx], lines[idx - 1]):
            keep.append(idx)
    return keep


def _deduplicate_stretched_patch(patch: np.ndarray) -> np.ndarray:
    row_starts = _run_start_indices(patch)
    col_starts = _run_start_indices(patch.T)
    return patch[np.ix_(row_starts, col_starts)].copy()


def _extract_square_and_pattern(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    bg = get_bg(grid)
    components = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) == 2, "expected exactly two foreground objects"

    square_patch = None
    pattern_patch = None
    for comp in components:
        patch = crop_bbox(grid, comp["bbox"])
        colors = set(map(int, np.unique(patch)))
        is_solid_square = patch.shape[0] == patch.shape[1] and np.all(patch != bg) and len(colors) >= 2
        is_binary_pattern = bg in colors and len(colors) == 2

        if is_solid_square:
            assert square_patch is None, "expected a unique multicolor square"
            square_patch = patch
        elif is_binary_pattern:
            assert pattern_patch is None, "expected a unique stretched binary pattern"
            pattern_patch = patch
        else:
            raise AssertionError("unrecognized foreground object")

    assert square_patch is not None and pattern_patch is not None
    assert square_patch.shape[0] <= pattern_patch.shape[0]
    assert square_patch.shape[1] <= pattern_patch.shape[1]
    return square_patch, pattern_patch, int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """Use a deduplicated non-uniformly stretched binary template as a mask on a multicolor square.

    Input structure:
    - The grid has a black background and exactly two separated foreground objects.
    - One object is a larger two-color pattern area whose rows and columns were stretched by unequal adjacent repetitions.
    - The other object is a solid kxk multicolor square with no background holes.

    Rule:
    - Crop the stretched pattern area and collapse each maximal run of identical neighboring rows and identical neighboring columns.
    - The deduplicated result is a kxk binary mask aligned to the multicolor square.
    - Keep square pixels where the recovered mask is non-background and set all other cells to black.

    Key insight:
    - The large pattern is not an integer-scale image; only row-and-column deduplication recovers the true template, which then acts as a mask on the separate colored square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    square_patch, pattern_patch, bg = _extract_square_and_pattern(g)
    mask = _deduplicate_stretched_patch(pattern_patch) != bg

    k = square_patch.shape[0]
    assert square_patch.shape == (k, k)
    assert mask.shape == square_patch.shape

    out = np.full_like(square_patch, bg)
    out[mask] = square_patch[mask]
    return out


_GEN_ATTEMPTS = 500
_GEN_K_CHOICES = (3, 4, 5, 6)
_GEN_PATTERN_REPEAT_MAX = 4
_GEN_PATTERN_MAX_DIM = 15
_GEN_MARGIN_MAX = 3
_GEN_GAP_MIN = 2
_GEN_GAP_MAX = 4


def _gen_random_mask(rng: np.random.Generator, k: int) -> np.ndarray | None:
    for _ in range(300):
        density = float(rng.uniform(0.38, 0.72))
        mask = rng.random((k, k)) < density
        if mask.sum() < max(4, k):
            continue
        if mask.sum() > k * k - 2:
            continue
        if not np.all(np.any(mask, axis=0)) or not np.all(np.any(mask, axis=1)):
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=8):
            continue
        if any(np.array_equal(mask[r], mask[r - 1]) for r in range(1, k)):
            continue
        if any(np.array_equal(mask[:, c], mask[:, c - 1]) for c in range(1, k)):
            continue
        zeros = k * k - int(mask.sum())
        if zeros < 2:
            continue
        return mask
    return None


def _gen_run_lengths(rng: np.random.Generator, k: int) -> np.ndarray | None:
    for _ in range(300):
        runs = rng.integers(1, _GEN_PATTERN_REPEAT_MAX + 1, size=k, dtype=int)
        if np.all(runs == runs[0]):
            continue
        if np.max(runs) == 1:
            continue
        if int(runs.sum()) > _GEN_PATTERN_MAX_DIM:
            continue
        return runs
    return None


def _uniform_partition_reduce(stretched: np.ndarray, k: int) -> np.ndarray | None:
    h, w = stretched.shape
    if h % k != 0 or w % k != 0:
        return None

    rh = h // k
    cw = w // k
    out = np.empty((k, k), dtype=stretched.dtype)
    for r in range(k):
        for c in range(k):
            block = stretched[r * rh : (r + 1) * rh, c * cw : (c + 1) * cw]
            vals = np.unique(block)
            if len(vals) != 1:
                return None
            out[r, c] = vals[0]
    return out


def _gen_square(rng: np.random.Generator, k: int, forbidden_color: int) -> np.ndarray:
    palette = [c for c in Color.NOT_BLACK if c != forbidden_color]
    while True:
        n_colors = int(rng.integers(3, min(7, len(palette)) + 1))
        colors = rng.choice(palette, size=n_colors, replace=False)
        square = rng.choice(colors, size=(k, k), replace=True).astype(int)
        if len(set(map(int, np.unique(square)))) < 3:
            continue
        return square


def _layout_positions(
    rng: np.random.Generator,
    pattern_shape: tuple[int, int],
    square_shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    ph, pw = map(int, pattern_shape)
    sh, sw = map(int, square_shape)
    gap = int(rng.integers(_GEN_GAP_MIN, _GEN_GAP_MAX + 1))
    margins = rng.integers(0, _GEN_MARGIN_MAX + 1, size=4, dtype=int)
    top, bottom, left, right = [int(x) for x in margins]
    mode = int(rng.integers(4))

    if mode == 0:
        height = max(ph, sh) + top + bottom
        width = pw + sw + gap + left + right
        pattern_rc = (top + int(rng.integers(0, height - top - bottom - ph + 1)), left)
        square_rc = (top + int(rng.integers(0, height - top - bottom - sh + 1)), left + pw + gap)
    elif mode == 1:
        height = max(ph, sh) + top + bottom
        width = pw + sw + gap + left + right
        square_rc = (top + int(rng.integers(0, height - top - bottom - sh + 1)), left)
        pattern_rc = (top + int(rng.integers(0, height - top - bottom - ph + 1)), left + sw + gap)
    elif mode == 2:
        height = ph + sh + gap + top + bottom
        width = max(pw, sw) + left + right
        pattern_rc = (top, left + int(rng.integers(0, width - left - right - pw + 1)))
        square_rc = (top + ph + gap, left + int(rng.integers(0, width - left - right - sw + 1)))
    else:
        height = ph + sh + gap + top + bottom
        width = max(pw, sw) + left + right
        square_rc = (top, left + int(rng.integers(0, width - left - right - sw + 1)))
        pattern_rc = (top + sh + gap, left + int(rng.integers(0, width - left - right - pw + 1)))

    if height > 30 or width > 30:
        return None
    return (height, width), pattern_rc, square_rc


def _expected_output(square: np.ndarray, mask: np.ndarray, bg: int = Color.BLACK) -> np.ndarray:
    out = np.full(square.shape, int(bg), dtype=int)
    out[mask] = square[mask]
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    k = int(rng.choice(_GEN_K_CHOICES))
    pattern_color = int(rng.choice(Color.NOT_BLACK))
    mask = _gen_random_mask(rng, k)
    if mask is None:
        return None

    row_runs = _gen_run_lengths(rng, k)
    col_runs = _gen_run_lengths(rng, k)
    if row_runs is None or col_runs is None:
        return None

    template = np.where(mask, pattern_color, Color.BLACK).astype(int)
    stretched = np.repeat(np.repeat(template, row_runs, axis=0), col_runs, axis=1)
    if _uniform_partition_reduce(stretched, k) is not None:
        return None

    square = _gen_square(rng, k, pattern_color)
    if len(set(map(int, np.unique(square[mask])))) < 2:
        return None
    if np.count_nonzero(mask) < 2 or np.count_nonzero(~mask) < 2:
        return None

    layout = _layout_positions(rng, stretched.shape, square.shape)
    if layout is None:
        return None
    (height, width), pattern_rc, square_rc = layout

    grid = np.full((height, width), Color.BLACK, dtype=int)
    paste(grid, stretched, pattern_rc[0], pattern_rc[1], transparent=Color.BLACK)
    paste(grid, square, square_rc[0], square_rc[1], transparent=Color.BLACK)

    expected = _expected_output(square, mask)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        square_patch, pattern_patch, bg = _extract_square_and_pattern(grid)
    except Exception:
        return None
    recovered_mask = _deduplicate_stretched_patch(pattern_patch) != bg
    if not np.array_equal(recovered_mask, mask):
        return None
    if not np.array_equal(square_patch, square):
        return None
    if np.array_equal(expected, square):
        return None
    if np.array_equal(expected, recovered_mask.astype(int) * pattern_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
