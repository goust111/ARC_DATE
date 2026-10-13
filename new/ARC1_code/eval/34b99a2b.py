from __future__ import annotations

import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the disagreement between two same-size foreground masks separated by a center bar.

    Input structure
    The grid has odd width. The middle column is a single-color separator, with
    equal-size binary masks on its left and right. The two masks share one
    background color and each side has exactly one foreground color.

    Rule
    Remove the separator and compare the two side masks cell by cell. Output the
    shared background color except where exactly one side has foreground; those
    differing cells become red.

    Key insight
    The separator is not part of the answer; it aligns two colored masks whose
    exclusive-or is rendered as a compact red mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _, width = map(int, g.shape)
    assert width % 2 == 1, "expected an odd-width grid with a center separator"

    mid = width // 2
    separator_values = np.unique(g[:, mid])
    assert separator_values.size == 1, "expected a solid center separator column"
    separator = int(separator_values[0])

    left = g[:, :mid]
    right = g[:, mid + 1 :]
    assert left.shape == right.shape, "expected equal-size masks around the separator"

    left_colors = set(map(int, np.unique(left)))
    right_colors = set(map(int, np.unique(right)))
    common_colors = left_colors & right_colors
    assert len(common_colors) == 1, "expected one shared background color"
    bg = next(iter(common_colors))

    assert separator != bg, "separator must be foreground"
    assert separator not in left_colors and separator not in right_colors, "separator color must only occur in the bar"
    assert len(left_colors - {bg}) == 1, "left side must be a binary mask"
    assert len(right_colors - {bg}) == 1, "right side must be a binary mask"

    differing = np.logical_xor(left != bg, right != bg)
    out = np.full(left.shape, bg, dtype=int)
    out[differing] = Color.RED
    return out

# --------------- generator (o5x) ---------------

# Panel geometry; upper bounds are exclusive for np.random.randint.
PANEL_H_RANGE, PANEL_W_RANGE = (4, 8), (3, 7)
# Mask density and XOR-density constraints.
FILL_RATIO_RANGE, XOR_RATIO_RANGE = (0.4, 0.6), (0.35, 0.7)
# Palette and retry budget.
BG_COLOR, OUT_COLOR = Color.BLACK, Color.RED
GENERATION_RETRIES = 300
FOREGROUND_PALETTE = [c for c in Color.NOT_BLACK if c != OUT_COLOR]


def _sample_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return paired masks satisfying the foreground, overlap, and XOR bounds."""
    n = h * w
    lo = int(np.ceil(FILL_RATIO_RANGE[0] * n))
    hi = int(np.floor(FILL_RATIO_RANGE[1] * n))
    left_count = int(np.random.randint(lo, hi + 1))
    right_count = int(np.random.randint(lo, hi + 1))
    min_overlap = max(1, left_count + right_count - n)
    max_overlap = min(left_count, right_count)
    if left_count == right_count:
        max_overlap -= 1
    xor_min = max(1, int(np.ceil(XOR_RATIO_RANGE[0] * n)))
    xor_max = max(xor_min, int(np.floor(XOR_RATIO_RANGE[1] * n)))
    min_overlap = max(min_overlap, int(np.ceil((left_count + right_count - xor_max) / 2)))
    max_overlap = min(max_overlap, int(np.floor((left_count + right_count - xor_min) / 2)))
    if min_overlap > max_overlap:
        return None
    overlap_count = int(np.random.randint(min_overlap, max_overlap + 1))
    order = np.random.permutation(n)
    overlap_idx = order[:overlap_count]
    left_only_idx = order[overlap_count:left_count]
    right_only_idx = order[left_count : left_count + right_count - overlap_count]
    left_mask = np.zeros(n, dtype=bool)
    right_mask = np.zeros(n, dtype=bool)
    left_mask[overlap_idx] = True
    right_mask[overlap_idx] = True
    left_mask[left_only_idx] = True
    right_mask[right_only_idx] = True
    return left_mask.reshape(h, w), right_mask.reshape(h, w)


def _render_input(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    colors = np.random.permutation(FOREGROUND_PALETTE)[:3]
    left_fg, right_fg, separator = map(int, colors)
    left = np.full(left_mask.shape, BG_COLOR, dtype=int)
    right = np.full(right_mask.shape, BG_COLOR, dtype=int)
    left[left_mask] = left_fg
    right[right_mask] = right_fg
    sep = np.full((left_mask.shape[0], 1), separator, dtype=int)
    return np.concatenate([left, sep, right], axis=1)


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(*PANEL_H_RANGE))
    w = int(np.random.randint(*PANEL_W_RANGE))
    masks = _sample_masks(h, w)
    if masks is None:
        return None
    left_mask, right_mask = masks

    inp = _render_input(left_mask, right_mask)
    expected = np.full(left_mask.shape, BG_COLOR, dtype=int)
    expected[np.logical_xor(left_mask, right_mask)] = OUT_COLOR
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
