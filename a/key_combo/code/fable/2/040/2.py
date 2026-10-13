from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
PINK = 6
MASK_COLORS = [2, 3, 4, 7, 8, 9]

MIN_PANEL_SIZE = 8
MAX_PANEL_HEIGHT = 14
MAX_PANEL_WIDTH = 18
GENERATION_ATTEMPTS = 500


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == GREY))]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _panel_anchor_and_mask(panel: np.ndarray) -> tuple[tuple[int, int], np.ndarray, int]:
    anchors = np.argwhere(panel == BLUE)
    assert anchors.shape[0] == 1, "expected exactly one blue anchor pixel per panel"
    ar, ac = map(int, anchors[0])

    colors = set(map(int, panel.ravel())) - {BLACK, BLUE}
    assert len(colors) == 1, "expected exactly one foreground mask color per panel"
    color = int(colors.pop())
    assert color != GREY, "grey is reserved for the separator row"

    mask = panel == color
    assert bool(mask.any()), "expected a non-empty foreground mask"
    assert not bool(mask[ar, ac]), "the blue anchor is not mask content"
    return (ar, ac), mask, color


def _translated_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    moved = coords + np.array([[int(dr), int(dc)]])
    h, w = map(int, mask.shape)
    assert bool(np.all((0 <= moved[:, 0]) & (moved[:, 0] < h))), "translated mask leaves the panel vertically"
    assert bool(np.all((0 <= moved[:, 1]) & (moved[:, 1] < w))), "translated mask leaves the panel horizontally"

    out = np.zeros(mask.shape, dtype=bool)
    out[moved[:, 0], moved[:, 1]] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align two masks by their blue anchor pixels and emit pink where exactly one mask covers a cell.

    Input structure:
    - The input is two equal-size black panels stacked vertically with one full-width grey separator row between them.
    - Each panel contains exactly one blue anchor pixel and one non-blue, non-grey foreground mask color.
    - The two mask colors are different, and blue anchor pixels are treated as background rather than mask cells.

    Rule:
    - Remove the separator row and create a black output canvas with the same shape as one panel.
    - Translate the lower panel mask so its blue anchor coincides with the upper panel anchor.
    - Paint pink exactly where one aligned mask has foreground and the other has background; leave overlaps, shared background, and the anchor location black.

    Key insight:
    - Correspondence is defined by anchor-relative coordinates, so positional XOR is only valid in examples where the two anchors already coincide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    sep = _separator_row(g)
    assert sep > 0, "separator must have a panel above it"
    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.shape == bottom.shape, "top and bottom panels must be equal size"
    assert bool(np.all(top != GREY)) and bool(np.all(bottom != GREY)), "grey may only appear in the separator row"

    top_anchor, top_mask, top_color = _panel_anchor_and_mask(top)
    bottom_anchor, bottom_mask, bottom_color = _panel_anchor_and_mask(bottom)
    assert top_color != bottom_color, "the two panel masks must use different colors"

    dr = top_anchor[0] - bottom_anchor[0]
    dc = top_anchor[1] - bottom_anchor[1]
    aligned_bottom = _translated_mask(bottom_mask, dr, dc)

    out = np.full(top.shape, BLACK, dtype=int)
    out[top_mask ^ aligned_bottom] = PINK
    out[top_anchor] = BLACK
    return out


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _random_connected_offsets(radius_r: int, radius_c: int, count: int) -> list[tuple[int, int]]:
    domain = {
        (r, c)
        for r in range(-int(radius_r), int(radius_r) + 1)
        for c in range(-int(radius_c), int(radius_c) + 1)
        if (r, c) != (0, 0)
    }
    assert count <= len(domain)

    coords = {random.choice(tuple(domain))}
    while len(coords) < count:
        candidates = set()
        for cell in coords:
            for nbr in _neighbors4(cell):
                if nbr in domain and nbr not in coords:
                    candidates.add(nbr)
        if not candidates:
            raise RuntimeError("failed to grow a connected relative mask")
        coords.add(random.choice(tuple(candidates)))
    return list(coords)


def _mask_from_offsets(shape: tuple[int, int], anchor: tuple[int, int], offsets: set[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    ar, ac = anchor
    for dr, dc in offsets:
        r = ar + int(dr)
        c = ac + int(dc)
        assert 0 <= r < shape[0] and 0 <= c < shape[1]
        out[r, c] = True
    return out


def _render_panel(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    offsets: set[tuple[int, int]],
    color: int,
) -> np.ndarray:
    panel = np.full(shape, BLACK, dtype=int)
    mask = _mask_from_offsets(shape, anchor, offsets)
    panel[mask] = int(color)
    panel[anchor] = BLUE
    return panel


def _render_expected(
    shape: tuple[int, int],
    top_anchor: tuple[int, int],
    top_offsets: set[tuple[int, int]],
    bottom_offsets: set[tuple[int, int]],
) -> np.ndarray:
    top = _mask_from_offsets(shape, top_anchor, top_offsets)
    bottom_aligned = _mask_from_offsets(shape, top_anchor, bottom_offsets)
    out = np.full(shape, BLACK, dtype=int)
    out[top ^ bottom_aligned] = PINK
    out[top_anchor] = BLACK
    return out


def _sample_anchor(shape: tuple[int, int], radius_r: int, radius_c: int) -> tuple[int, int]:
    h, w = shape
    return (
        random.randint(int(radius_r), h - 1 - int(radius_r)),
        random.randint(int(radius_c), w - 1 - int(radius_c)),
    )


def _split_relative_masks(
    radius_r: int,
    radius_c: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    common_count = random.randint(2, 5)
    top_only_count = random.randint(3, 8)
    bottom_only_count = random.randint(3, 8)
    total = common_count + top_only_count + bottom_only_count

    offsets = _random_connected_offsets(radius_r, radius_c, total)
    random.shuffle(offsets)
    common = set(offsets[:common_count])
    top_only = set(offsets[common_count : common_count + top_only_count])
    bottom_only = set(offsets[common_count + top_only_count :])
    top = common | top_only
    bottom = common | bottom_only
    return top, bottom, top_only, bottom_only


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(MIN_PANEL_SIZE, MAX_PANEL_HEIGHT)
    panel_w = random.randint(MIN_PANEL_SIZE, MAX_PANEL_WIDTH)
    shape = (panel_h, panel_w)

    max_radius_r = min(4, (panel_h - 1) // 2)
    max_radius_c = min(4, (panel_w - 1) // 2)
    if max_radius_r < 3 or max_radius_c < 3:
        return None
    radius_r = random.randint(3, max_radius_r)
    radius_c = random.randint(3, max_radius_c)

    top_anchor = _sample_anchor(shape, radius_r, radius_c)
    if random.random() < 0.25:
        bottom_anchor = top_anchor
    else:
        candidates = [
            (r, c)
            for r in range(radius_r, panel_h - radius_r)
            for c in range(radius_c, panel_w - radius_c)
            if (r, c) != top_anchor
        ]
        if not candidates:
            return None
        bottom_anchor = random.choice(candidates)

    top_offsets, bottom_offsets, top_only, bottom_only = _split_relative_masks(radius_r, radius_c)
    if not top_only or not bottom_only:
        return None
    if len(top_offsets & bottom_offsets) < 2:
        return None

    top_color, bottom_color = random.sample(MASK_COLORS, 2)
    top_panel = _render_panel(shape, top_anchor, top_offsets, top_color)
    bottom_panel = _render_panel(shape, bottom_anchor, bottom_offsets, bottom_color)
    separator = np.full((1, panel_w), GREY, dtype=int)
    grid = np.vstack([top_panel, separator, bottom_panel])

    expected = _render_expected(shape, top_anchor, top_offsets, bottom_offsets)
    pink_count = int(np.count_nonzero(expected == PINK))
    if pink_count != len(top_only) + len(bottom_only) or not (6 <= pink_count <= 16):
        return None

    if bottom_anchor != top_anchor:
        top_mask = _mask_from_offsets(shape, top_anchor, top_offsets)
        unaligned_bottom = _mask_from_offsets(shape, bottom_anchor, bottom_offsets)
        unaligned = np.full(shape, BLACK, dtype=int)
        unaligned[top_mask ^ unaligned_bottom] = PINK
        if np.array_equal(unaligned, expected):
            return None
        if int(np.count_nonzero(unaligned != expected)) < 2:
            return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-aligned XOR input")
