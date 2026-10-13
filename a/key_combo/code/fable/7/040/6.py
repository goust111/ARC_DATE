"""ARC task: repair a vertically symmetric template and stamp it in two orientations."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREY = 5
TEMPLATE_COLORS = (3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (damaged template, marker mask, background)."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    assert len(separator_rows) == 1, "expected exactly one solid grey separator row"
    sep = int(separator_rows[0])
    template = g[:sep]
    mask = g[sep + 1 :]
    assert template.shape == mask.shape and template.size > 0, (
        "template and mask must be non-empty regions of the same shape"
    )

    mask_colors = set(map(int, np.unique(mask)))
    background_colors = mask_colors - {RED, BLUE}
    assert len(background_colors) == 1, "mask must have one background color"
    bg = background_colors.pop()
    assert set(map(int, np.unique(mask))) <= {bg, RED, BLUE}, (
        "mask may contain only background, red, and blue"
    )
    assert np.any(mask == RED) and np.any(mask == BLUE), (
        "both stamp orientations must be represented"
    )
    return template, mask, int(bg)


def _repair_vertical_symmetry(template: np.ndarray, bg: int) -> np.ndarray:
    """Fill one-sided holes from their partners across the horizontal axis."""
    repaired = np.asarray(template, dtype=int).copy()
    height, width = repaired.shape
    for r in range(height // 2):
        mirror_r = height - 1 - r
        for c in range(width):
            upper = int(repaired[r, c])
            lower = int(repaired[mirror_r, c])
            if upper == bg and lower != bg:
                repaired[r, c] = lower
            elif lower == bg and upper != bg:
                repaired[mirror_r, c] = upper
            elif upper != bg and lower != bg:
                assert upper == lower, "opposite intact template cells must agree"
    assert np.array_equal(repaired, np.flipud(repaired)), (
        "template must become vertically symmetric after one-sided holes are filled"
    )
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a vertically symmetric template, then stamp original and mirrored copies according to a mask.

    Input structure:
    - One solid grey row separates equal-sized upper and lower panels.
    - The upper panel is a multicolor, vertically symmetric template with some
      foreground cells replaced by its background; every hole's vertical
      mirror partner remains intact.
    - The lower panel contains only background, red markers, and blue markers.

    Rule:
    - Complete each one-sided hole from the cell across the template's
      horizontal symmetry axis. Treat every lower-panel cell as one output
      block the size of the repaired template: paste the repaired template at
      red cells, its left-right flip at blue cells, and background elsewhere.

    Key insight:
    - The visible upper panel is not the stamp itself: vertical reflection must
      first recover the missing pixels, while marker color independently
      chooses the stamp's horizontal orientation.
    """
    template, mask, bg = _split_input(grid)
    stamp = _repair_vertical_symmetry(template, bg)
    flipped_stamp = np.fliplr(stamp)
    tile_h, tile_w = stamp.shape

    out = np.full(
        (mask.shape[0] * tile_h, mask.shape[1] * tile_w), bg, dtype=int
    )
    for marker_r, marker_c in np.argwhere(mask != bg):
        marker_r, marker_c = int(marker_r), int(marker_c)
        r0, c0 = marker_r * tile_h, marker_c * tile_w
        if int(mask[marker_r, marker_c]) == RED:
            patch = stamp
        else:
            assert int(mask[marker_r, marker_c]) == BLUE
            patch = flipped_stamp
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = patch
    return out


def _make_complete_template(height: int, width: int) -> np.ndarray:
    """Sample a colorful vertical-symmetry pattern that is not LR-symmetric."""
    half_height = (height + 1) // 2
    for _ in range(100):
        color_count = random.randint(2, min(4, len(TEMPLATE_COLORS)))
        colors = random.sample(TEMPLATE_COLORS, color_count)
        upper = np.full((half_height, width), BACKGROUND, dtype=int)

        # Moderate density keeps both the motif and its background silhouette clear.
        for r in range(half_height):
            for c in range(width):
                if random.random() < 0.58:
                    upper[r, c] = random.choice(colors)
        complete = np.vstack([upper, np.flipud(upper[: height // 2])])

        pair_cells = [
            (r, c)
            for r in range(height // 2)
            for c in range(width)
            if complete[r, c] != BACKGROUND
        ]
        used_colors = set(map(int, complete[complete != BACKGROUND]))
        if (
            len(pair_cells) >= 3
            and len(used_colors) >= 2
            and not np.array_equal(complete, np.fliplr(complete))
        ):
            return complete
    raise RuntimeError("could not sample a non-degenerate complete template")


def _damage_template(complete: np.ndarray) -> np.ndarray:
    """Make holes on both sides of the symmetry axis, never in mirror pairs."""
    height, width = complete.shape
    candidates = [
        (r, c)
        for r in range(height // 2)
        for c in range(width)
        if complete[r, c] != BACKGROUND
    ]
    hole_count = random.randint(2, min(4, len(candidates)))
    chosen = random.sample(candidates, hole_count)
    damaged = complete.copy()

    # These forced sides rule out simply trusting either the upper or lower half.
    for index, (r, c) in enumerate(chosen):
        mirror_r = height - 1 - r
        if index == 0:
            damaged[r, c] = BACKGROUND
        elif index == 1:
            damaged[mirror_r, c] = BACKGROUND
        elif random.random() < 0.5:
            damaged[r, c] = BACKGROUND
        else:
            damaged[mirror_r, c] = BACKGROUND
    return damaged


def _make_mask(height: int, width: int) -> np.ndarray:
    """Create a sparse mask with both marker branches and several empty blocks."""
    cell_count = height * width
    marker_count = random.randint(3, min(6, cell_count - 2))
    red_count = random.randint(1, marker_count - 1)
    cells = random.sample(range(cell_count), marker_count)
    random.shuffle(cells)
    mask = np.full((height, width), BACKGROUND, dtype=int)
    for index, flat in enumerate(cells):
        r, c = divmod(flat, width)
        mask[r, c] = RED if index < red_count else BLUE
    return mask


def _expected_output(complete: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Independent construction used as the generator oracle."""
    height, width = complete.shape
    expected = np.full((height * height, width * width), BACKGROUND, dtype=int)
    for r in range(height):
        for c in range(width):
            if mask[r, c] == BACKGROUND:
                continue
            patch = complete if mask[r, c] == RED else np.fliplr(complete)
            expected[r * height : (r + 1) * height, c * width : (c + 1) * width] = patch
    return expected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    complete = _make_complete_template(height, width)
    damaged = _damage_template(complete)
    mask = _make_mask(height, width)
    separator = np.full((1, width), GREY, dtype=int)
    inp = np.vstack([damaged, separator, mask])
    expected = _expected_output(complete, mask)

    # Semantic checks: each advertised operation changes the result.
    if np.array_equal(damaged, complete):
        return None
    if np.array_equal(complete, np.fliplr(complete)):
        return None
    raw_stamp_result = _expected_output(damaged, mask)
    if np.array_equal(raw_stamp_result, expected):
        return None
    all_original_mask = np.where(mask == BLUE, RED, mask)
    if np.array_equal(_expected_output(complete, all_original_mask), expected):
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(inp.shape) > 30 or max(actual.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance satisfying all solver assumptions."""
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-repair stamp instance")
