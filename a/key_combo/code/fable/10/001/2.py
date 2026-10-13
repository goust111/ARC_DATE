from __future__ import annotations

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

CORNER_ORDER = ("tl", "tr", "bl", "br")
FLIP_STATES = ((False, False), (False, True), (True, False), (True, True))


def _corner_pos(kind: str, height: int, width: int) -> tuple[int, int]:
    if kind == "tl":
        return 0, 0
    if kind == "tr":
        return 0, width - 1
    if kind == "bl":
        return height - 1, 0
    if kind == "br":
        return height - 1, width - 1
    raise AssertionError(f"unknown corner kind: {kind}")


def _corner_bits(kind: str) -> tuple[int, int]:
    return (0 if kind[0] == "t" else 1, 0 if kind[1] == "l" else 1)


def _bits_to_corner(row_bit: int, col_bit: int) -> str:
    return ("t" if row_bit == 0 else "b") + ("l" if col_bit == 0 else "r")


def _toggle_corner(kind: str, flip_v: bool, flip_h: bool) -> str:
    row_bit, col_bit = _corner_bits(kind)
    if flip_v:
        row_bit = 1 - row_bit
    if flip_h:
        col_bit = 1 - col_bit
    return _bits_to_corner(row_bit, col_bit)


def _apply_flips(arr: np.ndarray, flip_v: bool, flip_h: bool) -> np.ndarray:
    out = arr.copy()
    if flip_v:
        out = out[::-1, :]
    if flip_h:
        out = out[:, ::-1]
    return out


def _panel_anchor_kind(panel: np.ndarray) -> str:
    height, width = panel.shape
    yellow_coords = np.argwhere(panel == YELLOW)
    assert yellow_coords.shape[0] == 1, "each panel must contain exactly one yellow anchor"
    r, c = map(int, yellow_coords[0])
    for kind in CORNER_ORDER:
        if (r, c) == _corner_pos(kind, height, width):
            return kind
    raise AssertionError("yellow anchor must sit on a corner cell")


def _alignment_flips(left_kind: str, right_kind: str) -> tuple[bool, bool]:
    left_row, left_col = _corner_bits(left_kind)
    right_row, right_col = _corner_bits(right_kind)
    return left_row != right_row, left_col != right_col


def _diff_output(left_mask: np.ndarray, right_input_mask: np.ndarray, left_kind: str, right_kind: str) -> np.ndarray:
    flip_v, flip_h = _alignment_flips(left_kind, right_kind)
    aligned_right = _apply_flips(right_input_mask, flip_v, flip_h)
    out = np.full(left_mask.shape, BLACK, dtype=int)
    out[left_mask & ~aligned_right] = PINK
    out[aligned_right & ~left_mask] = TEAL
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    height, width = mask.shape
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mask_is_asymmetric(mask: np.ndarray) -> bool:
    return (
        not np.array_equal(mask, mask[::-1, :])
        and not np.array_equal(mask, mask[:, ::-1])
        and not np.array_equal(mask, mask[::-1, ::-1])
    )


def _generate_left_mask(height: int, width: int, anchor_kind: str, rng: np.random.Generator) -> np.ndarray | None:
    anchor = _corner_pos(anchor_kind, height, width)
    area = height * width
    target = int(rng.integers(max(7, area // 5), max(8, area // 3) + 1))

    allowed = [(r, c) for r in range(height) for c in range(width) if (r, c) != anchor]
    if len(allowed) < target + 3:
        return None

    for _ in range(80):
        start = allowed[int(rng.integers(len(allowed)))]
        cells = {start}
        frontier = set()
        r0, c0 = start
        for nr, nc in ((r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) != anchor:
                frontier.add((nr, nc))

        while len(cells) < target and frontier:
            frontier_list = list(frontier)
            pick = frontier_list[int(rng.integers(len(frontier_list)))]
            frontier.remove(pick)
            cells.add(pick)
            pr, pc = pick
            for nr, nc in ((pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) != anchor and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows, cols = np.where(mask)
        row_span = int(rows.max() - rows.min() + 1)
        col_span = int(cols.max() - cols.min() + 1)
        if row_span < max(3, height // 2) or col_span < max(3, width // 2):
            continue
        if not _connected(mask):
            continue
        if not _mask_is_asymmetric(mask):
            continue
        if int(mask[: height // 2 + 1, :].sum()) == 0 or int(mask[height // 2 :, :].sum()) == 0:
            continue
        if int(mask[:, : width // 2 + 1].sum()) == 0 or int(mask[:, width // 2 :].sum()) == 0:
            continue
        return mask
    return None


def _sample_positions(mask: np.ndarray, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    pick_idx = rng.choice(len(coords), size=count, replace=False)
    return [coords[int(i)] for i in np.atleast_1d(pick_idx)]


def _build_case(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(6, 15))
    width = int(rng.integers(6, 15))
    left_kind = CORNER_ORDER[int(rng.integers(len(CORNER_ORDER)))]
    flip_v, flip_h = FLIP_STATES[int(rng.integers(len(FLIP_STATES)))]
    right_kind = _toggle_corner(left_kind, flip_v, flip_h)

    left_mask = _generate_left_mask(height, width, left_kind, rng)
    if left_mask is None:
        return None

    anchor = _corner_pos(left_kind, height, width)
    fg_candidates = left_mask.copy()
    bg_candidates = ~left_mask
    fg_candidates[anchor] = False
    bg_candidates[anchor] = False

    fg_count = int(fg_candidates.sum())
    bg_count = int(bg_candidates.sum())
    if fg_count < 2 or bg_count < 2:
        return None

    pink_diff_n = int(rng.integers(1, min(4, fg_count) + 1))
    teal_diff_n = int(rng.integers(1, min(4, bg_count) + 1))
    pink_only = _sample_positions(fg_candidates, pink_diff_n, rng)
    teal_only = _sample_positions(bg_candidates, teal_diff_n, rng)

    aligned_right = left_mask.copy()
    for r, c in pink_only:
        aligned_right[r, c] = False
    for r, c in teal_only:
        aligned_right[r, c] = True

    if not _mask_is_asymmetric(aligned_right):
        return None

    right_input_mask = _apply_flips(aligned_right, flip_v, flip_h)

    grid = np.full((height, width * 2 + 1), BLACK, dtype=int)
    grid[:, width] = GREY
    grid[:, :width][left_mask] = PINK
    grid[:, width + 1 :][right_input_mask] = TEAL
    grid[_corner_pos(left_kind, height, width)] = YELLOW
    rr, rc = _corner_pos(right_kind, height, width)
    grid[rr, width + 1 + rc] = YELLOW

    expected = np.full((height, width), BLACK, dtype=int)
    for r, c in pink_only:
        expected[r, c] = PINK
    for r, c in teal_only:
        expected[r, c] = TEAL

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    correct = actual
    for wrong_v, wrong_h in FLIP_STATES:
        if (wrong_v, wrong_h) == (flip_v, flip_h):
            continue
        wrong_kind = _toggle_corner(left_kind, wrong_v, wrong_h)
        wrong = _diff_output(left_mask, right_input_mask, left_kind, wrong_kind)
        if np.array_equal(wrong, correct):
            return None

    if len({tuple(row) for row in correct}) == 1 and np.all(correct == BLACK):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Align the right panel by its anchor corner, then color the left-only and right-only defects.

    Input structure:
    - A single full-height grey separator column splits the grid into two equal-size panels.
    - The left panel contains only black background, a pink mask, and one yellow anchor pixel.
    - The right panel contains only black background, a teal mask, and one yellow anchor pixel.
    - In each panel, the yellow anchor is the only yellow cell and it sits on exactly one corner cell.

    Rule:
    - Read the anchor corner on the left panel and the anchor corner on the right panel.
    - Mirror the right panel horizontally and/or vertically so that its anchor lands on the same corner as the left anchor.
    - Compare the two panel masks cell by cell after that alignment; anchor cells are not part of either mask.
    - Output one black panel-sized grid where left-only foreground cells become pink and right-only foreground cells become teal.

    Key insight:
    - The anchors do not mark fixed absolute corners such as top-left; their relative corner mismatch tells you which flips are needed before the defect comparison.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    grey_cols = [c for c in range(width) if np.all(g[:, c] == GREY)]
    assert len(grey_cols) == 1, "expected exactly one full-height grey separator column"
    sep = grey_cols[0]
    assert 0 < sep < width - 1, "separator must leave one panel on each side"
    assert sep == width - sep - 1, "panels must have equal width"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    panel_h, panel_w = left.shape
    assert right.shape == left.shape, "panels must match in size"
    assert not np.any(left == GREY) and not np.any(right == GREY), "grey belongs only to the separator"

    left_colors = set(map(int, np.unique(left))) - {BLACK}
    right_colors = set(map(int, np.unique(right))) - {BLACK}
    assert left_colors <= {PINK, YELLOW}, "left panel must contain only pink mask and yellow anchor"
    assert right_colors <= {TEAL, YELLOW}, "right panel must contain only teal mask and yellow anchor"
    assert PINK in left_colors and TEAL in right_colors, "both panels must contain a foreground mask"

    left_kind = _panel_anchor_kind(left)
    right_kind = _panel_anchor_kind(right)
    flip_v, flip_h = _alignment_flips(left_kind, right_kind)

    left_mask = left == PINK
    right_mask = right == TEAL
    aligned_right = _apply_flips(right_mask, flip_v, flip_h)

    out = np.full((panel_h, panel_w), BLACK, dtype=int)
    out[left_mask & ~aligned_right] = PINK
    out[aligned_right & ~left_mask] = TEAL
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _build_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-alignment defect case")
