from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEFT_SIZE_RANGE = (3, 5)
GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 120


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    separator_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)]
    assert len(separator_cols) == 1, "expected exactly one grey separator column"
    sep = int(separator_cols[0])
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0, "both panels must be non-empty"
    return left, right


def _extract_candidate_masks(right: np.ndarray) -> dict[int, np.ndarray]:
    comps = _components(right != BACKGROUND)
    assert len(comps) == 3, "expected exactly three candidate template blocks"

    masks: dict[int, np.ndarray] = {}
    for coords in comps:
        patch = _crop_bbox(right, _bbox(coords))
        assert patch.shape == (3, 3), "each candidate must occupy a 3x3 bbox"
        colors = {int(v) for v in patch.ravel() if int(v) != BACKGROUND}
        diversity = len(colors)
        assert 1 <= diversity <= 3, "candidate diversity must be in 1..3"
        assert diversity not in masks, "candidate diversities must be unique"
        masks[diversity] = patch != BACKGROUND

    assert set(masks) == {1, 2, 3}, "candidates must realize diversities 1, 2, and 3"
    return masks


def _stamp_output(left: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = left.shape
    out = np.zeros((h * 3, w * 3), dtype=int)
    for r, c in np.argwhere(left != BACKGROUND):
        block = out[r * 3 : (r + 1) * 3, c * 3 : (c + 1) * 3]
        block[mask] = int(left[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the right-hand template whose color diversity matches the left panel's diversity.

    Input structure:
    - A single grey separator column splits the grid into a left source panel and a right candidate panel.
    - The left panel is on a black background and contains colored pixels using K distinct non-black colors.
    - The right panel contains exactly three separated 3x3 candidate templates on black, whose non-black colors have diversities 1, 2, and 3.

    Rule:
    - Count K, the number of distinct non-background colors in the left panel.
    - Select the unique right-hand 3x3 candidate whose own non-background color diversity is also K, then discard its colors and keep only its non-background mask.
    - Build a 3x-expanded output: for every non-background left pixel at (r, c), stamp that selected mask recolored to the pixel's color into output block (r, c); blocks for black left cells stay black.

    Key insight:
    - The left panel contributes two independent signals at once: its non-black locations say where to stamp, while its color diversity says which right-hand template shape to stamp.
    """
    left, right = _split_panels(grid)
    left_colors = {int(v) for v in left.ravel() if int(v) != BACKGROUND}
    diversity = len(left_colors)
    assert 1 <= diversity <= 3, "left panel must use 1..3 non-background colors"
    assert int(np.count_nonzero(left != BACKGROUND)) >= 2, "left panel needs at least two source pixels"

    candidate_masks = _extract_candidate_masks(right)
    return _stamp_output(left, candidate_masks[diversity])


def _random_connected_mask() -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        target_area = random.randint(5, 8)
        mask = np.zeros((3, 3), dtype=bool)
        start = (random.randrange(3), random.randrange(3))
        mask[start] = True
        cells = {start}

        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 3 and 0 <= nc < 3 and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            pick = random.choice(tuple(frontier))
            mask[pick] = True
            cells.add(pick)

        coords = np.argwhere(mask)
        if coords.shape[0] != target_area:
            continue
        if _bbox(coords) != (0, 0, 2, 2):
            continue
        if len(_components(mask)) != 1:
            continue
        if np.all(mask):
            continue
        return mask

    raise RuntimeError("failed to generate a valid 3x3 template mask")


def _make_left_panel(size: int, diversity: int) -> tuple[np.ndarray, set[int]]:
    panel = np.zeros((size, size), dtype=int)
    all_positions = [(r, c) for r in range(size) for c in range(size)]
    min_fg = max(2, diversity)
    max_fg = max(min_fg, min(size * size - 1, (size * size * 3) // 5))
    fg_count = random.randint(min_fg, max_fg)
    positions = random.sample(all_positions, fg_count)
    colors = random.sample(PALETTE, diversity)
    assigned = colors[:] + [random.choice(colors) for _ in range(fg_count - diversity)]
    random.shuffle(assigned)
    for (r, c), color in zip(positions, assigned):
        panel[r, c] = int(color)
    return panel, set(colors)


def _paint_candidate(mask: np.ndarray, diversity: int, forbidden: set[int]) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    usable = [color for color in PALETTE if color not in forbidden]
    if len(usable) < diversity:
        usable = PALETTE[:]
    colors = random.sample(usable, diversity)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assigned = colors[:] + [random.choice(colors) for _ in range(len(cells) - diversity)]
    random.shuffle(assigned)
    for (r, c), color in zip(cells, assigned):
        patch[r, c] = int(color)
    return patch


def _make_candidate_masks() -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    used: set[bytes] = set()
    diversities = [1, 2, 3]
    random.shuffle(diversities)
    for diversity in diversities:
        while True:
            mask = _random_connected_mask()
            key = mask.tobytes()
            if key not in used:
                used.add(key)
                masks[diversity] = mask
                break
    return masks


def _make_right_panel(height: int, left_colors: set[int], masks: dict[int, np.ndarray]) -> np.ndarray:
    gap1 = random.randint(1, 2)
    gap2 = random.randint(1, 2)
    outer_left = random.randint(0, 1)
    outer_right = random.randint(0, 1)
    width = outer_left + 3 + gap1 + 3 + gap2 + 3 + outer_right

    panel = np.zeros((height, width), dtype=int)
    order = [1, 2, 3]
    random.shuffle(order)
    col = outer_left

    for idx, diversity in enumerate(order):
        row = random.randint(0, height - 3)
        patch = _paint_candidate(masks[diversity], diversity, left_colors)
        panel[row : row + 3, col : col + 3] = patch
        if idx == 0:
            col += 3 + gap1
        elif idx == 1:
            col += 3 + gap2
    return panel


def _validate_generated(grid: np.ndarray, expected: np.ndarray, selected_diversity: int, masks: dict[int, np.ndarray]) -> bool:
    left, right = _split_panels(grid)
    parsed_masks = _extract_candidate_masks(right)
    parsed_diversity = len({int(v) for v in left.ravel() if int(v) != BACKGROUND})

    if parsed_diversity != selected_diversity:
        return False
    if any(np.array_equal(parsed_masks[k], parsed_masks[j]) for k in parsed_masks for j in parsed_masks if k < j):
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    for other_diversity, other_mask in masks.items():
        alt = _stamp_output(left, other_mask)
        if other_diversity == selected_diversity:
            if not np.array_equal(alt, expected):
                return False
        else:
            if np.array_equal(alt, expected):
                return False

    right_color_counts = Counter(
        len({int(v) for v in _crop_bbox(right, _bbox(coords)).ravel() if int(v) != BACKGROUND})
        for coords in _components(right != BACKGROUND)
    )
    return right_color_counts == Counter({1: 1, 2: 1, 3: 1})


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        size = random.randint(*LEFT_SIZE_RANGE)
        selected_diversity = random.randint(1, 3)
        left, left_colors = _make_left_panel(size, selected_diversity)
        masks = _make_candidate_masks()
        right = _make_right_panel(size, left_colors, masks)
        grid = np.concatenate(
            [left, np.full((size, 1), SEPARATOR, dtype=int), right],
            axis=1,
        )
        expected = _stamp_output(left, masks[selected_diversity])
        if _validate_generated(grid, expected, selected_diversity, masks):
            return grid
    raise RuntimeError("failed to generate a valid input for the puzzle")
