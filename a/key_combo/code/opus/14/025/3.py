from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
LEFT_COLOR = 1
OUTPUT_COLOR = 2
RIGHT_COLOR = 3
SEPARATOR_COLOR = 5

PANEL_WIDTH = 5
INPUT_WIDTH = PANEL_WIDTH * 2 + 1
HEIGHT_RANGE = (7, 12)
VISIBLE_COUNT_RANGE = (3, 6)
MAX_ROW_OCCUPANCY = 2
MAX_GENERATION_ATTEMPTS = 500
ORBIT_COLUMNS = (0, 1, 2)


def _complete_horizontally(mask: np.ndarray) -> np.ndarray:
    """Mirror a panel mask across its own vertical axis and union the result."""
    return np.logical_or(mask, np.fliplr(mask))


def _panel_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert w == INPUT_WIDTH, f"expected width {INPUT_WIDTH}, got {w}"
    mid = w // 2
    assert np.all(g[:, mid] == SEPARATOR_COLOR), "expected a solid separator column"

    left = g[:, :mid] == LEFT_COLOR
    right = g[:, mid + 1 :] == RIGHT_COLOR
    assert left.shape == right.shape == (h, PANEL_WIDTH)
    return left, right


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each side to vertical symmetry, then output the XOR disagreement mask.

    Input structure:
    - The grid has odd width with a solid grey separator column in the middle.
    - The left half is a sparse blue pattern and the right half is a sparse green pattern.
    - Each half uses its own 5-column coordinate system and is not yet symmetric about its own vertical axis.

    Rule:
    - Mirror every blue cell in the left half across the left half's own vertical axis, keeping originals.
    - Mirror every green cell in the right half across the right half's own vertical axis, keeping originals.
    - Remove the separator and compare the two completed 5-column masks cell by cell.
    - Output red where exactly one completed mask has foreground, otherwise output black.

    Key insight:
    - The visible sparse pixels are only compressed representatives; the true comparison happens after each half is first expanded into its implicit symmetric pattern.
    """
    left_visible, right_visible = _panel_masks(grid)
    left_completed = _complete_horizontally(left_visible)
    right_completed = _complete_horizontally(right_visible)

    out = np.full(left_visible.shape, BACKGROUND, dtype=int)
    out[np.logical_xor(left_completed, right_completed)] = OUTPUT_COLOR
    return out


def _sample_orbits(height: int) -> list[tuple[int, int]] | None:
    count = random.randint(*VISIBLE_COUNT_RANGE)
    must_include_center = random.random() < 0.45
    all_orbits = [(r, orbit) for r in range(height) for orbit in ORBIT_COLUMNS]

    for _ in range(150):
        picked = random.sample(all_orbits, count)
        row_counts = np.bincount([r for r, _ in picked], minlength=height)
        side_count = sum(orbit < 2 for _, orbit in picked)
        center_count = count - side_count

        if side_count == 0:
            continue
        if side_count < center_count:
            continue
        if must_include_center and center_count == 0:
            continue
        if not np.any(row_counts):
            continue
        if row_counts.max() > MAX_ROW_OCCUPANCY:
            continue
        if np.count_nonzero(row_counts) < 2:
            continue
        return picked
    return None


def _visible_mask_from_orbits(height: int, orbits: list[tuple[int, int]], *, right_side: bool) -> np.ndarray:
    mask = np.zeros((height, PANEL_WIDTH), dtype=bool)
    for r, orbit in orbits:
        col = PANEL_WIDTH - 1 - orbit if right_side else orbit
        mask[int(r), int(col)] = True
    return mask


def _render_input(left_visible: np.ndarray, right_visible: np.ndarray) -> np.ndarray:
    h = left_visible.shape[0]
    grid = np.full((h, INPUT_WIDTH), BACKGROUND, dtype=int)
    grid[:, PANEL_WIDTH] = SEPARATOR_COLOR
    grid[:, :PANEL_WIDTH][left_visible] = LEFT_COLOR
    grid[:, PANEL_WIDTH + 1 :][right_visible] = RIGHT_COLOR
    return grid


def _is_valid_instance(left_visible: np.ndarray, right_visible: np.ndarray) -> bool:
    left_completed = _complete_horizontally(left_visible)
    right_completed = _complete_horizontally(right_visible)
    full_xor = np.logical_xor(left_completed, right_completed)

    if np.array_equal(left_completed, left_visible):
        return False
    if np.array_equal(right_completed, right_visible):
        return False
    if np.array_equal(left_completed, right_completed):
        return False
    if not np.any(left_completed & right_completed):
        return False
    if not np.any(~left_completed & ~right_completed):
        return False

    raw_xor = np.logical_xor(left_visible, right_visible)
    left_only_xor = np.logical_xor(left_completed, right_visible)
    right_only_xor = np.logical_xor(left_visible, right_completed)
    if np.array_equal(full_xor, raw_xor):
        return False
    if np.array_equal(full_xor, left_only_xor):
        return False
    if np.array_equal(full_xor, right_only_xor):
        return False

    red_count = int(full_xor.sum())
    total = int(full_xor.size)
    if red_count < 2 or red_count > total - 2:
        return False

    inp = _render_input(left_visible, right_visible)
    expected = np.full((left_visible.shape[0], PANEL_WIDTH), BACKGROUND, dtype=int)
    expected[full_xor] = OUTPUT_COLOR
    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    left_orbits = _sample_orbits(height)
    right_orbits = _sample_orbits(height)
    if left_orbits is None or right_orbits is None:
        return None

    left_visible = _visible_mask_from_orbits(height, left_orbits, right_side=False)
    right_visible = _visible_mask_from_orbits(height, right_orbits, right_side=True)
    if not _is_valid_instance(left_visible, right_visible):
        return None
    return _render_input(left_visible, right_visible)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-xor ARC input")
