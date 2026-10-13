from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, neighbors4, paste, random_free_location


PANEL_HEIGHT = 8
PANEL_WIDTH = 8
INPUT_SHAPE = (PANEL_HEIGHT, PANEL_WIDTH * 2)

BACKGROUND = Color.BLACK
LEFT_COLOR = Color.BLUE
RIGHT_COLOR = Color.RED
EMPTY_COLOR = Color.GREY

TOTAL_MIN = 12
TOTAL_MAX = 20
COUNT_DIFF_MAX = 4
OVERLAP_MIN = 2
OVERLAP_MAX = 5

OBJECT_SIZE_MIN = 3
OBJECT_SIZE_MAX = 8
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4

PANEL_RENDER_ATTEMPTS = 120
GENERATION_ATTEMPTS = 500

_BRANCH_OFFSET = random.randint(0, 1)
_GENERATION_INDEX = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two aligned side-by-side panels and resolve collision pixels by the panel with more total foreground cells.

    Input structure:
    - The input is split into two equal side-by-side panels on a black background.
    - The left panel contains blue 4-connected foreground objects and the right panel contains red 4-connected foreground objects.
    - The two panels have the same shape, and some aligned positions are occupied in both panels.

    Rule:
    - Align the left and right panels by row and column and return one output panel of that shared size.
    - Paint cells blue where only the left panel is occupied and red where only the right panel is occupied.
    - Paint cells grey where both aligned panel cells are background.
    - Count total foreground pixels in each panel; when both panels occupy the same aligned cell, paint that collision cell with the color of the panel whose total foreground count is larger.

    Key insight:
    - Collision cells cannot be resolved from their local neighborhood alone; they depend on a global comparison between the total occupied-pixel counts of the two panels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 0, "expected two equal-width panels"

    bg = int(get_bg(g))
    mid = width // 2
    left = g[:, :mid]
    right = g[:, mid:]

    left_mask = left != bg
    right_mask = right != bg
    left_total = int(left_mask.sum())
    right_total = int(right_mask.sum())
    assert left_total != right_total, "collision color must be uniquely determined"

    out = np.full((height, mid), EMPTY_COLOR, dtype=int)
    left_only = left_mask & ~right_mask
    right_only = right_mask & ~left_mask
    overlap = left_mask & right_mask

    out[left_only] = LEFT_COLOR
    out[right_only] = RIGHT_COLOR
    out[overlap] = LEFT_COLOR if left_total > right_total else RIGHT_COLOR
    return out


def _partition_total(total: int) -> list[int] | None:
    feasible_counts = [
        count
        for count in range(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX + 1)
        if count * OBJECT_SIZE_MIN <= total <= count * OBJECT_SIZE_MAX
    ]
    random.shuffle(feasible_counts)

    for count in feasible_counts:
        remaining = int(total)
        sizes: list[int] = []
        for slots_left in range(count, 0, -1):
            lo = max(OBJECT_SIZE_MIN, remaining - OBJECT_SIZE_MAX * (slots_left - 1))
            hi = min(OBJECT_SIZE_MAX, remaining - OBJECT_SIZE_MIN * (slots_left - 1))
            if lo > hi:
                sizes = []
                break
            sizes.append(random.randint(lo, hi))
            remaining -= sizes[-1]
        if remaining == 0 and sizes:
            random.shuffle(sizes)
            return sizes
    return None


def _grow_blob(size: int, color: int) -> np.ndarray | None:
    target = int(size)
    coords = {(0, 0)}
    bbox_limit = random.choice((3, 4)) if target <= 5 else random.choice((4, 5))

    while len(coords) < target:
        candidates: list[tuple[int, int]] = []
        weights: list[float] = []
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]

        for r, c in list(coords):
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in coords:
                    continue
                test_rows = rows + [int(nr)]
                test_cols = cols + [int(nc)]
                height = max(test_rows) - min(test_rows) + 1
                width = max(test_cols) - min(test_cols) + 1
                if height > bbox_limit or width > bbox_limit:
                    continue
                adjacency = sum((ar, ac) in coords for ar, ac in neighbors4(nr, nc))
                span = max(height, width)
                weight = 1.0 + 2.0 * adjacency + max(0, bbox_limit + 1 - span)
                if min(height, width) == 1 and target >= 5:
                    weight *= 0.5
                candidates.append((int(nr), int(nc)))
                weights.append(weight)

        if not candidates:
            return None
        coords.add(random.choices(candidates, weights=weights, k=1)[0])

    norm = np.asarray(sorted(coords), dtype=int)
    norm -= norm.min(axis=0)
    height, width = norm.max(axis=0) + 1
    patch = np.full((height, width), BACKGROUND, dtype=int)
    patch[norm[:, 0], norm[:, 1]] = int(color)
    return patch


def _render_panel(total: int, color: int) -> np.ndarray | None:
    for _ in range(PANEL_RENDER_ATTEMPTS):
        sizes = _partition_total(total)
        if sizes is None:
            return None

        panel = np.full((PANEL_HEIGHT, PANEL_WIDTH), BACKGROUND, dtype=int)
        ok = True
        for size in sizes:
            patch = _grow_blob(size, color)
            if patch is None:
                ok = False
                break
            try:
                r0, c0 = random_free_location(
                    panel,
                    patch,
                    transparent=BACKGROUND,
                    padding=1,
                    padding_connectivity=4,
                )
            except ValueError:
                ok = False
                break
            paste(panel, patch, r0, c0, transparent=BACKGROUND)

        if not ok:
            continue

        components = find_components(
            panel,
            bg=BACKGROUND,
            connectivity=4,
            monochromatic=True,
            colors=[color],
        )
        areas = sorted(int(component["area"]) for component in components)
        if areas == sorted(sizes):
            return panel
    return None


def _desired_branch() -> bool:
    return ((_GENERATION_INDEX + _BRANCH_OFFSET) % 2) == 0


def _sample_totals(left_wins: bool) -> tuple[int, int]:
    pairs = [
        (left_total, right_total)
        for left_total in range(TOTAL_MIN, TOTAL_MAX + 1)
        for right_total in range(TOTAL_MIN, TOTAL_MAX + 1)
        if 1 <= abs(left_total - right_total) <= COUNT_DIFF_MAX
        and ((left_total > right_total) == left_wins)
    ]
    return random.choice(pairs)


def _expected_output(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    left_total = int(left_mask.sum())
    right_total = int(right_mask.sum())
    out = np.full(left_mask.shape, EMPTY_COLOR, dtype=int)
    left_only = left_mask & ~right_mask
    right_only = right_mask & ~left_mask
    overlap = left_mask & right_mask
    out[left_only] = LEFT_COLOR
    out[right_only] = RIGHT_COLOR
    out[overlap] = LEFT_COLOR if left_total > right_total else RIGHT_COLOR
    return out


def _panel_components(panel: np.ndarray, color: int) -> list[dict]:
    return find_components(
        panel,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        colors=[color],
    )


def _validate_candidate(grid: np.ndarray, expected_left_wins: bool) -> bool:
    if not isinstance(grid, np.ndarray) or grid.shape != INPUT_SHAPE:
        return False
    if int(get_bg(grid)) != BACKGROUND:
        return False

    left = grid[:, :PANEL_WIDTH]
    right = grid[:, PANEL_WIDTH:]
    if not np.all(np.isin(left, (BACKGROUND, LEFT_COLOR))):
        return False
    if not np.all(np.isin(right, (BACKGROUND, RIGHT_COLOR))):
        return False

    left_components = _panel_components(left, LEFT_COLOR)
    right_components = _panel_components(right, RIGHT_COLOR)
    if not (OBJECT_COUNT_MIN <= len(left_components) <= OBJECT_COUNT_MAX):
        return False
    if not (OBJECT_COUNT_MIN <= len(right_components) <= OBJECT_COUNT_MAX):
        return False
    if any(not (OBJECT_SIZE_MIN <= int(component["area"]) <= OBJECT_SIZE_MAX) for component in left_components):
        return False
    if any(not (OBJECT_SIZE_MIN <= int(component["area"]) <= OBJECT_SIZE_MAX) for component in right_components):
        return False

    left_mask = left != BACKGROUND
    right_mask = right != BACKGROUND
    left_total = int(left_mask.sum())
    right_total = int(right_mask.sum())
    if not (TOTAL_MIN <= left_total <= TOTAL_MAX):
        return False
    if not (TOTAL_MIN <= right_total <= TOTAL_MAX):
        return False
    if left_total == right_total or (left_total > right_total) != expected_left_wins:
        return False

    overlap = left_mask & right_mask
    overlap_count = int(overlap.sum())
    if not (OVERLAP_MIN <= overlap_count <= OVERLAP_MAX):
        return False

    left_only_count = int((left_mask & ~right_mask).sum())
    right_only_count = int((right_mask & ~left_mask).sum())
    empty_count = int((~left_mask & ~right_mask).sum())
    if min(left_only_count, right_only_count, empty_count) <= 0:
        return False

    expected = _expected_output(left_mask, right_mask)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    flipped = expected.copy()
    flipped[overlap] = RIGHT_COLOR if expected_left_wins else LEFT_COLOR
    if np.array_equal(flipped, expected):
        return False
    return True


def _gen_attempt(left_wins: bool) -> np.ndarray | None:
    left_total, right_total = _sample_totals(left_wins)
    left_panel = _render_panel(left_total, LEFT_COLOR)
    right_panel = _render_panel(right_total, RIGHT_COLOR)
    if left_panel is None or right_panel is None:
        return None

    grid = np.concatenate([left_panel, right_panel], axis=1)
    if _validate_candidate(grid, left_wins):
        return grid
    return None


def generate_input() -> np.ndarray:
    global _GENERATION_INDEX

    left_wins = _desired_branch()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(left_wins)
        if grid is not None:
            _GENERATION_INDEX += 1
            return grid
    raise RuntimeError("failed to generate a valid collision-count overlay input")
