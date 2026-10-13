from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg


BACKGROUND = 0
DIVIDER_WIDTH = 1
HEIGHT_RANGE = (5, 12)
PANEL_WIDTH_RANGE = (4, 9)
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))

_LEFT_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
)

_RIGHT_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
)


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        col = grid[:, c]
        if int(col[0]) != bg and np.all(col == col[0]):
            candidates.append(c)

    valid = []
    for c1 in candidates:
        left_w = c1
        if left_w <= 0:
            continue
        c2 = 2 * left_w + DIVIDER_WIDTH
        if c2 >= w:
            continue
        if w != 3 * left_w + 2 * DIVIDER_WIDTH:
            continue
        col2 = grid[:, c2]
        if int(col2[0]) == bg or not np.all(col2 == col2[0]):
            continue
        valid.append((c1, c2))

    assert len(valid) == 1, "expected exactly one equal-width divider pair"
    return valid[0]


def _split_panels(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    c1, c2 = _find_dividers(grid, bg)
    left = grid[:, :c1].copy()
    middle = grid[:, c1 + DIVIDER_WIDTH : c2].copy()
    right = grid[:, c2 + DIVIDER_WIDTH :].copy()
    assert left.shape == middle.shape == right.shape
    return left, middle, right, (c1, c2)


def _compose_middle(left: np.ndarray, right: np.ndarray, bg: int) -> np.ndarray:
    middle = np.fliplr(left).copy()
    overlay = right != bg
    middle[overlay] = right[overlay]
    return middle


def _assemble_full(
    left: np.ndarray,
    middle: np.ndarray,
    right: np.ndarray,
    sep1: np.ndarray,
    sep2: np.ndarray,
) -> np.ndarray:
    return np.concatenate((left, sep1, middle, sep2, right), axis=1)


def _recover_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))
    left, middle, right, divs = _split_panels(g, bg)
    missing = [idx for idx, panel in enumerate((left, middle, right)) if np.all(panel == bg)]
    assert len(missing) == 1, "expected exactly one fully blank panel"
    missing_idx = missing[0]

    if missing_idx == 1:
        middle = _compose_middle(left, right, bg)
    elif missing_idx == 2:
        mirrored_left = np.fliplr(left)
        right = np.where(middle != mirrored_left, middle, bg)
    else:
        mirrored_left = np.where(right == bg, middle, bg)
        left = np.fliplr(mirrored_left)

    expected_middle = _compose_middle(left, right, bg)
    assert np.array_equal(middle, expected_middle), "panels must obey the composition law"
    assert np.all((right != bg) <= (np.fliplr(left) == bg)), "right overlay must avoid mirrored-left foreground"
    assert np.any(right != bg), "right panel must contain at least one overlay pixel"
    assert np.any((np.fliplr(left) != bg) & (right == bg)), "middle must retain some mirrored-left pixels"
    return left, middle, right, divs


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the missing panel in a three-column assembly where the center is the mirrored-left image with right-panel patches overlaid.

    Input structure:
    - The grid is split into equal-width left, middle, and right panels by two full-height divider columns.
    - Exactly one entire panel has been erased to the background color; the other two panels remain intact.
    - In a complete example, the middle panel equals the left panel mirrored left-to-right, then covered by the right panel's non-background pixels.

    Rule:
    - Detect the divider columns and split the grid into the three panels.
    - If the middle panel is blank, rebuild it by mirroring the left panel and overlaying the right panel.
    - If the right panel is blank, recover it as the pixels where the middle differs from the mirrored left; if the left panel is blank, remove right-panel patches from the middle and mirror back.
    - Return the fully reconstructed three-panel grid, preserving the divider columns.

    Key insight:
    - The three panels are causally linked: left is the source, right is a sparse patch layer, and the no-overlap guarantee makes the missing panel uniquely recoverable in either forward or reverse direction.
    """
    g = np.asarray(input_grid, dtype=int)
    bg = int(get_bg(g))
    left, middle, right, (c1, c2) = _recover_panels(g)
    sep1 = g[:, c1 : c1 + DIVIDER_WIDTH].copy()
    sep2 = g[:, c2 : c2 + DIVIDER_WIDTH].copy()
    return _assemble_full(left, middle, right, sep1, sep2)


def _cells_with_gap_ok(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h - 1, r + 1)
        c0 = max(0, c - 1)
        c1 = min(w - 1, c + 1)
        if np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]):
            return False
    return True


def _place_shape(
    panel: np.ndarray,
    occupied: np.ndarray,
    color: int,
    shapes: tuple[tuple[tuple[int, int], ...], ...],
    allowed_mask: np.ndarray | None,
    attempts: int,
) -> bool:
    h, w = panel.shape
    for _ in range(attempts):
        offsets = random.choice(shapes)
        max_r = max(r for r, _ in offsets)
        max_c = max(c for _, c in offsets)
        if max_r >= h or max_c >= w:
            continue
        r0 = random.randint(0, h - max_r - 1)
        c0 = random.randint(0, w - max_c - 1)
        cells = [(r0 + dr, c0 + dc) for dr, dc in offsets]
        if not _cells_with_gap_ok(cells, occupied):
            continue
        if allowed_mask is not None and not all(bool(allowed_mask[r, c]) for r, c in cells):
            continue
        for r, c in cells:
            panel[r, c] = color
            occupied[r, c] = True
        return True
    return False


def _has_separator_like_column(panel: np.ndarray, bg: int) -> bool:
    return any(int(panel[0, c]) != bg and np.all(panel[:, c] == panel[0, c]) for c in range(panel.shape[1]))


def _make_left_panel(h: int, w: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(200):
        panel = np.full((h, w), BACKGROUND, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        shape_count = random.randint(2, min(5, h * w // 3))
        used_colors = random.sample(colors, k=min(len(colors), random.randint(2, min(4, len(colors)))))

        placed = 0
        for _ in range(shape_count):
            color = random.choice(used_colors)
            if _place_shape(panel, occupied, color, _LEFT_SHAPES, None, attempts=50):
                placed += 1

        filled = int(np.count_nonzero(panel))
        if placed < 2:
            continue
        if filled < max(4, (h * w) // 8):
            continue
        if filled > (h * w * 3) // 5:
            continue
        if len({int(v) for v in panel[panel != BACKGROUND]}) < 2:
            continue
        if _has_separator_like_column(panel, BACKGROUND):
            continue
        return panel
    return None


def _make_right_panel(left: np.ndarray, colors: list[int]) -> np.ndarray | None:
    mirror = np.fliplr(left)
    allowed = mirror == BACKGROUND
    if int(np.count_nonzero(allowed)) < 2:
        return None

    h, w = left.shape
    for _ in range(200):
        panel = np.full((h, w), BACKGROUND, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        shape_count = random.randint(1, min(4, int(np.count_nonzero(allowed))))
        used = 0
        for _ in range(shape_count):
            color = random.choice(colors)
            if _place_shape(panel, occupied, color, _RIGHT_SHAPES, allowed, attempts=60):
                used += 1
        if used == 0:
            continue
        if _has_separator_like_column(panel, BACKGROUND):
            continue
        if not np.any(panel != BACKGROUND):
            continue
        if np.count_nonzero(panel) > max(1, int(np.count_nonzero(allowed) * 0.45)):
            continue
        if not np.any((mirror != BACKGROUND) & (panel == BACKGROUND)):
            continue
        return panel
    return None


def _semantic_checks(
    inp: np.ndarray,
    full: np.ndarray,
    missing_idx: int,
    divider_cols: tuple[int, int],
) -> None:
    bg = int(get_bg(inp))
    left, middle, right, found_cols = _split_panels(inp, bg)
    assert found_cols == divider_cols
    missing = [idx for idx, panel in enumerate((left, middle, right)) if np.all(panel == bg)]
    assert missing == [missing_idx]

    full_left, full_middle, full_right, _ = _split_panels(full, bg)
    assert np.array_equal(full_middle, _compose_middle(full_left, full_right, bg))
    assert np.all((full_right != bg) <= (np.fliplr(full_left) == bg))
    assert np.any(full_right != bg)
    assert np.any((np.fliplr(full_left) != bg) & (full_right == bg))

    if missing_idx == 0:
        recovered_left = np.fliplr(np.where(full_right == bg, full_middle, bg))
        assert np.array_equal(recovered_left, full_left)
    elif missing_idx == 1:
        rebuilt = _compose_middle(full_left, full_right, bg)
        assert np.array_equal(rebuilt, full_middle)
    else:
        recovered_right = np.where(full_middle != np.fliplr(full_left), full_middle, bg)
        assert np.array_equal(recovered_right, full_right)

    assert np.array_equal(main(inp), full)


def _generate_case(return_meta: bool = False):
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(*HEIGHT_RANGE)
        panel_w = random.randint(*PANEL_WIDTH_RANGE)
        total_w = 3 * panel_w + 2 * DIVIDER_WIDTH
        if total_w > 30 or h > 30:
            continue

        divider_color = random.choice(PALETTE)
        object_colors = [color for color in PALETTE if color != divider_color]

        left = _make_left_panel(h, panel_w, object_colors)
        if left is None:
            continue

        right = _make_right_panel(left, object_colors)
        if right is None:
            continue

        middle = _compose_middle(left, right, BACKGROUND)
        sep = np.full((h, DIVIDER_WIDTH), divider_color, dtype=int)
        full = _assemble_full(left, middle, right, sep, sep)
        if _find_dividers(full, BACKGROUND) != (panel_w, 2 * panel_w + DIVIDER_WIDTH):
            continue

        missing_idx = random.randint(0, 2)
        inp = full.copy()
        if missing_idx == 0:
            inp[:, :panel_w] = BACKGROUND
        elif missing_idx == 1:
            inp[:, panel_w + DIVIDER_WIDTH : 2 * panel_w + DIVIDER_WIDTH] = BACKGROUND
        else:
            inp[:, 2 * panel_w + 2 * DIVIDER_WIDTH :] = BACKGROUND

        try:
            _semantic_checks(inp, full, missing_idx, (panel_w, 2 * panel_w + DIVIDER_WIDTH))
        except AssertionError:
            continue

        if return_meta:
            return inp, {"output": full, "missing_idx": missing_idx, "panel_width": panel_w, "height": h}
        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
