from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


BG = Color.BLACK
RECT_COLOR = Color.TEAL
STRIPE_PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]
GEN_RETRY_LIMIT = 300


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_stripes(grid: np.ndarray) -> tuple[str, list[int]]:
    horizontal = []
    for r in range(grid.shape[0]):
        row = grid[r, :]
        if np.all(row == row[0]) and int(row[0]) in STRIPE_PALETTE:
            horizontal.append((r, int(row[0])))

    vertical = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) in STRIPE_PALETTE:
            vertical.append((c, int(col[0])))

    if horizontal and not vertical:
        horizontal.sort()
        colors = [color for _, color in horizontal]
        assert len(colors) == len(set(colors)) and 2 <= len(colors) <= 4
        return "horizontal", colors
    if vertical and not horizontal:
        vertical.sort()
        colors = [color for _, color in vertical]
        assert len(colors) == len(set(colors)) and 2 <= len(colors) <= 4
        return "vertical", colors
    raise AssertionError("expected exactly one stripe orientation")


def _coords_fit_bbox(coords: np.ndarray, box: tuple[int, int, int, int], orientation: str, k: int) -> bool:
    r0, c0, r1, c1 = box
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 2 or width < 2:
        return False
    if orientation == "horizontal" and height != k:
        return False
    if orientation == "vertical" and width != k:
        return False
    on_border = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    return bool(np.all(on_border))


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _find_rectangles(grid: np.ndarray, orientation: str, k: int) -> list[tuple[int, int, int, int]]:
    comps = components_mask(grid == RECT_COLOR, connectivity=8)
    assert comps, "expected at least one teal rectangle fragment"

    direct_boxes = [bbox_coords(comp) for comp in comps]
    if (
        1 <= len(direct_boxes) <= 2
        and all(_coords_fit_bbox(comp, box, orientation, k) for comp, box in zip(comps, direct_boxes))
        and (len(direct_boxes) == 1 or not _boxes_overlap(direct_boxes[0], direct_boxes[1]))
    ):
        return sorted(direct_boxes)

    n = len(comps)
    solutions: list[list[tuple[int, int, int, int]]] = []
    full_mask = (1 << n) - 1
    for split in range(1, full_mask + 1):
        if (split & 1) == 0:
            continue
        group_masks = [split]
        if split != full_mask:
            group_masks.append(full_mask ^ split)
        if len(group_masks) > 2:
            continue

        boxes = []
        valid = True
        for group_mask in group_masks:
            coords = np.concatenate(
                [comps[idx] for idx in range(n) if (group_mask >> idx) & 1],
                axis=0,
            )
            box = bbox_coords(coords)
            if not _coords_fit_bbox(coords, box, orientation, k):
                valid = False
                break
            boxes.append(box)

        if not valid:
            continue
        if len(boxes) == 2 and _boxes_overlap(boxes[0], boxes[1]):
            continue
        solutions.append(sorted(boxes))

    assert solutions, "failed to infer rectangle bounding boxes"
    solutions.sort(key=lambda boxes: (-len(boxes), tuple(boxes)))
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the missing border pixels of incomplete teal rectangles using the ordered stripe colors along the stripe axis.

    Input structure:
    - A black background contains one or two incomplete teal rectangular perimeters with some border pixels replaced by black.
    - Outside the rectangles there are two to four full monochrome stripes, all horizontal or all vertical.
    - Stripe colors are distinct and the remaining teal pixels still determine each rectangle's bounding box.

    Rule:
    - Infer each rectangle's complete bounding box from its remaining teal border pixels.
    - Find the full-row or full-column stripes, sort them by position, and read their colors in that order.
    - For every black gap on a rectangle border, fill it by row if the stripes are horizontal or by column if the stripes are vertical, using the corresponding ordered stripe color.
    - Preserve all original teal pixels, stripes, and background cells that are not rectangle-border gaps.

    Key insight:
    - The teal fragments say where each rectangle border should exist, while the ordered stripes say which color each border row or border column of the missing gaps should receive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orientation, stripe_colors = _detect_stripes(g)
    k = len(stripe_colors)
    rectangles = _find_rectangles(g, orientation, k)

    out = g.copy()
    for r0, c0, r1, c1 in rectangles:
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(region.shape[0], region.shape[1])
        gaps = border & (region == BG)

        if orientation == "horizontal":
            for rel_r, color in enumerate(stripe_colors):
                row_mask = gaps[rel_r, :]
                if np.any(row_mask):
                    out[r0 + rel_r, c0 : c1 + 1][row_mask] = int(color)
        else:
            for rel_c, color in enumerate(stripe_colors):
                col_mask = gaps[:, rel_c]
                if np.any(col_mask):
                    out[r0 : r1 + 1, c0 + rel_c][col_mask] = int(color)

    return out


def _draw_rectangle_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = RECT_COLOR
    grid[r1, c0 : c1 + 1] = RECT_COLOR
    grid[r0 : r1 + 1, c0] = RECT_COLOR
    grid[r0 : r1 + 1, c1] = RECT_COLOR


def _remove_horizontal_gaps(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    anchor_left = random.choice([True, False])
    gap_col = c1 if anchor_left else c0
    for r in range(r0 + 1, r1):
        grid[r, gap_col] = BG

    top_choices = list(range(c0 + 1, c1 + 1)) if anchor_left else list(range(c0, c1))
    bottom_choices = list(range(c0 + 1, c1 + 1)) if anchor_left else list(range(c0, c1))
    grid[r0, random.choice(top_choices)] = BG
    grid[r1, random.choice(bottom_choices)] = BG


def _remove_vertical_gaps(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    anchor_top = random.choice([True, False])
    gap_row = r1 if anchor_top else r0
    for c in range(c0 + 1, c1):
        grid[gap_row, c] = BG

    left_choices = list(range(r0 + 1, r1 + 1)) if anchor_top else list(range(r0, r1))
    right_choices = list(range(r0 + 1, r1 + 1)) if anchor_top else list(range(r0, r1))
    grid[random.choice(left_choices), c0] = BG
    grid[random.choice(right_choices), c1] = BG


def _attempt_generate() -> np.ndarray:
    orientation = random.choice(["horizontal", "vertical"])
    stripe_count = random.randint(2, 4)
    rect_count = 1 if random.random() < 0.45 else 2
    stripe_colors = random.sample(STRIPE_PALETTE, stripe_count)

    if orientation == "horizontal":
        widths = [random.randint(3, 8) for _ in range(rect_count)]
        total_width = sum(widths) + (rect_count - 1) * 2
        width_lo = max(12, total_width + 2)
        width_hi = max(width_lo, min(28, total_width + 8))
        height_lo = max(10, 2 * stripe_count + 4)
        height_hi = max(height_lo, min(24, 2 * stripe_count + 12))
        width = random.randint(width_lo, width_hi)
        height = random.randint(height_lo, height_hi)
        grid = np.full((height, width), BG, dtype=int)

        rect_r0 = random.randint(1, height - stripe_count - 2)
        rect_rows = set(range(rect_r0, rect_r0 + stripe_count))
        available_rows = [r for r in range(height) if r not in rect_rows]
        assert len(available_rows) >= stripe_count
        stripe_rows = sorted(random.sample(available_rows, stripe_count))
        for r, color in zip(stripe_rows, stripe_colors):
            grid[r, :] = int(color)

        free_width = width - total_width
        left_margin = random.randint(1, max(1, free_width - 1))
        c = left_margin
        for idx, rect_width in enumerate(widths):
            r0 = rect_r0
            c0 = c
            r1 = rect_r0 + stripe_count - 1
            c1 = c0 + rect_width - 1
            _draw_rectangle_border(grid, r0, c0, r1, c1)
            _remove_horizontal_gaps(grid, r0, c0, r1, c1)
            c = c1 + 3 if idx + 1 < rect_count else c1 + 1
    else:
        heights = [random.randint(3, 8) for _ in range(rect_count)]
        total_height = sum(heights) + (rect_count - 1) * 2
        height_lo = max(12, total_height + 2)
        height_hi = max(height_lo, min(28, total_height + 8))
        width_lo = max(10, 2 * stripe_count + 4)
        width_hi = max(width_lo, min(24, 2 * stripe_count + 12))
        height = random.randint(height_lo, height_hi)
        width = random.randint(width_lo, width_hi)
        grid = np.full((height, width), BG, dtype=int)

        rect_c0 = random.randint(1, width - stripe_count - 2)
        rect_cols = set(range(rect_c0, rect_c0 + stripe_count))
        available_cols = [c for c in range(width) if c not in rect_cols]
        assert len(available_cols) >= stripe_count
        stripe_cols = sorted(random.sample(available_cols, stripe_count))
        for c, color in zip(stripe_cols, stripe_colors):
            grid[:, c] = int(color)

        free_height = height - total_height
        top_margin = random.randint(1, max(1, free_height - 1))
        r = top_margin
        for idx, rect_height in enumerate(heights):
            r0 = r
            c0 = rect_c0
            r1 = r0 + rect_height - 1
            c1 = rect_c0 + stripe_count - 1
            _draw_rectangle_border(grid, r0, c0, r1, c1)
            _remove_vertical_gaps(grid, r0, c0, r1, c1)
            r = r1 + 3 if idx + 1 < rect_count else r1 + 1

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "generated puzzle must have visible filled gaps"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            return _attempt_generate()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
