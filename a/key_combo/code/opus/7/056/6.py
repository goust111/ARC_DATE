from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


BACKGROUND = 0
SEPARATOR = 4
SIDES = (1, 3, 5, 7, 9)
TRANSFORMED_SIDE = {1: 9, 3: 1, 5: 3, 7: 5, 9: 7}
COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 600

PANEL_H_RANGE = (9, 13)
PANEL_W_RANGE = (9, 13)
OVERLAP_BRANCHES = ("left", "right", "tie")
BRANCH_SIDE_PAIRS = {
    "left": [(9, 1), (9, 3), (7, 1), (7, 3), (7, 5), (5, 1), (5, 3)],
    "right": [(1, 9), (3, 9), (1, 7), (3, 7), (5, 7), (1, 5), (3, 5)],
    "tie": [(3, 3), (5, 5), (7, 7), (9, 9), (1, 1)],
}
EXTRA_SIDE_WEIGHTS = (1, 3, 3, 5, 5, 7, 7, 9)


def _bbox_from_center(center_r: int, center_c: int, side: int) -> tuple[int, int, int, int]:
    half = side // 2
    return center_r - half, center_c - half, center_r + half, center_c + half


def _fits(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    height, width = shape
    return 0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if not _boxes_overlap(a, b):
        return 0
    return (min(a[2], b[2]) - max(a[0], b[0]) + 1) * (min(a[3], b[3]) - max(a[1], b[1]) + 1)


def _draw_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _render_transformed_panel(
    objects: list[dict],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    colors = np.zeros(shape, dtype=int)
    priorities = np.zeros(shape, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["out_box"]
        region = colors[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(region != BACKGROUND), "transformed squares inside one panel must not overlap"
        region[:, :] = int(obj["color"])
        priorities[r0 : r1 + 1, c0 : c1 + 1] = int(obj["side"])
    return colors, priorities


def _combine_rendered_panels(left: list[dict], right: list[dict], shape: tuple[int, int]) -> np.ndarray:
    left_colors, left_priorities = _render_transformed_panel(left, shape)
    right_colors, right_priorities = _render_transformed_panel(right, shape)

    out = np.zeros(shape, dtype=int)
    left_mask = left_colors != BACKGROUND
    right_mask = right_colors != BACKGROUND
    both = left_mask & right_mask

    out[left_mask & ~right_mask] = left_colors[left_mask & ~right_mask]
    out[right_mask & ~left_mask] = right_colors[right_mask & ~left_mask]

    left_wins = both & (left_priorities >= right_priorities)
    right_wins = both & (left_priorities < right_priorities)
    out[left_wins] = left_colors[left_wins]
    out[right_wins] = right_colors[right_wins]
    return out


def _extract_panel_objects(panel: np.ndarray) -> list[dict]:
    comps = find_components(panel, bg=BACKGROUND, connectivity=4, monochromatic=True, return_format="dict")
    assert 2 <= len(comps) <= 3, "expected 2-3 squares per panel"

    objects: list[dict] = []
    seen_colors: set[int] = set()
    height, width = panel.shape
    for comp in comps:
        color = int(comp["color"])
        assert color not in seen_colors, "panel colors must be distinct"
        seen_colors.add(color)

        r0, c0, r1, c1 = map(int, comp["bbox"])
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1, "foreground object must be square"
        assert side in SIDES, "foreground square must use an odd side from the cycle"
        assert int(comp["area"]) == side * side, "foreground square must be solid"
        assert np.all(panel[r0 : r1 + 1, c0 : c1 + 1] == color), "foreground square must fill its bbox"

        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        out_side = TRANSFORMED_SIDE[side]
        out_box = _bbox_from_center(center_r, center_c, out_side)
        assert _fits(out_box, (height, width)), "transformed square must stay inside its panel"

        objects.append(
            {
                "color": color,
                "side": side,
                "center": (center_r, center_c),
                "in_box": (r0, c0, r1, c1),
                "out_box": out_box,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle the odd squares in two side-by-side panels, then overlay the resized panels using original side length priority.

    Input structure:
    - The grid has black background and one unique full-height separator column.
    - The separator splits the grid into two equal-size panels of the same shape.
    - Each panel contains 2-3 separated solid odd-sided squares, each with a distinct non-background color.

    Rule:
    - Record every square's original side length.
    - Resize each square in place around its center using the cycle 1->9, 3->1, 5->3, 7->5, 9->7.
    - Remove the separator and align the two resized panels cell by cell.
    - If exactly one panel paints a cell, keep that color; if both paint it, choose the color from the square with the larger original side length, breaking ties in favor of the left panel.

    Key insight:
    - The resized footprint decides where color can appear, but the winner in overlaps is chosen by each square's pre-resize side length rather than by its visible post-resize size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected black background"

    separator_cols = []
    for c in range(width):
        col_color = int(g[0, c])
        if col_color == bg:
            continue
        if np.all(g[:, c] == col_color) and int(np.count_nonzero(g == col_color)) == height:
            separator_cols.append(c)
    assert len(separator_cols) == 1, "expected exactly one unique separator column"

    separator = separator_cols[0]
    left = g[:, :separator]
    right = g[:, separator + 1 :]
    assert left.shape == right.shape and left.size > 0, "expected two non-empty equal-size panels"
    assert separator == left.shape[1], "separator must split the grid between equal panels"

    left_objects = _extract_panel_objects(left)
    right_objects = _extract_panel_objects(right)
    return _combine_rendered_panels(left_objects, right_objects, left.shape)


def _valid_centers(side: int, out_side: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    margin = max(side // 2, out_side // 2)
    return [(r, c) for r in range(margin, height - margin) for c in range(margin, width - margin)]


def _make_object(color: int, side: int, center: tuple[int, int]) -> dict:
    center_r, center_c = center
    out_side = TRANSFORMED_SIDE[side]
    return {
        "color": int(color),
        "side": int(side),
        "center": (int(center_r), int(center_c)),
        "in_box": _bbox_from_center(int(center_r), int(center_c), int(side)),
        "out_box": _bbox_from_center(int(center_r), int(center_c), int(out_side)),
    }


def _can_add_object(existing: list[dict], candidate: dict) -> bool:
    return all(
        not _boxes_overlap(candidate["in_box"], obj["in_box"]) and not _boxes_overlap(candidate["out_box"], obj["out_box"])
        for obj in existing
    )


def _place_forced_overlap_pair(
    left_side: int,
    right_side: int,
    shape: tuple[int, int],
    colors: list[int],
) -> tuple[list[dict], list[dict]] | None:
    left_out = TRANSFORMED_SIDE[left_side]
    right_out = TRANSFORMED_SIDE[right_side]
    left_centers = _valid_centers(left_side, left_out, shape)
    right_centers = _valid_centers(right_side, right_out, shape)
    if not left_centers or not right_centers:
        return None

    random.shuffle(left_centers)
    random.shuffle(right_centers)
    overlap_cap = max(1, min(12, min(left_out * left_out, right_out * right_out) // 2))

    for left_center in left_centers[:120]:
        left_obj = _make_object(colors[0], left_side, left_center)
        for right_center in right_centers[:120]:
            right_obj = _make_object(colors[1], right_side, right_center)
            area = _overlap_area(left_obj["out_box"], right_obj["out_box"])
            if 1 <= area <= overlap_cap:
                return [left_obj], [right_obj]
    return None


def _sample_total_counts(shape: tuple[int, int]) -> tuple[int, int]:
    area = shape[0] * shape[1]
    if area >= 120 and random.random() < 0.55:
        return random.choice((2, 3)), random.choice((2, 3))
    return 2, 2


def _add_random_objects(
    existing: list[dict],
    target_count: int,
    shape: tuple[int, int],
    colors: list[int],
    color_index: int,
) -> tuple[list[dict], int] | tuple[None, int]:
    panel_objects = list(existing)
    next_color_index = color_index
    attempts = 0
    while len(panel_objects) < target_count and attempts < 220:
        attempts += 1
        if next_color_index >= len(colors):
            return None, next_color_index
        side = int(random.choice(EXTRA_SIDE_WEIGHTS))
        out_side = TRANSFORMED_SIDE[side]
        centers = _valid_centers(side, out_side, shape)
        random.shuffle(centers)
        candidate = None
        for center in centers:
            obj = _make_object(colors[next_color_index], side, center)
            if _can_add_object(panel_objects, obj):
                candidate = obj
                break
        if candidate is None:
            continue
        panel_objects.append(candidate)
        next_color_index += 1
    if len(panel_objects) != target_count:
        return None, next_color_index
    return panel_objects, next_color_index


def _overlap_profile(left: list[dict], right: list[dict], shape: tuple[int, int]) -> tuple[int, int]:
    left_colors, _ = _render_transformed_panel(left, shape)
    right_colors, _ = _render_transformed_panel(right, shape)
    overlap_pixels = int(np.count_nonzero((left_colors != BACKGROUND) & (right_colors != BACKGROUND)))
    overlapping_pairs = sum(1 for lo in left for ro in right if _boxes_overlap(lo["out_box"], ro["out_box"]))
    return overlap_pixels, overlapping_pairs


def _render_input_grid(left: list[dict], right: list[dict], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    grid = np.zeros((height, width * 2 + 1), dtype=int)
    grid[:, width] = SEPARATOR
    for obj in left:
        _draw_box(grid[:, :width], obj["in_box"], obj["color"])
    for obj in right:
        _draw_box(grid[:, width + 1 :], obj["in_box"], obj["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(PANEL_H_RANGE[0], PANEL_H_RANGE[1]),
        random.randint(PANEL_W_RANGE[0], PANEL_W_RANGE[1]),
    )
    branch = random.choice(OVERLAP_BRANCHES)
    left_side, right_side = random.choice(BRANCH_SIDE_PAIRS[branch])
    colors = random.sample(list(COLORS), 6)

    forced_pair = _place_forced_overlap_pair(left_side, right_side, shape, colors)
    if forced_pair is None:
        return None
    left_objects, right_objects = forced_pair

    left_target, right_target = _sample_total_counts(shape)
    left_objects, color_index = _add_random_objects(left_objects, left_target, shape, colors, 2)
    if left_objects is None:
        return None
    right_objects, color_index = _add_random_objects(right_objects, right_target, shape, colors, color_index)
    if right_objects is None:
        return None

    overlap_pixels, overlapping_pairs = _overlap_profile(left_objects, right_objects, shape)
    max_allowed_overlap = max(10, (shape[0] * shape[1]) // 5)
    if not (1 <= overlap_pixels <= max_allowed_overlap):
        return None
    if overlapping_pairs > 3:
        return None

    inp = _render_input_grid(left_objects, right_objects, shape)
    expected = _combine_rendered_panels(left_objects, right_objects, shape)
    if np.count_nonzero(expected) == 0 or np.count_nonzero(expected == BACKGROUND) == 0:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
