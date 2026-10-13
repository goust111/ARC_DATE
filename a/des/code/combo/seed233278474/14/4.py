from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = tuple(color for color in range(10) if color not in (BLACK, GREY))
GEN_RETRIES = 400

PROFILE_SIDES: dict[str, tuple[str, ...]] = {
    "top": ("top",),
    "bottom": ("bottom",),
    "left": ("left",),
    "right": ("right",),
    "top_left": ("top", "left"),
    "top_right": ("top", "right"),
    "bottom_left": ("bottom", "left"),
    "bottom_right": ("bottom", "right"),
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _intersection(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _render_rectangles(
    shape: tuple[int, int],
    rectangles: dict[int, tuple[int, int, int, int]],
    bottom_to_top: list[int],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for color in bottom_to_top:
        r0, c0, r1, c1 = rectangles[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _find_separator_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [c for c in range(1, w - 1) if np.all(grid[:, c] == GREY)]
    assert candidates, "expected at least one full grey separator column"
    assert len(candidates) == 1, "separator column must be unique"
    return int(candidates[0])


def _read_legend(left: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for r in range(left.shape[0]):
        row_colors: list[int] = []
        for value in left[r]:
            color = int(value)
            if color in (BLACK, GREY):
                continue
            if color not in row_colors:
                row_colors.append(color)
        assert len(row_colors) <= 1, "each legend row should contain at most one color"
        if not row_colors:
            continue
        color = row_colors[0]
        assert color not in seen, "legend colors must be distinct"
        seen.add(color)
        order.append(color)
    assert len(order) >= 2, "legend must list at least two colors"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack overlapping color rectangles according to the legend on the left.

    The legend lists colors from top layer to bottom layer. On the right, each color's
    currently visible pixels still reveal the original rectangle bbox, even if overlap is
    displayed in the wrong order. Reconstruct each full rectangle from its visible bbox,
    redraw from bottom to top, and return only the right-hand puzzle area.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_column(g)
    legend = _read_legend(g[:, :sep])
    right = g[:, sep + 1 :]

    rectangles: dict[int, tuple[int, int, int, int]] = {}
    for color in legend:
        mask = right == int(color)
        assert mask.any(), "every legend color must remain visible in the puzzle area"
        rectangles[color] = _bbox(mask)

    return _render_rectangles(right.shape, rectangles, list(reversed(legend)))


def _sample_extra(max_extra: int, emphasize: bool) -> int:
    if max_extra <= 0:
        return 0
    if emphasize:
        low = max(1, (max_extra + 1) // 2)
        return random.randint(low, max_extra)
    return random.randint(0, max_extra // 2)


def _sample_rectangles(height: int, width: int, colors: list[int]) -> dict[int, tuple[int, int, int, int]] | None:
    profiles = random.sample(list(PROFILE_SIDES), len(colors))
    core_h = random.randint(1, min(3, height - 6))
    core_w = random.randint(1, min(3, width - 6))
    core_r0 = random.randint(1, height - core_h - 2)
    core_c0 = random.randint(1, width - core_w - 2)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    rectangles: dict[int, tuple[int, int, int, int]] = {}
    used_boxes: set[tuple[int, int, int, int]] = set()

    for color, profile in zip(colors, profiles, strict=False):
        favored = set(PROFILE_SIDES[profile])
        if random.random() < 0.3:
            extra_side = random.choice([side for side in ("top", "bottom", "left", "right") if side not in favored])
            favored.add(extra_side)

        top = _sample_extra(core_r0, "top" in favored)
        bottom = _sample_extra(height - 1 - core_r1, "bottom" in favored)
        left = _sample_extra(core_c0, "left" in favored)
        right = _sample_extra(width - 1 - core_c1, "right" in favored)

        box = (
            core_r0 - top,
            core_c0 - left,
            core_r1 + bottom,
            core_c1 + right,
        )
        if box[2] - box[0] + 1 < 3 or box[3] - box[1] + 1 < 3:
            return None
        if box in used_boxes:
            return None
        used_boxes.add(box)
        rectangles[color] = box

    boxes = list(rectangles.values())
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1 :]:
            overlap = _intersection(box_a, box_b)
            assert overlap is not None
            overlap_area = _box_area(overlap)
            smaller = min(_box_area(box_a), _box_area(box_b))
            if _contains(box_a, box_b) or _contains(box_b, box_a):
                return None
            if overlap_area * 5 >= smaller * 4:
                return None

    union = np.zeros((height, width), dtype=bool)
    for box in boxes:
        r0, c0, r1, c1 = box
        union[r0 : r1 + 1, c0 : c1 + 1] = True
    coverage = float(union.mean())
    if coverage < 0.2 or coverage > 0.8:
        return None

    return rectangles


def _visible_bboxes_match(
    rendered: np.ndarray,
    rectangles: dict[int, tuple[int, int, int, int]],
    legend: list[int],
) -> bool:
    for color in legend:
        mask = rendered == int(color)
        if not mask.any():
            return False
        if int(mask.sum()) < 4:
            return False
        if _bbox(mask) != rectangles[color]:
            return False
    return True


def _candidate_orders(correct_bottom_to_top: list[int]) -> list[list[int]]:
    orders = list(itertools.permutations(correct_bottom_to_top))
    random.shuffle(orders)
    return [list(order) for order in orders if list(order) != correct_bottom_to_top]


def _build_input_grid(
    left_width: int,
    legend: list[int],
    right: np.ndarray,
) -> np.ndarray:
    height = right.shape[0]
    grid = np.full((height, left_width + 1 + right.shape[1]), BLACK, dtype=int)
    grid[:, left_width] = GREY

    start_row = random.randint(0, height - len(legend))
    for offset, color in enumerate(legend):
        row = start_row + offset
        col = random.randint(0, left_width - 1)
        grid[row, col] = int(color)

    grid[:, left_width + 1 :] = right
    return grid


def _generate_attempt() -> np.ndarray | None:
    right_h = random.randint(9, 16)
    right_w = random.randint(9, 17)
    left_w = random.randint(1, 4)
    color_count = random.choice((2, 2, 3, 3, 4))
    legend = random.sample(list(OBJECT_COLORS), color_count)

    rectangles = _sample_rectangles(right_h, right_w, legend)
    if rectangles is None:
        return None

    correct_order = list(reversed(legend))
    expected = _render_rectangles((right_h, right_w), rectangles, correct_order)

    for wrong_order in _candidate_orders(correct_order):
        current = _render_rectangles((right_h, right_w), rectangles, wrong_order)
        if np.array_equal(current, expected):
            continue
        if not _visible_bboxes_match(current, rectangles, legend):
            continue

        grid = _build_input_grid(left_w, legend, current)
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
