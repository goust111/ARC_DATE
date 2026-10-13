from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, random_free_location


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
GREY = Color.GREY
PINK = Color.PINK
ORANGE = Color.ORANGE

COUNT_TO_COLOR = {1: GREEN, 2: YELLOW, 3: ORANGE, 4: PINK}
ATTEMPT_LIMIT = 500

# The prompt's "each color 1..4 and unequal" makes minority count 4 unreachable.
# The generator therefore allows one color to reach 5 so that the 4->pink branch exists.
COUNT_TEMPLATES = [
    ((1, 4), (2, 3), (4, 5)),
    ((1, 5), (2, 5), (3, 4)),
    ((1, 2), (2, 4), (3, 5), (4, 5)),
    ((1, 4), (2, 4), (1, 2), (3, 5)),
    ((2, 3), (1, 5), (2, 4), (4, 5)),
]
_template_index = 0


def _rect_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    grey_components = components_mask(grid == GREY, connectivity=4)
    for coords in grey_components:
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        border = _rect_border_mask(grid.shape, box)
        comp_mask = np.zeros(grid.shape, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        if int(comp_mask.sum()) != int(border.sum()):
            continue
        if not np.array_equal(comp_mask, border):
            continue
        frames.append(box)

    assert len(frames) in (3, 4), f"expected 3 or 4 rectangular frames, found {len(frames)}"
    cols = [box[1] for box in frames]
    assert len(set(cols)) == len(cols), "frame order by top-left column must be unique"
    frames.sort(key=lambda box: box[1])
    return frames


def _minority_count(grid: np.ndarray, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    red_count = int(np.count_nonzero(interior == RED))
    blue_count = int(np.count_nonzero(interior == BLUE))
    assert red_count > 0 and blue_count > 0, "each frame interior must contain both marker colors"
    assert red_count != blue_count, "marker colors must have a unique minority"
    minority = min(red_count, blue_count)
    assert minority in COUNT_TO_COLOR, f"unsupported minority count: {minority}"
    return minority


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each frame border using the minority red/blue count from the previous frame in left-to-right cyclic order.

    Input structure:
    - The background is black.
    - The foreground contains 3 or 4 non-overlapping one-pixel-thick grey rectangular frames.
    - Inside every frame, the strict interior contains scattered red and blue marker pixels; red/blue pixels outside the frames are distractors.

    Rule:
    - Detect the grey rectangular frames and sort them by the column of their top-left corner.
    - For each frame, count only the red and blue pixels strictly inside that frame, ignoring the border and all outside distractors.
    - Take the minority of those two counts and map it as 1->green, 2->yellow, 3->orange, 4->pink.
    - Paint the entire border of the next frame in the cyclic order with that mapped color, while leaving all interiors, distractors, and background unchanged.

    Key insight:
    - A frame's output color is determined by another frame's interior statistic, so the puzzle requires both frame-local counting and a global cyclic ordering of the frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert get_bg(g) == BG, "expected black background"

    frames = _find_frames(g)
    minorities = [_minority_count(g, box) for box in frames]

    out = g.copy()
    for index, minority in enumerate(minorities):
        target_box = frames[(index + 1) % len(frames)]
        out[_rect_border_mask(g.shape, target_box)] = COUNT_TO_COLOR[minority]
    return out


def _next_template() -> tuple[tuple[int, int], ...]:
    global _template_index
    template = COUNT_TEMPLATES[_template_index % len(COUNT_TEMPLATES)]
    _template_index += 1
    return template


def _sample_frame_size(total_markers: int) -> tuple[int, int]:
    roomy = []
    fallback = []
    for height in range(5, 9):
        for width in range(5, 9):
            interior_area = (height - 2) * (width - 2)
            if interior_area < total_markers:
                continue
            fallback.append((height, width))
            if interior_area >= total_markers + 2:
                roomy.append((height, width))
    options = roomy if roomy else fallback
    return random.choice(options)


def _place_boxes(
    height: int, width: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    occupancy = np.zeros((height, width), dtype=int)
    indexed = sorted(enumerate(sizes), key=lambda item: item[1][0] * item[1][1], reverse=True)
    placed: dict[int, tuple[int, int, int, int]] = {}
    try:
        for idx, (box_h, box_w) in indexed:
            patch = np.ones((box_h, box_w), dtype=int)
            r0, c0 = random_free_location(occupancy, patch, transparent=0, border=1, padding=1)
            occupancy[r0 : r0 + box_h, c0 : c0 + box_w] = 1
            placed[idx] = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
    except ValueError:
        return None

    boxes = [placed[idx] for idx in range(len(sizes))]
    cols = [box[1] for box in boxes]
    if len(set(cols)) != len(cols):
        return None

    ordered = sorted(boxes, key=lambda box: box[1])
    rows = [box[0] for box in ordered]
    is_nondec = all(a <= b for a, b in zip(rows, rows[1:]))
    is_noninc = all(a >= b for a, b in zip(rows, rows[1:]))
    if is_nondec or is_noninc:
        return None
    return ordered


def _fill_frame_interior(
    grid: np.ndarray, box: tuple[int, int, int, int], red_count: int, blue_count: int
) -> None:
    r0, c0, r1, c1 = box
    interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    chosen = random.sample(interior_cells, red_count + blue_count)
    red_cells = chosen[:red_count]
    blue_cells = chosen[red_count:]
    for r, c in red_cells:
        grid[r, c] = RED
    for r, c in blue_cells:
        grid[r, c] = BLUE


def _noise_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    allowed = np.ones(shape, dtype=bool)
    for r0, c0, r1, c1 in boxes:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(shape[0] - 1, r1 + 1)
        cc1 = min(shape[1] - 1, c1 + 1)
        allowed[rr0 : rr1 + 1, cc0 : cc1 + 1] = False
    return allowed


def _scatter_noise(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    mask = _noise_mask(grid.shape, boxes) & (grid == BG)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(coords) < 4:
        outside = np.ones(grid.shape, dtype=bool)
        for r0, c0, r1, c1 in boxes:
            outside[r0 : r1 + 1, c0 : c1 + 1] = False
        coords = [tuple(map(int, rc)) for rc in np.argwhere(outside & (grid == BG))]
    assert len(coords) >= 4, "not enough room for distractor markers"
    random.shuffle(coords)
    red_noise = random.randint(1, min(4, len(coords) - 1))
    blue_noise = random.randint(1, min(4, len(coords) - red_noise))
    for r, c in coords[:red_noise]:
        grid[r, c] = RED
    for r, c in coords[red_noise : red_noise + blue_noise]:
        grid[r, c] = BLUE


def _anti_shortcut_ok(pairs: list[tuple[int, int]]) -> bool:
    minorities = [min(pair) for pair in pairs]
    if len(set(minorities)) < 2:
        return False
    totals = [sum(pair) for pair in pairs]
    majorities = [max(pair) for pair in pairs]
    diffs = [abs(pair[0] - pair[1]) for pair in pairs]
    same_total = any(
        totals[i] == totals[j] and minorities[i] != minorities[j]
        for i in range(len(pairs))
        for j in range(i + 1, len(pairs))
    )
    same_majority = any(
        majorities[i] == majorities[j] and minorities[i] != minorities[j]
        for i in range(len(pairs))
        for j in range(i + 1, len(pairs))
    )
    same_diff = any(
        diffs[i] == diffs[j] and minorities[i] != minorities[j]
        for i in range(len(pairs))
        for j in range(i + 1, len(pairs))
    )
    return same_total or same_majority or same_diff


def _expected_output(
    grid: np.ndarray, boxes: list[tuple[int, int, int, int]], pairs: list[tuple[int, int]]
) -> np.ndarray:
    out = grid.copy()
    for index, pair in enumerate(pairs):
        target_box = boxes[(index + 1) % len(boxes)]
        out[_rect_border_mask(grid.shape, target_box)] = COUNT_TO_COLOR[min(pair)]
    return out


def _build_instance() -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    template = _next_template()
    pairs = [random.choice((pair, pair[::-1])) for pair in template]
    assert _anti_shortcut_ok(pairs)

    n_frames = len(pairs)
    min_width = 16 if n_frames == 3 else 20
    height = random.randint(16, 24)
    width = random.randint(min_width, 28)

    sizes = [_sample_frame_size(sum(pair)) for pair in pairs]
    boxes = _place_boxes(height, width, sizes)
    if boxes is None:
        raise RuntimeError("placement failed")

    grid = np.zeros((height, width), dtype=int)
    for box in boxes:
        grid[_rect_border_mask(grid.shape, box)] = GREY
    for box, (red_count, blue_count) in zip(boxes, pairs):
        _fill_frame_interior(grid, box, red_count, blue_count)
    _scatter_noise(grid, boxes)
    return grid, boxes, pairs


def _validate_instance(
    grid: np.ndarray, boxes: list[tuple[int, int, int, int]], pairs: list[tuple[int, int]]
) -> bool:
    if get_bg(grid) != BG:
        return False
    if not _anti_shortcut_ok(pairs):
        return False
    if not np.any(grid == RED) or not np.any(grid == BLUE):
        return False

    try:
        found = _find_frames(grid)
    except AssertionError:
        return False
    if found != boxes:
        return False

    for box, pair in zip(boxes, pairs):
        r0, c0, r1, c1 = box
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if int(np.count_nonzero(interior == RED)) != pair[0]:
            return False
        if int(np.count_nonzero(interior == BLUE)) != pair[1]:
            return False

    expected = _expected_output(grid, boxes, pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    for index, box in enumerate(boxes):
        border = _rect_border_mask(grid.shape, box)
        expected_color = COUNT_TO_COLOR[min(pairs[index - 1])]
        if set(map(int, np.unique(actual[border]))) != {expected_color}:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(ATTEMPT_LIMIT):
        try:
            grid, boxes, pairs = _build_instance()
        except RuntimeError:
            continue
        if _validate_instance(grid, boxes, pairs):
            return grid
    raise RuntimeError("failed to generate a valid loop-frame-coloring input")
