from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from solvers.arc_utils import colors_present, get_bg


SIDE_NAMES = ("top", "right", "bottom", "left")
PRESSURE_TO_MOVE = {
    "top": "down",
    "right": "left",
    "bottom": "up",
    "left": "right",
}

BG = 0
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 800
PLACEMENT_TRIES = 180


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _side_coords(bbox: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    return {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
    }


def _bbox_from_visible(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("each frame color must stay visible")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _render_frames(
    shape: tuple[int, int],
    order: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    *,
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in order:
        out[_frame_mask(shape, bboxes[color])] = int(color)
    return out


def _infer_layer_order(grid: np.ndarray, frame_masks: dict[int, np.ndarray]) -> list[int]:
    colors = list(frame_masks)
    above: dict[int, set[int]] = {color: set() for color in colors}
    indeg: dict[int, int] = {color: 0 for color in colors}

    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            inter = frame_masks[a] & frame_masks[b]
            if not np.any(inter):
                continue
            vals = set(map(int, grid[inter]))
            a_seen = a in vals
            b_seen = b in vals
            if a_seen and b_seen:
                raise AssertionError("pairwise overlap cannot show both layer orders")
            if a_seen:
                if a not in above[b]:
                    above[b].add(a)
                    indeg[a] += 1
            elif b_seen:
                if b not in above[a]:
                    above[a].add(b)
                    indeg[b] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        zeros = sorted(color for color in remaining if indeg[color] == 0)
        if len(zeros) != 1:
            raise AssertionError("occlusion constraints must determine a unique total order")
        color = zeros[0]
        order.append(color)
        remaining.remove(color)
        for upper in above[color]:
            indeg[upper] -= 1
    return order


def _occlusion_counts(
    order: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> dict[int, list[int]]:
    masks = {color: _frame_mask(shape, bbox) for color, bbox in bboxes.items()}
    counts: dict[int, list[int]] = {}
    for idx, color in enumerate(order):
        upper_union = np.zeros(shape, dtype=bool)
        for upper in order[idx + 1 :]:
            upper_union |= masks[upper]
        side_count = []
        for side in SIDE_NAMES:
            hits = 0
            for r, c in _side_coords(bboxes[color])[side]:
                hits += int(upper_union[r, c])
            side_count.append(hits)
        counts[color] = side_count
    return counts


def _move_from_counts(counts: list[int]) -> str:
    pressure_side = SIDE_NAMES[int(np.argmax(np.asarray(counts, dtype=int)))]
    return PRESSURE_TO_MOVE[pressure_side]


def _shift_bbox(
    bbox: tuple[int, int, int, int],
    move: str,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    H, W = shape
    if move == "up":
        return 0, c0, h - 1, c1
    if move == "down":
        return H - h, c0, H - 1, c1
    if move == "left":
        return r0, 0, r1, w - 1
    return r0, W - w, r1, W - 1


def _solve_geometry(grid: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int, int, int]], dict[int, str]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert 3 <= len(colors) <= 4

    bboxes = {color: _bbox_from_visible(g == color) for color in colors}
    frame_masks = {color: _frame_mask(g.shape, bbox) for color, bbox in bboxes.items()}
    order = _infer_layer_order(g, frame_masks)
    counts = _occlusion_counts(order, bboxes, g.shape)
    moves = {color: _move_from_counts(counts[color]) for color in colors}
    return order, bboxes, moves


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct overlapping hollow rectangles, infer which edge is most occluded, and push each frame to the opposite boundary.

    Input structure:
    - The background is the dominant color.
    - The foreground consists of 3 or 4 single-pixel-thick hollow rectangular frames, one color per frame.
    - Frames partially overlap, and at shared border cells only the top visible layer keeps its color.

    Rule:
    - For each foreground color, recover the full rectangle bbox from the visible pixels and rebuild the complete hollow frame.
    - Compare reconstructed frames pairwise on shared border cells to infer the unique bottom-to-top occlusion order.
    - For each frame, count how many of its top, right, bottom, and left edge pixels are covered by higher layers; the most-covered edge chooses the escape direction by taking the opposite side.
    - Clear the input and redraw the complete frames in bottom-to-top order after translating each one until its leading edge reaches the matching grid boundary.

    Key insight:
    - The visible pixels encode two hidden facts at once: extrema recover each missing rectangle, while overlap cells reveal both the layer order and which side is under the strongest pressure.
    """
    g = np.asarray(grid, dtype=int)
    order, bboxes, moves = _solve_geometry(g)
    bg = int(get_bg(g))
    shifted = {color: _shift_bbox(bboxes[color], moves[color], g.shape) for color in bboxes}
    return _render_frames(g.shape, order, shifted, bg=bg)


def _random_bbox(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    H, W = shape
    h = random.randint(4, min(8, H - 2))
    w = random.randint(4, min(8, W - 2))
    r0 = random.randint(1, H - h - 1)
    c0 = random.randint(1, W - w - 1)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _sample_overlapping_bbox(
    shape: tuple[int, int],
    anchor: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    H, W = shape
    ar0, ac0, ar1, ac1 = anchor
    for _ in range(PLACEMENT_TRIES):
        h = random.randint(4, min(8, H - 2))
        w = random.randint(4, min(8, W - 2))
        r_lo = max(1, ar0 - h + 2, ar0 - 4)
        r_hi = min(H - h - 1, ar1 - 1, ar1 + 2)
        c_lo = max(1, ac0 - w + 2, ac0 - 4)
        c_hi = min(W - w - 1, ac1 - 1, ac1 + 2)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if bbox == anchor:
            continue
        inter = _frame_mask(shape, bbox) & _frame_mask(shape, anchor)
        overlap = int(np.count_nonzero(inter))
        if 1 <= overlap <= 8:
            return bbox
    return None


def _visible_side_checks(
    grid: np.ndarray,
    color: int,
    bbox: tuple[int, int, int, int],
) -> tuple[bool, bool]:
    any_full = False
    for side in SIDE_NAMES:
        visible = [int(grid[r, c]) == int(color) for r, c in _side_coords(bbox)[side]]
        if not any(visible):
            return False, False
        any_full |= all(visible)
    return True, any_full


def _shifted_frames_disjoint(
    shape: tuple[int, int],
    bboxes: dict[int, tuple[int, int, int, int]],
    moves: dict[int, str],
) -> bool:
    masks: list[np.ndarray] = []
    for color, bbox in bboxes.items():
        shifted = _shift_bbox(bbox, moves[color], shape)
        mask = _frame_mask(shape, shifted)
        if any(np.any(mask & other) for other in masks):
            return False
        masks.append(mask)
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(14, 20), random.randint(14, 20))
    n = random.randint(3, 4)
    colors = random.sample(COLOR_POOL, n)
    order = random.sample(colors, n)

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    bboxes[order[0]] = _random_bbox(shape)
    for color in order[1:]:
        bbox = _sample_overlapping_bbox(shape, bboxes[order[len(bboxes) - 1]])
        if bbox is None:
            return None
        bboxes[color] = bbox

    grid = _render_frames(shape, order, bboxes, bg=BG)

    try:
        visible_bboxes = {color: _bbox_from_visible(grid == color) for color in colors}
    except AssertionError:
        return None
    if visible_bboxes != bboxes:
        return None

    side_status = [_visible_side_checks(grid, color, bboxes[color]) for color in colors]
    if not all(ok for ok, _ in side_status):
        return None
    if not all(full for _, full in side_status):
        return None

    try:
        inferred_order, inferred_bboxes, moves = _solve_geometry(grid)
    except AssertionError:
        return None
    if inferred_order != order or inferred_bboxes != bboxes:
        return None

    counts = _occlusion_counts(order, bboxes, shape)
    if any(sum(counts[color]) == 0 for color in order[:-1]):
        return None
    if len(set(moves.values())) < 2:
        return None
    if not _shifted_frames_disjoint(shape, bboxes, moves):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded-frame puzzle")
