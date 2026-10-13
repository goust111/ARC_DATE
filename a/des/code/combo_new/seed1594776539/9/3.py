from __future__ import annotations

import random

import numpy as np


BG = 8  # azure
ARC_MAX = 30
PALETTE = tuple(c for c in range(10) if c != BG)

# Generator configuration.
RECT_COUNT_RANGE = (3, 4)
BASE_SIDE_RANGE = (6, 10)
RECT_SIDE_RANGE = (4, 8)
OVERLAP_RANGE = (2, 4)
MARGIN_RANGE = (1, 3)
RIGHT_MARGIN_RANGE = (1, 3)
GEN_ATTEMPTS = 400
CHAIN_DIRECTIONS = ("ul", "ur", "dl", "dr")


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert rows.size > 0, "mask must contain at least one pixel"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _fill_rect(canvas: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    canvas[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _scene_rectangles(scene: np.ndarray) -> dict[int, dict[str, np.ndarray | tuple[int, int, int, int]]]:
    rectangles: dict[int, dict[str, np.ndarray | tuple[int, int, int, int]]] = {}
    for color in np.unique(scene):
        color = int(color)
        if color == BG:
            continue
        mask = scene == color
        rectangles[color] = {"mask": mask, "bbox": _bbox_from_mask(mask)}
    assert rectangles, "scene must contain at least one rectangle"
    return rectangles


def _resolve_top_order(scene: np.ndarray) -> tuple[list[int], dict[int, dict[str, np.ndarray | tuple[int, int, int, int]]]]:
    rectangles = _scene_rectangles(scene)
    covered = np.zeros(scene.shape, dtype=bool)
    unresolved = set(rectangles)
    order: list[int] = []

    while unresolved:
        candidates: list[int] = []
        for color in unresolved:
            mask = rectangles[color]["mask"]
            r0, c0, r1, c1 = rectangles[color]["bbox"]
            window = covered[r0 : r1 + 1, c0 : c1 + 1] | mask[r0 : r1 + 1, c0 : c1 + 1]
            if bool(np.all(window)):
                candidates.append(color)

        assert len(candidates) == 1, "occlusion order must be uniquely recoverable"
        color = candidates[0]
        order.append(color)
        r0, c0, r1, c1 = rectangles[color]["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
        unresolved.remove(color)

    return order, rectangles


def _find_legend_column(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    for c in range(w - 1, -1, -1):
        col = grid[:, c]
        rows = np.flatnonzero(col != BG)
        if rows.size < 2:
            continue
        if c < w - 1 and np.any(grid[:, c + 1 :] != BG):
            continue
        if c > 0 and not np.all(grid[rows, c - 1] == BG):
            continue
        legend = [int(col[r]) for r in rows.tolist()]
        if len(legend) != len(set(legend)):
            continue
        scene = np.delete(grid, c, axis=1)
        if len([v for v in np.unique(scene) if int(v) != BG]) != len(legend):
            continue
        return c, legend
    raise AssertionError("expected one right-side legend column")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full occluded rectangles and recolor them by stacking order from the right-side legend.

    Input structure:
    - The background is azure.
    - Several distinct solid rectangles of different colors overlap each other, so lower rectangles are only partially visible.
    - On the right side, one vertical legend column contains one single-pixel color entry per rectangle, ordered from top to bottom.

    Rule:
    - Remove the legend column from the scene and treat each remaining non-background color as one partially occluded rectangle.
    - Recover each rectangle's full bounding box, then infer the top-to-bottom occlusion order by repeatedly selecting the rectangle whose missing cells are already fully explained by rectangles known to be above it.
    - Repaint the recovered full rectangles using the legend colors: the topmost rectangle gets the first legend color, the next rectangle gets the second, and so on; draw them from bottom to top so upper rectangles still cover lower ones.

    Key insight:
    - A rectangle's visible pixels reveal its full outline, but its output color depends on where it sits in the occlusion stack rather than on its original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_col, legend_colors = _find_legend_column(g)
    scene = np.delete(g, legend_col, axis=1)
    order, rectangles = _resolve_top_order(scene)
    assert len(order) == len(legend_colors), "legend length must match the rectangle count"

    rank_to_color = {color: legend_colors[i] for i, color in enumerate(order)}
    out = np.full(scene.shape, BG, dtype=int)
    for color in reversed(order):
        _fill_rect(out, rectangles[color]["bbox"], rank_to_color[color])
    return out


def _shift_rectangles(
    rectangles: list[tuple[int, int, int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rectangles]


def _render_scene(
    shape: tuple[int, int],
    rectangles: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    scene = np.full(shape, BG, dtype=int)
    for rect, color in zip(rectangles, colors, strict=True):
        _fill_rect(scene, rect, color)
    return scene


def _sample_next_rectangle(prev: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    pr0, pc0, pr1, pc1 = prev
    prev_h = pr1 - pr0 + 1
    prev_w = pc1 - pc0 + 1

    for _ in range(80):
        h = random.randint(*RECT_SIDE_RANGE)
        w = random.randint(*RECT_SIDE_RANGE)
        max_oh = min(prev_h - 2, h - 1, OVERLAP_RANGE[1])
        max_ow = min(prev_w - 2, w - 1, OVERLAP_RANGE[1])
        if max_oh < OVERLAP_RANGE[0] or max_ow < OVERLAP_RANGE[0]:
            continue

        oh = random.randint(OVERLAP_RANGE[0], max_oh)
        ow = random.randint(OVERLAP_RANGE[0], max_ow)
        top = random.randint(pr0 + 1, pr1 - oh)
        left = random.randint(pc0 + 1, pc1 - ow)
        bottom = top + oh - 1
        right = left + ow - 1
        direction = random.choice(CHAIN_DIRECTIONS)

        if direction == "ul":
            rect = (bottom - h + 1, right - w + 1, bottom, right)
        elif direction == "ur":
            rect = (bottom - h + 1, left, bottom, left + w - 1)
        elif direction == "dl":
            rect = (top, right - w + 1, top + h - 1, right)
        else:
            rect = (top, left, top + h - 1, left + w - 1)
        return rect
    return None


def _sample_rectangle_chain(count: int) -> list[tuple[int, int, int, int]] | None:
    h = random.randint(*BASE_SIDE_RANGE)
    w = random.randint(*BASE_SIDE_RANGE)
    rectangles = [(0, 0, h - 1, w - 1)]

    for _ in range(1, count):
        nxt = _sample_next_rectangle(rectangles[-1])
        if nxt is None:
            return None
        rectangles.append(nxt)
    return rectangles


def _expected_output(
    shape: tuple[int, int],
    rectangles: list[tuple[int, int, int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for idx, rect in enumerate(rectangles):
        _fill_rect(out, rect, legend_colors[len(rectangles) - 1 - idx])
    return out


def _valid_layout(
    scene: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
    colors: list[int],
) -> bool:
    intended_top_order = list(reversed(colors))
    for idx, (rect, color) in enumerate(zip(rectangles, colors, strict=True)):
        mask = scene == color
        if not np.any(mask):
            return False
        if _bbox_from_mask(mask) != rect:
            return False
        area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
        visible = int(np.sum(mask))
        if idx == len(rectangles) - 1:
            if visible != area:
                return False
        elif visible >= area:
            return False

    try:
        order, _ = _resolve_top_order(scene)
    except AssertionError:
        return False
    return order == intended_top_order


def _sample_input() -> np.ndarray | None:
    count = random.randint(*RECT_COUNT_RANGE)
    rectangles = _sample_rectangle_chain(count)
    if rectangles is None:
        return None

    min_r = min(r0 for r0, _, _, _ in rectangles)
    min_c = min(c0 for _, c0, _, _ in rectangles)
    max_r = max(r1 for _, _, r1, _ in rectangles)
    max_c = max(c1 for _, _, _, c1 in rectangles)
    union_h = max_r - min_r + 1
    union_w = max_c - min_c + 1

    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*RIGHT_MARGIN_RANGE)
    scene_h = union_h + top_margin + bottom_margin
    scene_w = union_w + left_margin + right_margin
    if scene_h > ARC_MAX or scene_w + 1 > ARC_MAX:
        return None

    shifted = _shift_rectangles(rectangles, top_margin - min_r, left_margin - min_c)
    available = list(PALETTE)
    orig_colors = random.sample(available, count)
    remaining = [c for c in available if c not in orig_colors]
    legend_colors = random.sample(remaining, count)
    scene = _render_scene((scene_h, scene_w), shifted, orig_colors)
    if not _valid_layout(scene, shifted, orig_colors):
        return None

    legend_rows = sorted(random.sample(range(scene_h), count))
    if np.any(scene[legend_rows, scene_w - 1] != BG):
        return None

    grid = np.full((scene_h, scene_w + 1), BG, dtype=int)
    grid[:, :scene_w] = scene
    for row, color in zip(legend_rows, legend_colors, strict=True):
        grid[row, scene_w] = color

    expected = _expected_output(scene.shape, shifted, legend_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_input()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
