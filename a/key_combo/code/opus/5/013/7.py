from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7)
MAX_RETRIES = 500


def _intersect(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(int(b1[0]), int(b2[0]))
    c0 = max(int(b1[1]), int(b2[1]))
    r1 = min(int(b1[2]), int(b2[2]))
    c1 = min(int(b1[3]), int(b2[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_visible(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == int(color))
    assert pts.size > 0, f"color {color} must be visible"
    return tuple(map(int, bbox_coords(pts)))


def _priority_order(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    colors = colors_present(g, exclude={BG})
    boxes = {color: _bbox_from_visible(g, color) for color in colors}

    wins = {color: 0 for color in colors}
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            ia = boxes[a]
            ib = boxes[b]
            overlap = _intersect(ia, ib)
            if overlap is None:
                winner = a if (ia[0], ia[1]) < (ib[0], ib[1]) else b
            else:
                r0, c0, r1, c1 = overlap
                patch_colors = {
                    int(x)
                    for x in np.unique(g[r0 : r1 + 1, c0 : c1 + 1])
                    if int(x) != BG
                }
                assert len(patch_colors) == 1, "overlap region must reveal a single top color"
                winner = next(iter(patch_colors))
                assert winner in (a, b), "generator must avoid triple-overlap ambiguity"
            wins[winner] += 1

    order = sorted(colors, key=lambda color: -wins[color])
    assert [wins[c] for c in order] == list(range(len(colors) - 1, -1, -1))
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the full rectangle stacking order from visible overlaps and compress it into color stripes.

    Input structure:
    - The grid has a black (0) background and 3 to 5 distinct non-black rectangle colors.
    - Each non-black color is the visible remainder of one solid axis-aligned rectangle; its visible bbox is the rectangle's full bbox.
    - Some rectangle pairs overlap, and an overlap cell shows the color of the higher-priority rectangle.

    Rule:
    - Recover each rectangle's bbox from the visible pixels of its color.
    - For every overlapping bbox pair, the visible color inside the intersection is higher priority than the other rectangle.
    - For every non-overlapping pair, compare bbox top-left corners: smaller row wins, then smaller column wins.
    - Combine these pairwise relations into the unique global order from highest to lowest priority, then output one 1x3 solid stripe per rectangle in that order.

    Key insight:
    The overlap colors reveal only a partial order, so the solver must merge all overlap evidence with the explicit top-left tie-break rule to recover the final ranking.
    """
    order = _priority_order(np.asarray(grid, dtype=int))
    return np.array([[color, color, color] for color in order], dtype=int)


def _make_single() -> dict:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    return {
        "locals": [((0, 0, h - 1, w - 1), 0)],
        "union": (0, 0, h - 1, w - 1),
        "size": 1,
    }


def _make_pair() -> dict | None:
    for _ in range(120):
        h1 = random.randint(4, 6)
        w1 = random.randint(4, 6)
        mode = random.choice(("inner", "corner"))
        if mode == "inner":
            if h1 < 5 or w1 < 5:
                continue
            dr = random.randint(0, h1 - 4)
            dc = random.randint(0, w1 - 4)
            if dr == 0 and dc == 0:
                dc = 1
            max_h2 = min(5, h1 - dr - 1)
            max_w2 = min(5, w1 - dc - 1)
            if max_h2 < 3 or max_w2 < 3:
                continue
            h2 = random.randint(3, max_h2)
            w2 = random.randint(3, max_w2)
        else:
            dr = random.randint(1, h1 - 2)
            dc = random.randint(1, w1 - 2)
            h2 = random.randint(max(3, h1 - dr), min(6, h1 - dr + 2))
            w2 = random.randint(max(3, w1 - dc), min(6, w1 - dc + 2))

        b1 = (0, 0, h1 - 1, w1 - 1)
        b2 = (dr, dc, dr + h2 - 1, dc + w2 - 1)
        if _intersect(b1, b2) is None:
            continue
        u = (
            0,
            0,
            max(int(b1[2]), int(b2[2])),
            max(int(b1[3]), int(b2[3])),
        )
        return {
            "locals": [(b1, 0), (b2, 1)],
            "union": u,
            "size": 2,
        }
    return None


def _layout_groups(groups: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    gh = [int(g["union"][2] + 1) for g in groups]
    gw = [int(g["union"][3] + 1) for g in groups]
    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    gap_small = random.randint(2, 4)
    gap_big = random.randint(3, 5)

    if len(groups) == 2:
        if random.random() < 0.45:
            anchors = [
                (top_margin, left_margin),
                (top_margin, left_margin + gw[0] + gap_big),
            ]
        else:
            anchors = [
                (top_margin, left_margin),
                (top_margin + gh[0] + gap_big, left_margin + random.randint(0, 3)),
            ]
    else:
        pattern = random.choice(("diag", "top_two", "bottom_two"))
        if pattern == "diag":
            anchors = [
                (top_margin, left_margin),
                (top_margin + gh[0] + gap_small, left_margin + random.randint(0, 3)),
                (
                    top_margin + gh[0] + gap_small + gh[1] + gap_small,
                    left_margin + random.randint(0, 5),
                ),
            ]
        elif pattern == "top_two":
            anchors = [
                (top_margin, left_margin),
                (top_margin, left_margin + gw[0] + gap_big),
                (
                    top_margin + max(gh[0], gh[1]) + gap_big,
                    left_margin + random.randint(0, 4),
                ),
            ]
        else:
            anchors = [
                (top_margin, left_margin),
                (top_margin + gh[0] + gap_big, left_margin),
                (
                    top_margin + gh[0] + gap_big,
                    left_margin + gw[1] + gap_big,
                ),
            ]

    union_boxes = []
    for (ar, ac), g in zip(anchors, groups):
        ur0, uc0, ur1, uc1 = map(int, g["union"])
        union_boxes.append((ar + ur0, ac + uc0, ar + ur1, ac + uc1))

    for i, box_i in enumerate(union_boxes):
        for j, box_j in enumerate(union_boxes[i + 1 :], start=i + 1):
            if _intersect(box_i, box_j) is not None:
                return None
            if (box_i[0], box_i[1]) >= (box_j[0], box_j[1]):
                return None

    h = max(box[2] for box in union_boxes) + 1 + bottom_margin
    w = max(box[3] for box in union_boxes) + 1 + right_margin
    if h > 30 or w > 30:
        return None
    return int(h), int(w), [(int(r), int(c)) for r, c in anchors]


def _render_case(rects: dict[int, tuple[int, int, int, int]], order: list[int]) -> np.ndarray:
    h = max(b[2] for b in rects.values()) + 1
    w = max(b[3] for b in rects.values()) + 1
    grid = np.full((h, w), BG, dtype=int)
    for color in reversed(order):
        r0, c0, r1, c1 = rects[color]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _build_case() -> tuple[np.ndarray, list[int], dict[int, tuple[int, int, int, int]], list[int]] | None:
    n = random.choice((4, 5))
    group_sizes = [2, 2] if n == 4 else random.choice(([2, 1, 2], [2, 2, 1], [1, 2, 2]))
    groups: list[dict] = []
    for size in group_sizes:
        group = _make_pair() if size == 2 else _make_single()
        if group is None:
            return None
        groups.append(group)

    laid_out = _layout_groups(groups)
    if laid_out is None:
        return None
    _h, _w, anchors = laid_out

    colors = random.sample(PALETTE, n)
    rects: dict[int, tuple[int, int, int, int]] = {}
    hidden_order: list[int] = []
    group_ids: list[int] = []
    spatial_order: list[int] = []

    color_idx = 0
    for group_index, (group, anchor) in enumerate(zip(groups, anchors)):
        local_colors = colors[color_idx : color_idx + group["size"]]
        color_idx += group["size"]
        ar, ac = anchor
        local_rects = []
        for local_box, local_rank in group["locals"]:
            color = local_colors[local_rank]
            r0, c0, r1, c1 = map(int, local_box)
            rects[color] = (ar + r0, ac + c0, ar + r1, ac + c1)
            local_rects.append(color)
        if group["size"] == 2:
            spatial_order.extend(local_rects)
            hidden_order.extend(local_rects[::-1])
        else:
            spatial_order.extend(local_rects)
            hidden_order.extend(local_rects)
        group_ids.extend([group_index] * group["size"])

    grid = _render_case(rects, hidden_order)
    return grid, hidden_order, rects, spatial_order


def _validate_case(
    grid: np.ndarray,
    expected_order: list[int],
    rects: dict[int, tuple[int, int, int, int]],
    spatial_order: list[int],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    colors = colors_present(grid, exclude={BG})
    if sorted(colors) != sorted(expected_order):
        return False

    visible_boxes = {color: _bbox_from_visible(grid, color) for color in colors}
    if any(visible_boxes[color] != rects[color] for color in colors):
        return False

    coverage = np.zeros_like(grid, dtype=int)
    overlap_pairs = 0
    for color, box in rects.items():
        r0, c0, r1, c1 = box
        coverage[r0 : r1 + 1, c0 : c1 + 1] += 1
        vis = grid == int(color)
        if vis.sum() == 0:
            return False
        if not vis[r0, c0 : c1 + 1].any():
            return False
        if not vis[r1, c0 : c1 + 1].any():
            return False
        if not vis[r0 : r1 + 1, c0].any():
            return False
        if not vis[r0 : r1 + 1, c1].any():
            return False
    if int(coverage.max()) > 2:
        return False

    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            if _intersect(rects[a], rects[b]) is not None:
                overlap_pairs += 1
    if overlap_pairs < 2:
        return False

    if _priority_order(grid) != expected_order:
        return False

    position_only = sorted(colors, key=lambda color: (rects[color][0], rects[color][1]))
    if position_only == expected_order:
        return False

    if not any(
        _intersect(rects[a], rects[b]) is None
        for i, a in enumerate(colors)
        for b in colors[i + 1 :]
    ):
        return False

    if len({tuple(rects[c][:2]) for c in colors}) != len(colors):
        return False

    if spatial_order == expected_order:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        case = _build_case()
        if case is None:
            continue
        grid, expected_order, rects, spatial_order = case
        try:
            if _validate_case(grid, expected_order, rects, spatial_order):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid overlapping-rectangle ordering input")
