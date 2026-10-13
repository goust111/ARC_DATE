from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
LEGEND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_DEPTH_CHOICES = (1, 2, 3, 4)
ROOT_COUNT_BY_DEPTH = {
    1: (2, 3),
    2: (1, 3),
    3: (1, 2),
    4: (1, 2),
}
LEAF_SIZE_BY_ROOT_DEPTH = {
    1: (3, 8),
    2: (3, 7),
    3: (3, 6),
    4: (3, 5),
}
CHILD_GAP_RANGE = (1, 3)
INNER_SLACK_RANGE = (0, 4)
ROOT_GAP_RANGE = (2, 4)
OUTER_MARGIN_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 400


@dataclass
class RectNode:
    h: int
    w: int
    children: list[tuple["RectNode", int, int]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested grey rectangles by depth and turn each one into a hollow frame.

    Input structure:
    - The background is black.
    - The main area contains several same-color grey rectangles, possibly nested with at least one black cell of separation between levels.
    - The bottom row is a contiguous legend strip whose left-to-right colors encode the colors for depth 1, depth 2, and so on.

    Rule:
    - Find every grey rectangular object and determine its nesting depth from strict bbox containment.
    - Read the legend colors from left to right so the k-th legend color is the border color for depth k.
    - Replace each grey rectangle by only its outermost one-pixel border in the color for its depth, clear its interior to black, and remove the legend row.

    Key insight:
    - Rectangle depth is not visible from color in the main area, so the solver must infer levels from geometry and only then use the legend as a depth-to-color lookup table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.zeros_like(g)
    if g.shape[0] == 0:
        return out

    legend = [int(v) for v in g[-1] if int(v) != BLACK]
    top = g[:-1]
    rects: list[tuple[int, int, int, int]] = []

    for coords, color in _monochrome_components(top):
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        if _has_full_bbox_border(top, color, r0, c0, r1, c1):
            rects.append((r0, c0, r1, c1))

    if not rects or not legend:
        return out

    depths: list[int] = []
    max_depth = 0
    for rect in rects:
        depth = 1 + sum(_strictly_contains(other, rect) for other in rects if other != rect)
        depths.append(depth)
        max_depth = max(max_depth, depth)

    assert len(legend) >= max_depth
    for rect, depth in zip(rects, depths):
        _draw_border(out, rect, legend[depth - 1])

    return out


def _monochrome_components(grid: np.ndarray) -> list[tuple[np.ndarray, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[np.ndarray, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BLACK:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((np.array(pts, dtype=int), color))
    return comps


def _has_full_bbox_border(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(
        (sub[0, :] == color).all()
        and (sub[-1, :] == color).all()
        and (sub[:, 0] == color).all()
        and (sub[:, -1] == color).all()
    )


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _draw_border(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _randint(rng: random.Random, bounds: tuple[int, int]) -> int:
    return rng.randint(bounds[0], bounds[1])


def _sample_depth(rng: random.Random) -> int:
    return rng.choices(MAX_DEPTH_CHOICES, weights=(1, 2, 2, 1), k=1)[0]


def _generate_subtree(depth: int, rng: random.Random, root_depth: int) -> RectNode:
    if depth == 1:
        lo, hi = LEAF_SIZE_BY_ROOT_DEPTH[root_depth]
        return RectNode(_randint(rng, (lo, hi)), _randint(rng, (lo, hi)), [])

    child_depths = [depth - 1]
    max_extra = 1 if depth >= 4 else 2
    extra_children = rng.randint(0, max_extra)
    for _ in range(extra_children):
        child_depths.append(rng.randint(1, depth - 1))
    rng.shuffle(child_depths)

    children = [_generate_subtree(child_depth, rng, root_depth) for child_depth in child_depths]
    if len(children) == 1:
        child = children[0]
        slack_h = _randint(rng, INNER_SLACK_RANGE)
        slack_w = _randint(rng, INNER_SLACK_RANGE)
        top = 2 + rng.randint(0, slack_h)
        left = 2 + rng.randint(0, slack_w)
        return RectNode(child.h + 4 + slack_h, child.w + 4 + slack_w, [(child, top, left)])

    layout = rng.choice(("row", "column"))
    gaps = [_randint(rng, CHILD_GAP_RANGE) for _ in range(len(children) - 1)]

    if layout == "row":
        inner_h = max(child.h for child in children) + _randint(rng, INNER_SLACK_RANGE)
        inner_w = sum(child.w for child in children) + sum(gaps)
        align = rng.choice(("top", "center", "bottom"))
        placements: list[tuple[RectNode, int, int]] = []
        left = 2
        for idx, child in enumerate(children):
            slack = inner_h - child.h
            if align == "top":
                top = 2
            elif align == "bottom":
                top = 2 + slack
            else:
                top = 2 + slack // 2
            placements.append((child, top, left))
            if idx < len(gaps):
                left += child.w + gaps[idx]
        return RectNode(inner_h + 4, inner_w + 4, placements)

    inner_h = sum(child.h for child in children) + sum(gaps)
    inner_w = max(child.w for child in children) + _randint(rng, INNER_SLACK_RANGE)
    align = rng.choice(("left", "center", "right"))
    placements = []
    top = 2
    for idx, child in enumerate(children):
        slack = inner_w - child.w
        if align == "left":
            left = 2
        elif align == "right":
            left = 2 + slack
        else:
            left = 2 + slack // 2
        placements.append((child, top, left))
        if idx < len(gaps):
            top += child.h + gaps[idx]
    return RectNode(inner_h + 4, inner_w + 4, placements)


def _row_group_positions(roots: list[RectNode], rng: random.Random) -> tuple[int, int, list[tuple[RectNode, int, int]]]:
    row_h = max(root.h for root in roots)
    gaps = [_randint(rng, ROOT_GAP_RANGE) for _ in range(len(roots) - 1)]
    positions: list[tuple[RectNode, int, int]] = []
    left = 0
    for idx, root in enumerate(roots):
        top = rng.randint(0, row_h - root.h)
        positions.append((root, top, left))
        if idx < len(gaps):
            left += root.w + gaps[idx]
    width = sum(root.w for root in roots) + sum(gaps)
    return row_h, width, positions


def _pack_roots(roots: list[RectNode], rng: random.Random) -> tuple[int, int, list[tuple[RectNode, int, int]]] | None:
    roots = roots[:]
    rng.shuffle(roots)
    modes = ["row", "column"]
    if len(roots) >= 3:
        modes.append("grid")
    rng.shuffle(modes)

    for mode in modes:
        left_margin = _randint(rng, OUTER_MARGIN_RANGE)
        right_margin = _randint(rng, OUTER_MARGIN_RANGE)
        top_margin = _randint(rng, OUTER_MARGIN_RANGE)
        bottom_margin = _randint(rng, OUTER_MARGIN_RANGE)

        if mode == "row":
            row_h, inner_w, row_positions = _row_group_positions(roots, rng)
            height = top_margin + row_h + bottom_margin
            width = left_margin + inner_w + right_margin
            positions = [(root, top_margin + top, left_margin + left) for root, top, left in row_positions]
        elif mode == "column":
            gaps = [_randint(rng, ROOT_GAP_RANGE) for _ in range(len(roots) - 1)]
            col_w = max(root.w for root in roots)
            top = top_margin
            positions = []
            for idx, root in enumerate(roots):
                left = left_margin + rng.randint(0, col_w - root.w)
                positions.append((root, top, left))
                top += root.h
                if idx < len(gaps):
                    top += gaps[idx]
            height = top + bottom_margin
            width = left_margin + col_w + right_margin
        else:
            split = rng.randint(1, len(roots) - 1)
            row1 = roots[:split]
            row2 = roots[split:]
            row1_h, row1_w, pos1 = _row_group_positions(row1, rng)
            row2_h, row2_w, pos2 = _row_group_positions(row2, rng)
            row_gap = _randint(rng, ROOT_GAP_RANGE)
            inner_w = max(row1_w, row2_w)
            off1 = rng.randint(0, inner_w - row1_w)
            off2 = rng.randint(0, inner_w - row2_w)
            height = top_margin + row1_h + row_gap + row2_h + bottom_margin
            width = left_margin + inner_w + right_margin
            positions = [
                (root, top_margin + top, left_margin + off1 + left)
                for root, top, left in pos1
            ]
            positions.extend(
                (root, top_margin + row1_h + row_gap + top, left_margin + off2 + left)
                for root, top, left in pos2
            )

        if height <= 29 and width <= 30:
            return height, width, positions

    return None


def _paint_input_subtree(grid: np.ndarray, node: RectNode, top: int, left: int) -> None:
    grid[top : top + node.h, left : left + node.w] = GREY
    for child, child_top, child_left in node.children:
        r0 = top + child_top - 1
        c0 = left + child_left - 1
        r1 = top + child_top + child.h
        c1 = left + child_left + child.w
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        _paint_input_subtree(grid, child, top + child_top, left + child_left)


def _paint_output_subtree(grid: np.ndarray, node: RectNode, top: int, left: int, depth: int, legend: list[int]) -> None:
    _draw_border(grid, (top, left, top + node.h - 1, left + node.w - 1), legend[depth - 1])
    for child, child_top, child_left in node.children:
        _paint_output_subtree(grid, child, top + child_top, left + child_left, depth + 1, legend)


def _build_candidate(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    max_depth = _sample_depth(rng)
    root_count = rng.randint(*ROOT_COUNT_BY_DEPTH[max_depth])
    root_depths = [max_depth] + [rng.randint(1, max_depth) for _ in range(root_count - 1)]
    rng.shuffle(root_depths)
    roots = [_generate_subtree(depth, rng, depth) for depth in root_depths]

    packed = _pack_roots(roots, rng)
    if packed is None:
        return None
    top_h, width, placements = packed

    legend = rng.sample(LEGEND_COLORS, max_depth)
    width = max(width, len(legend) + 2)
    if width > 30 or top_h + 1 > 30:
        return None

    grid = np.zeros((top_h + 1, width), dtype=int)
    expected = np.zeros_like(grid)

    for root, top, left in placements:
        _paint_input_subtree(grid, root, top, left)
        _paint_output_subtree(expected, root, top, left, 1, legend)

    legend_start = rng.randint(0, width - len(legend))
    grid[-1, legend_start : legend_start + len(legend)] = np.array(legend, dtype=int)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_candidate(rng)
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle input")
