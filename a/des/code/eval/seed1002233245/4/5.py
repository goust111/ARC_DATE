from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
SHADOW = 5
GAP = 1
MIN_RECTS = 2
MAX_RECTS = 4
MIN_SIDE = 3
MAX_ATTEMPTS = 500
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 9)
BRANCHES = ("right", "bottom", "both")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _shadow_segments(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = shape
    r0, c0, r1, c1 = rect
    bottom = []
    right = []
    if r1 + 1 < h:
        for c in range(c0 + 1, c1 + 2):
            if 0 <= c < w:
                bottom.append((r1 + 1, c))
    if c1 + 1 < w:
        for r in range(r0 + 1, r1 + 2):
            if 0 <= r < h:
                right.append((r, c1 + 1))
    return bottom, right


def _render_scene(rects: list[tuple[int, int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for r0, c0, r1, c1, color in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    bodies = grid.copy()
    for r0, c0, r1, c1, _color in rects:
        bottom, right = _shadow_segments((r0, c0, r1, c1), shape)
        for rr, cc in bottom + right:
            if bodies[rr, cc] == BG:
                grid[rr, cc] = SHADOW
    return grid


def _pack_output(rects: list[tuple[int, int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    col = 0
    for r0, c0, r1, c1, color in rects:
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        out[:rect_h, col : col + rect_w] = color
        col += rect_w + GAP
    return out


def _candidate_ok(
    grid: np.ndarray,
    color: int,
    vis_bbox: tuple[int, int, int, int],
    rect: tuple[int, int, int, int],
    higher_mask: np.ndarray,
) -> bool:
    r0, c0, vr1, vc1 = vis_bbox
    cr0, cc0, r1, c1 = rect
    assert (r0, c0) == (cr0, cc0)

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    cover = higher_mask[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all((sub == color) | cover):
        return False

    bottom, right = _shadow_segments(rect, grid.shape)
    visible_bottom = [(rr, cc) for rr, cc in bottom if not higher_mask[rr, cc]]
    visible_right = [(rr, cc) for rr, cc in right if not higher_mask[rr, cc]]

    if any(grid[rr, cc] != SHADOW for rr, cc in visible_bottom):
        return False
    if any(grid[rr, cc] != SHADOW for rr, cc in visible_right):
        return False

    if r1 > vr1 and not visible_bottom:
        return False
    if c1 > vc1 and not visible_right:
        return False

    return True


def _recover_rectangles(grid: np.ndarray) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    body_colors = [int(v) for v in np.unique(g) if int(v) not in (BG, SHADOW)]
    assert MIN_RECTS <= len(body_colors) <= MAX_RECTS

    visible_masks: dict[int, np.ndarray] = {}
    infos: dict[int, tuple[int, int, int, int]] = {}
    for color in body_colors:
        mask = g == color
        comps = _components(mask)
        assert len(comps) == 1, "each color must form one visible component"
        coords = comps[0]
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        infos[color] = (r0, c0, r1, c1)
        visible_masks[color] = mask

    top_to_bottom = sorted(body_colors, key=lambda col: infos[col][:2], reverse=True)

    higher_mask = np.zeros_like(g, dtype=bool)
    rects: dict[int, tuple[int, int, int, int]] = {}
    for color in top_to_bottom:
        r0, c0, vr1, vc1 = infos[color]
        candidates: list[tuple[int, int, int, int]] = []
        for r1 in range(vr1, g.shape[0] - 1):
            for c1 in range(vc1, g.shape[1] - 1):
                rect = (r0, c0, r1, c1)
                if _candidate_ok(g, color, infos[color], rect, higher_mask):
                    candidates.append(rect)
        assert candidates, f"no rectangle fits color {color}"
        max_area = max((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in candidates)
        best = [rect for rect in candidates if (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1) == max_area]
        assert len(best) == 1, f"ambiguous rectangle for color {color}"
        rects[color] = best[0]
        br0, bc0, br1, bc1 = best[0]
        higher_mask[br0 : br1 + 1, bc0 : bc1 + 1] = True

    return rects, visible_masks


def _stack_order(
    rects: dict[int, tuple[int, int, int, int]],
    visible_masks: dict[int, np.ndarray],
) -> list[int]:
    colors = list(rects)
    edges = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}

    for a in colors:
        ar0, ac0, ar1, ac1 = rects[a]
        for b in colors:
            if a == b:
                continue
            pts = np.argwhere(visible_masks[b])
            if pts.size == 0:
                continue
            inside = (
                (pts[:, 0] >= ar0)
                & (pts[:, 0] <= ar1)
                & (pts[:, 1] >= ac0)
                & (pts[:, 1] <= ac1)
            )
            if bool(np.any(inside)) and b not in edges[a]:
                edges[a].add(b)
                indeg[b] += 1

    queue = sorted((color for color in colors if indeg[color] == 0), key=lambda col: rects[col][:2])
    order: list[int] = []
    while queue:
        color = queue.pop(0)
        order.append(color)
        for nxt in sorted(edges[color], key=lambda col: rects[col][:2]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
                queue.sort(key=lambda col: rects[col][:2])

    assert len(order) == len(colors), "stack relation must be acyclic"
    return order


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rects, visible_masks = _recover_rectangles(g)
    order = _stack_order(rects, visible_masks)

    full_rects = []
    for color in order:
        r0, c0, r1, c1 = rects[color]
        full_rects.append((r0, c0, r1, c1, color))

    packed_width = sum(c1 - c0 + 1 for r0, c0, r1, c1, _ in full_rects) + len(full_rects) - 1
    packed_height = max(r1 - r0 + 1 for r0, c0, r1, c1, _ in full_rects)
    assert packed_width <= g.shape[1]
    assert packed_height <= g.shape[0]
    return _pack_output(full_rects, g.shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each stacked rectangle from its visible fragment and down-right shadow,
    infer the bottom-to-top stack order from occlusion, and output the full
    rectangles packed left-to-right with one background column between them.
    """
    return _solve(grid)


def _sample_chain() -> list[tuple[int, int, int, int, int]] | None:
    count = random.randint(MIN_RECTS, MAX_RECTS)
    colors = random.sample(OBJECT_COLORS, count)

    h0 = random.randint(5, 8)
    w0 = random.randint(5, 8)
    rects = [(0, 0, h0 - 1, w0 - 1, colors[0])]

    for i in range(1, count):
        pr0, pc0, pr1, pc1, _ = rects[-1]
        prev_h = pr1 - pr0 + 1
        prev_w = pc1 - pc0 + 1
        dr_max = min(3, prev_h - MIN_SIDE)
        dc_max = min(3, prev_w - MIN_SIDE)
        if dr_max < 1 or dc_max < 1:
            return None

        r0 = pr0 + random.randint(1, dr_max)
        c0 = pc0 + random.randint(1, dc_max)
        branch = random.choice(BRANCHES)

        if branch == "right":
            r1_lo = max(r0 + MIN_SIDE - 1, pr1 - 2)
            r1_hi = pr1
            c1_lo = max(c0 + MIN_SIDE - 1, pc1 + 1)
            c1_hi = pc1 + 3
        elif branch == "bottom":
            r1_lo = max(r0 + MIN_SIDE - 1, pr1 + 1)
            r1_hi = pr1 + 3
            c1_lo = max(c0 + MIN_SIDE - 1, pc1 - 2)
            c1_hi = pc1
        else:
            r1_lo = max(r0 + MIN_SIDE - 1, pr1 + 1)
            r1_hi = pr1 + 3
            c1_lo = max(c0 + MIN_SIDE - 1, pc1 + 1)
            c1_hi = pc1 + 3

        if r1_lo > r1_hi or c1_lo > c1_hi:
            return None

        r1 = random.randint(r1_lo, r1_hi)
        c1 = random.randint(c1_lo, c1_hi)
        rects.append((r0, c0, r1, c1, colors[i]))

    return rects


def _shift_and_size(
    rects: list[tuple[int, int, int, int, int]]
) -> tuple[list[tuple[int, int, int, int, int]], tuple[int, int]] | None:
    widths = [c1 - c0 + 1 for r0, c0, r1, c1, _ in rects]
    heights = [r1 - r0 + 1 for r0, c0, r1, c1, _ in rects]
    packed_w = sum(widths) + len(widths) - 1
    packed_h = max(heights)

    max_r1 = max(r1 for _r0, _c0, r1, _c1, _color in rects)
    max_c1 = max(c1 for _r0, _c0, _r1, c1, _color in rects)

    top_pad = random.randint(1, 3)
    left_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    right_pad = random.randint(1, 4)

    shifted = [
        (r0 + top_pad, c0 + left_pad, r1 + top_pad, c1 + left_pad, color)
        for r0, c0, r1, c1, color in rects
    ]

    h = max(packed_h, max_r1 + top_pad + 2 + bottom_pad)
    w = max(packed_w, max_c1 + left_pad + 2 + right_pad)
    if h > 30 or w > 30:
        return None
    return shifted, (h, w)


def _visible_shadow_cells(rect: tuple[int, int, int, int], bodies: np.ndarray, grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    bottom, right = _shadow_segments(rect, grid.shape)
    vis_bottom = {(rr, cc) for rr, cc in bottom if bodies[rr, cc] == BG and grid[rr, cc] == SHADOW}
    vis_right = {(rr, cc) for rr, cc in right if bodies[rr, cc] == BG and grid[rr, cc] == SHADOW}
    return vis_bottom, vis_right


def _valid_instance(rects: list[tuple[int, int, int, int, int]], shape: tuple[int, int]) -> bool:
    grid = _render_scene(rects, shape)
    bodies = np.full(shape, BG, dtype=int)
    for r0, c0, r1, c1, color in rects:
        bodies[r0 : r1 + 1, c0 : c1 + 1] = color

    for idx, (r0, c0, r1, c1, color) in enumerate(rects):
        mask = grid == color
        comps = _components(mask)
        if len(comps) != 1:
            return False
        coords = comps[0]
        vr0, vc0 = map(int, coords.min(axis=0))
        vr1, vc1 = map(int, coords.max(axis=0))
        if (vr0, vc0) != (r0, c0):
            return False
        visible_area = int(mask.sum())
        full_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if idx < len(rects) - 1 and visible_area == full_area:
            return False
        vis_bottom, vis_right = _visible_shadow_cells((r0, c0, r1, c1), bodies, grid)
        if vr1 < r1 and not vis_bottom:
            return False
        if vc1 < c1 and not vis_right:
            return False

    expected = _pack_output(rects, shape)
    try:
        actual = _solve(grid)
    except AssertionError:
        return False
    return np.array_equal(expected, actual)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rects = _sample_chain()
        if rects is None:
            continue
        shifted = _shift_and_size(rects)
        if shifted is None:
            continue
        placed_rects, shape = shifted
        if not _valid_instance(placed_rects, shape):
            continue
        return _render_scene(placed_rects, shape)
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
