import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _grey_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != GREY or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == GREY:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _find_rectangles(grid):
    rects = []
    for coords in _grey_components(grid):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        if bottom - top >= 2 and right - left >= 2:
            rects.append((top, left, bottom, right))
    return rects


def _contains(parent, child):
    pt, pl, pb, pr = parent
    ct, cl, cb, cr = child
    return pt < ct and cl > pl and cb < pb and cr < pr


def _legend_colors(grid):
    for r in range(grid.shape[0] - 1, -1, -1):
        colors = [int(v) for v in grid[r] if int(v) not in (BG, GREY)]
        if colors:
            return colors
    return []


def _parents_and_depths(rects):
    parents = [None] * len(rects)
    areas = [(b - t + 1) * (r - l + 1) for t, l, b, r in rects]
    for i, child in enumerate(rects):
        candidates = [j for j, parent in enumerate(rects) if i != j and _contains(parent, child)]
        if candidates:
            parents[i] = min(candidates, key=lambda j: areas[j])

    depths = [-1] * len(rects)

    def depth(i):
        if depths[i] != -1:
            return depths[i]
        parent = parents[i]
        depths[i] = 0 if parent is None else depth(parent) + 1
        return depths[i]

    for i in range(len(rects)):
        depth(i)
    return parents, depths


def _draw_rect(out, rect, color):
    top, left, bottom, right = rect
    out[top, left : right + 1] = color
    out[bottom, left : right + 1] = color
    out[top : bottom + 1, left] = color
    out[top : bottom + 1, right] = color


def _draw_connector(out, parent, child, color):
    pt, pl, pb, pr = parent
    ct, cl, cb, cr = child
    choices = [
        (ct - pt, "top"),
        (pb - cb, "bottom"),
        (cl - pl, "left"),
        (pr - cr, "right"),
    ]
    side = min(choices, key=lambda item: item[0])[1]
    if side == "top":
        col = (cl + cr) // 2
        out[pt : ct + 1, col] = color
    elif side == "bottom":
        col = (cl + cr) // 2
        out[cb : pb + 1, col] = color
    elif side == "left":
        row = (ct + cb) // 2
        out[row, pl : cl + 1] = color
    else:
        row = (ct + cb) // 2
        out[row, cr : pr + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Color nested rectangular frames by depth and connect each child to its direct parent.

    Input structure:
    The background is black. Grey pixels form separate one-pixel-thick closed rectangular
    frames, and frames are either fully nested inside another frame or fully disjoint. A
    bottom legend row contains colored single pixels separated by black cells; those colors
    are the visible depth palette from left to right.
    Rule:
    Build the containment tree of the grey frames. Recolor every frame border with the
    legend color for its depth, where outer frames have depth 0. For every direct
    parent-child pair, choose the child side closest to the parent border and draw a
    one-pixel straight connector from the midpoint of that child side to the corresponding
    parent side, using the child's color.
    Key insight:
    The legend indexes containment depth, while connector direction comes from nearest
    enclosing side and connector color follows the child rather than the parent.
    """
    out = np.array(grid, copy=True)
    rects = _find_rectangles(grid)
    colors = _legend_colors(grid)
    if not rects or not colors:
        return out

    parents, depths = _parents_and_depths(rects)
    for rect, d in zip(rects, depths):
        color = colors[min(d, len(colors) - 1)]
        _draw_rect(out, rect, color)
    for i, parent in enumerate(parents):
        if parent is None:
            continue
        d = depths[i]
        color = colors[min(d, len(colors) - 1)]
        _draw_connector(out, rects[parent], rects[i], color)
    return out


def _split_slots(start, end, count, gap):
    total = end - start + 1 - gap * (count - 1)
    base = total // count
    rem = total % count
    slots = []
    cur = start
    for i in range(count):
        size = base + (1 if i < rem else 0)
        slots.append((cur, cur + size - 1))
        cur += size + gap
    return slots


def _min_dim_for_links(links):
    return 3 + 6 * links


def _feasible_sides(box, remaining_after):
    top, left, bottom, right = box
    height = bottom - top + 1
    width = right - left + 1
    need = _min_dim_for_links(remaining_after)
    sides = []
    if height - 5 >= need and width - 6 >= need:
        sides.extend(["top", "bottom"])
    if width - 5 >= need and height - 6 >= need:
        sides.extend(["left", "right"])
    return sides


def _make_child(box, side, remaining_after):
    top, left, bottom, right = box
    height = bottom - top + 1
    width = right - left + 1
    need = _min_dim_for_links(remaining_after)
    if side in ("top", "bottom"):
        max_h = height - 5
        max_w = width - 6
        if max_h < need or max_w < need:
            return None
        child_h = random.randint(need, max_h)
        child_w = random.randint(need, max_w)
        min_left = left + 3
        max_left = right - 3 - child_w + 1
        child_left = random.randint(min_left, max_left)
        child_right = child_left + child_w - 1
        if side == "top":
            child_top = top + 2
            child_bottom = child_top + child_h - 1
        else:
            child_bottom = bottom - 2
            child_top = child_bottom - child_h + 1
    else:
        max_h = height - 6
        max_w = width - 5
        if max_h < need or max_w < need:
            return None
        child_h = random.randint(need, max_h)
        child_w = random.randint(need, max_w)
        min_top = top + 3
        max_top = bottom - 3 - child_h + 1
        child_top = random.randint(min_top, max_top)
        child_bottom = child_top + child_h - 1
        if side == "left":
            child_left = left + 2
            child_right = child_left + child_w - 1
        else:
            child_right = right - 2
            child_left = child_right - child_w + 1
    return (child_top, child_left, child_bottom, child_right)


def _draw_grey_frame(grid, rect):
    _draw_rect(grid, rect, GREY)


def _build_roots(height, width, count, orientation):
    if orientation == "vertical":
        slots = _split_slots(1, height - 3, count, 1)
        if min(b - a + 1 for a, b in slots) < 8:
            return None
        roots = []
        for top_slot, bottom_slot in slots:
            top = top_slot + random.randint(0, min(1, bottom_slot - top_slot - 7))
            bottom = bottom_slot - random.randint(0, min(1, bottom_slot - top - 7))
            left = random.randint(1, min(3, width - 9))
            right = width - 2 - random.randint(0, min(2, width - left - 9))
            if bottom - top + 1 < 8 or right - left + 1 < 8:
                return None
            roots.append((top, left, bottom, right))
    else:
        slots = _split_slots(1, width - 2, count, 1)
        if min(b - a + 1 for a, b in slots) < 8:
            return None
        roots = []
        for left_slot, right_slot in slots:
            left = left_slot + random.randint(0, min(1, right_slot - left_slot - 7))
            right = right_slot - random.randint(0, min(1, right_slot - left - 7))
            top = random.randint(1, min(3, height - 10))
            bottom = height - 3 - random.randint(0, min(2, height - top - 10))
            if bottom - top + 1 < 8 or right - left + 1 < 8:
                return None
            roots.append((top, left, bottom, right))
    return roots


def generate_input() -> np.ndarray:
    for _ in range(300):
        height = random.randint(18, 30)
        width = random.randint(18, 30)
        root_count = random.randint(1, 3)
        orientation = random.choice(["vertical", "horizontal"])
        roots = _build_roots(height, width, root_count, orientation)
        if roots is None:
            continue

        rects = []
        max_depth = 0
        made_child = False
        for root_index, root in enumerate(roots):
            rects.append(root)
            current = root
            available = 0
            probe = current
            while available < 4 and _feasible_sides(probe, 0):
                sides = _feasible_sides(probe, 0)
                probe_child = _make_child(probe, random.choice(sides), 0)
                if probe_child is None:
                    break
                available += 1
                probe = probe_child

            forced = 1 if root_index == 0 and not made_child else 0
            levels = random.randint(forced, max(forced, min(4, available)))
            for level in range(levels):
                remaining_after = levels - level - 1
                sides = _feasible_sides(current, remaining_after)
                if not sides:
                    break
                child = _make_child(current, random.choice(sides), remaining_after)
                if child is None:
                    break
                rects.append(child)
                current = child
                max_depth = max(max_depth, level + 1)
                made_child = True

        if not made_child:
            continue

        grid = np.zeros((height, width), dtype=int)
        for rect in rects:
            _draw_grey_frame(grid, rect)

        colors = random.sample(PALETTE, max_depth + 1)
        legend_width = 2 * len(colors) - 1
        if legend_width > width:
            continue
        start = random.randint(0, width - legend_width)
        for i, color in enumerate(colors):
            grid[height - 1, start + 2 * i] = color
        return grid

    grid = np.zeros((24, 24), dtype=int)
    rects = [(2, 2, 18, 21), (4, 5, 13, 16), (7, 7, 10, 12)]
    for rect in rects:
        _draw_grey_frame(grid, rect)
    for i, color in enumerate([1, 2, 3]):
        grid[23, 2 + 2 * i] = color
    return grid
