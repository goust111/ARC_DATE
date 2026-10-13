import random
from collections import defaultdict, deque
from itertools import combinations

import numpy as np


BLACK = 0


def _components_by_color(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = defaultdict(list)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps[color].append(comp)
    return comps


def _line_info(comp):
    if len(comp) < 3:
        return None
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    if len(rows) == 1:
        row = next(iter(rows))
        lo, hi = min(cols), max(cols)
        if len(comp) == hi - lo + 1:
            return ("h", row, lo, hi)
    if len(cols) == 1:
        col = next(iter(cols))
        lo, hi = min(rows), max(rows)
        if len(comp) == hi - lo + 1:
            return ("v", col, lo, hi)
    return None


def _find_lines_and_anchors(grid):
    comps = _components_by_color(grid)
    lines = []
    used_anchors = set()
    line_cells = set()
    for color, color_comps in comps.items():
        singles = [comp[0] for comp in color_comps if len(comp) == 1]
        candidates = []
        for comp in color_comps:
            info = _line_info(comp)
            if info is not None:
                candidates.append((info, comp))
        for info, comp in candidates:
            orient, fixed, lo, hi = info
            anchor_choices = []
            for ar, ac in singles:
                if (ar, ac) in used_anchors:
                    continue
                if orient == "h" and ar != fixed and lo <= ac <= hi:
                    anchor_choices.append((abs(ar - fixed), ar, ac, ac))
                elif orient == "v" and ac != fixed and lo <= ar <= hi:
                    anchor_choices.append((abs(ac - fixed), ar, ac, ar))
            if not anchor_choices:
                continue
            _, ar, ac, fold = min(anchor_choices)
            used_anchors.add((ar, ac))
            line_cells.update(comp)
            lines.append(
                {
                    "color": int(color),
                    "orient": orient,
                    "fixed": fixed,
                    "lo": lo,
                    "hi": hi,
                    "fold": fold,
                    "anchor": (ar, ac),
                }
            )
    return lines, line_cells, used_anchors


def _parse_mix_map(grid, legend_cells):
    h, w = grid.shape
    mix = {}
    for r in range(h - 1):
        for c in range(w - 1):
            left = (r, c)
            right = (r, c + 1)
            below_left = (r + 1, c)
            below_right = (r + 1, c + 1)
            if left in legend_cells and right in legend_cells:
                a = int(grid[left])
                b = int(grid[right])
                if a == BLACK or b == BLACK or a == b:
                    continue
                if below_left in legend_cells and below_right not in legend_cells:
                    res = int(grid[below_left])
                    if res != BLACK:
                        mix[frozenset((a, b))] = res
                elif below_right in legend_cells and below_left not in legend_cells:
                    res = int(grid[below_right])
                    if res != BLACK:
                        mix[frozenset((a, b))] = res
    return mix


def _folded_cells(line):
    orient = line["orient"]
    fixed = line["fixed"]
    lo = line["lo"]
    hi = line["hi"]
    fold = line["fold"]
    left_len = fold - lo
    right_len = hi - fold
    if left_len <= right_len:
        coords = range(fold, hi + 1)
    else:
        coords = range(lo, fold + 1)
    if orient == "h":
        return [(fixed, c) for c in coords]
    return [(r, fixed) for r in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """Fold each colored segment at its same-colored anchor and mix crossing colors.

    Input structure:
    The background is black.  Each task line is a one-pixel-wide horizontal or
    vertical monochrome segment with one off-line singleton anchor of the same
    color.  A compact legend made from isolated colored pixels shows unordered
    color pairs in a top row and their combined result just below one of them.

    Rule:
    For every segment, the same-colored singleton whose perpendicular projection
    lands on the segment is the anchor; that projection is the fold point.  The
    shorter side of the segment folds onto the longer side, so the visible result
    is the longer side including the fold point.  Anchors are erased.  When final
    folded segments of two different colors occupy the same cell, the output
    color is read from the legend; same-color self-overlap keeps the line color.
    The legend pixels are copied unchanged.

    Key insight:
    Separate the structural clues: line color identifies the anchor, anchor
    projection determines the fold, and the detached legend defines how any
    cross-color overlap is rendered.
    """
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    lines, line_cells, anchor_cells = _find_lines_and_anchors(grid)
    all_colored = {
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) != BLACK
    }
    legend_cells = all_colored - line_cells - anchor_cells
    mix = _parse_mix_map(grid, legend_cells)

    occupancy = defaultdict(list)
    for line in lines:
        color = line["color"]
        for cell in _folded_cells(line):
            if color not in occupancy[cell]:
                occupancy[cell].append(color)

    for cell, colors in occupancy.items():
        if len(colors) == 1:
            out[cell] = colors[0]
        else:
            key = frozenset(colors[:2])
            out[cell] = mix.get(key, colors[0])

    for r, c in legend_cells:
        out[r, c] = grid[r, c]
    return out


def _near_existing(cells, occupied, h, w):
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _pick_fold_index(length):
    mid2 = length - 1
    left_side = random.choice([True, False])
    if left_side:
        choices = [i for i in range(1, length - 1) if 2 * i < mid2]
    else:
        choices = [i for i in range(1, length - 1) if 2 * i > mid2]
    if not choices:
        choices = [i for i in range(1, length - 1) if 2 * i != mid2]
    return random.choice(choices)


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        n_lines = random.choice([2, 3])
        pair_count = n_lines * (n_lines - 1) // 2
        legend_width = 3 * pair_count - 1
        if legend_width + 1 > w:
            continue

        grid = np.zeros((h, w), dtype=int)
        line_colors = random.sample(range(1, 10), n_lines)
        result_pool = [c for c in range(1, 10) if c not in line_colors]
        result_colors = random.sample(result_pool, pair_count)
        pairs = list(combinations(line_colors, 2))
        random.shuffle(pairs)

        legend_r = random.choice([0, 1])
        legend_c = random.randint(0, w - legend_width)
        occupied = set()
        for i, ((a, b), res) in enumerate(zip(pairs, result_colors)):
            c = legend_c + 3 * i
            grid[legend_r, c] = a
            grid[legend_r, c + 1] = b
            grid[legend_r + 1, c] = res
            occupied.update({(legend_r, c), (legend_r, c + 1), (legend_r + 1, c)})

        ok = True
        for color in line_colors:
            placed = False
            for _attempt in range(250):
                orient = random.choice(["h", "v"])
                length = random.randint(5, 9)
                fold_i = _pick_fold_index(length)
                dist = random.choice([2, 3])
                side = random.choice([-1, 1])
                if orient == "h":
                    row = random.randint(legend_r + 5, h - 2)
                    col = random.randint(0, w - length)
                    fold_r, fold_c = row, col + fold_i
                    anchor = (fold_r + side * dist, fold_c)
                    cells = [(row, c) for c in range(col, col + length)] + [anchor]
                else:
                    row = random.randint(legend_r + 4, h - length)
                    col = random.randint(1, w - 2)
                    fold_r, fold_c = row + fold_i, col
                    anchor = (fold_r, fold_c + side * dist)
                    cells = [(r, col) for r in range(row, row + length)] + [anchor]
                if _near_existing(cells, occupied, h, w):
                    continue
                for r, c in cells[:-1]:
                    grid[r, c] = color
                ar, ac = anchor
                grid[ar, ac] = color
                occupied.update(cells)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("could not generate a valid grid")

