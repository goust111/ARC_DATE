import random
from collections import deque

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _contains(outer, inner):
    return outer["r0"] < inner["r0"] and outer["c0"] < inner["c0"] and outer["r1"] > inner["r1"] and outer["c1"] > inner["c1"]


def _frame_from_component(grid, cells):
    if len(cells) < 12:
        return None
    colors = {int(grid[r, c]) for r, c in cells}
    if len(colors) != 1 or BG in colors:
        return None
    color = next(iter(colors))
    r0, c0, r1, c1 = _bbox(cells)
    height, width = r1 - r0 + 1, c1 - c0 + 1
    if height < 5 or width < 5:
        return None
    cell_set = set(cells)
    perim = set()
    for c in range(c0, c1 + 1):
        perim.add((r0, c))
        perim.add((r1, c))
    for r in range(r0, r1 + 1):
        perim.add((r, c0))
        perim.add((r, c1))
    if any((r, c) not in perim for r, c in cell_set):
        return None
    missing = sorted(p for p in perim if p not in cell_set)
    if len(missing) != 1:
        return None
    mr, mc = missing[0]
    if (mr, mc) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if mr == r0:
        side, coord = "up", mc
    elif mr == r1:
        side, coord = "down", mc
    elif mc == c0:
        side, coord = "left", mr
    elif mc == c1:
        side, coord = "right", mr
    else:
        return None
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "color": color,
        "side": side,
        "coord": coord,
        "cells": cell_set,
    }


def _find_legend(grid):
    h, w = grid.shape
    best = None
    for c in list(range(min(3, w))) + list(range(max(0, w - 3), w)):
        r = 0
        while r < h:
            if int(grid[r, c]) == BG:
                r += 1
                continue
            start = r
            colors = []
            while r < h and int(grid[r, c]) != BG:
                left_clear = c == 0 or int(grid[r, c - 1]) == BG
                right_clear = c == w - 1 or int(grid[r, c + 1]) == BG
                if not (left_clear and right_clear):
                    break
                colors.append(int(grid[r, c]))
                r += 1
            if len(colors) >= 2 and len(set(colors)) == len(colors):
                cand = {"cells": {(rr, c) for rr in range(start, start + len(colors))}, "colors": colors}
                if best is None or len(colors) > len(best["colors"]):
                    best = cand
            r = max(r + 1, start + 1)
    if best is None:
        colors = [2, 3, 4, 6]
        return {"cells": set(), "colors": colors}
    return best


def _analyze(grid):
    non_bg = grid != BG
    comps = _components(non_bg)
    frames = []
    for cells in comps:
        fr = _frame_from_component(grid, cells)
        if fr is not None:
            frames.append(fr)
    frames.sort(key=lambda f: (f["r1"] - f["r0"] + 1) * (f["c1"] - f["c0"] + 1), reverse=True)
    legend = _find_legend(grid)
    frame_cells = set().union(*(f["cells"] for f in frames)) if frames else set()
    blocked = frame_cells | legend["cells"]

    fragments = []
    for cells in comps:
        s = set(cells)
        if s & blocked:
            continue
        if 2 <= len(cells) <= 4:
            fragments.append({"cells": s, "bbox": _bbox(cells), "color": int(grid[cells[0]])})

    parents = {}
    for i, inner in enumerate(frames):
        containing = [j for j, outer in enumerate(frames) if i != j and _contains(outer, inner)]
        if containing:
            parents[i] = min(containing, key=lambda j: (frames[j]["r1"] - frames[j]["r0"]) * (frames[j]["c1"] - frames[j]["c0"]))
    children = set(parents.keys())
    inners = [i for i in range(len(frames)) if i not in set(parents.values())]
    return frames, parents, inners, fragments, legend["colors"]


def _chain_for_frame(idx, frames, parents):
    chain = [idx]
    while idx in parents:
        idx = parents[idx]
        chain.append(idx)
    return chain


def _fragment_inside(frame, frag):
    r0, c0, r1, c1 = frag["bbox"]
    return frame["r0"] < r0 and frame["c0"] < c0 and frame["r1"] > r1 and frame["c1"] > c1


def _norm_shape(cells):
    r0, c0, _, _ = _bbox(cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _place_group(out, outer, fragments, color):
    if not fragments:
        return
    h, w = out.shape
    side = outer["side"]
    coord = outer["coord"]
    data = []
    for frag in fragments:
        cells = frag["cells"]
        r0, c0, r1, c1 = frag["bbox"]
        shape = _norm_shape(cells)
        data.append({"shape": shape, "height": r1 - r0 + 1, "width": c1 - c0 + 1, "bbox": frag["bbox"]})
        for r, c in cells:
            out[r, c] = BG

    if side in ("up", "down"):
        data.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        total_w = sum(x["width"] for x in data)
        max_h = max(x["height"] for x in data)
        row = outer["r0"] - max_h if side == "up" else outer["r1"] + 1
        col = max(0, min(w - total_w, coord - total_w // 2))
        if not (0 <= row and row + max_h <= h):
            return
        for item in data:
            rr = row + (max_h - item["height"] if side == "up" else 0)
            for dr, dc in item["shape"]:
                out[rr + dr, col + dc] = color
            col += item["width"]
    else:
        data.sort(key=lambda x: (x["bbox"][0], x["bbox"][1]))
        total_h = sum(x["height"] for x in data)
        max_w = max(x["width"] for x in data)
        col = outer["c0"] - max_w if side == "left" else outer["c1"] + 1
        row = max(0, min(h - total_h, coord - total_h // 2))
        if not (0 <= col and col + max_w <= w):
            return
        for item in data:
            cc = col + (max_w - item["width"] if side == "left" else 0)
            for dr, dc in item["shape"]:
                out[row + dr, cc + dc] = color
            row += item["height"]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    frames, parents, inners, fragments, legend = _analyze(grid)
    if not frames:
        return out
    assigned = set()
    for inner_idx in inners:
        chain = _chain_for_frame(inner_idx, frames, parents)
        inner = frames[inner_idx]
        group_frags = []
        for i, frag in enumerate(fragments):
            if i in assigned:
                continue
            if _fragment_inside(inner, frag):
                group_frags.append(frag)
                assigned.add(i)
        if not group_frags:
            continue
        depth = len(chain)
        color = legend[min(depth - 1, len(legend) - 1)]
        outer = frames[chain[-1]]
        _place_group(out, outer, group_frags, color)
    return out


def _draw_frame(grid, r0, c0, r1, c1, color, side, coord):
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if side == "up":
        grid[r0, coord] = BG
    elif side == "down":
        grid[r1, coord] = BG
    elif side == "left":
        grid[coord, c0] = BG
    else:
        grid[coord, c1] = BG


SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
]


def _try_place_shape(grid, occupied, rmin, cmin, rmax, cmax, color):
    shape = random.choice(SHAPES)
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    if rmax - rmin + 1 < sh or cmax - cmin + 1 < sw:
        return False
    for _ in range(100):
        r0 = random.randint(rmin, rmax - sh + 1)
        c0 = random.randint(cmin, cmax - sw + 1)
        cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
        expanded = set(cells)
        for r, c in list(cells):
            expanded.update(_neighbors4(r, c, grid.shape[0], grid.shape[1]))
        if occupied & expanded:
            continue
        for r, c in cells:
            grid[r, c] = color
        occupied.update(cells)
        return True
    return False


def _gen_attempt():
    h = random.randint(28, 34)
    w = random.randint(28, 34)
    depth = random.randint(2, 4)
    grid = np.zeros((h, w), dtype=int)

    legend_colors = random.sample([1, 2, 3, 4, 6, 9], depth)
    legend_col = 1 if random.random() < 0.5 else w - 2
    legend_start = random.randint(1, h - depth - 2)
    for i, color in enumerate(legend_colors):
        grid[legend_start + i, legend_col] = color

    outer_h = random.randint(17, min(22, h - 10))
    outer_w = random.randint(17, min(22, w - 10))
    r0 = random.randint(5, h - outer_h - 5)
    c0 = random.randint(5, w - outer_w - 5)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    sides = list(DIRS)
    frame_colors = random.sample([5, 7, 8, 9, 6], depth)
    bboxes = []
    for level in range(depth):
        rr0, cc0 = r0 + 3 * level, c0 + 3 * level
        rr1, cc1 = r1 - 3 * level, c1 - 3 * level
        if rr1 - rr0 < 5 or cc1 - cc0 < 5:
            return None
        side = random.choice(sides)
        if side in ("up", "down"):
            coord = random.randint(cc0 + 2, cc1 - 2)
        else:
            coord = random.randint(rr0 + 2, rr1 - 2)
        _draw_frame(grid, rr0, cc0, rr1, cc1, frame_colors[level], side, coord)
        bboxes.append((rr0, cc0, rr1, cc1))

    ir0, ic0, ir1, ic1 = bboxes[-1]
    occupied = set()
    frag_colors = [c for c in range(1, 10) if c not in frame_colors]
    for _ in range(random.randint(2, 4)):
        ok = _try_place_shape(
            grid,
            occupied,
            ir0 + 2,
            ic0 + 2,
            ir1 - 2,
            ic1 - 2,
            random.choice(frag_colors),
        )
        if not ok:
            return None

    before = grid.copy()
    out = main(before)
    if not isinstance(out, np.ndarray) or out.shape != before.shape:
        return None
    if np.array_equal(out, before):
        return None
    return before


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
