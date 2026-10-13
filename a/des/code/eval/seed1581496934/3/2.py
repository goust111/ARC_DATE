import random
from collections import deque

import numpy as np


BG = 8
COLORS = [1, 2, 3, 4, 6, 7, 9]


def _component_cells(grid: np.ndarray, excluded: set[tuple[int, int]]):
    h, w = grid.shape
    seen = set(excluded)
    comps = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen.add((r, c))
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                        continue
                    if int(grid[nr, nc]) == color:
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _anchors(grid: np.ndarray):
    h, w = grid.shape
    result = []
    excluded = set()
    edge_cells = []
    for c in range(w):
        edge_cells.append((0, c, "top"))
        edge_cells.append((h - 1, c, "bottom"))
    for r in range(1, h - 1):
        edge_cells.append((r, 0, "left"))
        edge_cells.append((r, w - 1, "right"))

    for r, c, edge in edge_cells:
        color = int(grid[r, c])
        if color == BG:
            continue
        ok = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
                ok = False
                break
        if ok:
            result.append({"pos": (r, c), "edge": edge, "color": color})
            excluded.add((r, c))
    return result, excluded


def _shape(comp):
    rows = [r for r, _ in comp["cells"]]
    cols = [c for _, c in comp["cells"]]
    top, left = min(rows), min(cols)
    height = max(rows) - top + 1
    width = max(cols) - left + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in comp["cells"]:
        mask[r - top, c - left] = True
    return top, left, mask


def _find_destination(rect: np.ndarray, anchors, grid_shape):
    h, w = grid_shape
    rh, rw = rect.shape
    for a in anchors:
        for b in anchors:
            if a is b or a["edge"] != b["edge"]:
                continue
            (ar, ac), (br, bc) = a["pos"], b["pos"]
            if a["edge"] == "top" and ar == br == 0 and abs(ac - bc) == rw - 1:
                left = min(ac, bc)
                if int(rect[0, ac - left]) == a["color"] and int(rect[0, bc - left]) == b["color"]:
                    return 0, left
            if a["edge"] == "bottom" and ar == br == h - 1 and abs(ac - bc) == rw - 1:
                left = min(ac, bc)
                top = h - rh
                if int(rect[-1, ac - left]) == a["color"] and int(rect[-1, bc - left]) == b["color"]:
                    return top, left
            if a["edge"] == "left" and ac == bc == 0 and abs(ar - br) == rh - 1:
                top = min(ar, br)
                if int(rect[ar - top, 0]) == a["color"] and int(rect[br - top, 0]) == b["color"]:
                    return top, 0
            if a["edge"] == "right" and ac == bc == w - 1 and abs(ar - br) == rh - 1:
                top = min(ar, br)
                left = w - rw
                if int(rect[ar - top, -1]) == a["color"] and int(rect[br - top, -1]) == b["color"]:
                    return top, left
    return None


def main(grid: np.ndarray) -> np.ndarray:
    anchors, excluded = _anchors(grid)
    comps = _component_cells(grid, excluded)
    shaped = []
    for comp in comps:
        top, left, mask = _shape(comp)
        shaped.append({**comp, "top": top, "left": left, "mask": mask})

    out = grid.copy()
    used = set()
    placements = []
    for i, a in enumerate(shaped):
        if i in used:
            continue
        for j, b in enumerate(shaped):
            if j <= i or j in used or a["color"] == b["color"]:
                continue
            if a["mask"].shape != b["mask"].shape:
                continue
            if not np.all(a["mask"] ^ b["mask"]):
                continue
            rect = np.full(a["mask"].shape, b["color"], dtype=int)
            rect[a["mask"]] = a["color"]
            dest = _find_destination(rect, anchors, grid.shape)
            if dest is None:
                rect2 = np.full(a["mask"].shape, a["color"], dtype=int)
                rect2[b["mask"]] = b["color"]
                dest = _find_destination(rect2, anchors, grid.shape)
                rect = rect2 if dest is not None else rect
            if dest is None:
                continue
            used.update({i, j})
            placements.append((dest, rect, a["cells"] + b["cells"]))
            break

    by_color = {}
    for idx, comp in enumerate(shaped):
        by_color.setdefault(comp["color"], []).append((idx, comp))

    for a in anchors:
        for b in anchors:
            if a is b or a["edge"] != b["edge"]:
                continue
            if a["color"] not in by_color or b["color"] not in by_color:
                continue
            (ar, ac), (br, bc) = a["pos"], b["pos"]
            if a["edge"] in ("top", "bottom"):
                if ar != br:
                    continue
                total_width = abs(ac - bc) + 1
                left_anchor, right_anchor = (a, b) if ac < bc else (b, a)
                dest = (0, min(ac, bc)) if a["edge"] == "top" else (grid.shape[0], min(ac, bc))
                for li, left_comp in by_color.get(left_anchor["color"], []):
                    if li in used:
                        continue
                    for ri, right_comp in by_color.get(right_anchor["color"], []):
                        if ri in used or li == ri:
                            continue
                        lm, rm = left_comp["mask"], right_comp["mask"]
                        if lm.shape[0] != rm.shape[0] or lm.shape[1] + rm.shape[1] != total_width:
                            continue
                        rect = np.full((lm.shape[0], total_width), BG, dtype=int)
                        rect[:, : lm.shape[1]][lm] = left_comp["color"]
                        rect[:, total_width - rm.shape[1] :][rm] = right_comp["color"]
                        if np.any(rect == BG):
                            continue
                        top = 0 if a["edge"] == "top" else grid.shape[0] - rect.shape[0]
                        used.update({li, ri})
                        placements.append(((top, dest[1]), rect, left_comp["cells"] + right_comp["cells"]))
                        break
                    if li in used:
                        break
            else:
                if ac != bc:
                    continue
                total_height = abs(ar - br) + 1
                upper_anchor, lower_anchor = (a, b) if ar < br else (b, a)
                for ui, upper_comp in by_color.get(upper_anchor["color"], []):
                    if ui in used:
                        continue
                    for li, lower_comp in by_color.get(lower_anchor["color"], []):
                        if li in used or ui == li:
                            continue
                        um, lm = upper_comp["mask"], lower_comp["mask"]
                        if um.shape[1] != lm.shape[1] or um.shape[0] + lm.shape[0] != total_height:
                            continue
                        rect = np.full((total_height, um.shape[1]), BG, dtype=int)
                        rect[: um.shape[0], :][um] = upper_comp["color"]
                        rect[total_height - lm.shape[0] :, :][lm] = lower_comp["color"]
                        if np.any(rect == BG):
                            continue
                        left = 0 if a["edge"] == "left" else grid.shape[1] - rect.shape[1]
                        used.update({ui, li})
                        placements.append(((min(ar, br), left), rect, upper_comp["cells"] + lower_comp["cells"]))
                        break
                    if ui in used:
                        break

    for _, _, cells in placements:
        for r, c in cells:
            out[r, c] = BG
    for (top, left), rect, _ in placements:
        rh, rw = rect.shape
        out[top : top + rh, left : left + rw] = rect
    return out


def _connected(mask: np.ndarray) -> bool:
    cells = list(zip(*np.nonzero(mask)))
    if not cells:
        return False
    want = set((int(r), int(c)) for r, c in cells)
    queue = deque([cells[0]])
    seen = {(int(cells[0][0]), int(cells[0][1]))}
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = int(r) + dr, int(c) + dc
            if (nr, nc) in want and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen == want


def _full_bbox(mask: np.ndarray) -> bool:
    rows, cols = np.nonzero(mask)
    return rows.min() == 0 and cols.min() == 0 and rows.max() == mask.shape[0] - 1 and cols.max() == mask.shape[1] - 1


def _random_mask(height: int, width: int) -> np.ndarray:
    target = random.randint(height * width // 3, height * width * 2 // 3)
    for _ in range(800):
        cells = {(0, 0), (height - 1, width - 1)}
        frontier = set(cells)
        while len(cells) < target:
            r, c = random.choice(tuple(frontier))
            choices = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    choices.append((nr, nc))
            if not choices:
                frontier.discard((r, c))
                if not frontier:
                    break
                continue
            nxt = random.choice(choices)
            cells.add(nxt)
            frontier.add(nxt)
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        comp = ~mask
        if (
            mask[0, 0]
            and comp[0, width - 1]
            and _connected(mask)
            and _connected(comp)
            and _full_bbox(mask)
            and _full_bbox(comp)
        ):
            return mask
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    mask[0, 0] = True
    mask[1:, -1] = True
    mask[0, -1] = False
    if _connected(mask) and _connected(~mask) and _full_bbox(~mask):
        return mask
    raise RuntimeError("failed to build complementary masks")


def _can_place(grid: np.ndarray, top: int, left: int, mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = grid[max(0, top - 1) : min(grid.shape[0], top + h + 1), max(0, left - 1) : min(grid.shape[1], left + w + 1)]
    return top >= 1 and left >= 1 and top + h < grid.shape[0] - 1 and left + w < grid.shape[1] - 1 and np.all(area == BG)


def _place_piece(grid: np.ndarray, mask: np.ndarray, color: int):
    h, w = mask.shape
    for _ in range(300):
        top = random.randint(3, grid.shape[0] - h - 2)
        left = random.randint(1, grid.shape[1] - w - 2)
        if _can_place(grid, top, left, mask):
            grid[top : top + h, left : left + w][mask] = color
            return
    raise RuntimeError("failed to place piece")


def _gen_attempt() -> np.ndarray:
    height = random.randint(20, 28)
    width = random.randint(22, 30)
    grid = np.full((height, width), BG, dtype=int)
    pair_count = random.randint(1, min(3, max(1, width // 8)))
    cursor = random.randint(1, 3)
    colors = random.sample(COLORS, pair_count * 2)

    for k in range(pair_count):
        rh = random.randint(3, 6)
        rw = random.randint(5, 8)
        if cursor + rw >= width - 1:
            break
        split = random.randint(2, rw - 2)
        left_mask = np.ones((rh, split), dtype=bool)
        right_mask = np.ones((rh, rw - split), dtype=bool)
        c1, c2 = colors[2 * k], colors[2 * k + 1]
        grid[0, cursor] = c1
        grid[0, cursor + rw - 1] = c2
        _place_piece(grid, left_mask, c1)
        _place_piece(grid, right_mask, c2)
        cursor += rw + random.randint(2, 4)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = _gen_attempt()
        out = main(grid)
        if isinstance(out, np.ndarray) and out.shape == grid.shape and np.any(out != grid):
            return grid
    raise RuntimeError("failed to generate valid complementary-anchor sample")
