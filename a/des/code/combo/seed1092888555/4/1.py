import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _label_components(mask: np.ndarray):
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            cid += 1
            q = deque([(r, c)])
            labels[r, c] = cid
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            comps.append(pts)
    return labels, comps


def _find_legend(grid: np.ndarray, bg: int):
    h, w = grid.shape
    row = h - 1
    nz = np.flatnonzero(grid[row] != bg)
    assert len(nz) == 3, "legend row must contain exactly three non-bg pixels"
    assert nz[-1] - nz[0] == 2, "legend pixels must be contiguous"
    colors = [int(grid[row, c]) for c in nz]
    assert len(set(colors)) == 3, "legend colors must be distinct"
    return row, int(nz[0]), colors


def _extract_objects(grid: np.ndarray, bg: int, legend_row: int):
    work = grid.copy()
    work[legend_row, :] = bg
    fg = work != bg
    _labels, comps = _label_components(fg)
    assert comps, "expected at least one object"
    objects = []
    sizes = set()
    for pts in comps:
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        color = int(work[pts[0][0], pts[0][1]])
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        assert side_r == side_c, "objects must be square"
        side = side_r
        assert side in (2, 3), "object side must be 2 or 3"
        assert len(pts) == side * side, "object must be solid"
        assert np.all(work[r0 : r1 + 1, c0 : c1 + 1] == color), "object bbox must be filled"
        objects.append({"bbox": (r0, c0, r1, c1), "side": side, "color": color})
        sizes.add(side)
    assert len(sizes) == 1, "all objects must have equal size"
    return objects


def _adjacent(a, b) -> bool:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    same_rows = ar0 == br0 and ar1 == br1
    same_cols = ac0 == bc0 and ac1 == bc1
    if same_rows and (ac1 + 2 == bc0 or bc1 + 2 == ac0):
        return True
    if same_cols and (ar1 + 2 == br0 or br1 + 2 == ar0):
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_row, _legend_col, (cold_color, warm_color, connect_color) = _find_legend(g, bg)
    objects = _extract_objects(g, bg, legend_row)

    neighbors = {i: set() for i in range(len(objects))}
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _adjacent(objects[i], objects[j]):
                neighbors[i].add(j)
                neighbors[j].add(i)

    out = np.full_like(g, bg)
    for i, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["bbox"]
        fill = warm_color if neighbors[i] else cold_color
        out[r0 : r1 + 1, c0 : c1 + 1] = fill

    for i in range(len(objects)):
        for j in neighbors[i]:
            if j < i:
                continue
            ar0, ac0, ar1, ac1 = objects[i]["bbox"]
            br0, bc0, br1, bc1 = objects[j]["bbox"]
            if ar0 == br0 and ar1 == br1:
                gap_c = min(ac1, bc1) + 1
                out[ar0 : ar1 + 1, gap_c] = connect_color
            else:
                gap_r = min(ar1, br1) + 1
                out[gap_r, ac0 : ac1 + 1] = connect_color

    return out


def _object_cells(top: int, left: int, side: int):
    return {(r, c) for r in range(top, top + side) for c in range(left, left + side)}


def _can_place(existing, top: int, left: int, side: int, h: int, w: int, legend_row: int) -> bool:
    if top < 0 or left < 0 or top + side > legend_row or left + side > w:
        return False
    new_bbox = (top, left, top + side - 1, left + side - 1)
    for bbox in existing:
        r0, c0, r1, c1 = bbox
        if not (new_bbox[2] < r0 or r1 < new_bbox[0] or new_bbox[3] < c0 or c1 < new_bbox[1]):
            return False
        row_overlap = max(new_bbox[0], r0) <= min(new_bbox[2], r1)
        col_overlap = max(new_bbox[1], c0) <= min(new_bbox[3], c1)
        if row_overlap and (new_bbox[1] == c1 + 1 or c0 == new_bbox[3] + 1):
            return False
        if col_overlap and (new_bbox[0] == r1 + 1 or r0 == new_bbox[2] + 1):
            return False
        same_rows = new_bbox[0] == r0 and new_bbox[2] == r1
        same_cols = new_bbox[1] == c0 and new_bbox[3] == c1
        horizontal_gap = max(c0, new_bbox[1]) - min(c1, new_bbox[3]) - 1
        vertical_gap = max(r0, new_bbox[0]) - min(r1, new_bbox[2]) - 1
        if same_rows and horizontal_gap == 1:
            return False
        if same_cols and vertical_gap == 1:
            return False
    return True


def _count_neighbors(boxes):
    objs = [{"bbox": b} for b in boxes]
    deg = [0] * len(objs)
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if _adjacent(objs[i], objs[j]):
                deg[i] += 1
                deg[j] += 1
    return deg


def _gen_attempt():
    side = random.choice([2, 3])
    h = random.randint(max(8, side * 3 + 3), 18)
    w = random.randint(max(9, side * 4 + 3), 18)
    legend_row = h - 1

    palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
    square_color, cold_color, warm_color, connect_color = palette

    boxes = []

    orient = random.choice(["h", "v"])
    if orient == "h":
        row = random.randint(0, legend_row - side)
        left1_max = w - (2 * side + 1)
        if left1_max < 0:
            return None
        left1 = random.randint(0, left1_max)
        left2 = left1 + side + 1
        boxes.extend(
            [
                (row, left1, row + side - 1, left1 + side - 1),
                (row, left2, row + side - 1, left2 + side - 1),
            ]
        )
    else:
        col = random.randint(0, w - side)
        top1_max = legend_row - (2 * side + 1)
        if top1_max < 0:
            return None
        top1 = random.randint(0, top1_max)
        top2 = top1 + side + 1
        boxes.extend(
            [
                (top1, col, top1 + side - 1, col + side - 1),
                (top2, col, top2 + side - 1, col + side - 1),
            ]
        )

    target_count = random.randint(3, 7)
    attempts = 0
    while len(boxes) < target_count and attempts < 300:
        attempts += 1
        top = random.randint(0, legend_row - side)
        left = random.randint(0, w - side)
        if _can_place(boxes, top, left, side, h, w, legend_row):
            boxes.append((top, left, top + side - 1, left + side - 1))

    if len(boxes) < 3:
        return None

    deg = _count_neighbors(boxes)
    if not any(d == 0 for d in deg):
        return None
    if not any(d > 0 for d in deg):
        return None

    grid = np.zeros((h, w), dtype=int)
    for r0, c0, r1, c1 in boxes:
        grid[r0 : r1 + 1, c0 : c1 + 1] = square_color

    legend_col = random.randint(0, w - 3)
    grid[legend_row, legend_col : legend_col + 3] = np.array([cold_color, warm_color, connect_color], dtype=int)

    if len(set(map(int, np.unique(grid)))) < 5:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is None:
            continue
        out = main(grid)
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("failed to generate valid input")
