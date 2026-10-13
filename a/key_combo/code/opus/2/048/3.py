from __future__ import annotations

import numpy as np

RED = 2
YELLOW = 4
BLUE = 1
GREEN = 3


def _neighbors4(r: int, c: int, H: int, W: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < H:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < W:
        yield r, c + 1


def _components(mask: np.ndarray):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for rr, cc in _neighbors4(r, c, H, W):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _component_map(mask: np.ndarray):
    comps = _components(mask)
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=int)
    for i, comp in enumerate(comps, start=1):
        for r, c in comp:
            lab[r, c] = i
    return lab


def _count_bends(mask: np.ndarray) -> int:
    # Count 2x2 L patterns (three-pixel corners), unique by position
    H, W = mask.shape
    cnt = 0
    for r in range(H - 1):
        for c in range(W - 1):
            sub = mask[r : r + 2, c : c + 2]
            if int(sub.sum()) == 3:
                cnt += 1
    return cnt


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    H, W = grid.shape
    out = np.array(grid, copy=True)
    yellow_pos = np.argwhere(grid == YELLOW)
    if len(yellow_pos) == 0:
        return out
    sy, sx = map(int, yellow_pos[0])

    blue_mask = grid == BLUE
    comp_label = _component_map(blue_mask)
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    bend_to_dir = {1: (-1, 0), 2: (0, 1), 3: (1, 0)}

    # initial ray from marker right, start from neighboring red cell
    if sx + 1 >= W:
        return out
    r, c = sy, sx + 1
    dr, dc = 0, 1

    def _find_component_id(x: int, y: int):
        return comp_label[x, y] if 0 <= x < H and 0 <= y < W else 0

    while 0 <= r < H and 0 <= c < W:
        cid = _find_component_id(r, c)
        out[r, c] = RED
        if cid == 0:
            r += dr
            c += dc
            continue

        # activate shape
        shape_mask = comp_label == cid
        out[shape_mask] = RED
        bends = _count_bends(shape_mask)
        if bends not in bend_to_dir:
            break

        # find unique green port adjacent to the component
        ports = []
        pts = np.argwhere(shape_mask)
        for rr, cc in pts:
            for nr, nc in _neighbors4(int(rr), int(cc), H, W):
                if out[nr, nc] == GREEN:
                    ports.append((nr, nc))
        ports = list(dict.fromkeys(ports))
        if len(ports) == 0:
            break
        pr, pc = ports[0]
        out[pr, pc] = RED
        dr, dc = bend_to_dir[bends]
        r, c = pr, pc
        continue
    return out


def _template_count2():
    return np.array([(0, 0), (1, 0), (1, 1), (2, 0)], dtype=int)


def generate_input() -> np.ndarray:
    out = np.zeros((20, 20), dtype=int)
    rng = np.random.default_rng()
    sy = int(rng.integers(5, 15))
    out[sy, 0] = YELLOW
    out[sy, 1] = RED
    out[sy, 2] = 0
    shape = _template_count2()
    gap = int(rng.integers(3, 6))
    # three independent shapes, all map to bend count 2 -> chain to right
    anchors = [(sy - 1, 4 + i * (3 + gap)) for i in range(3)]
    for ar, ac in anchors:
        if ac + 2 >= 18:
            break
        for r, c in shape:
            out[ar + r, ac + c] = BLUE
        # port on right side (shape can only be activated right)
        out[ar + 1, ac + 2] = GREEN

    # extra maroon noise-free background only to keep grid sparse
    return out
