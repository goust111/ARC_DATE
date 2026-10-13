import random
from collections import Counter, deque
from typing import List, Tuple

import numpy as np


def _components(mask: np.ndarray):
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for sr in range(H):
        for sc in range(W):
            if not mask[sr, sc] or visited[sr, sc]:
                continue

            queue = deque([(sr, sc)])
            visited[sr, sc] = True
            comp = []
            min_r, max_r = sr, sr
            min_c, max_c = sc, sc

            while queue:
                r, c = queue.popleft()
                comp.append((r, c))

                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c

                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        continue
                    if visited[nr, nc] or not mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            yield comp, (min_r, max_r, min_c, max_c)


def _stripe_score_line(values: List[int], bg: int):
    if not values:
        return 0.0, 0.0

    filtered = [v for v in values if v != -1]
    if len(filtered) < 2:
        return 0.0, 0.0

    changes = 0
    same = 0
    for i in range(len(filtered) - 1):
        if filtered[i] == filtered[i + 1]:
            same += 1
        else:
            changes += 1

    total = len(filtered) - 1
    return changes / total, same / total


def _infer_stripe_direction(block: np.ndarray, bg: int) -> str:
    rows = []
    for r in range(block.shape[0]):
        vals = block[r][block[r] != bg]
        if vals.size == 0:
            rows.append(-1)
            continue
        rows.append(int(Counter(vals.tolist()).most_common(1)[0][0]))

    cols = []
    for c in range(block.shape[1]):
        vals = block[:, c][block[:, c] != bg]
        if vals.size == 0:
            cols.append(-1)
            continue
        cols.append(int(Counter(vals.tolist()).most_common(1)[0][0]))

    row_change, row_same = _stripe_score_line(rows, bg)
    col_change, col_same = _stripe_score_line(cols, bg)

    if row_change > col_change + 1e-9:
        return "H"
    if col_change > row_change + 1e-9:
        return "V"
    if row_same < col_same:
        return "H"
    if col_same < row_same:
        return "V"
    return "H"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add protruding structures around each striped rectangle according to its internal hole geometry.

    Input structure:
    - A dominant background color fills the canvas.
    - Each object is a medium rectangle (about 5x5 to 7x7) using two stripe colors, with one small internal background hole of size 2-4.

    Rule:
    - Detect each rectangle and infer stripe direction from its interior pattern.
    - If stripes are horizontal, grow protrusions upward and downward; if stripes are vertical, grow protrusions left and right.
    - Project the hole onto the growth axis-perpendicular direction to determine protrusion width, and use hole pixel count as protrusion length.
    - Paint protrusions with the main stripe color; fill the hole itself with the secondary stripe color.

    Key insight:
    - Stripe orientation determines growth direction, while the hole geometry separately controls protrusion width and extension length.
    """

    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    out = g.copy()
    bg = int(np.bincount(g.ravel()).argmax())
    non_bg = g != bg

    for comp, (r0, r1, c0, c1) in _components(non_bg):
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue

        area = len(comp)
        if area > h * w - 2 or area < h * w - 4:
            # outside expected hole size 2..4
            continue

        block = g[r0 : r1 + 1, c0 : c1 + 1]
        hole_expected = h * w - area
        if hole_expected < 2 or hole_expected > 4:
            continue

        local_bg = block == bg
        hole_cells = []
        touched = np.zeros_like(local_bg, dtype=bool)
        inner_dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for rr in range(h):
            for cc in range(w):
                if not local_bg[rr, cc] or touched[rr, cc]:
                    continue

                q = deque([(rr, cc)])
                touched[rr, cc] = True
                cells = [(rr, cc)]
                touches_border = rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1

                while q:
                    cr, cc2 = q.popleft()
                    for dr, dc in inner_dirs:
                        nr, nc = cr + dr, cc2 + dc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            continue
                        if not local_bg[nr, nc] or touched[nr, nc]:
                            continue
                        touched[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
                        if nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1:
                            touches_border = True

                if not touches_border:
                    hole_cells.append(cells)

        if len(hole_cells) != 1:
            continue

        hole = hole_cells[0]
        if len(hole) != hole_expected or len(hole) < 2 or len(hole) > 4:
            continue

        non_bg_cells = block[block != bg]
        if non_bg_cells.size == 0:
            continue
        colors, cnts = np.unique(non_bg_cells, return_counts=True)
        if colors.size == 0:
            continue
        order = np.argsort(cnts)[::-1]
        main_color = int(colors[order[0]])
        if colors.size >= 2:
            sec_color = int(colors[order[1]])
        else:
            sec_color = main_color

        ori = _infer_stripe_direction(block, bg)
        length = len(hole)
        hole_rows = [x[0] for x in hole]
        hole_cols = [x[1] for x in hole]

        if ori == "H":
            left = min(hole_cols)
            right = max(hole_cols)
            for step in range(1, length + 1):
                up_r = r0 - step
                down_r = r1 + step
                if 0 <= up_r < H:
                    out[up_r, c0 + left : c0 + right + 1] = main_color
                if 0 <= down_r < H:
                    out[down_r, c0 + left : c0 + right + 1] = main_color
        else:
            top = min(hole_rows)
            bot = max(hole_rows)
            for step in range(1, length + 1):
                left_c = c0 - step
                right_c = c1 + step
                if 0 <= left_c < W:
                    out[r0 + top : r0 + bot + 1, left_c] = main_color
                if 0 <= right_c < W:
                    out[r0 + top : r0 + bot + 1, right_c] = main_color

        for rr, cc in hole:
            out[r0 + rr, c0 + cc] = sec_color

    return out


def _normalize(coords: np.ndarray) -> List[Tuple[int, int]]:
    coords = coords.copy()
    coords[:, 0] -= coords[:, 0].min()
    coords[:, 1] -= coords[:, 1].min()
    return [(int(r), int(c)) for r, c in coords.tolist()]


def _random_hole(coords_rng: random.Random, h: int, w: int) -> List[Tuple[int, int]]:
    inner_h = h - 2
    inner_w = w - 2
    if inner_h <= 0 or inner_w <= 0:
        return []

    shapes = {
        2: [
            [(0, 0), (0, 1)],
            [(0, 0), (1, 0)],
        ],
        3: [
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 0), (1, 0), (0, 1)],
        ],
        4: [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 1), (1, 0), (1, 1), (1, 2)],
        ],
    }

    for _ in range(120):
        size = coords_rng.randint(2, 4)
        base = coords_rng.choice(shapes[size])
        arr = np.array(base, dtype=int)

        if coords_rng.random() < 0.5:
            arr = arr[:, [1, 0]]

        if coords_rng.random() < 0.5:
            arr[:, 0] = arr[:, 0].max() - arr[:, 0]
        if coords_rng.random() < 0.5:
            arr[:, 1] = arr[:, 1].max() - arr[:, 1]

        arr = _normalize(arr)
        rs = [p[0] for p in arr]
        cs = [p[1] for p in arr]
        ph = max(rs) - min(rs) + 1
        pw = max(cs) - min(cs) + 1

        if ph > inner_h or pw > inner_w:
            continue

        r0 = coords_rng.randint(1, inner_h - ph + 1)
        c0 = coords_rng.randint(1, inner_w - pw + 1)

        return [(r0 + rr, c0 + cc) for rr, cc in arr]

    return []


def _rect_intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    a_r0, a_r1, a_c0, a_c1 = a
    b_r0, b_r1, b_c0, b_c1 = b
    return not (
        a_r1 < b_r0
        or b_r1 < a_r0
        or a_c1 < b_c0
        or b_c1 < a_c0
    )


def generate_input() -> np.ndarray:
    palette = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    pad = 4

    for _ in range(260):
        H = random.randint(20, 30)
        W = random.randint(20, 30)

        p = random.random()
        if p < 0.2:
            n_obj = 1
        elif p < 0.8:
            n_obj = 2
        else:
            n_obj = 3

        if min(H, W) < 24:
            n_obj = min(n_obj, 2)

        objects = []
        reserved = []
        used = set()
        success = True

        for _obj_i in range(n_obj):
            h = random.randint(5, 7)
            w = random.randint(5, 7)
            ori = random.choice(("H", "V"))

            hole = []
            for _ in range(80):
                hole = _random_hole(random, h, w)
                if hole:
                    break
            if not hole:
                success = False
                break

            y_min = pad
            y_max = H - h - pad
            x_min = pad
            x_max = W - w - pad
            if y_max < y_min or x_max < x_min:
                success = False
                break

            placed = False
            for _ in range(260):
                y = random.randint(y_min, y_max)
                x = random.randint(x_min, x_max)
                local = (y - pad, y + h - 1 + pad, x - pad, x + w - 1 + pad)

                conflict = False
                for prev in reserved:
                    if _rect_intersects(local, prev):
                        conflict = True
                        break
                if conflict:
                    continue

                used_main = random.choice([c for c in palette if c not in used] or palette)
                used_sec = random.choice([c for c in palette if c != used_main])
                used.update([used_main, used_sec])

                objects.append(
                    {
                        "r": y,
                        "c": x,
                        "h": h,
                        "w": w,
                        "ori": ori,
                        "main": used_main,
                        "sec": used_sec,
                        "hole": hole,
                    }
                )
                reserved.append(local)
                placed = True
                break

            if not placed:
                success = False
                break

        if not success:
            continue

        grid = np.zeros((H, W), dtype=int)

        for obj in objects:
            r, c, h, w = obj["r"], obj["c"], obj["h"], obj["w"]
            main_col = obj["main"]
            sec_col = obj["sec"]
            ori = obj["ori"]
            start = random.randint(0, 1)

            if ori == "H":
                for rr in range(h):
                    col = main_col if ((rr + start) % 2 == 0) else sec_col
                    grid[r + rr, c : c + w] = col
            else:
                for cc in range(w):
                    col = main_col if ((cc + start) % 2 == 0) else sec_col
                    grid[r : r + h, c + cc] = col

            for rr, cc in obj["hole"]:
                grid[r + rr, c + cc] = 0

        return grid

    # Fallback: always-valid single object to avoid hard failure.
    H = W = 20
    grid = np.zeros((H, W), dtype=int)
    h, w = 5, 5
    r, c = 6, 7
    for rr in range(h):
        col = 1 if rr % 2 == 0 else 2
        grid[r + rr, c : c + w] = col
    for cc in range(2):
        grid[r + 2, c + cc + 1] = 0
    return grid
