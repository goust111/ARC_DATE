import random
from collections import deque

import numpy as np


BG = 0
TOWER_COLORS = [1, 5, 6, 7, 8, 9]
RED = 2
GREEN = 3
YELLOW = 4

DIRS = ["up", "right", "down", "left"]
ROT_K = {"up": 0, "right": -1, "down": 2, "left": 1}


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(coords)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_receivers(g):
    h, w = g.shape
    receivers = []
    for coords in _components(g == GREEN):
        r0, c0, r1, c1 = _bbox(coords)
        fh, fw = r1 - r0 + 1, c1 - c0 + 1
        if fh < 3 or fw < 3:
            continue
        frame = g[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(frame[0, :] == GREEN)
            and np.all(frame[-1, :] == GREEN)
            and np.all(frame[:, 0] == GREEN)
            and np.all(frame[:, -1] == GREEN)
        ):
            continue
        if np.any(frame[1:-1, 1:-1] == GREEN):
            continue

        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        candidates = []
        checks = {
            "up": (r0 - 1, mid_c),
            "right": (mid_r, c1 + 1),
            "down": (r1 + 1, mid_c),
            "left": (mid_r, c0 - 1),
        }
        for side, (rr, cc) in checks.items():
            if 0 <= rr < h and 0 <= cc < w and g[rr, cc] == RED:
                candidates.append(side)
        if len(candidates) != 1:
            continue
        receivers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "dir": candidates[0],
                "inner_shape": (fh - 2, fw - 2),
            }
        )
    return receivers


def _analyze_tower(g, color, coords):
    rmin, cmin, rmax, cmax = _bbox(coords)
    for ant_len in range(1, 4):
        br0 = rmin + ant_len
        if br0 + 2 > rmax or cmin + 2 > cmax:
            continue
        mid_c = (cmin + cmax) // 2
        if any(g[r, mid_c] != color for r in range(rmin, br0)):
            continue

        antenna_ok = True
        for r in range(rmin, br0):
            for c in range(cmin, cmax + 1):
                if c != mid_c and g[r, c] == color:
                    antenna_ok = False
        if not antenna_ok:
            continue

        body = g[br0 : rmax + 1, cmin : cmax + 1]
        if not (
            np.all(body[0, :] == color)
            and np.all(body[-1, :] == color)
            and np.all(body[:, 0] == color)
            and np.all(body[:, -1] == color)
        ):
            continue

        core = g[br0 + 1 : rmax, cmin + 1 : cmax]
        if core.size == 0 or not np.any(core == YELLOW):
            continue
        if np.any((core != color) & (core != YELLOW)):
            continue
        pattern = np.where(core == YELLOW, YELLOW, BG)
        return {
            "bbox": (rmin, cmin, rmax, cmax),
            "body_bbox": (br0, cmin, rmax, cmax),
            "pattern": pattern,
            "color": color,
        }
    return None


def _find_towers(g):
    towers = []
    for color in TOWER_COLORS:
        for coords in _components(g == color):
            tower = _analyze_tower(g, color, coords)
            if tower is not None:
                towers.append(tower)
    return towers


def _direction_distance(receiver, tower):
    rr0, rc0, rr1, rc1 = receiver["bbox"]
    tr0, tc0, tr1, tc1 = tower["bbox"]
    side = receiver["dir"]
    if side in ("up", "down"):
        if tc1 < rc0 or tc0 > rc1:
            return None
        if side == "up" and tr1 < rr0:
            return rr0 - tr1
        if side == "down" and tr0 > rr1:
            return tr0 - rr1
    else:
        if tr1 < rr0 or tr0 > rr1:
            return None
        if side == "left" and tc1 < rc0:
            return rc0 - tc1
        if side == "right" and tc0 > rc1:
            return tc0 - rc1
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    towers = _find_towers(g)
    receivers = _find_receivers(g)

    for receiver in receivers:
        candidates = []
        for tower in towers:
            dist = _direction_distance(receiver, tower)
            if dist is None:
                continue
            rotated = np.rot90(tower["pattern"], ROT_K[receiver["dir"]])
            if rotated.shape == receiver["inner_shape"]:
                candidates.append((dist, tower, rotated))
        if not candidates:
            continue
        _, _, pattern = min(candidates, key=lambda item: item[0])
        r0, c0, r1, c1 = receiver["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[:, :] = BG
        inner[pattern == YELLOW] = YELLOW
    return out


def _expanded_free(occ, r0, c0, r1, c1):
    h, w = occ.shape
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    return not occ[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any()


def _reserve(occ, r0, c0, r1, c1):
    occ[r0 : r1 + 1, c0 : c1 + 1] = True


def _random_pattern(h, w):
    mask = np.zeros((h, w), dtype=bool)
    count = random.randint(1, max(1, min(h * w - 1, h + w)))
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for r, c in cells[:count]:
        mask[r, c] = True
    return mask


def _draw_tower(grid, occ, r0, c0, core_h, core_w, color, pattern_mask, ant_len):
    body_h = core_h + 2
    body_w = core_w + 2
    br0 = r0 + ant_len
    r1 = br0 + body_h - 1
    c1 = c0 + body_w - 1
    grid[br0 : r1 + 1, c0 : c1 + 1] = color
    grid[br0 + 1 : r1, c0 + 1 : c1][pattern_mask] = YELLOW
    mid_c = c0 + body_w // 2
    grid[r0:br0, mid_c] = color
    _reserve(occ, r0, c0, r1, c1)
    return (r0, c0, r1, c1)


def _draw_receiver(grid, occ, r0, c0, inner_h, inner_w, side):
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0 : r1 + 1, c0] = GREEN
    grid[r0 : r1 + 1, c1] = GREEN
    grid[r0, c0 : c1 + 1] = GREEN
    grid[r1, c0 : c1 + 1] = GREEN
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    markers = {
        "up": (r0 - 1, mid_c),
        "right": (mid_r, c1 + 1),
        "down": (r1 + 1, mid_c),
        "left": (mid_r, c0 - 1),
    }
    mr, mc = markers[side]
    grid[mr, mc] = RED
    _reserve(occ, min(r0, mr), min(c0, mc), max(r1, mr), max(c1, mc))
    return (r0, c0, r1, c1)


def _try_place_pair(grid, occ, side):
    h, w = grid.shape
    recv_inner_h = random.randint(2, 5)
    recv_inner_w = random.choice([3, 4, 5])
    if side in ("up", "down"):
        core_h, core_w = recv_inner_h, recv_inner_w
    else:
        core_h, core_w = recv_inner_w, recv_inner_h
    ant_len = random.randint(1, 3)
    color = random.choice(TOWER_COLORS)
    pattern = _random_pattern(core_h, core_w)

    body_h = core_h + 2
    body_w = core_w + 2
    tower_h = ant_len + body_h
    tower_w = body_w
    recv_h = recv_inner_h + 2
    recv_w = recv_inner_w + 2

    for _ in range(250):
        rr0 = random.randint(2, h - recv_h - 3)
        rc0 = random.randint(2, w - recv_w - 3)
        marker = {
            "up": (rr0 - 1, rc0 + recv_w // 2),
            "right": (rr0 + recv_h // 2, rc0 + recv_w),
            "down": (rr0 + recv_h, rc0 + recv_w // 2),
            "left": (rr0 + recv_h // 2, rc0 - 1),
        }[side]
        rec_r0 = min(rr0, marker[0])
        rec_c0 = min(rc0, marker[1])
        rec_r1 = max(rr0 + recv_h - 1, marker[0])
        rec_c1 = max(rc0 + recv_w - 1, marker[1])
        if not _expanded_free(occ, rec_r0, rec_c0, rec_r1, rec_c1):
            continue

        if side == "up":
            max_tr0 = rr0 - tower_h - 2
            if max_tr0 < 1:
                continue
            tr0 = random.randint(1, max_tr0)
            tc0_min = max(1, rc0 - tower_w + 2)
            tc0_max = min(w - tower_w - 2, rc0 + recv_w - 2)
            if tc0_min > tc0_max:
                continue
            tc0 = random.randint(tc0_min, tc0_max)
        elif side == "down":
            min_tr0 = rr0 + recv_h + 2
            if min_tr0 > h - tower_h - 2:
                continue
            tr0 = random.randint(min_tr0, h - tower_h - 2)
            tc0_min = max(1, rc0 - tower_w + 2)
            tc0_max = min(w - tower_w - 2, rc0 + recv_w - 2)
            if tc0_min > tc0_max:
                continue
            tc0 = random.randint(tc0_min, tc0_max)
        elif side == "left":
            max_tc0 = rc0 - tower_w - 2
            if max_tc0 < 1:
                continue
            tc0 = random.randint(1, max_tc0)
            tr0_min = max(1, rr0 - tower_h + 2)
            tr0_max = min(h - tower_h - 2, rr0 + recv_h - 2)
            if tr0_min > tr0_max:
                continue
            tr0 = random.randint(tr0_min, tr0_max)
        else:
            min_tc0 = rc0 + recv_w + 2
            if min_tc0 > w - tower_w - 2:
                continue
            tc0 = random.randint(min_tc0, w - tower_w - 2)
            tr0_min = max(1, rr0 - tower_h + 2)
            tr0_max = min(h - tower_h - 2, rr0 + recv_h - 2)
            if tr0_min > tr0_max:
                continue
            tr0 = random.randint(tr0_min, tr0_max)

        if not _expanded_free(occ, tr0, tc0, tr0 + tower_h - 1, tc0 + tower_w - 1):
            continue
        _draw_receiver(grid, occ, rr0, rc0, recv_inner_h, recv_inner_w, side)
        _draw_tower(grid, occ, tr0, tc0, core_h, core_w, color, pattern, ant_len)
        return True
    return False


def _gen_attempt():
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    sides = DIRS[:]
    random.shuffle(sides)
    pair_count = random.randint(2, 4)
    placed = 0
    for side in sides:
        if placed >= pair_count:
            break
        if _try_place_pair(grid, occ, side):
            placed += 1
    if placed < 2:
        return None
    if len(_find_receivers(grid)) != placed or len(_find_towers(grid)) != placed:
        return None
    solved = main(grid)
    if not isinstance(solved, np.ndarray) or solved.shape != grid.shape:
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal tower input")
