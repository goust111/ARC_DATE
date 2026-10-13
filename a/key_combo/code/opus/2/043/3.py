import random
from collections import deque

import numpy as np

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape

    # 1) detect yellow rectangle blocker (occluder)
    oc_candidates = []
    for col in np.unique(grid):
        if col in (BLACK, BLUE,):
            continue
        mask = (grid == col)
        visited = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or visited[r, c]:
                    continue
                q = deque([(r, c)])
                visited[r, c] = True
                comp = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                            comp.append((nr, nc))
                if len(comp) >= 6:
                    r0, c0, r1, c1 = _bbox(comp)
                    if (r1 - r0 + 1) * (c1 - c0 + 1) == len(comp):
                        # solid rectangle
                        oc_candidates.append((col, comp, (r0, c0, r1, c1)))

    if not oc_candidates:
        return np.array([[]], dtype=int)

    oc_color, _, (ro0, co0, ro1, co1) = min(oc_candidates, key=lambda x: x[1][0][0] if False else 0)
    oc_center = ((ro0 + ro1) // 2, (co0 + co1) // 2)

    work = grid.copy()
    work[ro0 : ro1 + 1, co0 : co1 + 1] = BLACK

    # 2) infer symmetric axis on blue object
    blue = np.argwhere(work == BLUE)
    if len(blue) == 0:
        return np.array([[]], dtype=int)

    best_axis = None
    best_score = -10**9
    for t in range(-1, w):
        axis = t + 0.5
        score = 0
        for r, c in blue:
            mc = int(round(2 * axis - c))
            if 0 <= mc < w:
                if work[r, mc] == BLUE:
                    score += 1
                else:
                    score -= 1
            else:
                score -= 2
        if score > best_score:
            best_score = score
            best_axis = axis
    if best_axis is None:
        return np.array([[]], dtype=int)

    # 3) complete symmetry
    for r, c in list(np.argwhere(work == BLUE)):
        mc = int(round(2 * best_axis - c))
        if 0 <= mc < w:
            work[int(r), mc] = BLUE

    # 4) crop quadrant
    blue_coords = np.argwhere(work == BLUE)
    if len(blue_coords) == 0:
        return np.array([[]], dtype=int)
    r0, c0, r1, c1 = _bbox([(int(x), int(y)) for x, y in blue_coords])
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2

    if oc_center[0] <= rc:
        rs0, rs1 = r0, rc
    else:
        rs0, rs1 = rc, r1
    if oc_center[1] <= cc:
        cs0, cs1 = c0, cc
    else:
        cs0, cs1 = cc, c1

    crop = work[rs0 : rs1 + 1, cs0 : cs1 + 1].copy()

    # 5) shoot rays from crop center
    cr = rc - rs0
    cc0 = cc - cs0
    h2, w2 = crop.shape

    def _ray(rr: int, cc: int, dr: int, dc: int):
        nr, nc = rr + dr, cc + dc
        while 0 <= nr < h2 and 0 <= nc < w2:
            if crop[nr, nc] == BLUE:
                break
            crop[nr, nc] = oc_color
            nr += dr
            nc += dc

    _ray(cr, cc0, -1, 0)
    _ray(cr, cc0, 1, 0)
    _ray(cr, cc0, 0, -1)
    _ray(cr, cc0, 0, 1)
    return crop


def _connected_growth(h: int, w: int, target: int, bound: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bound
    rr = random.randint(r0, r1)
    cc = random.randint(c0, c1)
    coords = [(rr, cc)]
    occ = {(rr, cc)}
    while len(coords) < target and len(occ) < (r1 - r0 + 1) * (c1 - c0 + 1):
        r, c = random.choice(coords)
        neigh = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(neigh)
        for nr, nc in neigh:
            if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in occ:
                coords.append((nr, nc))
                occ.add((nr, nc))
                break
    return coords


def _generate_input_single() -> np.ndarray:
    h = 15
    w = 15
    bg = np.full((h, w), BLACK, dtype=int)

    obj_h = random.choice([7, 9, 11])
    obj_w = random.choice([5, 7, 9])
    r0 = random.randint(2, h - obj_h - 2)
    c0 = random.randint(2, w - obj_w - 2)
    r1, c1 = r0 + obj_h - 1, c0 + obj_w - 1

    cell_count = random.randint(20, 40)
    half_w = obj_w // 2 + 1
    raw = _connected_growth(h, w, max(6, cell_count // 2), (r0, c0 + 1, r1, c0 + half_w - 1))
    raw = list(set(raw))
    obj_coords = []
    for r, c in raw:
        # mirror across vertical axis of bbox
        obj_coords.append((r, c))
        mc = c0 + (obj_w - 1 - (c - c0))
        mirror = (r, mc)
        if mirror not in obj_coords:
            obj_coords.append(mirror)
    for rr, cc in obj_coords:
        if 0 <= rr < h and 0 <= cc < w and r0 <= rr <= r1 and c0 <= cc <= c1:
            bg[rr, cc] = BLUE

    # 3x? irregularization: remove random few cells but keep odd bbox
    for _ in range(random.randint(2, 4)):
        rr = random.randint(r0, r1)
        cc = random.randint(c0, c1)
        if bg[rr, cc] == BLUE:
            # keep connectivity by only removing tiny protrusions
            bg[rr, cc] = BLACK

    # make sure odd bbox has enough blue area
    if np.sum(bg == BLUE) < 20:
        return _generate_input_single()

    # draw yellow occluder inside bbox
    oh = random.randint(3, 5)
    ow = random.randint(2, 4)
    ry0 = random.randint(r0, r1 - oh + 1)
    cx0 = random.randint(c0, c1 - ow + 1)
    ry1 = ry0 + oh - 1
    cx1 = cx0 + ow - 1
    if np.sum(bg[ry0 : ry1 + 1, cx0 : cx1 + 1] == BLUE) == 0:
        # no overlap with blue, regenerate
        return _generate_input_single()
    bg[ry0 : ry1 + 1, cx0 : cx1 + 1] = YELLOW

    return bg


def generate_input() -> np.ndarray:
    return _generate_input_single()


def generate_inputs() -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    attempts = 0
    while len(outputs) < 20 and attempts < 200:
        attempts += 1
        g = _generate_input_single()
        arr = np.array(g, dtype=int, copy=True)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2 or arr.shape[0] > 30 or arr.shape[1] > 30:
            continue
        outputs.append(arr)

    while len(outputs) < 20:
        if outputs:
            outputs.append(outputs[-1].copy())
        else:
            outputs.append(np.zeros((1, 1), dtype=int))
    return outputs


if __name__ == "__main__":
    random.seed(0)
    print(main(generate_input()[0]).shape)
