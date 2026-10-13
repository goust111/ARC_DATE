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
AZURE = 8
MAROON = 9

DIRS = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}
DIR_VECTORS = list(DIRS.keys())
COLOR_POOL = [BLUE, GREEN, YELLOW, PINK, AZURE, MAROON]
DIM_OPTIONS = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (2, 5), (5, 2), (3, 3)]


def _grey_components(grid):
    seen = np.zeros(grid.shape, dtype=bool)
    comps = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or grid[r, c] != GREY:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR_VECTORS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] == GREY:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rect_border(grid, r0, r1, c0, c1, comp_size):
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    border = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(window[border] == GREY) and np.all(window[~border] != GREY) and comp_size == int(border.sum()))


def _find_marker(grid, r0, r1, c0, c1):
    reds = []
    oranges = set()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if grid[r, c] == RED:
                reds.append((r, c))
            elif grid[r, c] == ORANGE:
                oranges.add((r, c))
    for rr, rc in reds:
        for dr, dc in DIR_VECTORS:
            if (rr + dr, rc + dc) in oranges:
                return dr, dc
    return None


def _extract_objects(grid):
    objects = []
    for comp in _grey_components(grid):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if not _is_rect_border(grid, r0, r1, c0, c1, len(comp)):
            continue
        marker = _find_marker(grid, r0, r1, c0, c1)
        if marker is None:
            continue
        area = (r1 - r0 - 1) * (c1 - c0 - 1)
        objects.append(
            {
                "bbox": (r0, r1, c0, c1),
                "area": area,
                "delta": (marker[0] * area, marker[1] * area),
                "patch": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )
    objects.sort(key=lambda obj: (obj["area"], obj["bbox"]))
    return objects


def _paint_patch(out, patch, top, left):
    rows, cols = out.shape
    ph, pw = patch.shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(rows, top + ph)
    c1 = min(cols, left + pw)
    if r0 >= r1 or c0 >= c1:
        return
    sr0 = r0 - top
    sc0 = c0 - left
    sub = patch[sr0 : sr0 + (r1 - r0), sc0 : sc0 + (c1 - c0)]
    mask = sub != BLACK
    out[r0:r1, c0:c1][mask] = sub[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each bordered region in the compass direction by its interior area.

    Input structure:
    The grid has a black background with several closed rectangular regions made from
    one-pixel grey borders. Each interior is filled with one ordinary color, except
    for one adjacent red-orange compass pair placed near an interior corner.

    Rule:
    For every grey-bordered rectangle, read the direction from the red pixel to its
    adjacent orange pixel. Compute the interior area excluding the grey border, then
    move the entire rectangle, including border and marker, by that many cells in the
    compass direction. Draw all moved rectangles on a fresh black canvas, with smaller
    interior areas drawn first and larger areas drawn later when overlaps occur.

    Key insight:
    The red-orange pair gives only direction; the hidden distance is the rectangle's
    own interior area, so identifying both the compass marker and the enclosed size is
    necessary to reconstruct the output.
    """
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    for obj in _extract_objects(grid):
        r0, _, c0, _ = obj["bbox"]
        dr, dc = obj["delta"]
        _paint_patch(out, obj["patch"], r0 + dr, c0 + dc)
    return out


def _bbox_intersects(a, b, pad=0):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _ranges_for_original(rows, cols, spec):
    h, w, area, dr, dc = spec["oh"], spec["ow"], spec["area"], spec["dr"], spec["dc"]
    r_min, r_max = 0, rows - h
    c_min, c_max = 0, cols - w
    if dr < 0:
        r_min = area
    elif dr > 0:
        r_max = rows - h - area
    if dc < 0:
        c_min = area
    elif dc > 0:
        c_max = cols - w - area
    if r_min > r_max or c_min > c_max:
        return None
    return r_min, r_max, c_min, c_max


def _ranges_for_output(rows, cols, spec):
    original = _ranges_for_original(rows, cols, spec)
    if original is None:
        return None
    r_min, r_max, c_min, c_max = original
    shift_r = spec["dr"] * spec["area"]
    shift_c = spec["dc"] * spec["area"]
    return r_min + shift_r, r_max + shift_r, c_min + shift_c, c_max + shift_c


def _random_spec(color):
    ih, iw = random.choice(DIM_OPTIONS)
    dr, dc = random.choice(DIR_VECTORS)
    return {
        "ih": ih,
        "iw": iw,
        "oh": ih + 2,
        "ow": iw + 2,
        "area": ih * iw,
        "dr": dr,
        "dc": dc,
        "color": color,
        "corner_side": random.randrange(2),
    }


def _make_patch(spec):
    patch = np.full((spec["oh"], spec["ow"]), GREY, dtype=int)
    patch[1:-1, 1:-1] = spec["color"]
    ih, iw = spec["ih"], spec["iw"]
    dr, dc = spec["dr"], spec["dc"]
    if (dr, dc) == (-1, 0):
        col = 1 if spec["corner_side"] == 0 else iw
        patch[1, col] = ORANGE
        patch[2, col] = RED
    elif (dr, dc) == (1, 0):
        col = 1 if spec["corner_side"] == 0 else iw
        patch[ih - 1, col] = RED
        patch[ih, col] = ORANGE
    elif (dr, dc) == (0, -1):
        row = 1 if spec["corner_side"] == 0 else ih
        patch[row, 1] = ORANGE
        patch[row, 2] = RED
    else:
        row = 1 if spec["corner_side"] == 0 else ih
        patch[row, iw - 1] = RED
        patch[row, iw] = ORANGE
    return patch


def _place_object(grid, spec, top, left):
    patch = _make_patch(spec)
    grid[top : top + spec["oh"], left : left + spec["ow"]] = patch


def _try_forced_overlap(rows, cols, specs):
    first, second = specs[0], specs[1]
    if first["area"] == second["area"]:
        return None
    range_a = _ranges_for_output(rows, cols, first)
    range_b = _ranges_for_output(rows, cols, second)
    if range_a is None or range_b is None:
        return None
    ar0, ar1, ac0, ac1 = range_a
    br0, br1, bc0, bc1 = range_b
    for _ in range(200):
        out_a_r = random.randint(ar0, ar1)
        out_a_c = random.randint(ac0, ac1)
        out_b_r_min = max(br0, out_a_r - second["oh"] + 1)
        out_b_r_max = min(br1, out_a_r + first["oh"] - 1)
        out_b_c_min = max(bc0, out_a_c - second["ow"] + 1)
        out_b_c_max = min(bc1, out_a_c + first["ow"] - 1)
        if out_b_r_min > out_b_r_max or out_b_c_min > out_b_c_max:
            continue
        out_b_r = random.randint(out_b_r_min, out_b_r_max)
        out_b_c = random.randint(out_b_c_min, out_b_c_max)
        in_a_r = out_a_r - first["dr"] * first["area"]
        in_a_c = out_a_c - first["dc"] * first["area"]
        in_b_r = out_b_r - second["dr"] * second["area"]
        in_b_c = out_b_c - second["dc"] * second["area"]
        bbox_a = (in_a_r, in_a_r + first["oh"] - 1, in_a_c, in_a_c + first["ow"] - 1)
        bbox_b = (in_b_r, in_b_r + second["oh"] - 1, in_b_c, in_b_c + second["ow"] - 1)
        if not _bbox_intersects(bbox_a, bbox_b, pad=1):
            return [(in_a_r, in_a_c, bbox_a), (in_b_r, in_b_c, bbox_b)]
    return None


def _place_remaining(rows, cols, specs, placements):
    bboxes = [p[2] for p in placements]
    for spec in specs[len(placements) :]:
        allowed = _ranges_for_original(rows, cols, spec)
        if allowed is None:
            return None
        r_min, r_max, c_min, c_max = allowed
        placed = False
        for _ in range(300):
            r = random.randint(r_min, r_max)
            c = random.randint(c_min, c_max)
            bbox = (r, r + spec["oh"] - 1, c, c + spec["ow"] - 1)
            if all(not _bbox_intersects(bbox, old, pad=1) for old in bboxes):
                placements.append((r, c, bbox))
                bboxes.append(bbox)
                placed = True
                break
        if not placed:
            return None
    return placements


def generate_input() -> np.ndarray:
    for _ in range(1000):
        rows = random.randint(18, 30)
        cols = random.randint(18, 30)
        count = random.randint(2, 5)
        colors = random.sample(COLOR_POOL, count)
        specs = [_random_spec(color) for color in colors]
        want_overlap = count >= 2 and random.random() < 0.4
        placements = None
        if want_overlap:
            for _ in range(20):
                if specs[0]["area"] != specs[1]["area"]:
                    break
                specs[1] = _random_spec(colors[1])
            forced = _try_forced_overlap(rows, cols, specs)
            if forced is not None:
                placements = _place_remaining(rows, cols, specs, forced)
        if placements is None:
            placements = _place_remaining(rows, cols, specs, [])
        if placements is None:
            continue
        grid = np.zeros((rows, cols), dtype=int)
        for spec, (r, c, _) in zip(specs, placements):
            _place_object(grid, spec, r, c)
        return grid
    raise RuntimeError("failed to generate a valid input")
