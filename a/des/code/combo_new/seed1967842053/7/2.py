import random
from collections import deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3

MARK_TO_MODE = {RED: "horizontal", GREEN: "vertical", BLUE: "rotation"}
MODE_TO_MARK = {v: k for k, v in MARK_TO_MODE.items()}
SIDES = {
    "horizontal": ("top", "bottom"),
    "vertical": ("left", "right"),
    "rotation": ("top", "right", "bottom", "left"),
}
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """Attach a transformed copy of each flawed seed on the side where its target symmetry is missing cells.

    Input structure:
    A mostly background grid contains one 1x1 corner marker and several separated connected seed patterns. The
    marker is red, green, or blue, selecting horizontal reflection, vertical reflection, or fourfold rotation. Each
    seed fits in a small bounding box and is either already symmetric for that selected type or is missing one or two
    colored cells on a single side of that symmetry.

    Rule:
    Read the corner marker to choose the symmetry type, then process every non-background component except the marker.
    For the selected symmetry, compare the seed with its reflected or rotated versions inside its bounding box to find
    the side containing the missing counterpart cells. Symmetric seeds are copied unchanged. For flawed seeds, make the
    corresponding reflected or rotated copy and paste it flush against the missing side, sharing the boundary row or
    column with the original seed.

    Key insight:
    The corner color only chooses which symmetry test to apply; the placement direction must be inferred separately for
    each seed from the location of the cells that would be supplied by the chosen symmetry.
    """
    grid = np.asarray(grid)
    bg = _background_color(grid)
    marker = _find_marker(grid, bg)
    if marker is None:
        return grid.copy()
    mr, mc, marker_color = marker
    mode = MARK_TO_MODE.get(marker_color)
    if mode is None:
        return grid.copy()

    out = grid.copy()
    ignore = {(mr, mc)}
    for cells in _components(grid, bg, ignore):
        r0, r1, c0, c1 = _bbox(cells)
        arr = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=grid.dtype)
        for r, c in cells:
            arr[r - r0, c - c0] = grid[r, c]

        side = _defect_side(arr, mode, bg)
        if side is None:
            continue
        copy = _transformed_copy(arr, mode)
        _paste(out, copy, bg, r0, r1, c0, c1, side)
    return out


def _background_color(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_marker(grid, bg):
    h, w = grid.shape
    for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        color = int(grid[r, c])
        if color != bg and color in MARK_TO_MODE:
            return r, c, color
    return None


def _components(grid, bg, ignore):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or (sr, sc) in ignore or int(grid[sr, sc]) == bg:
                continue
            seen[sr, sc] = True
            q = deque([(sr, sc)])
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or (nr, nc) in ignore or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _defect_side(arr, mode, bg):
    if mode == "horizontal":
        missing = _missing_from_transform(arr, np.flipud(arr), bg)
        if not missing:
            return None
        center = (arr.shape[0] - 1) / 2.0
        return "top" if sum(r - center for r, _ in missing) < 0 else "bottom"

    if mode == "vertical":
        missing = _missing_from_transform(arr, np.fliplr(arr), bg)
        if not missing:
            return None
        center = (arr.shape[1] - 1) / 2.0
        return "left" if sum(c - center for _, c in missing) < 0 else "right"

    if mode == "rotation":
        square = _square_view(arr, bg)
        missing = set()
        for k in (1, 2, 3):
            missing.update(_missing_from_transform(square, np.rot90(square, k), bg))
        if not missing:
            return None
        center = (square.shape[0] - 1) / 2.0
        scores = {"top": 0.0, "right": 0.0, "bottom": 0.0, "left": 0.0}
        for r, c in missing:
            dr = r - center
            dc = c - center
            if abs(dr) >= abs(dc):
                scores["bottom" if dr > 0 else "top"] += abs(dr) + 0.01
            else:
                scores["right" if dc > 0 else "left"] += abs(dc) + 0.01
        return max(scores, key=scores.get)

    return None


def _missing_from_transform(arr, transformed, bg):
    rows, cols = np.where((transformed != bg) & (arr != transformed))
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def _square_view(arr, bg):
    h, w = arr.shape
    if h == w:
        return arr
    n = max(h, w)
    square = np.full((n, n), bg, dtype=arr.dtype)
    r0 = (n - h) // 2
    c0 = (n - w) // 2
    square[r0:r0 + h, c0:c0 + w] = arr
    return square


def _transformed_copy(arr, mode):
    if mode == "horizontal":
        return np.flipud(arr)
    if mode == "vertical":
        return np.fliplr(arr)
    return np.rot90(_square_view(arr, BACKGROUND), 2)


def _paste(out, copy, bg, r0, r1, c0, c1, side):
    h, w = copy.shape
    if side == "top":
        base_r, base_c = r0 - h + 1, c0
    elif side == "bottom":
        base_r, base_c = r1, c0
    elif side == "left":
        base_r, base_c = r0, c0 - w + 1
    else:
        base_r, base_c = r0, c1

    rows, cols = out.shape
    for rr in range(h):
        for cc in range(w):
            if int(copy[rr, cc]) == bg:
                continue
            tr = base_r + rr
            tc = base_c + cc
            if 0 <= tr < rows and 0 <= tc < cols:
                out[tr, tc] = copy[rr, cc]


def generate_input() -> np.ndarray:
    mode = random.choice(tuple(SIDES))
    marker_color = MODE_TO_MARK[mode]

    for _ in range(200):
        rows = random.randint(18, 30)
        cols = random.randint(18, 30)
        grid = np.full((rows, cols), BACKGROUND, dtype=int)
        occupied = np.zeros((rows, cols), dtype=bool)

        marker_pos = random.choice(((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)))
        grid[marker_pos] = marker_color
        _reserve(occupied, marker_pos[0], marker_pos[0], marker_pos[1], marker_pos[1], pad=2)

        count = random.randint(2, 5)
        choices = list(SIDES[mode]) + [None]
        requested = [random.choice(SIDES[mode]), None]
        while len(requested) < count:
            requested.append(random.choice(choices))
        random.shuffle(requested)

        placed = True
        for side in requested:
            seed = _make_seed(mode, side)
            pos = _place_seed(grid, occupied, seed, mode, side)
            if pos is None:
                placed = False
                break
            r, c, footprint = pos
            mask = seed != BACKGROUND
            grid[r:r + seed.shape[0], c:c + seed.shape[1]][mask] = seed[mask]
            _reserve(occupied, *footprint, pad=1)
        if placed:
            return grid

    return _fallback_input(mode, marker_color)


def _random_color():
    return random.choice(PALETTE)


def _paint(arr, points, color=None):
    if color is None:
        color = _random_color()
    for r, c in points:
        arr[r, c] = color


def _make_seed(mode, side):
    if mode == "horizontal":
        return _make_horizontal_seed(side)
    if mode == "vertical":
        transposed_side = None
        if side == "left":
            transposed_side = "top"
        elif side == "right":
            transposed_side = "bottom"
        return _make_horizontal_seed(transposed_side).T.copy()
    return _make_rotation_seed(side)


def _make_horizontal_seed(side):
    for _ in range(80):
        h = random.choice((3, 5))
        w = random.randint(3, 5)
        mid = h // 2
        arr = np.full((h, w), BACKGROUND, dtype=int)
        for c in range(w):
            arr[mid, c] = _random_color()
        for r in range(mid):
            left = random.randint(0, w // 2)
            right = random.randint(w // 2, w - 1)
            if left == right:
                left = max(0, left - 1)
                right = min(w - 1, right + 1)
            for c in range(left, right + 1):
                color = _random_color()
                arr[r, c] = color
                arr[h - 1 - r, c] = color

        if side is None:
            if _connected(arr != BACKGROUND):
                return arr
            continue

        candidates = _edge_removal_candidates(arr, side)
        random.shuffle(candidates)
        before = _local_bbox(arr != BACKGROUND)
        for nremove in random.sample((1, 2), 2):
            if len(candidates) < nremove:
                continue
            for start in range(min(8, len(candidates))):
                removed = candidates[start:start + nremove]
                if len(removed) < nremove:
                    removed = candidates[:nremove]
                trial = arr.copy()
                for r, c in removed:
                    trial[r, c] = BACKGROUND
                if _local_bbox(trial != BACKGROUND) == before and _connected(trial != BACKGROUND):
                    return trial
    return _simple_horizontal_seed(side)


def _edge_removal_candidates(arr, side):
    h, w = arr.shape
    rows = range(0, h // 2) if side == "top" else range(h // 2 + 1, h)
    candidates = []
    for r in rows:
        cols = [c for c in range(w) if int(arr[r, c]) != BACKGROUND]
        if len(cols) <= 1:
            continue
        if cols[0] != w // 2:
            candidates.append((r, cols[0]))
        if cols[-1] != w // 2:
            candidates.append((r, cols[-1]))
    return candidates


def _simple_horizontal_seed(side):
    arr = np.array(
        [
            [4, 5, 4],
            [6, 7, 6],
            [4, 5, 4],
        ],
        dtype=int,
    )
    if side == "top":
        arr[0, 0] = BACKGROUND
    elif side == "bottom":
        arr[2, 2] = BACKGROUND
    return arr


def _make_rotation_seed(side):
    arr = np.full((5, 5), BACKGROUND, dtype=int)
    _paint(arr, [(2, 2)])
    orbits = [
        [(0, 2), (2, 4), (4, 2), (2, 0)],
        [(1, 2), (2, 3), (3, 2), (2, 1)],
        [(0, 1), (1, 4), (4, 3), (3, 0)],
        [(0, 3), (3, 4), (4, 1), (1, 0)],
        [(1, 1), (1, 3), (3, 3), (3, 1)],
    ]
    for orbit in orbits[:4]:
        _paint(arr, orbit)
    if random.random() < 0.7:
        _paint(arr, orbits[4])

    if side is None:
        return arr

    removable = {
        "top": [(0, 1), (0, 3), (1, 2)],
        "right": [(1, 4), (3, 4), (2, 3)],
        "bottom": [(4, 1), (4, 3), (3, 2)],
        "left": [(1, 0), (3, 0), (2, 1)],
    }[side]
    before = _local_bbox(arr != BACKGROUND)
    for nremove in random.sample((1, 2), 2):
        choices = removable[:]
        random.shuffle(choices)
        trial = arr.copy()
        for r, c in choices[:nremove]:
            trial[r, c] = BACKGROUND
        if _local_bbox(trial != BACKGROUND) == before and _connected(trial != BACKGROUND):
            return trial
    arr[removable[0]] = BACKGROUND
    return arr


def _connected(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(int(x) for x in points[0])
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _local_bbox(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return None
    rs = points[:, 0]
    cs = points[:, 1]
    return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())


def _place_seed(grid, occupied, seed, mode, side):
    rows, cols = grid.shape
    h, w = seed.shape
    for _ in range(300):
        r = random.randint(1, rows - h - 1)
        c = random.randint(1, cols - w - 1)
        footprint = _footprint(r, c, h, w, side)
        if footprint is None:
            footprint = (r, r + h - 1, c, c + w - 1)
        r0, r1, c0, c1 = footprint
        if r0 < 1 or c0 < 1 or r1 >= rows - 1 or c1 >= cols - 1:
            continue
        pr0 = max(0, r0 - 1)
        pr1 = min(rows - 1, r1 + 1)
        pc0 = max(0, c0 - 1)
        pc1 = min(cols - 1, c1 + 1)
        if not occupied[pr0:pr1 + 1, pc0:pc1 + 1].any():
            return r, c, footprint
    return None


def _footprint(r, c, h, w, side):
    if side == "top":
        return r - h + 1, r + h - 1, c, c + w - 1
    if side == "bottom":
        return r, r + 2 * h - 2, c, c + w - 1
    if side == "left":
        return r, r + h - 1, c - w + 1, c + w - 1
    if side == "right":
        return r, r + h - 1, c, c + 2 * w - 2
    return None


def _reserve(occupied, r0, r1, c0, c1, pad=0):
    rows, cols = occupied.shape
    r0 = max(0, r0 - pad)
    r1 = min(rows - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(cols - 1, c1 + pad)
    occupied[r0:r1 + 1, c0:c1 + 1] = True


def _fallback_input(mode, marker_color):
    grid = np.full((24, 24), BACKGROUND, dtype=int)
    grid[0, 0] = marker_color
    sides = list(SIDES[mode]) + [None]
    positions = [(7, 7), (7, 16), (16, 7), (16, 16), (12, 12)]
    for side, (r, c) in zip(sides, positions):
        seed = _make_seed(mode, side)
        mask = seed != BACKGROUND
        grid[r:r + seed.shape[0], c:c + seed.shape[1]][mask] = seed[mask]
    return grid
