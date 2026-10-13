from collections import Counter, deque
import random

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "right", "down", "left")


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _foreground_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == BG:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            pts = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    comps.sort(key=lambda a: _bbox(a)[:2])
    return comps


def _marker_color(grid, coords):
    counts = Counter(int(grid[r, c]) for r, c in coords)
    assert len(counts) == 2, "each stamp object must contain one body color and one marker color"
    singles = [color for color, count in counts.items() if count == 1]
    assert len(singles) == 1, "the marker must be the unique one-pixel color within its object"
    return int(singles[0])


def _tip_direction(coords):
    r0, c0, r1, c1 = _bbox(coords)
    counts = {
        "up": int(np.sum(coords[:, 0] == r0)),
        "down": int(np.sum(coords[:, 0] == r1)),
        "left": int(np.sum(coords[:, 1] == c0)),
        "right": int(np.sum(coords[:, 1] == c1)),
    }
    min_count = min(counts.values())
    tips = [direction for direction, count in counts.items() if count == min_count]
    assert len(tips) == 1, "tip direction must be the unique narrowest bbox side"
    return tips[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each marked object repeatedly in the direction encoded by its pointed shape.

    Input structure:
    - The background is black.
    - Several separated 4-connected foreground objects appear on the canvas.
    - Each object has one body color plus exactly one embedded single-pixel marker of a different color.
    - For each object's bounding box, exactly one side contains the fewest foreground pixels; that side is the object's pointed tip.

    Rule:
    - For every object, infer the pointed direction from the unique narrowest side of its bounding box.
    - Translate the whole object shape by one bounding-box length along that direction, then repeat by the same step until the next copy would leave the grid.
    - Draw every stamped copy using the object's embedded marker color.
    - Keep the original object unchanged, including both its body color and marker pixel.

    Key insight:
    The shape supplies both the stamp template and travel direction, while the lone internal marker supplies the ink color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()

    for coords in _foreground_components(g):
        marker = _marker_color(g, coords)
        direction = _tip_direction(coords)
        dr, dc = DIRS[direction]
        r0, c0, r1, c1 = _bbox(coords)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        step = box_h if direction in ("up", "down") else box_w

        k = 1
        while True:
            rr0 = r0 + dr * step * k
            rr1 = r1 + dr * step * k
            cc0 = c0 + dc * step * k
            cc1 = c1 + dc * step * k
            if rr0 < 0 or cc0 < 0 or rr1 >= h or cc1 >= w:
                break
            for r, c in coords:
                nr = int(r) + dr * step * k
                nc = int(c) + dc * step * k
                if int(out[nr, nc]) == BG:
                    out[nr, nc] = marker
            k += 1

    return out


def _right_pointing_mask(rng):
    height = rng.choice((5, 7))
    width = rng.randint(4, 7)
    mid = height // 2
    mode = rng.choice(("block_tip", "taper", "shoulder"))
    mask = np.zeros((height, width), dtype=bool)

    if mode == "block_tip":
        mask[:, : width - 1] = True
        mask[mid, width - 1] = True
    elif mode == "taper":
        for r in range(height):
            dist = abs(r - mid)
            row_width = max(2, width - dist)
            mask[r, :row_width] = True
    else:
        mask[:, : width - 2] = True
        mask[mid - 1 : mid + 2, width - 2] = True
        mask[mid, width - 1] = True

    return mask


def _orient_mask(mask, direction):
    turns = {"right": 0, "up": 1, "left": 2, "down": 3}[direction]
    return np.rot90(mask, turns).copy()


def _solid_internal_cells(mask):
    h, w = mask.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if all(mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                cells.append((r, c))
    return cells


def _make_stamp_shape(rng, direction):
    for _ in range(100):
        mask = _orient_mask(_right_pointing_mask(rng), direction)
        coords = np.argwhere(mask)
        if _tip_direction(coords) != direction:
            continue
        marker_choices = _solid_internal_cells(mask)
        if marker_choices:
            return mask, rng.choice(marker_choices)
    raise RuntimeError("failed to build a pointed stamp shape")


def _origin_ranges(height, width, mask, direction):
    mh, mw = mask.shape
    r_min, r_max = 0, height - mh
    c_min, c_max = 0, width - mw

    if direction == "up":
        r_min = mh
    elif direction == "down":
        r_max = height - 2 * mh
    elif direction == "left":
        c_min = mw
    elif direction == "right":
        c_max = width - 2 * mw

    if r_min > r_max or c_min > c_max:
        return None
    return r_min, r_max, c_min, c_max


def _projected_cells(row, col, mask, direction, height, width):
    coords = np.argwhere(mask)
    mh, mw = mask.shape
    dr, dc = DIRS[direction]
    step = mh if direction in ("up", "down") else mw

    all_cells = set()
    original_cells = set()
    copy_count = 0
    k = 0
    while True:
        r0 = row + dr * step * k
        c0 = col + dc * step * k
        r1 = r0 + mh - 1
        c1 = c0 + mw - 1
        if r0 < 0 or c0 < 0 or r1 >= height or c1 >= width:
            break
        copy_count += 1
        for rr, cc in coords:
            cell = (int(r0 + rr), int(c0 + cc))
            all_cells.add(cell)
            if k == 0:
                original_cells.add(cell)
        k += 1

    return all_cells, original_cells, copy_count


def _clear_of_existing(cells, occupied):
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _paint_original(grid, row, col, mask, marker_at, body_color, marker_color):
    for r, c in np.argwhere(mask):
        grid[row + int(r), col + int(c)] = body_color
    mr, mc = marker_at
    grid[row + mr, col + mc] = marker_color


def _gen_attempt(rng):
    height = rng.randint(16, 30)
    width = rng.randint(16, 30)
    count = rng.choice((2, 3, 4))
    directions = rng.sample(DIR_ORDER, count)
    colors = rng.sample(list(range(1, 10)), 2 * count)

    grid = np.zeros((height, width), dtype=int)
    occupied_output = set()

    for i, direction in enumerate(directions):
        body_color = colors[2 * i]
        marker_color = colors[2 * i + 1]
        placed = False

        for _ in range(250):
            mask, marker_at = _make_stamp_shape(rng, direction)
            ranges = _origin_ranges(height, width, mask, direction)
            if ranges is None:
                continue
            r_min, r_max, c_min, c_max = ranges
            row = rng.randint(r_min, r_max)
            col = rng.randint(c_min, c_max)

            all_cells, original_cells, copy_count = _projected_cells(
                row, col, mask, direction, height, width
            )
            if copy_count < 2:
                continue
            if not _clear_of_existing(all_cells, occupied_output):
                continue

            _paint_original(grid, row, col, mask, marker_at, body_color, marker_color)
            occupied_output.update(all_cells)
            assert original_cells
            placed = True
            break

        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate valid stamp-repetition input")
