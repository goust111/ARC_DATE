import random
from collections import deque

import numpy as np


BLACK = 0
FENCE_COLOR = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("up", "right", "down", "left")


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _rect_frame(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    if set(cells) != expected:
        return None
    return r0, c0, r1, c1


def _nearest_side(r0, c0, r1, c1, sr, sc):
    distances = (
        (sr - r0 - 1, "up"),
        (c1 - sc - 1, "right"),
        (r1 - sr - 1, "down"),
        (sc - c0 - 1, "left"),
    )
    best = min(d for d, _ in distances)
    for distance, side in distances:
        if distance == best:
            return side
    return "up"


def _erase_side(out, rect, side):
    r0, c0, r1, c1 = rect
    if side == "up":
        out[r0, c0 : c1 + 1] = BLACK
    elif side == "right":
        out[r0 : r1 + 1, c1] = BLACK
    elif side == "down":
        out[r1, c0 : c1 + 1] = BLACK
    else:
        out[r0 : r1 + 1, c0] = BLACK


def _draw_ray(out, seed, side, color):
    sr, sc = seed
    rows, cols = out.shape
    if side == "up":
        out[: sr + 1, sc] = color
    elif side == "right":
        out[sr, sc:cols] = color
    elif side == "down":
        out[sr:rows, sc] = color
    else:
        out[sr, : sc + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Open each rectangular fence on the side nearest its seed, then emit a seed-colored ray through that opening.

    Input structure:
    A black grid contains one or more 1-pixel-thick closed rectangular fences. Each fence is a single non-black
    color component whose interior is black except for exactly one non-fence colored seed pixel. Fence rectangles
    are separated from each other, and each seed lies strictly inside its own fence.

    Rule:
    For every fence, compare the seed's grid-cell distance to the top, right, bottom, and left inner walls. The
    closest side is selected, with ties resolved in top, right, bottom, left order. All pixels on that selected
    fence side are erased to black. From the seed, draw a straight ray in the selected direction using the seed
    color, extending through the opened side to the edge of the grid. The seed remains colored and the other fence
    sides remain unchanged.

    Key insight:
    The seed's off-center position inside the rectangle is the visible cue that chooses both which wall opens and
    which direction the colored ray travels.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    operations = []

    for color in [x for x in np.unique(arr) if x != BLACK]:
        for cells in _components(arr == color):
            rect = _rect_frame(cells)
            if rect is None:
                continue
            r0, c0, r1, c1 = rect
            seeds = []
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    if arr[r, c] != BLACK and arr[r, c] != color:
                        seeds.append((r, c))
            if len(seeds) != 1:
                continue
            seed = seeds[0]
            seed_color = int(arr[seed])
            side = _nearest_side(r0, c0, r1, c1, seed[0], seed[1])
            operations.append((rect, seed, seed_color, side))

    for rect, _, _, side in operations:
        _erase_side(out, rect, side)
    for _, seed, seed_color, side in operations:
        _draw_ray(out, seed, side, seed_color)
    return out


def _draw_frame(grid, r0, c0, r1, c1, color):
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _seed_candidates(height, width, side):
    candidates = []
    tie_candidates = []
    for sr in range(1, height - 1):
        for sc in range(1, width - 1):
            chosen = _nearest_side(0, 0, height - 1, width - 1, sr, sc)
            if chosen != side:
                continue
            distances = (
                sr - 1,
                width - sc - 2,
                height - sr - 2,
                sc - 1,
            )
            candidates.append((sr, sc))
            if distances.count(min(distances)) > 1:
                tie_candidates.append((sr, sc))
    if tie_candidates and random.random() < 0.25:
        return tie_candidates
    return candidates


def _ray_cells(rows, cols, seed, side):
    sr, sc = seed
    if side == "up":
        return [(r, sc) for r in range(sr, -1, -1)]
    if side == "right":
        return [(sr, c) for c in range(sc, cols)]
    if side == "down":
        return [(r, sc) for r in range(sr, rows)]
    return [(sr, c) for c in range(sc, -1, -1)]


def _frame_cells(r0, c0, r1, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _make_grid(rows, cols, count):
    grid = np.zeros((rows, cols), dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)
    used_rays = set()
    seed_colors = random.sample(SEED_COLORS, count)

    for index in range(count):
        side = random.choice(DIRECTIONS)
        placed = False
        for _ in range(300):
            height = random.randint(5, min(9, rows - 2))
            width = random.randint(5, min(9, cols - 2))
            candidates = _seed_candidates(height, width, side)
            if not candidates:
                continue
            r0 = random.randint(1, rows - height - 1)
            c0 = random.randint(1, cols - width - 1)
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            mr0, mr1 = max(0, r0 - 1), min(rows, r1 + 2)
            mc0, mc1 = max(0, c0 - 1), min(cols, c1 + 2)
            if blocked[mr0:mr1, mc0:mc1].any():
                continue

            rel_seed = random.choice(candidates)
            seed = (r0 + rel_seed[0], c0 + rel_seed[1])
            ray = _ray_cells(rows, cols, seed, side)
            if any(grid[r, c] != BLACK for r, c in ray):
                continue
            if any(cell in used_rays for cell in ray):
                continue

            frame = _frame_cells(r0, c0, r1, c1)
            occupied = set(frame)
            occupied.add(seed)
            if any(cell in used_rays for cell in occupied):
                continue

            _draw_frame(grid, r0, c0, r1, c1, FENCE_COLOR)
            grid[seed] = seed_colors[index]
            blocked[mr0:mr1, mc0:mc1] = True
            used_rays.update(ray)
            placed = True
            break
        if not placed:
            return None
    return grid


def generate_input() -> np.ndarray:
    rows = random.randint(14, 30)
    cols = random.randint(14, 30)
    for _ in range(100):
        count = random.randint(1, 4)
        grid = _make_grid(rows, cols, count)
        if grid is not None:
            return grid
        rows = random.randint(14, 30)
        cols = random.randint(14, 30)

    side = random.choice(DIRECTIONS)
    grid = np.zeros((15, 15), dtype=int)
    height = width = 7
    r0 = c0 = 4
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    candidates = _seed_candidates(height, width, side)
    sr, sc = random.choice(candidates)
    _draw_frame(grid, r0, c0, r1, c1, FENCE_COLOR)
    grid[r0 + sr, c0 + sc] = random.choice(SEED_COLORS)
    return grid
