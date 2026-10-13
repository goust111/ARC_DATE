from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
OBJECT_COLORS = (1, 3, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_RETRIES = 400
MAX_LEFT = 18


def _detect_wall(grid: np.ndarray) -> tuple[str, tuple[int, int]]:
    walls = []
    if np.all(grid[:, 0] == YELLOW):
        walls.append(("left", (0, -1)))
    if np.all(grid[:, -1] == YELLOW):
        walls.append(("right", (0, 1)))
    if np.all(grid[0, :] == YELLOW):
        walls.append(("top", (-1, 0)))
    if np.all(grid[-1, :] == YELLOW):
        walls.append(("bottom", (1, 0)))
    assert len(walls) == 1, "expected exactly one full yellow light wall"
    return walls[0]


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, YELLOW) or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            objects.append({"id": len(objects), "color": color, "coords": coords})

    return objects


def _collect_shadow_hits(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], set[int]]]:
    _wall_side, (dr, dc) = _detect_wall(grid)
    h, w = grid.shape
    objects = _find_objects(grid)
    hits: dict[tuple[int, int], set[int]] = {}

    for obj in objects:
        oid = int(obj["id"])
        for r, c in obj["coords"]:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != YELLOW:
                if int(grid[nr, nc]) == BLACK:
                    hits.setdefault((int(nr), int(nc)), set()).add(oid)
                nr += dr
                nc += dc

    return objects, hits


def _shadow_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects, hits = _collect_shadow_hits(g)
    color_by_id = {int(obj["id"]): int(obj["color"]) for obj in objects}

    for (r, c), owners in hits.items():
        out[r, c] = RED if len(owners) >= 2 else color_by_id[next(iter(owners))]

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    filled = mask.copy()
    filled[~mask & ~outside] = True
    return filled


def _random_object_mask() -> np.ndarray:
    for _ in range(80):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target = random.randint(4, min(h * w, 8))

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        mask = _crop_mask(_fill_mask_holes(mask))
        area = int(mask.sum())
        if area < 4:
            continue
        if min(mask.shape) == 1 and random.random() < 0.7:
            continue
        return mask

    return np.array([[1, 1], [1, 1]], dtype=bool)


def _sample_colors(count: int) -> list[int]:
    if count <= len(OBJECT_COLORS) and random.random() < 0.75:
        return random.sample(list(OBJECT_COLORS), count)
    return [random.choice(OBJECT_COLORS) for _ in range(count)]


def _paint_scene(placements: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[:, 0] = YELLOW
    for obj in placements:
        mask = obj["mask"]
        top = int(obj["top"])
        left = int(obj["left"])
        color = int(obj["color"])
        rows, cols = np.nonzero(mask)
        grid[top + rows, left + cols] = color
    return grid


def _generate_canonical_scene(want_overlap: bool) -> np.ndarray | None:
    n_objects = random.randint(2, 4)
    masks = [_random_object_mask() for _ in range(n_objects)]
    colors = _sample_colors(n_objects)
    placements = []

    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    right_pad = random.randint(1, 5)
    row_cursor = top_pad

    if want_overlap:
        h0, w0 = masks[0].shape
        h1, w1 = masks[1].shape
        if random.random() < 0.5:
            top0 = row_cursor
            top1 = row_cursor + random.randint(0, min(h0 - 1, 2))
        else:
            top1 = row_cursor
            top0 = row_cursor + random.randint(0, min(h1 - 1, 2))

        near_idx, far_idx = (0, 1) if random.random() < 0.5 else (1, 0)
        lefts = {}
        lefts[near_idx] = random.randint(2, 5)
        far_min = lefts[near_idx] + masks[near_idx].shape[1] + random.randint(1, 2)
        far_max = min(far_min + 6, MAX_LEFT - masks[far_idx].shape[1])
        if far_min > far_max:
            return None
        lefts[far_idx] = random.randint(far_min, far_max)

        placements.append({"mask": masks[0], "top": top0, "left": lefts[0], "color": colors[0]})
        placements.append({"mask": masks[1], "top": top1, "left": lefts[1], "color": colors[1]})
        row_cursor = max(top0 + h0, top1 + h1) + random.randint(1, 3)

        for i in range(2, n_objects):
            h, w = masks[i].shape
            left_max = MAX_LEFT - w
            if left_max < 2:
                return None
            top = row_cursor
            left = random.randint(2, left_max)
            placements.append({"mask": masks[i], "top": top, "left": left, "color": colors[i]})
            row_cursor = top + h + random.randint(1, 3)
    else:
        for i in range(n_objects):
            h, w = masks[i].shape
            left_max = MAX_LEFT - w
            if left_max < 2:
                return None
            top = row_cursor
            left = random.randint(2, left_max)
            placements.append({"mask": masks[i], "top": top, "left": left, "color": colors[i]})
            row_cursor = top + h + random.randint(1, 3)

    height = max(int(obj["top"]) + obj["mask"].shape[0] for obj in placements) + bottom_pad
    width = max(int(obj["left"]) + obj["mask"].shape[1] for obj in placements) + right_pad
    if height > 30 or width > 30:
        return None

    return _paint_scene(placements, height, width)


def _is_valid_generated_case(inp: np.ndarray, want_overlap: bool) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    objects = _find_objects(inp)
    if len(objects) < 2:
        return False

    out = _shadow_output(inp)
    changed = (inp == BLACK) & (out != BLACK)
    shadow_count = int(np.count_nonzero(changed))
    red_count = int(np.count_nonzero((inp == BLACK) & (out == RED)))
    regular_shadow_count = int(np.count_nonzero(changed & (out != RED)))

    if shadow_count < 3 or regular_shadow_count == 0:
        return False
    if want_overlap and red_count == 0:
        return False
    if not want_overlap and red_count != 0:
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast each solid object toward the yellow light wall and mark shared shadow cells in red.

    Input structure:
    - The background is black.
    - Exactly one full outer edge is a continuous yellow row or column, which acts as the light wall.
    - The remaining foreground consists of one or more 4-connected solid monochrome objects in non-black colors.
    - Objects do not overlap the yellow wall, and the grid size stays unchanged.

    Rule:
    - Determine the wall side from the unique full yellow edge, then infer that each object projects only toward that wall.
    - For every pixel of every object, walk cell by cell from that pixel toward the wall until the yellow edge is reached.
    - Every black cell crossed by rays from exactly one object becomes that object's color; pre-existing object pixels are never overwritten.
    - If the same black cell is crossed by rays from two or more different objects, paint that cell red instead.

    Key insight:
    - The full yellow border is not a drawable object but a directional cue: it defines which side is lit, so shadows fill only the wall-facing corridor and shared projections must be resolved globally rather than by last-write wins.
    """
    return _shadow_output(np.asarray(grid, dtype=int))


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        want_overlap = bool(random.getrandbits(1))
        canonical = _generate_canonical_scene(want_overlap)
        if canonical is None:
            continue

        grid = np.rot90(canonical, random.randrange(4)).copy()
        if _is_valid_generated_case(grid, want_overlap):
            return grid

    raise RuntimeError("failed to generate a valid shadow-overlap ARC input")
