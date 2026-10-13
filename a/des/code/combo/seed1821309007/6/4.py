import random
from collections import Counter, deque

import numpy as np


BG = 7       # orange
MARKER = 9   # maroon
PALETTE = [1, 2, 3, 4, 5, 6, 8]
ANGLES = [0, 90, 180, 270]
MAX_ATTEMPTS = 400
SIZE_RANGE = (10, 19)
OBJECT_COUNT_RANGE = (1, 5)
SHAPE_SIZES = [
    (1, 1),
    (1, 3),
    (3, 1),
    (1, 5),
    (5, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 5),
    (5, 3),
    (4, 4),
]


def _majority_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_marker(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    h, w = grid.shape
    candidates = []
    for coords in _components(grid != bg):
        if len(coords) != 2:
            continue
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        rows = {int(r) for r, _c in coords}
        cols = {int(c) for _r, c in coords}
        angle = None
        if len(rows) == 1 and max(cols) - min(cols) == 1:
            row = next(iter(rows))
            if row == 0:
                angle = 0
            elif row == h - 1:
                angle = 180
        elif len(cols) == 1 and max(rows) - min(rows) == 1:
            col = next(iter(cols))
            if col == w - 1:
                angle = 90
            elif col == 0:
                angle = 270
        if angle is not None:
            candidates.append((coords, angle, next(iter(colors))))
    assert len(candidates) == 1, "expected exactly one 2-pixel edge marker"
    coords, angle, color = candidates[0]
    assert color != bg, "marker color must differ from background"
    return coords, angle


def _rotate_patch_about_center(patch: np.ndarray, angle: int) -> tuple[np.ndarray, int, int]:
    turns = (angle // 90) % 4
    rotated = np.rot90(patch, -turns)
    h, w = patch.shape
    rh, rw = rotated.shape
    row_shift = int(round((h - rh) / 2.0))
    col_shift = int(round((w - rw) / 2.0))
    return rotated, row_shift, col_shift


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a global rotation from a 2-pixel edge marker, then rotate each interior object around its own bbox center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _majority_color(g)
    marker_coords, angle = _find_marker(g, bg)

    work = g.copy()
    for r, c in marker_coords:
        work[r, c] = bg

    out = np.full_like(g, bg)
    for coords in _components(work != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = work[r0 : r1 + 1, c0 : c1 + 1]
        rotated, dr, dc = _rotate_patch_about_center(patch, angle)
        rr0 = r0 + dr
        cc0 = c0 + dc
        for rr in range(rotated.shape[0]):
            for cc in range(rotated.shape[1]):
                val = int(rotated[rr, cc])
                if val == bg:
                    continue
                gr = rr0 + rr
                gc = cc0 + cc
                if 0 <= gr < g.shape[0] and 0 <= gc < g.shape[1]:
                    assert out[gr, gc] in (bg, val), "rotated objects should not conflict"
                    out[gr, gc] = val
    return out


def _dilate(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    result = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result.add((nr, nc))
    return result


def _random_connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    area_min = 1 if h * w <= 2 else 2
    area_max = min(h * w, max(area_min, (h * w * 2) // 3))
    target = rng.randint(area_min, area_max)
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = {start}
    while len(cells) < target and frontier:
        base = rng.choice(tuple(frontier))
        nbrs = []
        for nr, nc in _neighbors4(*base):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove(base)
            continue
        nxt = rng.choice(nbrs)
        cells.add(nxt)
        frontier.add(nxt)
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    if max(rows) - min(rows) + 1 != h or max(cols) - min(cols) + 1 != w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _marker_cells(angle: int, h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    if angle == 0:
        center = w // 2
        options = [c for c in range(1, w - 2) if c + 1 < w - 1 and ((c + (c + 1)) / 2.0) != center]
        if not options:
            options = list(range(1, w - 2))
        c0 = rng.choice(options)
        return [(0, c0), (0, c0 + 1)]
    if angle == 180:
        center = w // 2
        options = [c for c in range(1, w - 2) if c + 1 < w - 1 and ((c + (c + 1)) / 2.0) != center]
        if not options:
            options = list(range(1, w - 2))
        c0 = rng.choice(options)
        return [(h - 1, c0), (h - 1, c0 + 1)]
    if angle == 90:
        center = h // 2
        options = [r for r in range(1, h - 2) if r + 1 < h - 1 and ((r + (r + 1)) / 2.0) != center]
        if not options:
            options = list(range(1, h - 2))
        r0 = rng.choice(options)
        return [(r0, w - 1), (r0 + 1, w - 1)]
    center = h // 2
    options = [r for r in range(1, h - 2) if r + 1 < h - 1 and ((r + (r + 1)) / 2.0) != center]
    if not options:
        options = list(range(1, h - 2))
    r0 = rng.choice(options)
    return [(r0, 0), (r0 + 1, 0)]


def _rotated_visible_cells(
    top: int, left: int, mask: np.ndarray, angle: int, grid_h: int, grid_w: int
) -> set[tuple[int, int]]:
    patch = np.where(mask, 1, 0)
    rotated, dr, dc = _rotate_patch_about_center(patch, angle)
    rr0 = top + dr
    cc0 = left + dc
    cells = set()
    for r in range(rotated.shape[0]):
        for c in range(rotated.shape[1]):
            if rotated[r, c] != 1:
                continue
            gr = rr0 + r
            gc = cc0 + c
            if 0 <= gr < grid_h and 0 <= gc < grid_w:
                cells.add((gr, gc))
    return cells


def _placeable(
    top: int,
    left: int,
    mask: np.ndarray,
    used_in: set[tuple[int, int]],
    used_out: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    angle: int,
    grid_h: int,
    grid_w: int,
) -> bool:
    input_cells = {(top + r, left + c) for r, c in np.argwhere(mask)}
    if any(cell in blocked or cell in used_in for cell in input_cells):
        return False
    if any(r in (0, grid_h - 1) or c in (0, grid_w - 1) for r, c in input_cells):
        return False
    output_cells = _rotated_visible_cells(top, left, mask, angle, grid_h, grid_w)
    if not output_cells:
        return False
    if used_out & output_cells:
        return False
    return True


def _gen_attempt(rng: random.Random, angle: int) -> np.ndarray | None:
    h = rng.randint(*SIZE_RANGE)
    w = rng.randint(*SIZE_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    marker_cells = _marker_cells(angle, h, w, rng)
    for r, c in marker_cells:
        grid[r, c] = MARKER
    blocked = _dilate(set(marker_cells), h, w, radius=1)

    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    colors = rng.sample(PALETTE, object_count)
    used_in: set[tuple[int, int]] = set()
    used_out: set[tuple[int, int]] = set()
    changed = angle == 0

    for color in colors:
        placed = False
        for _ in range(120):
            sh, sw = rng.choice(SHAPE_SIZES)
            if sh + 2 > h or sw + 2 > w:
                continue
            mask = _random_connected_mask(rng, sh, sw)
            if mask is None:
                continue
            top = rng.randint(1, h - sh)
            left = rng.randint(1, w - sw)
            if not _placeable(top, left, mask, used_in, used_out, blocked, angle, h, w):
                continue

            input_cells = {(top + r, left + c) for r, c in np.argwhere(mask)}
            output_cells = _rotated_visible_cells(top, left, mask, angle, h, w)
            if angle != 0 and input_cells != output_cells:
                changed = True

            for r, c in input_cells:
                grid[r, c] = color
            used_in |= _dilate(input_cells, h, w, radius=1)
            used_out |= output_cells
            placed = True
            break
        if not placed:
            return None

    if not changed:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    angles = ANGLES.copy()
    rng.shuffle(angles)
    for angle in angles:
        for _ in range(MAX_ATTEMPTS // len(ANGLES)):
            grid = _gen_attempt(rng, angle)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate valid input")
