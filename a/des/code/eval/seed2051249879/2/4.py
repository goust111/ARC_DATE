import random
from collections import deque

import numpy as np


AZURE = 8
BLACK = 0
OBJECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_RETRIES = 400
OBJECT_RETRIES = 120
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _enclosed_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    passable = (g == AZURE) | (g == BLACK)
    regions: list[dict[str, object]] = []
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        black_cells = [(r, c) for r, c in comp if int(g[r, c]) == BLACK]
        border_colors = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not passable[nr, nc]:
                    border_colors.add(int(g[nr, nc]))
        regions.append(
            {
                "coords": comp,
                "bbox": _bbox(comp),
                "black": black_cells,
                "border_colors": border_colors,
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _enclosed_regions(g):
        coords = region["coords"]
        black_cells = region["black"]
        if not black_cells:
            continue

        ir0, ic0, _, _ = region["bbox"]
        sr0, sc0, sr1, sc1 = _bbox(black_cells)
        tile = g[sr0 : sr1 + 1, sc0 : sc1 + 1] == BLACK
        th, tw = tile.shape

        for r, c in coords:
            rr = (r - ir0) % th
            cc = (c - ic0) % tw
            out[r, c] = BLACK if tile[rr, cc] else AZURE

    return out


def _random_stamp(rng: random.Random) -> np.ndarray:
    stamp_h = rng.randint(1, 3)
    stamp_w = rng.randint(1, 3)
    cells = {(0, 0)}

    r = 0
    c = 0
    steps = ["D"] * (stamp_h - 1) + ["R"] * (stamp_w - 1)
    rng.shuffle(steps)
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        cells.add((r, c))

    target_area = rng.randint(len(cells), stamp_h * stamp_w)
    while len(cells) < target_area:
        frontier = set()
        for cr, cc in cells:
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < stamp_h and 0 <= nc < stamp_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(rng.choice(tuple(frontier)))

    stamp = np.zeros((stamp_h, stamp_w), dtype=bool)
    for cr, cc in cells:
        stamp[cr, cc] = True
    return stamp


def _row_shape(
    rng: random.Random, inner_h: int, inner_w: int, stamp_h: int, stamp_w: int
) -> np.ndarray:
    lengths: list[int] = []
    for r in range(inner_h):
        floor = stamp_w if r < stamp_h else 1
        if r == 0:
            length = rng.randint(floor, inner_w)
        else:
            length = lengths[-1] + rng.randint(-2, 2)
            length = max(floor, min(inner_w, length))
            if rng.random() < 0.25:
                length = rng.randint(floor, inner_w)
        lengths.append(length)
    lengths[rng.randrange(inner_h)] = inner_w

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, length in enumerate(lengths):
        mask[r, :length] = True
    return mask


def _col_shape(
    rng: random.Random, inner_h: int, inner_w: int, stamp_h: int, stamp_w: int
) -> np.ndarray:
    heights: list[int] = []
    for c in range(inner_w):
        floor = stamp_h if c < stamp_w else 1
        if c == 0:
            height = rng.randint(floor, inner_h)
        else:
            height = heights[-1] + rng.randint(-2, 2)
            height = max(floor, min(inner_h, height))
            if rng.random() < 0.25:
                height = rng.randint(floor, inner_h)
        heights.append(height)
    heights[rng.randrange(inner_w)] = inner_h

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for c, height in enumerate(heights):
        mask[:height, c] = True
    return mask


def _corner_notch_shape(
    rng: random.Random, inner_h: int, inner_w: int, stamp_h: int, stamp_w: int
) -> np.ndarray:
    mask = np.ones((inner_h, inner_w), dtype=bool)
    if inner_h > stamp_h and inner_w > stamp_w:
        cut_r = rng.randint(stamp_h, inner_h - 1)
        cut_c = rng.randint(stamp_w, inner_w - 1)
        mask[cut_r:, cut_c:] = False
    return mask


def _random_interior_shape(
    rng: random.Random, inner_h: int, inner_w: int, stamp_h: int, stamp_w: int
) -> tuple[np.ndarray, str]:
    mode = rng.choice(("rect", "row", "col", "notch"))
    if mode == "rect":
        mask = np.ones((inner_h, inner_w), dtype=bool)
    elif mode == "row":
        mask = _row_shape(rng, inner_h, inner_w, stamp_h, stamp_w)
    elif mode == "col":
        mask = _col_shape(rng, inner_h, inner_w, stamp_h, stamp_w)
    else:
        mask = _corner_notch_shape(rng, inner_h, inner_w, stamp_h, stamp_w)
    return mask, mode


def _add_protrusions(
    rng: random.Random, patch: np.ndarray, color: int, outer_r0: int, outer_c0: int, outer_h: int, outer_w: int
) -> None:
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    count = rng.choice((0, 1, 1, 2))
    for side in sides[:count]:
        if side in ("top", "bottom"):
            height = rng.randint(1, 2)
            width = rng.randint(max(2, outer_w // 3), outer_w)
            start = rng.randint(0, outer_w - width)
            if side == "top":
                r0 = outer_r0 - height
                r1 = outer_r0
                c0 = outer_c0 + start
                c1 = c0 + width
            else:
                r0 = outer_r0 + outer_h
                r1 = r0 + height
                c0 = outer_c0 + start
                c1 = c0 + width
        else:
            width = rng.randint(1, 2)
            height = rng.randint(max(2, outer_h // 3), outer_h)
            start = rng.randint(0, outer_h - height)
            if side == "left":
                r0 = outer_r0 + start
                r1 = r0 + height
                c0 = outer_c0 - width
                c1 = outer_c0
            else:
                r0 = outer_r0 + start
                r1 = r0 + height
                c0 = outer_c0 + outer_w
                c1 = c0 + width
        patch[r0:r1, c0:c1] = color


def _crop_non_bg(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != AZURE)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1]


def _build_object_patch(
    rng: random.Random, color: int, interior_mask: np.ndarray, stamp: np.ndarray, wall: int
) -> np.ndarray:
    inner_h, inner_w = interior_mask.shape
    outer_h = inner_h + 2 * wall
    outer_w = inner_w + 2 * wall
    pad = 3
    patch = np.full((outer_h + 2 * pad, outer_w + 2 * pad), AZURE, dtype=int)

    outer_r0 = pad
    outer_c0 = pad
    patch[outer_r0 : outer_r0 + outer_h, outer_c0 : outer_c0 + outer_w] = color

    _add_protrusions(rng, patch, color, outer_r0, outer_c0, outer_h, outer_w)

    inner_r0 = outer_r0 + wall
    inner_c0 = outer_c0 + wall
    for r in range(inner_h):
        for c in range(inner_w):
            if interior_mask[r, c]:
                patch[inner_r0 + r, inner_c0 + c] = AZURE

    stamp_h, stamp_w = stamp.shape
    for r in range(stamp_h):
        for c in range(stamp_w):
            if stamp[r, c]:
                patch[inner_r0 + r, inner_c0 + c] = BLACK

    return _crop_non_bg(patch)


def _make_object_patch(rng: random.Random, color: int, inner_span_max: int) -> np.ndarray | None:
    for _ in range(OBJECT_RETRIES):
        stamp = _random_stamp(rng)
        stamp_h, stamp_w = stamp.shape

        min_h = max(4, stamp_h + 1)
        min_w = max(4, stamp_w + 1)
        if min_h > inner_span_max or min_w > inner_span_max + 1:
            continue

        inner_h = rng.randint(min_h, inner_span_max)
        inner_w = rng.randint(min_w, inner_span_max + 1)
        interior_mask, mode = _random_interior_shape(rng, inner_h, inner_w, stamp_h, stamp_w)

        area = int(interior_mask.sum())
        if area <= int(stamp.sum()) + 2:
            continue
        if not np.any(interior_mask[stamp_h:, :]) and not np.any(interior_mask[:, stamp_w:]):
            continue
        if mode != "rect" and np.array_equal(interior_mask, np.ones_like(interior_mask)):
            continue

        wall = rng.choice((1, 1, 2))
        patch = _build_object_patch(rng, color, interior_mask, stamp, wall)
        if np.array_equal(main(patch), patch):
            continue
        return patch
    return None


def _can_place(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not np.any(reserved[r0:r1, c0:c1])


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    height, width = patch.shape
    target = grid[top : top + height, left : left + width]
    assert np.all(target == AZURE)
    target[:, :] = patch
    reserved[top : top + height, left : left + width] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(22, 30)
    grid_w = rng.randint(22, 30)
    grid = np.full((grid_h, grid_w), AZURE, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    object_count = rng.choice((2, 3, 4))
    inner_span_max = {2: 9, 3: 8, 4: 7}[object_count]
    colors = rng.sample(OBJECT_COLORS, object_count)

    for color in colors:
        patch = _make_object_patch(rng, color, inner_span_max)
        if patch is None:
            return None

        ph, pw = patch.shape
        if ph >= grid_h - 2 or pw >= grid_w - 2:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(1, grid_h - ph - 1)
            left = rng.randint(1, grid_w - pw - 1)
            if _can_place(reserved, top, left, ph, pw):
                _place_patch(grid, reserved, patch, top, left)
                placed = True
                break
        if not placed:
            return None

    regions = _enclosed_regions(grid)
    if len(regions) != object_count:
        return None
    if any(len(region["black"]) == 0 for region in regions):
        return None
    for region in regions:
        black_cells = region["black"]
        ir0, ic0, _, _ = region["bbox"]
        sr0, sc0, _, _ = _bbox(black_cells)
        if sr0 != ir0 or sc0 != ic0:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == BLACK) <= np.count_nonzero(grid == BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
