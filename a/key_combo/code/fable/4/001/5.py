import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
RETRY_LIMIT = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _seed_and_pairs(g: np.ndarray):
    vals, counts = np.unique(g, return_counts=True)
    seed = []
    pairs = {}
    for value, count in zip(vals, counts):
        color = int(value)
        n = int(count)
        if color == BG:
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        if n == 1:
            seed.append((color, coords[0]))
        elif n == 2:
            (r0, c0), (r1, c1) = coords
            assert r0 == r1 or c0 == c1, "paired markers must share a row or column"
            pairs[color] = (coords[0], coords[1])
        else:
            raise AssertionError("each non-seed color must occur exactly twice")

    assert len(seed) == 1, "expected exactly one singleton seed color"
    assert pairs, "expected at least one same-color pair"
    return seed[0], pairs


def _add_segment(layer: np.ndarray, color: int, a: tuple[int, int], b: tuple[int, int]) -> None:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        cells = [(r0, c) for c in range(lo, hi + 1)]
    else:
        lo, hi = sorted((r0, r1))
        cells = [(r, c0) for r in range(lo, hi + 1)]

    for r, c in cells:
        old = int(layer[r, c])
        assert old in (BG, int(color)), "same-orientation line overlap is ambiguous"
        layer[r, c] = int(color)


def _draw_wall_layers(g: np.ndarray, pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]]):
    horizontal = np.zeros_like(g, dtype=int)
    vertical = np.zeros_like(g, dtype=int)

    for color, (a, b) in pairs.items():
        if a[0] == b[0]:
            _add_segment(horizontal, color, a, b)
    for color, (a, b) in pairs.items():
        if a[1] == b[1]:
            _add_segment(vertical, color, a, b)
    return horizontal, vertical


def _draw_walls_only(g: np.ndarray):
    (seed_color, seed_coord), pairs = _seed_and_pairs(g)
    horizontal, vertical = _draw_wall_layers(g, pairs)
    sr, sc = seed_coord
    assert int(horizontal[sr, sc]) == BG and int(vertical[sr, sc]) == BG, "line must not cover seed"

    out = np.full_like(g, BG, dtype=int)
    mask_h = horizontal != BG
    mask_v = vertical != BG
    out[mask_h] = horizontal[mask_h]
    out[mask_v] = vertical[mask_v]
    out[sr, sc] = int(seed_color)
    return out, seed_color, seed_coord, horizontal, vertical


def _reachable_black(canvas: np.ndarray, seed_coord: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = canvas.shape
    seen = {seed_coord}
    reachable: set[tuple[int, int]] = set()
    q = deque([seed_coord])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                continue
            if int(canvas[nr, nc]) != BG:
                continue
            seen.add((nr, nc))
            reachable.add((nr, nc))
            q.append((nr, nc))
    return reachable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color marker pairs into walls, then flood-fill the seed chamber.

    Input structure:
    - The grid has a black background, exactly one non-black color appearing once
      as the seed pixel, and every other non-black color appearing exactly twice.
    - Each same-color pair is aligned in one row or one column, so it determines
      a single horizontal or vertical segment.

    Rule:
    - Draw every horizontal pair segment and every vertical pair segment between
      its two endpoints, with vertical segment colors kept at horizontal/vertical
      intersections.
    - After all new wall segments exist, flood-fill from the singleton seed using
      four-connectivity through black cells only.
    - Recolor every reachable black cell with the seed color, while preserving the
      seed pixel and all wall segment pixels.

    Key insight:
    - The enclosure is created by connecting the marker pairs first; only after
      those new walls are present is the seed's reachable chamber determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    out, seed_color, seed_coord, _horizontal, _vertical = _draw_walls_only(g)
    for r, c in _reachable_black(out, seed_coord):
        out[r, c] = int(seed_color)
    return out


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return {(r0, c) for c in range(lo, hi + 1)}
    lo, hi = sorted((r0, r1))
    return {(r, c0) for r in range(lo, hi + 1)}


def _pick_span(total: int) -> tuple[int, int]:
    interior = random.randint(5, min(11, total - 6))
    start = random.randint(2, total - interior - 4)
    end = start + interior + 1
    return start, end


def _pick_between(lo: int, hi: int) -> int:
    if hi - lo >= 6:
        return random.randint(lo + 3, hi - 3)
    return random.randint(lo + 2, hi - 2)


def _paint_pair(grid: np.ndarray, color: int, a: tuple[int, int], b: tuple[int, int]) -> bool:
    for r, c in (a, b):
        if int(grid[r, c]) != BG:
            return False
    grid[a] = int(color)
    grid[b] = int(color)
    return True


def _border_cell(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return r in (0, h - 1) or c in (0, w - 1)


def _try_add_distractor(
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]],
    grid: np.ndarray,
    color: int,
    rect: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    rt, cl, rb, cr = rect
    occupied = {a for _color, a, _b in pairs} | {b for _color, _a, b in pairs}

    for _ in range(80):
        orient = random.choice(("h", "v"))
        if orient == "h":
            zones = [range(1, max(2, rt - 1)), range(min(h - 1, rb + 2), h - 1)]
            rows = [r for zone in zones for r in zone]
            if not rows:
                continue
            r = random.choice(rows)
            length = random.randint(3, min(8, w - 3))
            c0 = random.randint(1, w - length - 1)
            c1 = c0 + length - 1
            a, b = (r, c0), (r, c1)
        else:
            zones = [range(1, max(2, cl - 1)), range(min(w - 1, cr + 2), w - 1)]
            cols = [c for zone in zones for c in zone]
            if not cols:
                continue
            c = random.choice(cols)
            length = random.randint(3, min(8, h - 3))
            r0 = random.randint(1, h - length - 1)
            r1 = r0 + length - 1
            a, b = (r0, c), (r1, c)

        if a in occupied or b in occupied:
            continue
        if not _line_cells(a, b).isdisjoint({cell for _col, p, q in pairs for cell in _line_cells(p, q) if p[0] == q[0] == a[0] or p[1] == q[1] == a[1]}):
            continue
        pairs.append((int(color), a, b))
        return True
    return False


def _render_candidate() -> np.ndarray | None:
    h = random.randint(13, 26)
    w = random.randint(13, 26)
    rt, rb = _pick_span(h)
    cl, cr = _pick_span(w)

    available = list(COLORS)
    random.shuffle(available)
    seed_color = available.pop()

    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    def add_pair(a: tuple[int, int], b: tuple[int, int]) -> None:
        pairs.append((available.pop(), a, b))

    add_pair((rt, cl), (rt, cr))
    add_pair((rb, cl), (rb, cr))
    add_pair((rt - 1, cl), (rb + 1, cl))
    add_pair((rt - 1, cr), (rb + 1, cr))

    split_kind = random.choice(("vertical", "horizontal"))
    cdiv = None
    rdiv = None
    if split_kind == "vertical":
        cdiv = _pick_between(cl, cr)
        add_pair((rt - 1, cdiv), (rb + 1, cdiv))
        if random.random() < 0.45 and available:
            rdiv = _pick_between(rt, rb)
            add_pair((rdiv, cl), (rdiv, cr))
    else:
        rdiv = _pick_between(rt, rb)
        add_pair((rdiv, cl), (rdiv, cr))
        if random.random() < 0.45 and available:
            cdiv = _pick_between(cl, cr)
            add_pair((rt - 1, cdiv), (rb + 1, cdiv))

    possible_rows = list(range(rt + 1, rb))
    possible_cols = list(range(cl + 1, cr))
    if rdiv is not None:
        side = random.choice(("above", "below"))
        possible_rows = [r for r in possible_rows if r < rdiv] if side == "above" else [r for r in possible_rows if r > rdiv]
    if cdiv is not None:
        side = random.choice(("left", "right"))
        possible_cols = [c for c in possible_cols if c < cdiv] if side == "left" else [c for c in possible_cols if c > cdiv]
    if not possible_rows or not possible_cols:
        return None
    seed = (random.choice(possible_rows), random.choice(possible_cols))

    grid = np.zeros((h, w), dtype=int)
    for color, a, b in pairs:
        if not _paint_pair(grid, color, a, b):
            return None

    for _ in range(random.randint(0, min(2, len(available)))):
        color = available.pop()
        if not _try_add_distractor(pairs, grid, color, (rt, cl, rb, cr)):
            return None
        _color, a, b = pairs[-1]
        if not _paint_pair(grid, color, a, b):
            return None

    if int(grid[seed]) != BG:
        return None
    grid[seed] = int(seed_color)

    return grid


def _generator_semantics_ok(grid: np.ndarray) -> bool:
    try:
        wall, seed_color, seed_coord, horizontal, vertical = _draw_walls_only(grid)
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False

    crossings = (horizontal != BG) & (vertical != BG)
    if int(np.count_nonzero(crossings)) < 4:
        return False
    if not np.all(wall[crossings] == vertical[crossings]):
        return False
    visible_overwrite = crossings & (grid == horizontal) & (horizontal != vertical)
    if not bool(np.any(visible_overwrite)):
        return False

    after_reach = _reachable_black(wall, seed_coord)
    if len(after_reach) < 4:
        return False
    if any(_border_cell(grid.shape, cell) for cell in after_reach):
        return False

    original_reach = _reachable_black(grid, seed_coord)
    if not any(_border_cell(grid.shape, cell) for cell in original_reach):
        return False
    if len(original_reach) <= len(after_reach):
        return False

    changed_to_seed = {(int(r), int(c)) for r, c in np.argwhere((out == seed_color) & (wall == BG))}
    if changed_to_seed != after_reach:
        return False

    wall_mask = wall != BG
    if not np.array_equal(out[wall_mask], wall[wall_mask]):
        return False
    if int(out[seed_coord]) != int(seed_color):
        return False

    inner_bbox = np.argwhere(wall != BG)
    if inner_bbox.size == 0:
        return False
    r0, c0 = inner_bbox.min(axis=0)
    r1, c1 = inner_bbox.max(axis=0)
    chamber_area = wall[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    unfilled_black = int(np.count_nonzero((chamber_area == BG) & (out[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)] == BG)))
    return unfilled_black > 0


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _render_candidate()
        if grid is None:
            continue
        if _generator_semantics_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid fable/4/5 input")
