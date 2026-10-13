import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _dedupe_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for cell in coords:
        if not out or out[-1] != cell:
            out.append(cell)
    return out


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _classify_open_contour(comp: dict) -> dict | None:
    coords = comp["coords"]
    if len(coords) < 5:
        return None
    coord_set = set(coords)
    degrees = {cell: sum((nbr in coord_set) for nbr in _neighbors4(*cell)) for cell in coord_set}
    if any(deg not in {1, 2} for deg in degrees.values()):
        return None
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return None
    (r1, c1), (r2, c2) = endpoints
    if r1 == r2:
        axis_kind = "h"
        axis = r1
        offsets = [r - axis for (r, c) in coords if (r, c) not in endpoints]
    elif c1 == c2:
        axis_kind = "v"
        axis = c1
        offsets = [c - axis for (r, c) in coords if (r, c) not in endpoints]
    else:
        return None
    if not offsets or any(offset == 0 for offset in offsets):
        return None
    side = 1 if offsets[0] > 0 else -1
    if any((1 if offset > 0 else -1) != side for offset in offsets):
        return None
    return {
        "color": int(comp["color"]),
        "coords": coords,
        "coord_set": coord_set,
        "axis_kind": axis_kind,
        "axis": int(axis),
    }


def _reflect_coords(coords: set[tuple[int, int]], axis_kind: str, axis: int) -> set[tuple[int, int]]:
    if axis_kind == "h":
        return {(2 * axis - r, c) for (r, c) in coords}
    return {(r, 2 * axis - c) for (r, c) in coords}


def _enclosed_region(shape: tuple[int, int], ring_coords: set[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    for r, c in ring_coords:
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError("ring left the grid")
        blocked[r, c] = True
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return (~blocked) & (~outside)


def _ring_from_contour(contour: dict, shape: tuple[int, int]) -> dict:
    full = set(contour["coord_set"])
    full |= _reflect_coords(contour["coord_set"], contour["axis_kind"], contour["axis"])
    inside = _enclosed_region(shape, full)
    ring_mask = np.zeros(shape, dtype=bool)
    for r, c in full:
        ring_mask[r, c] = True
    return {"color": contour["color"], "full": full, "ring_mask": ring_mask, "inside": inside}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each open contour across its own endpoint-defined axis, then keep only dots in the annulus between the two completed loops.

    Input structure:
    - A black background contains exactly two colored one-cell-wide open contours plus scattered colored dots.
    - Each contour is a single 4-connected path with exactly two endpoints, and those endpoints lie on one shared row or one shared column.
    - Aside from its endpoints, each contour stays strictly on one side of its own axis row or axis column.

    Rule:
    - Detect the two open contour components and infer each mirror axis from its endpoint pair.
    - Reflect each contour across its own axis to form a closed loop of the same color.
    - The two completed loops become nested; keep only non-contour pixels that lie inside the outer loop and outside the inner loop.
    - Remove all other dots and draw both completed loops.

    Key insight:
    - Nesting is not reliable in the half-contour input: each contour must be completed first, then the kept dots are the set difference of the two newly created interiors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    shape = tuple(map(int, g.shape))

    components = _find_components(g)
    contours = []
    contour_cells: set[tuple[int, int]] = set()
    for comp in components:
        contour = _classify_open_contour(comp)
        if contour is not None:
            contours.append(contour)
            contour_cells |= contour["coord_set"]
    assert len(contours) == 2, "expected exactly two open contours"

    rings = [_ring_from_contour(contour, shape) for contour in contours]
    rings.sort(key=lambda ring: int(ring["inside"].sum()))
    inner, outer = rings

    annulus = outer["inside"] & (~inner["inside"])
    out = np.zeros_like(g)
    for ring in rings:
        for r, c in ring["full"]:
            out[r, c] = int(ring["color"])

    for r in range(shape[0]):
        for c in range(shape[1]):
            if int(g[r, c]) != BG and (r, c) not in contour_cells and annulus[r, c]:
                out[r, c] = int(g[r, c])
    return out


def _partition(total: int, parts: int, min_piece: int = 2) -> list[int]:
    remain = total - parts * min_piece
    pieces = [min_piece] * parts
    for _ in range(remain):
        pieces[random.randrange(parts)] += 1
    random.shuffle(pieces)
    return pieces


def _sample_depths(count: int, max_depth: int) -> list[int]:
    depths = [random.randint(1, max_depth) for _ in range(count)]
    depths[random.randrange(count)] = max_depth
    return depths


def _half_horizontal(axis: int, c0: int, widths: list[int], depths: list[int], side: str) -> list[tuple[int, int]]:
    sign = -1 if side == "up" else 1
    coords = [(axis, c0)]
    col = c0
    cur_r = axis
    for width, depth in zip(widths, depths):
        target_r = axis + sign * depth
        step = 1 if target_r > cur_r else -1
        for r in range(cur_r + step, target_r + step, step):
            coords.append((r, col))
        cur_r = target_r
        for c in range(col + 1, col + width + 1):
            coords.append((cur_r, c))
        col += width
    step = 1 if axis > cur_r else -1
    for r in range(cur_r + step, axis + step, step):
        coords.append((r, col))
    return _dedupe_path(coords)


def _half_vertical(axis: int, r0: int, heights: list[int], depths: list[int], side: str) -> list[tuple[int, int]]:
    sign = -1 if side == "left" else 1
    coords = [(r0, axis)]
    row = r0
    cur_c = axis
    for height, depth in zip(heights, depths):
        target_c = axis + sign * depth
        step = 1 if target_c > cur_c else -1
        for c in range(cur_c + step, target_c + step, step):
            coords.append((row, c))
        cur_c = target_c
        for r in range(row + 1, row + height + 1):
            coords.append((r, cur_c))
        row += height
    step = 1 if axis > cur_c else -1
    for c in range(cur_c + step, axis + step, step):
        coords.append((row, c))
    return _dedupe_path(coords)


def _build_loop(shape: tuple[int, int], kind: str, span: int, depth: int, axis: int, start: int) -> dict:
    segment_count = random.randint(2, 4)
    steps = _partition(span, segment_count, min_piece=2)
    depths = _sample_depths(segment_count, depth)
    if kind == "h":
        up = _half_horizontal(axis, start, steps, depths, "up")
        down = _half_horizontal(axis, start, steps, depths, "down")
        full = set(up) | set(down)
    else:
        left = _half_vertical(axis, start, steps, depths, "left")
        right = _half_vertical(axis, start, steps, depths, "right")
        full = set(left) | set(right)
    inside = _enclosed_region(shape, full)
    return {"kind": kind, "axis": axis, "start": start, "steps": steps, "depths": depths, "full": full, "inside": inside}


def _random_loop(shape: tuple[int, int], kind: str, *, large: bool) -> dict:
    h, w = shape
    if kind == "h":
        span = random.randint(10 if large else 7, min(18 if large else 14, w - 5))
        depth = random.randint(3 if large else 2, min(7 if large else 5, h // 3, span // 2))
        axis = random.randint(depth + 2, h - depth - 3)
        start = random.randint(2, w - span - 2)
    else:
        span = random.randint(10 if large else 7, min(18 if large else 14, h - 5))
        depth = random.randint(3 if large else 2, min(7 if large else 5, w // 3, span // 2))
        axis = random.randint(depth + 2, w - depth - 3)
        start = random.randint(2, h - span - 2)
    return _build_loop(shape, kind, span, depth, axis, start)


def _loop_half(loop: dict, side: str) -> list[tuple[int, int]]:
    if loop["kind"] == "h":
        return _half_horizontal(loop["axis"], loop["start"], loop["steps"], loop["depths"], side)
    return _half_vertical(loop["axis"], loop["start"], loop["steps"], loop["depths"], side)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _contains_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _choose_display_sides(outer: dict, inner: dict) -> tuple[str, str]:
    outer_sides = ("up", "down") if outer["kind"] == "h" else ("left", "right")
    inner_sides = ("up", "down") if inner["kind"] == "h" else ("left", "right")
    scored = []
    for so in outer_sides:
        bo = _bbox(_loop_half(outer, so))
        for si in inner_sides:
            bi = _bbox(_loop_half(inner, si))
            contains = _contains_bbox(bo, bi) or _contains_bbox(bi, bo)
            overlap_h = max(0, min(bo[2], bi[2]) - max(bo[0], bi[0]) + 1)
            overlap_w = max(0, min(bo[3], bi[3]) - max(bo[1], bi[1]) + 1)
            scored.append((contains, -(overlap_h * overlap_w), random.random(), so, si))
    scored.sort()
    return scored[0][3], scored[0][4]


def _sample_nested_loops(shape: tuple[int, int]) -> tuple[dict, dict]:
    h, w = shape
    for _ in range(150):
        outer_kind = random.choice(("h", "v"))
        inner_kind = "v" if outer_kind == "h" else "h"
        outer = _random_loop(shape, outer_kind, large=True)
        if int(outer["inside"].sum()) < 60:
            continue
        rows = np.where(outer["inside"].any(axis=1))[0]
        cols = np.where(outer["inside"].any(axis=0))[0]
        if len(rows) < 6 or len(cols) < 6:
            continue
        for _ in range(250):
            if inner_kind == "h":
                span_hi = min(12, len(cols) - 2, w - 5)
                if span_hi < 6:
                    break
                span = random.randint(6, span_hi)
                depth = random.randint(2, min(4, span // 2, h // 3))
                axis_lo = max(depth + 2, int(rows[1]))
                axis_hi = min(h - depth - 3, int(rows[-2]))
                start_lo = max(2, int(cols[0]))
                start_hi = min(w - span - 2, int(cols[-1]) - span)
                if axis_lo > axis_hi or start_lo > start_hi:
                    continue
                inner = _build_loop(shape, "h", span, depth, random.randint(axis_lo, axis_hi), random.randint(start_lo, start_hi))
            else:
                span_hi = min(12, len(rows) - 2, h - 5)
                if span_hi < 6:
                    break
                span = random.randint(6, span_hi)
                depth = random.randint(2, min(4, span // 2, w // 3))
                axis_lo = max(depth + 2, int(cols[1]))
                axis_hi = min(w - depth - 3, int(cols[-2]))
                start_lo = max(2, int(rows[0]))
                start_hi = min(h - span - 2, int(rows[-1]) - span)
                if axis_lo > axis_hi or start_lo > start_hi:
                    continue
                inner = _build_loop(shape, "v", span, depth, random.randint(axis_lo, axis_hi), random.randint(start_lo, start_hi))
            inner_ring = np.zeros(shape, dtype=bool)
            for r, c in inner["full"]:
                inner_ring[r, c] = True
            if int(inner["inside"].sum()) < 8:
                continue
            if any(cell in outer["full"] for cell in inner["full"]):
                continue
            if not np.all(outer["inside"][inner_ring | inner["inside"]]):
                continue
            annulus = outer["inside"] & (~inner["inside"]) & (~inner_ring)
            if int(annulus.sum()) < 8:
                continue
            return outer, inner
    raise RuntimeError(f"failed to sample nested loops for shape {(h, w)}")


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]] | set[tuple[int, int]], color: int):
    for r, c in coords:
        grid[r, c] = int(color)


def _place_dots(grid: np.ndarray, region: np.ndarray, count: int, color_pool: list[int], forbidden: np.ndarray) -> list[tuple[int, int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(region)]
    random.shuffle(coords)
    placed: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r, c in coords:
        if len(placed) >= count:
            break
        if forbidden[r, c] or int(grid[r, c]) != BG:
            continue
        color = random.choice(color_pool)
        grid[r, c] = color
        placed.append((r, c, color))
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                forbidden[rr, cc] = True
    return placed


def _render_expected(shape: tuple[int, int], outer_full: set[tuple[int, int]], outer_color: int,
                     inner_full: set[tuple[int, int]], inner_color: int,
                     annulus: np.ndarray, dots: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    _paint_cells(out, outer_full, outer_color)
    _paint_cells(out, inner_full, inner_color)
    for r, c, color in dots:
        if annulus[r, c]:
            out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(20, 28), random.randint(20, 28))
    outer, inner = _sample_nested_loops(shape)
    outer_side, inner_side = _choose_display_sides(outer, inner)

    outer_half = _loop_half(outer, outer_side)
    inner_half = _loop_half(inner, inner_side)
    annulus = outer["inside"] & (~inner["inside"])
    inner_region = inner["inside"].copy()
    outer_ring = np.zeros(shape, dtype=bool)
    inner_ring = np.zeros(shape, dtype=bool)
    for r, c in outer["full"]:
        outer_ring[r, c] = True
    for r, c in inner["full"]:
        inner_ring[r, c] = True
    outside_region = (~outer["inside"]) & (~outer_ring) & (~inner_ring)
    annulus &= (~outer_ring) & (~inner_ring)
    inner_region &= (~inner_ring)

    colors = random.sample(PALETTE, k=6)
    outer_color, inner_color = colors[:2]
    dot_pool = colors[2:]

    inp = np.zeros(shape, dtype=int)
    _paint_cells(inp, outer_half, outer_color)
    _paint_cells(inp, inner_half, inner_color)

    forbidden = np.zeros(shape, dtype=bool)
    for r, c in outer_half + inner_half:
        for rr in range(max(0, r - 1), min(shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(shape[1], c + 2)):
                forbidden[rr, cc] = True

    annulus_count = random.randint(2, 5)
    inner_count = random.randint(1, 3)
    outer_count = random.randint(2, 5)
    annulus_dots = _place_dots(inp, annulus, annulus_count, dot_pool, forbidden)
    inner_dots = _place_dots(inp, inner_region, inner_count, dot_pool, forbidden)
    outer_dots = _place_dots(inp, outside_region, outer_count, dot_pool, forbidden)
    if len(annulus_dots) < 2 or len(inner_dots) < 1 or len(outer_dots) < 2:
        return None

    all_dots = annulus_dots + inner_dots + outer_dots
    expected = _render_expected(shape, outer["full"], outer_color, inner["full"], inner_color, annulus, all_dots)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
