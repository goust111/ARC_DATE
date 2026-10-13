from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
FENCE_COLOR = 5

GRID_MIN = 16
GRID_MAX = 30
OBJECT_COUNT_MIN = 1
OBJECT_COUNT_MAX = 4
INNER_MIN = 3
INNER_MAX = 6
GEN_RETRIES = 400
PLACEMENT_RETRIES = 180


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in DIR4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _marker_is_isolated(grid: np.ndarray, r: int, c: int) -> bool:
    color = int(grid[r, c])
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == color:
            return False
    return True


def _analyze_fence(
    grid: np.ndarray, comp: list[tuple[int, int]], color: int, bg: int
) -> dict | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    pts = set(comp)
    boundary = _boundary_cells(r0, c0, r1, c1)
    if not pts.issubset(boundary):
        return None
    if len(pts) != len(boundary) - 1:
        return None

    missing = list(boundary - pts)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]

    side = None
    marker = None
    if gap_r == r0 and c0 < gap_c < c1:
        side = "top"
        marker = (gap_r - 1, gap_c)
    elif gap_r == r1 and c0 < gap_c < c1:
        side = "bottom"
        marker = (gap_r + 1, gap_c)
    elif gap_c == c0 and r0 < gap_r < r1:
        side = "left"
        marker = (gap_r, gap_c - 1)
    elif gap_c == c1 and r0 < gap_r < r1:
        side = "right"
        marker = (gap_r, gap_c + 1)
    if side is None or marker is None:
        return None

    mr, mc = marker
    if not (0 <= mr < grid.shape[0] and 0 <= mc < grid.shape[1]):
        return None
    marker_color = int(grid[mr, mc])
    if marker_color in (bg, color):
        return None
    if not _marker_is_isolated(grid, mr, mc):
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    shape_mask = (inner != bg) & (inner != color)
    if not np.any(shape_mask):
        return None

    inner_colors = {int(v) for v in np.unique(inner[shape_mask])}
    if len(inner_colors) != 1:
        return None
    shape_color = inner_colors.pop()
    if shape_color == marker_color:
        return None

    local_shape = inner == shape_color
    shape_comps = _components(local_shape)
    if len(shape_comps) != 1:
        return None

    shape = [(r0 + 1 + r, c0 + 1 + c) for r, c in shape_comps[0]]
    return {
        "bbox": (r0, c0, r1, c1),
        "fence_color": color,
        "shape_color": shape_color,
        "shape": shape,
        "gap": (gap_r, gap_c),
        "marker": (mr, mc),
        "marker_color": marker_color,
        "side": side,
    }


def _find_fences(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    fences: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            info = _analyze_fence(g, comp, color, bg)
            if info is not None:
                fences.append(info)
    fences.sort(key=lambda item: item["bbox"])
    return fences


def _mirror_cell(cell: tuple[int, int], fence: dict) -> tuple[int, int]:
    r, c = cell
    r0, c0, r1, c1 = fence["bbox"]
    side = fence["side"]
    if side == "top":
        return 2 * r0 - r, c
    if side == "bottom":
        return 2 * r1 - r, c
    if side == "left":
        return r, 2 * c0 - c
    return r, 2 * c1 - c


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()
    for fence in _find_fences(g):
        for cell in fence["shape"]:
            rr, cc = _mirror_cell(cell, fence)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and int(out[rr, cc]) == bg:
                out[rr, cc] = fence["marker_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each fenced inner shape across the side containing the single border gap,
    then draw the mirrored projection outside the fence using the adjacent marker color.
    """
    return _transform(grid)


def _reservation_bbox(r0: int, c0: int, ih: int, iw: int, side: str, pad: int = 1) -> tuple[int, int, int, int]:
    r1 = r0 + ih + 1
    c1 = c0 + iw + 1
    if side == "top":
        ar0, ac0, ar1, ac1 = r0 - ih, c0, r1, c1
    elif side == "bottom":
        ar0, ac0, ar1, ac1 = r0, c0, r1 + ih, c1
    elif side == "left":
        ar0, ac0, ar1, ac1 = r0, c0 - iw, r1, c1
    else:
        ar0, ac0, ar1, ac1 = r0, c0, r1, c1 + iw
    return ar0 - pad, ac0 - pad, ar1 + pad, ac1 + pad


def _bbox_in_bounds(bbox: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _sample_shape(ih: int, iw: int, rng: random.Random) -> set[tuple[int, int]] | None:
    max_area = ih * iw - 1
    if max_area < 2:
        return None
    density = rng.choice((0.25, 0.4, 0.55, 0.7, 0.85))
    target = max(2, min(max_area, int(round(ih * iw * density))))
    lo = max(2, target - rng.randint(0, 2))
    hi = min(max_area, target + rng.randint(0, 2))
    area = rng.randint(lo, hi)

    seed = (rng.randrange(ih), rng.randrange(iw))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ih and 0 <= nc < iw and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        frontier = list(dict.fromkeys(frontier))
        weights = []
        for r, c in frontier:
            degree = 0
            for dr, dc in DIR4:
                if (r + dr, c + dc) in cells:
                    degree += 1
            weights.append(3 if degree == 1 else 1)
        cells.add(rng.choices(frontier, weights=weights, k=1)[0])
    return cells


def _build_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    bg: int,
    side: str,
    shape_color: int,
    marker_color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        ih = rng.randint(INNER_MIN, INNER_MAX)
        iw = rng.randint(INNER_MIN, INNER_MAX)
        fh = ih + 2
        fw = iw + 2
        if fh + 2 > h or fw + 2 > w:
            continue

        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        reserve = _reservation_bbox(r0, c0, ih, iw, side, pad=1)
        if not _bbox_in_bounds(reserve, h, w):
            continue
        rr0, cc0, rr1, cc1 = reserve
        if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue

        shape = _sample_shape(ih, iw, rng)
        if shape is None:
            continue

        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        if side == "top":
            gap = (r0, rng.randint(c0 + 1, c1 - 1))
            marker = (r0 - 1, gap[1])
        elif side == "bottom":
            gap = (r1, rng.randint(c0 + 1, c1 - 1))
            marker = (r1 + 1, gap[1])
        elif side == "left":
            gap = (rng.randint(r0 + 1, r1 - 1), c0)
            marker = (gap[0], c0 - 1)
        else:
            gap = (rng.randint(r0 + 1, r1 - 1), c1)
            marker = (gap[0], c1 + 1)

        global_shape = {(r0 + 1 + r, c0 + 1 + c) for r, c in shape}
        marker_hit_only = True
        fence_stub = {"bbox": (r0, c0, r1, c1), "side": side}
        for cell in global_shape:
            mirrored = _mirror_cell(cell, fence_stub)
            if mirrored != marker:
                marker_hit_only = False
                break
        if marker_hit_only:
            continue

        grid[r0, c0 : c1 + 1] = FENCE_COLOR
        grid[r1, c0 : c1 + 1] = FENCE_COLOR
        grid[r0 : r1 + 1, c0] = FENCE_COLOR
        grid[r0 : r1 + 1, c1] = FENCE_COLOR
        grid[gap] = bg
        for r, c in global_shape:
            grid[r, c] = shape_color
        grid[marker] = marker_color
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    min_dim = max(GRID_MIN, 10 + 4 * n_objects)
    h = rng.randint(min_dim, GRID_MAX)
    w = rng.randint(min_dim, GRID_MAX)

    bg = rng.choice([c for c in range(10) if c != FENCE_COLOR])
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    palette = [c for c in range(10) if c not in (bg, FENCE_COLOR)]
    rng.shuffle(palette)
    if len(palette) < 2 * n_objects:
        return None

    sides = list(SIDES)
    rng.shuffle(sides)
    sides = sides[:n_objects]

    for i in range(n_objects):
        if not _build_object(
            grid=grid,
            occupied=occupied,
            bg=bg,
            side=sides[i],
            shape_color=palette[2 * i],
            marker_color=palette[2 * i + 1],
            rng=rng,
        ):
            return None

    if _bg_color(grid) != bg:
        return None
    fences = _find_fences(grid)
    if len(fences) != n_objects:
        return None
    if sorted(f["side"] for f in fences) != sorted(sides):
        return None
    if np.array_equal(_transform(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
