import random
from typing import Iterable

import numpy as np


ARROW_COLOR = 3  # green
FENCE_COLOR = 5  # grey
BASE_ARROW_UP = np.array(
    [
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=int,
)
DIR_TO_TEMPLATE = {
    (-1, 0): BASE_ARROW_UP,
    (0, 1): np.rot90(BASE_ARROW_UP, 3),
    (1, 0): np.rot90(BASE_ARROW_UP, 2),
    (0, -1): np.rot90(BASE_ARROW_UP, 1),
}
DIRECTIONS = list(DIR_TO_TEMPLATE)

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
OBJECT_COUNT_RANGE = (1, 3)
BLOCK_SIZE_RANGE = (3, 6)
DISTANCE_RANGE = (4, 8)
FENCE_EXT_RANGE = (0, 2)
PLACEMENT_RETRIES = 100
GEN_RETRIES = 300


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
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
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_component(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in points:
        arr[r - r0, c - c0] = 1
    return arr


def _find_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rects: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in {bg, ARROW_COLOR, FENCE_COLOR}:
            continue
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 2 or w < 2:
                continue
            if len(pts) != h * w:
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            rects.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "h": h,
                    "w": w,
                }
            )
    return rects


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for pts in _components(grid == ARROW_COLOR):
        arr = _normalize_component(pts)
        if arr.shape != (3, 3):
            continue
        direction = None
        for dvec, template in DIR_TO_TEMPLATE.items():
            if np.array_equal(arr, template):
                direction = dvec
                break
        if direction is None:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        arrows.append({"dir": direction, "r0": r0, "c0": c0, "r1": r1, "c1": c1})
    return arrows


def _rect_for_arrow(arrow: dict, rects: list[dict], used: set[int]) -> dict | None:
    dr, dc = arrow["dir"]
    matches: list[tuple[int, dict]] = []
    for idx, rect in enumerate(rects):
        if idx in used:
            continue
        if (dr, dc) == (-1, 0):
            ok = rect["r0"] == arrow["r1"] + 1 and rect["c0"] <= arrow["c0"] and rect["c1"] >= arrow["c1"]
        elif (dr, dc) == (1, 0):
            ok = rect["r1"] + 1 == arrow["r0"] and rect["c0"] <= arrow["c0"] and rect["c1"] >= arrow["c1"]
        elif (dr, dc) == (0, 1):
            ok = rect["c1"] + 1 == arrow["c0"] and rect["r0"] <= arrow["r0"] and rect["r1"] >= arrow["r1"]
        else:
            ok = rect["c0"] == arrow["c1"] + 1 and rect["r0"] <= arrow["r0"] and rect["r1"] >= arrow["r1"]
        if ok:
            matches.append((idx, rect))
    if len(matches) != 1:
        return None
    idx, rect = matches[0]
    used.add(idx)
    return rect


def _find_fence(grid: np.ndarray, rect: dict, arrow: dict, bg: int) -> tuple[str, int] | None:
    h, w = grid.shape
    dr, dc = arrow["dir"]
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    if (dr, dc) == (0, 1):
        for col in range(arrow["c1"] + 1, w):
            vals = grid[r0 : r1 + 1, col]
            if np.all(vals != bg) and np.any(vals == FENCE_COLOR) and not np.any(vals == ARROW_COLOR):
                return ("col", col)
    elif (dr, dc) == (0, -1):
        for col in range(arrow["c0"] - 1, -1, -1):
            vals = grid[r0 : r1 + 1, col]
            if np.all(vals != bg) and np.any(vals == FENCE_COLOR) and not np.any(vals == ARROW_COLOR):
                return ("col", col)
    elif (dr, dc) == (1, 0):
        for row in range(arrow["r1"] + 1, h):
            vals = grid[row, c0 : c1 + 1]
            if np.all(vals != bg) and np.any(vals == FENCE_COLOR) and not np.any(vals == ARROW_COLOR):
                return ("row", row)
    else:
        for row in range(arrow["r0"] - 1, -1, -1):
            vals = grid[row, c0 : c1 + 1]
            if np.all(vals != bg) and np.any(vals == FENCE_COLOR) and not np.any(vals == ARROW_COLOR):
                return ("row", row)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each source rectangle across the grey fence indicated by its green arrow.

    The source rectangle is translated to the symmetric position across the fence.
    For horizontal projection, each copied row takes the color of the crossed fence
    pixel on that row unless the fence pixel is grey; vertical projection uses the
    analogous per-column rule. The input objects remain in place.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    out = g.copy()

    rects = _find_rectangles(g, bg)
    arrows = _find_arrows(g)
    used_rects: set[int] = set()

    for arrow in arrows:
        rect = _rect_for_arrow(arrow, rects, used_rects)
        if rect is None:
            continue
        fence = _find_fence(g, rect, arrow, bg)
        if fence is None:
            continue

        axis, fence_pos = fence
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        color = rect["color"]

        if axis == "col":
            shift = 2 * fence_pos - c0 - c1
            nc0 = c0 + shift
            nc1 = c1 + shift
            if not (0 <= nc0 <= nc1 < g.shape[1]):
                continue
            if not np.all(out[r0 : r1 + 1, nc0 : nc1 + 1] == bg):
                continue
            filters = g[r0 : r1 + 1, fence_pos]
            for i, rr in enumerate(range(r0, r1 + 1)):
                new_color = color if int(filters[i]) == FENCE_COLOR else int(filters[i])
                out[rr, nc0 : nc1 + 1] = new_color
        else:
            shift = 2 * fence_pos - r0 - r1
            nr0 = r0 + shift
            nr1 = r1 + shift
            if not (0 <= nr0 <= nr1 < g.shape[0]):
                continue
            if not np.all(out[nr0 : nr1 + 1, c0 : c1 + 1] == bg):
                continue
            filters = g[fence_pos, c0 : c1 + 1]
            for i, cc in enumerate(range(c0, c1 + 1)):
                new_color = color if int(filters[i]) == FENCE_COLOR else int(filters[i])
                out[nr0 : nr1 + 1, cc] = new_color

    return out


def _filter_sequence(length: int, bg: int, source_color: int, rng: random.Random) -> list[int]:
    pool = [c for c in range(10) if c not in {bg, ARROW_COLOR, FENCE_COLOR, source_color}]
    specials = rng.sample(pool, k=2)
    seq = [FENCE_COLOR] * length
    count = rng.randint(1, length - 1)
    for idx in rng.sample(range(length), k=count):
        seq[idx] = rng.choice(specials)
    return seq


def _arrow_patch(direction: tuple[int, int]) -> np.ndarray:
    return DIR_TO_TEMPLATE[direction] * ARROW_COLOR


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    direction: tuple[int, int],
    source_color: int,
    bg: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    bh = rng.randint(*BLOCK_SIZE_RANGE)
    bw = rng.randint(*BLOCK_SIZE_RANGE)
    dist = rng.randint(*DISTANCE_RANGE)

    if direction in {(-1, 0), (1, 0)}:
        ext_before = rng.randint(*FENCE_EXT_RANGE)
        ext_after = rng.randint(*FENCE_EXT_RANGE)
        c0_min = ext_before
        c0_max = w - bw - ext_after - 1
        if c0_min > c0_max:
            return False
        c0 = rng.randint(c0_min, c0_max)
        arrow_c = c0 + rng.randint(0, bw - 3)
        filters = _filter_sequence(bw, bg, source_color, rng)
        if direction == (-1, 0):
            r0_min = bh + 2 * dist + 1
            r0_max = h - bh - 1
            if r0_min > r0_max:
                return False
            r0 = rng.randint(r0_min, r0_max)
            arrow_r = r0 - 3
            fence_r = r0 - dist - 1
            proj_r0 = r0 - bh - 2 * dist - 1
            proj_r1 = proj_r0 + bh - 1
            fence_c0 = c0 - ext_before
            fence_c1 = c0 + bw - 1 + ext_after
        else:
            r0_min = 0
            r0_max = h - (2 * bh + 2 * dist) - 1
            if r0_min > r0_max:
                return False
            r0 = rng.randint(r0_min, r0_max)
            arrow_r = r0 + bh
            fence_r = r0 + bh + dist
            proj_r0 = r0 + bh + 2 * dist + 1
            proj_r1 = proj_r0 + bh - 1
            fence_c0 = c0 - ext_before
            fence_c1 = c0 + bw - 1 + ext_after
        proj_c0 = c0
        proj_c1 = c0 + bw - 1
        arrow_r0, arrow_c0 = arrow_r, arrow_c
        arrow_r1, arrow_c1 = arrow_r + 2, arrow_c + 2
        bounds = (min(proj_r0, arrow_r0, fence_r, r0), min(c0, arrow_c0, fence_c0), max(proj_r1, arrow_r1, fence_r, r0 + bh - 1), max(proj_c1, arrow_c1, fence_c1))
    else:
        ext_before = rng.randint(*FENCE_EXT_RANGE)
        ext_after = rng.randint(*FENCE_EXT_RANGE)
        r0_min = ext_before
        r0_max = h - bh - ext_after - 1
        if r0_min > r0_max:
            return False
        r0 = rng.randint(r0_min, r0_max)
        arrow_r = r0 + rng.randint(0, bh - 3)
        filters = _filter_sequence(bh, bg, source_color, rng)
        if direction == (0, -1):
            c0_min = bw + 2 * dist + 1
            c0_max = w - bw - 1
            if c0_min > c0_max:
                return False
            c0 = rng.randint(c0_min, c0_max)
            arrow_c = c0 - 3
            fence_c = c0 - dist - 1
            proj_c0 = c0 - bw - 2 * dist - 1
            proj_c1 = proj_c0 + bw - 1
            fence_r0 = r0 - ext_before
            fence_r1 = r0 + bh - 1 + ext_after
        else:
            c0_min = 0
            c0_max = w - (2 * bw + 2 * dist) - 1
            if c0_min > c0_max:
                return False
            c0 = rng.randint(c0_min, c0_max)
            arrow_c = c0 + bw
            fence_c = c0 + bw + dist
            proj_c0 = c0 + bw + 2 * dist + 1
            proj_c1 = proj_c0 + bw - 1
            fence_r0 = r0 - ext_before
            fence_r1 = r0 + bh - 1 + ext_after
        proj_r0 = r0
        proj_r1 = r0 + bh - 1
        arrow_r0, arrow_c0 = arrow_r, arrow_c
        arrow_r1, arrow_c1 = arrow_r + 2, arrow_c + 2
        bounds = (min(r0, arrow_r0, fence_r0), min(proj_c0, arrow_c0, fence_c, c0), max(proj_r1, arrow_r1, fence_r1), max(proj_c1, arrow_c1, fence_c, c0 + bw - 1))

    br0, bc0, br1, bc1 = bounds
    br0 = max(0, br0 - 1)
    bc0 = max(0, bc0 - 1)
    br1 = min(h - 1, br1 + 1)
    bc1 = min(w - 1, bc1 + 1)
    if np.any(reserved[br0 : br1 + 1, bc0 : bc1 + 1]):
        return False

    grid[r0 : r0 + bh, c0 : c0 + bw] = source_color
    arrow = _arrow_patch(direction)
    arrow_view = grid[arrow_r0 : arrow_r1 + 1, arrow_c0 : arrow_c1 + 1]
    arrow_view[arrow > 0] = ARROW_COLOR

    if direction in {(-1, 0), (1, 0)}:
        grid[fence_r, fence_c0 : fence_c1 + 1] = FENCE_COLOR
        grid[fence_r, c0 : c0 + bw] = np.array(filters, dtype=int)
    else:
        grid[fence_r0 : fence_r1 + 1, fence_c] = FENCE_COLOR
        grid[r0 : r0 + bh, fence_c] = np.array(filters, dtype=int)

    reserved[br0 : br1 + 1, bc0 : bc1 + 1] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice([c for c in range(10) if c not in {ARROW_COLOR, FENCE_COLOR}])
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    count = rng.randint(*OBJECT_COUNT_RANGE)
    palette = [c for c in range(10) if c not in {bg, ARROW_COLOR, FENCE_COLOR}]
    if count > len(palette):
        return None
    source_colors = rng.sample(palette, k=count)

    directions = [rng.choice(DIRECTIONS) for _ in range(count)]
    if count > 1 and rng.random() < 0.5:
        directions = rng.sample(DIRECTIONS, k=count)

    for direction, source_color in zip(directions, source_colors, strict=False):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            if _place_object(grid, reserved, direction, source_color, bg, rng):
                placed = True
                break
        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid grid")
