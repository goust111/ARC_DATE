import random

import numpy as np


PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400
PLACEMENT_TRIES = 250
GRID_MIN = 14
GRID_MAX = 30
GAP = 1


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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    rectangles: list[dict[str, object]] = []
    for pts in _components(g != 0):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(g[pts[0][0], pts[0][1]])
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(box == color):
            raise ValueError("each foreground component must be a solid monochromatic rectangle")
        area = int(box.size)
        if len(pts) != area:
            raise ValueError("each foreground component must fill its bounding box")
        rectangles.append({"color": color, "area": area, "bbox": (r0, c0, r1, c1)})
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild scattered solid rectangles into concentric filled rectangles.

    The largest rectangle keeps its original bounding box. Remaining rectangles
    are sorted by area descending, and each contributes only its color to the
    next inward layer, inset by one cell per rank.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    rectangles = _find_rectangles(g)
    if not rectangles:
        return out

    rectangles.sort(key=lambda rect: (-int(rect["area"]), rect["bbox"][0], rect["bbox"][1]))
    r0, c0, r1, c1 = rectangles[0]["bbox"]

    for depth, rect in enumerate(rectangles):
        top = r0 + depth
        left = c0 + depth
        bottom = r1 - depth
        right = c1 - depth
        if top > bottom or left > right:
            break
        out[top : bottom + 1, left : right + 1] = int(rect["color"])

    return out


def _max_layers(height: int, width: int) -> int:
    return (min(height, width) + 1) // 2


def _can_place(used: np.ndarray, r0: int, c0: int, height: int, width: int, gap: int = GAP) -> bool:
    h, w = used.shape
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h, r1 + gap + 1)
    cc1 = min(w, c1 + gap + 1)
    return not used[rr0:rr1, cc0:cc1].any()


def _place_rect(
    rng: random.Random,
    used: np.ndarray,
    height: int,
    width: int,
    gap: int = GAP,
) -> tuple[int, int] | None:
    h, w = used.shape
    if height + 2 > h or width + 2 > w:
        return None
    for _ in range(PLACEMENT_TRIES):
        r0 = rng.randint(1, h - height - 1)
        c0 = rng.randint(1, w - width - 1)
        if _can_place(used, r0, c0, height, width, gap=gap):
            used[r0 : r0 + height, c0 : c0 + width] = True
            return r0, c0
    return None


def _sample_outer_dims(rng: random.Random, rect_count: int, overflow_branch: bool) -> tuple[int, int] | None:
    if overflow_branch:
        capacity = rng.randint(2, rect_count - 1)
        min_side = rng.randint(2 * capacity - 1, 2 * capacity)
    else:
        min_side_lo = 2 * rect_count - 1
        min_side_hi = min(13, min_side_lo + 3)
        if min_side_lo > min_side_hi:
            return None
        min_side = rng.randint(min_side_lo, min_side_hi)

    long_side = rng.randint(min_side, min(15, min_side + 4))
    dims = (min_side, long_side)
    if rng.random() < 0.5:
        dims = dims[::-1]
    return dims


def _sample_other_shapes(
    rng: random.Random,
    outer_area: int,
    rect_count: int,
    grid_h: int,
    grid_w: int,
    overflow_branch: bool,
) -> list[tuple[int, int, int]] | None:
    side_max = min(7, grid_h - 2, grid_w - 2)
    area_to_dims: dict[int, list[tuple[int, int]]] = {}
    for height in range(1, side_max + 1):
        for width in range(1, side_max + 1):
            area = height * width
            if area >= outer_area:
                continue
            area_to_dims.setdefault(area, []).append((height, width))

    areas = sorted(area_to_dims)
    if len(areas) < rect_count - 1:
        return None

    if overflow_branch:
        pool_limit = max(rect_count + 1, len(areas) // 2)
        pool = areas[: min(len(areas), pool_limit)]
    else:
        pool_limit = max(rect_count + 2, (2 * len(areas)) // 3)
        pool = areas[: min(len(areas), pool_limit)]
    if len(pool) < rect_count - 1:
        pool = areas

    chosen_areas = rng.sample(pool, rect_count - 1)
    shapes: list[tuple[int, int, int]] = []
    for area in sorted(chosen_areas, reverse=True):
        dims = rng.choice(area_to_dims[area])
        shapes.append((area, dims[0], dims[1]))
    return shapes


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    overflow_branch = rng.random() < 0.5
    rect_count = rng.randint(4, 7) if overflow_branch else rng.randint(3, 6)

    outer_dims = _sample_outer_dims(rng, rect_count, overflow_branch)
    if outer_dims is None:
        return None
    outer_h, outer_w = outer_dims

    if overflow_branch and _max_layers(outer_h, outer_w) >= rect_count:
        return None
    if not overflow_branch and _max_layers(outer_h, outer_w) < rect_count:
        return None

    extra_h = rng.randint(5, 6 + rect_count)
    extra_w = rng.randint(5, 6 + rect_count)
    grid_h = outer_h + extra_h
    grid_w = outer_w + extra_w
    if not (GRID_MIN <= grid_h <= GRID_MAX and GRID_MIN <= grid_w <= GRID_MAX):
        return None

    other_shapes = _sample_other_shapes(
        rng,
        outer_h * outer_w,
        rect_count,
        grid_h,
        grid_w,
        overflow_branch,
    )
    if other_shapes is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    used = np.zeros((grid_h, grid_w), dtype=bool)
    colors = rng.sample(PALETTE, rect_count)

    outer_pos = _place_rect(rng, used, outer_h, outer_w)
    if outer_pos is None:
        return None
    rectangles = [
        {
            "area": outer_h * outer_w,
            "height": outer_h,
            "width": outer_w,
            "pos": outer_pos,
            "color": colors[0],
        }
    ]

    for index, (area, height, width) in enumerate(other_shapes, start=1):
        pos = _place_rect(rng, used, height, width)
        if pos is None:
            return None
        rectangles.append(
            {
                "area": area,
                "height": height,
                "width": width,
                "pos": pos,
                "color": colors[index],
            }
        )

    rng.shuffle(rectangles)
    for rect in rectangles:
        r0, c0 = rect["pos"]
        grid[r0 : r0 + rect["height"], c0 : c0 + rect["width"]] = rect["color"]

    try:
        found = _find_rectangles(grid)
    except ValueError:
        return None
    if len(found) != rect_count:
        return None
    if len({int(rect["area"]) for rect in found}) != rect_count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
