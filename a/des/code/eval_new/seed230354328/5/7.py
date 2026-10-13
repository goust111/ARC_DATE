import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each sparse same-color "constellation" into the convex region implied by its vertices.

    Input structure:
    - The background is black.
    - Away from the edge sample, each active color appears as exactly 3 or 4 isolated pixels.
    - A grey rectangular frame touches the grid edge; inside it, one color already shows a completed example of the transformation.

    Rule:
    - Detect the grey edge frame as the example region and remove that whole rectangle in the output.
    - Group the remaining non-background pixels by color. Treat each color's 3 or 4 pixels as polygon vertices and fill every cell whose center lies inside or on the boundary of their convex hull.
    - If a group's hull degenerates to a line (the collinear case), rasterize the segment between the extreme points with a 1-pixel-wide Bresenham line.
    - Draw 3-pixel groups before 4-pixel groups; later fills overwrite earlier ones where regions overlap.

    Key insight: The edge sample is only a legend for the operation: sparse same-color pixels are really polygon vertices whose convex hull must be painted.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    frame_bbox, frame_color = _find_example_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox

    out = g.copy()
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg

    inside_example = np.zeros((h, w), dtype=bool)
    inside_example[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    groups = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, frame_color):
            continue
        pts = [(int(r), int(c)) for r, c in np.argwhere((g == color) & ~inside_example)]
        if not pts:
            continue
        assert len(pts) in (3, 4), f"expected 3 or 4 vertices for color {color}, got {len(pts)}"
        groups.append((len(pts), min(pts), color, pts))

    for _, _, color, pts in sorted(groups):
        for r, c in _filled_region(pts):
            out[r, c] = color

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)

    sample_bbox, sample_color = _place_example_region(grid, rng)
    colors = COLOR_POOL[:]
    rng.shuffle(colors)
    if sample_color not in colors:
        colors.insert(0, sample_color)
    colors = [c for c in colors if c != sample_color]

    blocked = _expanded_mask(grid != BG, radius=1)
    forbidden_fill = np.zeros((h, w), dtype=bool)
    fr0, fc0, fr1, fc1 = sample_bbox
    forbidden_fill[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    group_count = rng.randint(2, 4)
    sizes = [3, 4] + [rng.choice([3, 4]) for _ in range(group_count - 2)]
    rng.shuffle(sizes)
    need_collinear = rng.random() < 0.5
    need_overlap = rng.random() < 0.5

    groups = []
    fills = []
    pending_sizes = sizes[:]
    collinear_done = False

    if need_overlap:
        first_three = _place_group(
            rng,
            h,
            w,
            blocked,
            fills,
            forbidden_fill,
            size=3,
            color=colors[len(groups)],
            collinear=need_collinear,
        )
        if first_three is None:
            return None
        groups.append(first_three)
        fills.append((first_three["size"], first_three["fill"]))
        _stamp_vertices(grid, first_three["vertices"], first_three["color"])
        blocked = _expanded_mask(grid != BG, radius=1)
        pending_sizes.remove(3)
        collinear_done = need_collinear

        second_four = _place_group(
            rng,
            h,
            w,
            blocked,
            fills,
            forbidden_fill,
            size=4,
            color=colors[len(groups)],
            overlap_with=first_three["fill"],
            overlap_bounds=_expand_bbox(_coords_bbox(first_three["fill"]), h, w, pad=4),
        )
        if second_four is None:
            return None
        groups.append(second_four)
        fills.append((second_four["size"], second_four["fill"]))
        _stamp_vertices(grid, second_four["vertices"], second_four["color"])
        blocked = _expanded_mask(grid != BG, radius=1)
        pending_sizes.remove(4)

    for size in pending_sizes:
        collinear = size == 3 and need_collinear and not collinear_done
        group = _place_group(
            rng,
            h,
            w,
            blocked,
            fills,
            forbidden_fill,
            size=size,
            color=colors[len(groups)],
            collinear=collinear,
        )
        if group is None:
            return None
        groups.append(group)
        fills.append((group["size"], group["fill"]))
        _stamp_vertices(grid, group["vertices"], group["color"])
        blocked = _expanded_mask(grid != BG, radius=1)
        collinear_done = collinear_done or collinear

    if need_collinear and not collinear_done:
        return None

    if not _vertices_are_isolated(grid, sample_bbox):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    if need_overlap:
        overlaps = 0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i]["size"] != groups[j]["size"] and groups[i]["fill"] & groups[j]["fill"]:
                    overlaps += 1
        if overlaps == 0:
            return None

    return grid


def _place_example_region(grid: np.ndarray, rng: random.Random) -> tuple[tuple[int, int, int, int], int]:
    h, w = grid.shape
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        box_h = rng.randint(7, min(10, h - 6))
        box_w = rng.randint(7, min(12, w - 4))
        r0 = 0 if side == "top" else h - box_h
        c0 = rng.randint(1, w - box_w - 1)
    else:
        box_h = rng.randint(7, min(12, h - 4))
        box_w = rng.randint(7, min(10, w - 6))
        r0 = rng.randint(1, h - box_h - 1)
        c0 = 0 if side == "left" else w - box_w
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1

    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME

    interior = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    sample_color = rng.choice(COLOR_POOL)
    example_kind = rng.choice([3, 4])
    example_collinear = example_kind == 3 and rng.random() < 0.5
    for _ in range(200):
        if example_kind == 3 and example_collinear:
            vertices, fill = _sample_line_triplet(rng, h, w, bounds=interior)
        elif example_kind == 3:
            vertices, fill = _sample_triangle(rng, h, w, bounds=interior)
        else:
            vertices, fill = _sample_quad(rng, h, w, bounds=interior)
        if vertices is None:
            continue
        if any(r in (r0, r1) or c in (c0, c1) for r, c in fill):
            continue
        if len(fill) < 4:
            continue
        for r, c in fill:
            grid[r, c] = sample_color
        return (r0, c0, r1, c1), sample_color
    raise RuntimeError("failed to place example region")


def _place_group(
    rng: random.Random,
    h: int,
    w: int,
    blocked: np.ndarray,
    fills: list[tuple[int, set[tuple[int, int]]]],
    forbidden_fill: np.ndarray,
    *,
    size: int,
    color: int,
    collinear: bool = False,
    overlap_with: set[tuple[int, int]] | None = None,
    overlap_bounds: tuple[int, int, int, int] | None = None,
) -> dict | None:
    bounds = overlap_bounds if overlap_bounds is not None else (0, 0, h - 1, w - 1)
    for _ in range(300):
        if size == 3 and collinear:
            vertices, fill = _sample_line_triplet(rng, h, w, bounds=bounds)
        elif size == 3:
            vertices, fill = _sample_triangle(rng, h, w, bounds=bounds)
        else:
            vertices, fill = _sample_quad(rng, h, w, bounds=bounds)
        if vertices is None:
            continue
        if any(blocked[r, c] for r, c in vertices):
            continue
        if any(forbidden_fill[r, c] for r, c in fill):
            continue
        if len(fill) < (5 if size == 3 else 8) or len(fill) > 60:
            continue
        if any(_cheb(a, b) <= 1 for a in vertices for b in vertices if a != b):
            continue

        bad = False
        for other_size, other_fill in fills:
            overlap = fill & other_fill
            if overlap_with is other_fill:
                if not overlap:
                    bad = True
                    break
                if len(overlap) >= min(len(fill), len(other_fill)):
                    bad = True
                    break
            elif overlap:
                bad = True
                break
            if other_size == size and overlap:
                bad = True
                break
        if bad:
            continue

        return {
            "size": size,
            "color": color,
            "vertices": vertices,
            "fill": fill,
        }
    return None


def _sample_line_triplet(
    rng: random.Random,
    h: int,
    w: int,
    *,
    bounds: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]] | None, set[tuple[int, int]]]:
    r0, c0, r1, c1 = bounds
    directions = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (2, -1), (1, 2), (1, -2)]
    for _ in range(120):
        dr, dc = rng.choice(directions)
        gap1 = rng.randint(2, 4)
        gap2 = rng.randint(2, 4)
        total = gap1 + gap2

        rr_min = r0 + max(0, -dr * total)
        rr_max = r1 - max(0, dr * total)
        cc_min = c0 + max(0, -dc * total)
        cc_max = c1 - max(0, dc * total)
        if rr_min > rr_max or cc_min > cc_max:
            continue

        sr = rng.randint(rr_min, rr_max)
        sc = rng.randint(cc_min, cc_max)
        p0 = (sr, sc)
        p1 = (sr + dr * gap1, sc + dc * gap1)
        p2 = (sr + dr * total, sc + dc * total)
        vertices = [p0, p1, p2]
        fill = set(_bresenham(p0, p2))
        if len(fill) >= 5:
            return vertices, fill
    return None, set()


def _sample_triangle(
    rng: random.Random,
    h: int,
    w: int,
    *,
    bounds: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]] | None, set[tuple[int, int]]]:
    r0, c0, r1, c1 = bounds
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None, set()
    for _ in range(120):
        box_h = rng.randint(4, min(9, r1 - r0 + 1))
        box_w = rng.randint(4, min(9, c1 - c0 + 1))
        if box_h > (r1 - r0 + 1) or box_w > (c1 - c0 + 1):
            continue
        br = rng.randint(r0, r1 - box_h + 1)
        bc = rng.randint(c0, c1 - box_w + 1)
        orient = rng.choice(["up", "down", "left", "right"])
        if orient == "up":
            vertices = [
                (br, bc + rng.randint(1, box_w - 2)),
                (br + box_h - 1, bc),
                (br + box_h - 1, bc + box_w - 1),
            ]
        elif orient == "down":
            vertices = [
                (br + box_h - 1, bc + rng.randint(1, box_w - 2)),
                (br, bc),
                (br, bc + box_w - 1),
            ]
        elif orient == "left":
            vertices = [
                (br + rng.randint(1, box_h - 2), bc),
                (br, bc + box_w - 1),
                (br + box_h - 1, bc + box_w - 1),
            ]
        else:
            vertices = [
                (br + rng.randint(1, box_h - 2), bc + box_w - 1),
                (br, bc),
                (br + box_h - 1, bc),
            ]
        fill = _filled_region(vertices)
        if len(fill) >= 5:
            return vertices, fill
    return None, set()


def _sample_quad(
    rng: random.Random,
    h: int,
    w: int,
    *,
    bounds: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]] | None, set[tuple[int, int]]]:
    r0, c0, r1, c1 = bounds
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None, set()
    for _ in range(160):
        box_h = rng.randint(5, min(10, r1 - r0 + 1))
        box_w = rng.randint(5, min(10, c1 - c0 + 1))
        if box_h > (r1 - r0 + 1) or box_w > (c1 - c0 + 1):
            continue
        br = rng.randint(r0, r1 - box_h + 1)
        bc = rng.randint(c0, c1 - box_w + 1)
        style = rng.choice(["diamond", "rectangle", "kite"])
        if style == "rectangle":
            vertices = [
                (br, bc),
                (br, bc + box_w - 1),
                (br + box_h - 1, bc + box_w - 1),
                (br + box_h - 1, bc),
            ]
        elif style == "kite":
            vertices = [
                (br, bc + rng.randint(1, box_w - 2)),
                (br + rng.randint(1, box_h - 3), bc + box_w - 1),
                (br + box_h - 1, bc + rng.randint(1, box_w - 2)),
                (br + rng.randint(2, box_h - 2), bc),
            ]
        else:
            vertices = [
                (br, bc + rng.randint(1, box_w - 2)),
                (br + rng.randint(1, box_h - 2), bc + box_w - 1),
                (br + box_h - 1, bc + rng.randint(1, box_w - 2)),
                (br + rng.randint(1, box_h - 2), bc),
            ]
        fill = _filled_region(vertices)
        if len(_convex_hull([(c, r) for r, c in vertices])) != 4:
            continue
        if len(fill) >= 8:
            return vertices, fill
    return None, set()


def _filled_region(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = [(c, r) for r, c in vertices]
    hull = _convex_hull(pts)
    if len(hull) <= 2 or _polygon_area2(hull) == 0:
        if len(hull) == 1:
            x, y = hull[0]
            return {(y, x)}
        (x0, y0), (x1, y1) = hull
        return set(_bresenham((y0, x0), (y1, x1)))

    xs = [x for x, _ in hull]
    ys = [y for _, y in hull]
    fill = set()
    for y in range(min(ys), max(ys) + 1):
        for x in range(min(xs), max(xs) + 1):
            if _point_in_convex((x, y), hull):
                fill.add((y, x))
    return fill


def _point_in_convex(p: tuple[int, int], hull: list[tuple[int, int]]) -> bool:
    x, y = p
    n = len(hull)
    for i in range(n):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % n]
        if (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) < 0:
            return False
    return True


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _polygon_area2(poly: list[tuple[int, int]]) -> int:
    area2 = 0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        area2 += x0 * y1 - x1 * y0
    return abs(area2)


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    out = []
    r, c = r0, c0
    while True:
        out.append((r, c))
        if (r, c) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return out


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_example_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    h, w = grid.shape
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _coords_bbox(comp)
            if min(r1 - r0 + 1, c1 - c0 + 1) < 4:
                continue
            if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
                continue
            border = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
            }
            if set(comp) == border:
                candidates.append(((r1 - r0 + 1) * (c1 - c0 + 1), color, (r0, c0, r1, c1)))
    assert candidates, "example frame not found"
    _, color, bbox = max(candidates)
    return bbox, color


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _coords_bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expand_bbox(bbox: tuple[int, int, int, int], h: int, w: int, pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _expanded_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                out[rr, cc] = True
    return out


def _stamp_vertices(grid: np.ndarray, vertices: list[tuple[int, int]], color: int) -> None:
    for r, c in vertices:
        grid[r, c] = color


def _vertices_are_isolated(grid: np.ndarray, sample_bbox: tuple[int, int, int, int]) -> bool:
    h, w = grid.shape
    fr0, fc0, fr1, fc1 = sample_bbox
    sample = np.zeros((h, w), dtype=bool)
    sample[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    points = [(int(r), int(c)) for r, c in np.argwhere((grid != BG) & ~sample & (grid != FRAME))]
    for i, (r, c) in enumerate(points):
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if (nr, nc) != (r, c) and grid[nr, nc] != BG and not sample[nr, nc]:
                    return False
        for rr, cc in points[i + 1 :]:
            if _cheb((r, c), (rr, cc)) <= 1:
                return False
    return True


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
