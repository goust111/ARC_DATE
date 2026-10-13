import random
from functools import lru_cache
from itertools import combinations

import numpy as np


BG = 0
YELLOW = 4
INPUT_COLOR_POOL = tuple(color for color in range(1, 10) if color != YELLOW)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _cross(a: tuple[int, int], b: tuple[int, int], p: tuple[int, int]) -> int:
    return (b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1])


def _triangle_area2(
    a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]
) -> int:
    return abs(_cross(a, b, c))


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    pts: list[tuple[int, int]] = []
    while True:
        pts.append((r0, c0))
        if (r0, c0) == (r1, c1):
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return pts


def _triangle_mask(
    shape: tuple[int, int], vertices: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
) -> np.ndarray:
    h, w = shape
    a, b, c = vertices
    mask = np.zeros((h, w), dtype=bool)
    area2 = _triangle_area2(a, b, c)
    if area2 == 0:
        for p0, p1 in ((a, b), (b, c), (a, c)):
            for r, col in _line_points(p0, p1):
                if 0 <= r < h and 0 <= col < w:
                    mask[r, col] = True
        return mask

    r0 = max(0, min(a[0], b[0], c[0]))
    r1 = min(h - 1, max(a[0], b[0], c[0]))
    c0 = max(0, min(a[1], b[1], c[1]))
    c1 = min(w - 1, max(a[1], b[1], c[1]))

    for r in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            p = (r, col)
            s1 = _cross(a, b, p)
            s2 = _cross(b, c, p)
            s3 = _cross(c, a, p)
            if (s1 >= 0 and s2 >= 0 and s3 >= 0) or (s1 <= 0 and s2 <= 0 and s3 <= 0):
                mask[r, col] = True

    mask[a] = True
    mask[b] = True
    mask[c] = True
    return mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rs = slice(max(0, -dr), min(h, h - dr))
            cs = slice(max(0, -dc), min(w, w - dc))
            rd = slice(max(0, dr), min(h, h + dr))
            cd = slice(max(0, dc), min(w, w + dc))
            out[rd, cd] |= mask[rs, cs]
    return out


def _legend_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    if h < 2:
        return []

    pairs = [
        ((0, 0), (1, 0)),
        ((0, w - 1), (1, w - 1)),
        ((h - 2, 0), (h - 1, 0)),
        ((h - 2, w - 1), (h - 1, w - 1)),
    ]
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for top, bottom in pairs:
        top_color = int(grid[top])
        bottom_color = int(grid[bottom])
        if top_color == BG or bottom_color == BG or top_color == bottom_color:
            continue
        isolated = True
        for cell, mate in ((top, bottom), (bottom, top)):
            for nr, nc in _neighbors4(*cell):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) == mate:
                    continue
                if int(grid[nr, nc]) != BG:
                    isolated = False
                    break
            if not isolated:
                break
        if isolated:
            candidates.append((top, bottom))
    return candidates


def _detect_legend(
    grid: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    candidates = _legend_candidates(grid)
    assert len(candidates) == 1, f"expected exactly one isolated corner legend, got {candidates}"
    top, bottom = candidates[0]
    return top, bottom, int(grid[top]), int(grid[bottom])


def _group_points_into_triangles(
    points: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    pts = tuple(sorted((int(r), int(c)) for r, c in points))
    n = len(pts)
    assert n % 3 == 0, f"point count must be divisible by 3, got {n}"
    if n == 0:
        return []

    d2 = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dr = pts[i][0] - pts[j][0]
            dc = pts[i][1] - pts[j][1]
            val = dr * dr + dc * dc
            d2[i][j] = val
            d2[j][i] = val

    triplet_cost: dict[tuple[int, int, int], int] = {}
    for i, j, k in combinations(range(n), 3):
        triplet_cost[(i, j, k)] = d2[i][j] + d2[i][k] + d2[j][k]

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, tuple[tuple[tuple[int, int], ...], ...]]:
        if mask == 0:
            return 0, ()

        first = (mask & -mask).bit_length() - 1
        remain = [idx for idx in range(first + 1, n) if mask & (1 << idx)]
        best: tuple[int, tuple[tuple[tuple[int, int], ...], ...]] | None = None
        for a_idx, j in enumerate(remain):
            for k in remain[a_idx + 1 :]:
                trip_mask = (1 << first) | (1 << j) | (1 << k)
                sub_cost, sub_groups = solve(mask ^ trip_mask)
                trip = tuple(sorted((pts[first], pts[j], pts[k])))
                total = triplet_cost[(first, j, k)] + sub_cost
                groups = (trip,) + sub_groups
                if best is None or total < best[0] or (total == best[0] and groups < best[1]):
                    best = (total, groups)
        assert best is not None
        return best

    _, groups = solve((1 << n) - 1)
    return [tuple(group) for group in groups]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    legend_top, legend_bottom, highest, middle = _detect_legend(g)
    legend_cells = {legend_top, legend_bottom}

    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    assert len(colors) == 3, f"expected exactly three non-background colors, got {colors}"
    lowest = [color for color in colors if color not in (highest, middle)]
    assert len(lowest) == 1, "legend must leave exactly one implied lowest-priority color"
    priority_low_to_high = [lowest[0], middle, highest]
    priority_rank = {color: rank for rank, color in enumerate(priority_low_to_high)}

    triangles: list[tuple[int, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]] = []
    marker_positions: dict[tuple[int, int], int] = {}
    for color in colors:
        pts = [
            (int(r), int(c))
            for r, c in np.argwhere(g == color)
            if (int(r), int(c)) not in legend_cells
        ]
        assert pts, f"color {color} must have at least one triangle"
        groups = _group_points_into_triangles(pts)
        for group in groups:
            assert _triangle_area2(*group) > 0, f"degenerate triangle for color {color}: {group}"
            triangles.append((color, group))
            for cell in group:
                marker_positions[cell] = color

    canvas = np.zeros((h, w), dtype=int)
    for color, group in sorted(triangles, key=lambda item: priority_rank[item[0]]):
        canvas[_triangle_mask((h, w), group)] = color

    out = canvas.copy()
    visible = canvas != BG
    for r in range(h):
        for c in range(w):
            if not visible[r, c]:
                continue
            if (
                r == 0
                or c == 0
                or r == h - 1
                or c == w - 1
                or any(
                    not (0 <= nr < h and 0 <= nc < w) or not visible[nr, nc]
                    for nr, nc in _neighbors4(r, c)
                )
            ):
                out[r, c] = YELLOW

    for (r, c), color in marker_positions.items():
        out[r, c] = color
    out[legend_top] = highest
    out[legend_bottom] = middle
    return out


def _rand_range(rng: random.Random, lo: int, hi: int) -> int:
    assert lo <= hi
    return rng.randint(lo, hi)


def _pick_in_box(
    rng: random.Random, r0: int, r1: int, c0: int, c1: int
) -> tuple[int, int]:
    return _rand_range(rng, r0, r1), _rand_range(rng, c0, c1)


def _sample_triangle_candidate(
    rng: random.Random, shape: tuple[int, int]
) -> tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], np.ndarray] | None:
    h, w = shape
    max_bh = min(10, h - 4)
    max_bw = min(10, w - 4)
    if max_bh < 6 or max_bw < 6:
        return None

    for _ in range(200):
        bh = _rand_range(rng, 6, max_bh)
        bw = _rand_range(rng, 6, max_bw)
        r0 = _rand_range(rng, 2, h - bh - 2)
        c0 = _rand_range(rng, 2, w - bw - 2)
        r1 = r0 + bh - 1
        c1 = c0 + bw - 1

        orient = rng.choice(("up", "down", "left", "right"))
        if orient == "up":
            p0 = _pick_in_box(rng, r0, r0 + 1, c0 + bw // 4, c1 - bw // 4)
            p1 = _pick_in_box(rng, r1 - 1, r1, c0, c0 + bw // 3)
            p2 = _pick_in_box(rng, r1 - 1, r1, c1 - bw // 3, c1)
        elif orient == "down":
            p0 = _pick_in_box(rng, r1 - 1, r1, c0 + bw // 4, c1 - bw // 4)
            p1 = _pick_in_box(rng, r0, r0 + 1, c0, c0 + bw // 3)
            p2 = _pick_in_box(rng, r0, r0 + 1, c1 - bw // 3, c1)
        elif orient == "left":
            p0 = _pick_in_box(rng, r0 + bh // 4, r1 - bh // 4, c0, c0 + 1)
            p1 = _pick_in_box(rng, r0, r0 + bh // 3, c1 - 1, c1)
            p2 = _pick_in_box(rng, r1 - bh // 3, r1, c1 - 1, c1)
        else:
            p0 = _pick_in_box(rng, r0 + bh // 4, r1 - bh // 4, c1 - 1, c1)
            p1 = _pick_in_box(rng, r0, r0 + bh // 3, c0, c0 + 1)
            p2 = _pick_in_box(rng, r1 - bh // 3, r1, c0, c0 + 1)

        verts = tuple(sorted({p0, p1, p2}))
        if len(verts) != 3:
            continue
        if _triangle_area2(*verts) < 18:
            continue
        if min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in combinations(verts, 2)) < 5:
            continue

        mask = _triangle_mask(shape, verts)
        area = int(mask.sum())
        if area < 12 or area > (h * w) // 4:
            continue
        return verts, mask
    return None


def _can_place_triangle(
    verts: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    mask: np.ndarray,
    color: int,
    placed: list[dict],
    reserved: np.ndarray,
    overlap_target: dict | None,
) -> bool:
    if any(reserved[r, c] for r, c in verts):
        return False
    if bool(mask[reserved].any()):
        return False

    dilated = _dilate8(mask)
    for tri in placed:
        inter = mask & tri["mask"]
        if overlap_target is not None and tri is overlap_target:
            overlap_cells = int(inter.sum())
            if overlap_cells < 6:
                return False
            if overlap_cells * 3 > int(mask.sum()) or overlap_cells * 3 > int(tri["mask"].sum()):
                return False
            if any(mask[r, c] for r, c in tri["verts"]):
                return False
            if any(tri["mask"][r, c] for r, c in verts):
                return False
            continue

        if bool(inter.any()):
            return False
        if bool((dilated & tri["mask"]).any()):
            return False

    same_color_verts = [cell for tri in placed if tri["color"] == color for cell in tri["verts"]]
    if same_color_verts:
        if min(abs(r - rr) + abs(c - cc) for r, c in verts for rr, cc in same_color_verts) < 7:
            return False
    return True


def _group_signature(
    groups: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]
) -> set[frozenset[tuple[int, int]]]:
    return {frozenset(group) for group in groups}


def _build_reserved(shape: tuple[int, int], legend_cells: set[tuple[int, int]]) -> np.ndarray:
    reserved = np.zeros(shape, dtype=bool)
    for r, c in legend_cells:
        reserved[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                reserved[nr, nc] = True
    return reserved


def _construct_grid(
    shape: tuple[int, int],
    colors: list[int],
    priority_high_to_low: list[int],
    legend_cells: tuple[tuple[int, int], tuple[int, int]],
    triangles: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    top, bottom = legend_cells
    grid[top] = priority_high_to_low[0]
    grid[bottom] = priority_high_to_low[1]
    for tri in triangles:
        for r, c in tri["verts"]:
            grid[r, c] = tri["color"]
    return grid


def _validate_generated_instance(
    grid: np.ndarray,
    colors: list[int],
    legend_cells: tuple[tuple[int, int], tuple[int, int]],
    intended_groups: dict[int, list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]],
    overlap_required: bool,
) -> bool:
    non_bg = [int(v) for v in np.unique(grid) if int(v) != BG]
    if set(non_bg) != set(colors):
        return False

    legend_set = set(legend_cells)
    for color in colors:
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == color) if (int(r), int(c)) not in legend_set]
        if len(pts) != 3 * len(intended_groups[color]):
            return False
        if _group_signature(_group_points_into_triangles(pts)) != _group_signature(intended_groups[color]):
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if int(np.count_nonzero(out == YELLOW)) == 0:
        return False

    if overlap_required:
        low_to_high = [color for color in colors if color not in (grid[legend_cells[0]], grid[legend_cells[1]])]
        low_to_high.append(int(grid[legend_cells[1]]))
        low_to_high.append(int(grid[legend_cells[0]]))
        rank = {color: idx for idx, color in enumerate(low_to_high)}
        canvas = np.zeros_like(grid)
        for color in sorted(colors, key=lambda color: rank[color]):
            pts_groups = intended_groups[color]
            for group in pts_groups:
                canvas[_triangle_mask(grid.shape, group)] = color
        if len(set(int(v) for v in np.unique(canvas) if int(v) != BG)) < 2:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = _rand_range(rng, 18, 28)
    w = _rand_range(rng, 18, 28)
    shape = (h, w)

    colors = rng.sample(list(INPUT_COLOR_POOL), 3)
    priority_high_to_low = rng.sample(colors, len(colors))
    counts = list(rng.choice(((1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2))))
    rng.shuffle(counts)
    overlap_required = rng.random() < 0.5

    corner = rng.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        legend_cells = ((0, 0), (1, 0))
    elif corner == "tr":
        legend_cells = ((0, w - 1), (1, w - 1))
    elif corner == "bl":
        legend_cells = ((h - 2, 0), (h - 1, 0))
    else:
        legend_cells = ((h - 2, w - 1), (h - 1, w - 1))

    reserved = _build_reserved(shape, set(legend_cells))
    placed: list[dict] = []
    intended_groups: dict[int, list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]] = {
        color: [] for color in colors
    }

    remaining = {color: count for color, count in zip(colors, counts)}

    if overlap_required:
        high_idx, low_idx = sorted(rng.sample(range(3), 2))
        overlap_colors = [priority_high_to_low[high_idx], priority_high_to_low[low_idx]]
        base_color, overlap_color = overlap_colors[0], overlap_colors[1]

        base = None
        for _ in range(160):
            cand = _sample_triangle_candidate(rng, shape)
            if cand is None:
                return None
            verts, mask = cand
            if _can_place_triangle(verts, mask, base_color, placed, reserved, None):
                base = {"color": base_color, "verts": verts, "mask": mask}
                break
        if base is None:
            return None
        placed.append(base)
        intended_groups[base_color].append(base["verts"])
        remaining[base_color] -= 1

        overlay = None
        for _ in range(220):
            cand = _sample_triangle_candidate(rng, shape)
            if cand is None:
                return None
            verts, mask = cand
            if _can_place_triangle(verts, mask, overlap_color, placed, reserved, base):
                overlay = {"color": overlap_color, "verts": verts, "mask": mask}
                break
        if overlay is None:
            return None
        placed.append(overlay)
        intended_groups[overlap_color].append(overlay["verts"])
        remaining[overlap_color] -= 1

    color_order = colors[:]
    rng.shuffle(color_order)
    for color in color_order:
        while remaining[color] > 0:
            tri = None
            for _ in range(220):
                cand = _sample_triangle_candidate(rng, shape)
                if cand is None:
                    return None
                verts, mask = cand
                if _can_place_triangle(verts, mask, color, placed, reserved, None):
                    tri = {"color": color, "verts": verts, "mask": mask}
                    break
            if tri is None:
                return None
            placed.append(tri)
            intended_groups[color].append(tri["verts"])
            remaining[color] -= 1

    grid = _construct_grid(shape, colors, priority_high_to_low, legend_cells, placed)
    if not _validate_generated_instance(grid, colors, legend_cells, intended_groups, overlap_required):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
