import random

import numpy as np


BG_COLOR = 7
FG_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
EPS = 1e-9

GRID_SIZE_RANGE = (18, 30)
GEN_RETRIES = 400
SHAPE_RETRIES = 120
PLACEMENT_MARGIN = 1
MIN_PAINT_CELLS = 8
MIN_OVERLAP_CELLS = 4


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    xy = sorted({(int(c), int(r)) for r, c in points})
    if len(xy) <= 1:
        return [(y, x) for x, y in xy]

    lower: list[tuple[int, int]] = []
    for p in xy:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[int, int]] = []
    for p in reversed(xy):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull_xy = lower[:-1] + upper[:-1]
    return [(y, x) for x, y in hull_xy]


def _scan_spans(
    hull: list[tuple[int, int]],
) -> tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = [int(r) for r, _c in hull]
    cols = [int(c) for _r, c in hull]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)

    row_left = np.full(rmax - rmin + 1, np.inf, dtype=float)
    row_right = np.full(rmax - rmin + 1, -np.inf, dtype=float)
    col_top = np.full(cmax - cmin + 1, np.inf, dtype=float)
    col_bottom = np.full(cmax - cmin + 1, -np.inf, dtype=float)

    hull_xy = [(float(c), float(r)) for r, c in hull]
    n = len(hull_xy)
    for i in range(n):
        x1, y1 = hull_xy[i]
        x2, y2 = hull_xy[(i + 1) % n]

        if abs(y1 - y2) <= EPS:
            row = int(round(y1))
            idx = row - rmin
            row_left[idx] = min(row_left[idx], min(x1, x2))
            row_right[idx] = max(row_right[idx], max(x1, x2))
        else:
            start = int(np.ceil(min(y1, y2) - EPS))
            end = int(np.floor(max(y1, y2) + EPS))
            for row in range(start, end + 1):
                x = x1 + (row - y1) * (x2 - x1) / (y2 - y1)
                idx = row - rmin
                row_left[idx] = min(row_left[idx], x)
                row_right[idx] = max(row_right[idx], x)

        if abs(x1 - x2) <= EPS:
            col = int(round(x1))
            idx = col - cmin
            col_top[idx] = min(col_top[idx], min(y1, y2))
            col_bottom[idx] = max(col_bottom[idx], max(y1, y2))
        else:
            start = int(np.ceil(min(x1, x2) - EPS))
            end = int(np.floor(max(x1, x2) + EPS))
            for col in range(start, end + 1):
                y = y1 + (col - x1) * (y2 - y1) / (x2 - x1)
                idx = col - cmin
                col_top[idx] = min(col_top[idx], y)
                col_bottom[idx] = max(col_bottom[idx], y)

    return rmin, rmax, cmin, cmax, row_left, row_right, col_top, col_bottom


def _region_from_markers(
    points: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    marker_set = {(int(r), int(c)) for r, c in points}
    hull = _convex_hull(list(marker_set))
    if len(hull) < 3:
        return hull, set(marker_set), set()

    rmin, rmax, cmin, cmax, row_left, row_right, col_top, col_bottom = _scan_spans(hull)
    region: set[tuple[int, int]] = set()
    for r in range(rmin, rmax + 1):
        left = row_left[r - rmin]
        right = row_right[r - rmin]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        for c in range(cmin, cmax + 1):
            top = col_top[c - cmin]
            bottom = col_bottom[c - cmin]
            if not np.isfinite(top) or not np.isfinite(bottom):
                continue
            if left - EPS <= c <= right + EPS and top - EPS <= r <= bottom + EPS:
                region.add((r, c))

    return hull, region, region - marker_set


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group single-pixel markers by color, interpret each group as the vertices of a
    convex polygon, and fill that polygon with the same color.

    Colors with more markers are painted first. Later groups may overwrite earlier
    fills on original background cells; marker pixels themselves are preserved.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    groups: dict[int, list[tuple[int, int]]] = {}
    first_seen: dict[int, int] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg:
                continue
            groups.setdefault(color, []).append((int(r), int(c)))
            first_seen.setdefault(color, int(r * w + c))

    order = sorted(groups, key=lambda color: (-len(groups[color]), first_seen[color]))
    out = g.copy()

    for color in order:
        _hull, _region, paint = _region_from_markers(groups[color])
        for r, c in paint:
            if int(g[r, c]) == bg:
                out[r, c] = int(color)

    return out


def _normalize(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _c in points)
    c0 = min(c for _r, c in points)
    return [(int(r - r0), int(c - c0)) for r, c in points]


def _transform_local(
    points: list[tuple[int, int]], height: int, width: int
) -> tuple[list[tuple[int, int]], int, int]:
    pts = list(points)
    h, w = int(height), int(width)

    if random.random() < 0.5:
        pts = [(h - 1 - r, c) for r, c in pts]
    if random.random() < 0.5:
        pts = [(r, w - 1 - c) for r, c in pts]
    if random.random() < 0.35:
        pts = [(c, r) for r, c in pts]
        h, w = w, h

    pts = _normalize(pts)
    h = max(r for r, _c in pts) + 1
    w = max(c for _r, c in pts) + 1
    return pts, h, w


def _point_on_side(side: str, height: int, width: int) -> tuple[int, int]:
    if side == "top":
        return (0, random.randint(1, width - 2))
    if side == "right":
        return (random.randint(1, height - 2), width - 1)
    if side == "bottom":
        return (height - 1, random.randint(1, width - 2))
    if side == "left":
        return (random.randint(1, height - 2), 0)
    raise ValueError(side)


def _make_shape(marker_count: int) -> dict[str, object] | None:
    for _ in range(SHAPE_RETRIES):
        base_h = random.randint(6, 10)
        base_w = random.randint(6, 10)

        if marker_count == 4:
            sides = ["top", "right", "bottom", "left"]
        else:
            missing = random.choice(["top", "right", "bottom", "left"])
            sides = [side for side in ["top", "right", "bottom", "left"] if side != missing]

        pts = [_point_on_side(side, base_h, base_w) for side in sides]
        pts, local_h, local_w = _transform_local(pts, base_h, base_w)

        hull, region, paint = _region_from_markers(pts)
        if len(hull) != marker_count:
            continue
        if local_h < 5 or local_w < 5:
            continue
        if len(paint) < MIN_PAINT_CELLS:
            continue

        return {
            "count": int(marker_count),
            "markers_local": tuple((int(r), int(c)) for r, c in pts),
            "region_local": frozenset((int(r), int(c)) for r, c in region),
            "paint_local": frozenset((int(r), int(c)) for r, c in paint),
            "height": int(local_h),
            "width": int(local_w),
        }

    return None


def _translate_points(
    points: tuple[tuple[int, int], ...] | frozenset[tuple[int, int]], dr: int, dc: int
) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in points}


def _offsets_for_shape(height: int, width: int, grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    max_r = grid_h - height - PLACEMENT_MARGIN
    max_c = grid_w - width - PLACEMENT_MARGIN
    if max_r < PLACEMENT_MARGIN or max_c < PLACEMENT_MARGIN:
        return []
    offsets = [
        (r0, c0)
        for r0 in range(PLACEMENT_MARGIN, max_r + 1)
        for c0 in range(PLACEMENT_MARGIN, max_c + 1)
    ]
    random.shuffle(offsets)
    return offsets


def _overlap_offsets(
    shape_h: int,
    shape_w: int,
    grid_h: int,
    grid_w: int,
    anchor_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    min_r = max(PLACEMENT_MARGIN, ar0 - shape_h + 1)
    max_r = min(grid_h - shape_h - PLACEMENT_MARGIN, ar1)
    min_c = max(PLACEMENT_MARGIN, ac0 - shape_w + 1)
    max_c = min(grid_w - shape_w - PLACEMENT_MARGIN, ac1)
    if min_r > max_r or min_c > max_c:
        return []
    offsets = [(r0, c0) for r0 in range(min_r, max_r + 1) for c0 in range(min_c, max_c + 1)]
    random.shuffle(offsets)
    return offsets


def _chebyshev_too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _place_specs(
    specs: list[dict[str, object]],
    grid_h: int,
    grid_w: int,
    overlap_pair: tuple[int, int] | None,
) -> list[dict[str, object]] | None:
    placed: dict[int, dict[str, object]] = {}

    def try_place(idx: int, offsets: list[tuple[int, int]]) -> bool:
        spec = specs[idx]
        count = int(spec["count"])
        shape_h = int(spec["height"])
        shape_w = int(spec["width"])

        for r0, c0 in offsets:
            markers = _translate_points(spec["markers_local"], r0, c0)
            region = _translate_points(spec["region_local"], r0, c0)
            paint = _translate_points(spec["paint_local"], r0, c0)

            ok = True
            overlap_size = 0
            for j, pdata in placed.items():
                other_count = int(pdata["count"])
                other_markers = pdata["markers"]
                other_region = pdata["region"]
                other_paint = pdata["paint"]

                if markers & other_markers:
                    ok = False
                    break
                if markers & other_region:
                    ok = False
                    break
                if other_markers & region:
                    ok = False
                    break
                for p in markers:
                    if any(_chebyshev_too_close(p, q) for q in other_markers):
                        ok = False
                        break
                if not ok:
                    break

                overlap = paint & other_paint
                if overlap:
                    if overlap_pair is None or {idx, j} != set(overlap_pair):
                        ok = False
                        break
                    if count == other_count:
                        ok = False
                        break
                    overlap_size = len(overlap)

            if not ok:
                continue

            if overlap_pair is not None and idx in overlap_pair:
                other = overlap_pair[0] if overlap_pair[1] == idx else overlap_pair[1]
                if other in placed and overlap_size < MIN_OVERLAP_CELLS:
                    continue

            placed[idx] = {
                "count": count,
                "markers": markers,
                "region": region,
                "paint": paint,
                "bbox": (int(r0), int(c0), int(r0 + shape_h - 1), int(c0 + shape_w - 1)),
            }
            return True

        return False

    indices = list(range(len(specs)))
    if overlap_pair is None:
        order = sorted(indices, key=lambda idx: -len(specs[idx]["region_local"]))
    else:
        a, b = overlap_pair
        order = [a, b] + sorted(
            [idx for idx in indices if idx not in overlap_pair],
            key=lambda idx: -len(specs[idx]["region_local"]),
        )

    for idx in order:
        spec = specs[idx]
        if overlap_pair is not None and idx == overlap_pair[1] and overlap_pair[0] in placed:
            offsets = _overlap_offsets(
                int(spec["height"]),
                int(spec["width"]),
                grid_h,
                grid_w,
                placed[overlap_pair[0]]["bbox"],
            )
            if not offsets:
                return None
        else:
            offsets = _offsets_for_shape(int(spec["height"]), int(spec["width"]), grid_h, grid_w)

        if not try_place(idx, offsets):
            return None

    return [placed[idx] for idx in range(len(specs))]


def _sample_counts(num_groups: int, overlap_case: bool) -> list[int]:
    counts = [random.choice((3, 4)) for _ in range(num_groups)]
    if overlap_case and len(set(counts)) == 1:
        counts[random.randrange(num_groups)] = 7 - counts[0]
    return counts


def _validate_generated(grid: np.ndarray, overlap_case: bool) -> bool:
    bg = _get_bg(grid)
    if bg != BG_COLOR:
        return False

    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    if len(colors) not in (3, 4):
        return False

    regions: dict[int, set[tuple[int, int]]] = {}
    paint_sets: dict[int, set[tuple[int, int]]] = {}
    counts: dict[int, int] = {}

    for color in colors:
        coords = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
        if len(coords) not in (3, 4):
            return False
        hull, region, paint = _region_from_markers(coords)
        if len(hull) != len(coords):
            return False
        if len(paint) < MIN_PAINT_CELLS:
            return False
        regions[color] = region
        paint_sets[color] = paint
        counts[color] = len(coords)

    had_distinct_overlap = False
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            overlap = paint_sets[color_a] & paint_sets[color_b]
            if not overlap:
                continue
            if counts[color_a] == counts[color_b]:
                return False
            had_distinct_overlap = True

    if overlap_case and not had_distinct_overlap:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    for color in colors:
        coords = np.argwhere(grid == color)
        if not np.all(out[coords[:, 0], coords[:, 1]] == color):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        num_groups = random.choice((3, 4))
        overlap_case = bool(random.getrandbits(1))
        counts = _sample_counts(num_groups, overlap_case)

        specs: list[dict[str, object]] = []
        for count in counts:
            spec = _make_shape(count)
            if spec is None:
                break
            specs.append(spec)
        if len(specs) != num_groups:
            continue

        overlap_pair = None
        if overlap_case:
            bigger = [idx for idx, count in enumerate(counts) if count == 4]
            smaller = [idx for idx, count in enumerate(counts) if count == 3]
            if not bigger or not smaller:
                continue
            overlap_pair = (random.choice(bigger), random.choice(smaller))

        grid_h = random.randint(*GRID_SIZE_RANGE)
        grid_w = random.randint(*GRID_SIZE_RANGE)
        placed = _place_specs(specs, grid_h, grid_w, overlap_pair)
        if placed is None:
            continue

        colors = random.sample(FG_POOL, num_groups)
        grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
        for color, pdata in zip(colors, placed, strict=False):
            for r, c in pdata["markers"]:
                grid[r, c] = int(color)

        if _validate_generated(grid, overlap_case):
            return grid

    raise RuntimeError("generate_input failed after repeated retries")
