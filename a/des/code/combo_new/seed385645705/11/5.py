import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
BOX_H = 5
BOX_W = 5
BOX_STRIDE_R = BOX_H + 1
BOX_STRIDE_C = BOX_W + 1
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _legend_info(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    grey_comps = _components(grid == GREY)
    candidates: list[tuple[int, list[int], np.ndarray]] = []
    for pts in grey_comps:
        r0, c0, r1, c1 = _bbox(pts)

        if r0 == 0 and (r1 - r0 + 1) == 3 and (c1 - c0 + 1) >= 5:
            if np.all(grid[r0, c0 : c1 + 1] == GREY) and np.all(grid[r1, c0 : c1 + 1] == GREY):
                if grid[r0 + 1, c0] == GREY and grid[r0 + 1, c1] == GREY:
                    colors = [int(v) for v in grid[r0 + 1, c0 + 1 : c1] if int(v) not in (BLACK, GREY)]
                    if len(colors) == (c1 - c0 - 1) and len(colors) >= 2:
                        mask = np.zeros((h, w), dtype=bool)
                        mask[r0 : r1 + 1, c0 : c1 + 1] = True
                        candidates.append((len(colors), colors, mask))

        if c0 == 0 and (c1 - c0 + 1) == 3 and (r1 - r0 + 1) >= 5:
            if np.all(grid[r0 : r1 + 1, c0] == GREY) and np.all(grid[r0 : r1 + 1, c1] == GREY):
                if grid[r0, c0 + 1] == GREY and grid[r1, c0 + 1] == GREY:
                    colors = [int(v) for v in grid[r0 + 1 : r1, c0 + 1] if int(v) not in (BLACK, GREY)]
                    if len(colors) == (r1 - r0 - 1) and len(colors) >= 2:
                        mask = np.zeros((h, w), dtype=bool)
                        mask[r0 : r1 + 1, c0 : c1 + 1] = True
                        candidates.append((len(colors), colors, mask))

    assert candidates, "a grey-framed legend strip on the top or left edge is required"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, colors, mask = candidates[0]
    return colors, mask


def _component_points(grid: np.ndarray, color: int, ignore_mask: np.ndarray) -> list[tuple[int, int]]:
    mask = (grid == color) & (~ignore_mask)
    comps = _components(mask)
    assert len(comps) == 1, f"color {color} must form exactly one path component outside the legend"
    return comps[0]


def _path_endpoints(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pset = set(points)
    ends: list[tuple[int, int]] = []
    for r, c in sorted(points):
        deg = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))
        if deg == 1:
            ends.append((r, c))
    assert len(ends) == 2, "each path must have exactly two 4-neighbor endpoints"
    return ends


def _candidate_pairs(
    grid: np.ndarray, ends_a: list[tuple[int, int]], ends_b: list[tuple[int, int]]
) -> list[tuple[int, int, tuple[int, int]]]:
    pairs: list[tuple[int, int, tuple[int, int]]] = []
    for ia, (ra, ca) in enumerate(ends_a):
        for ib, (rb, cb) in enumerate(ends_b):
            if abs(ra - rb) + abs(ca - cb) != 2:
                continue
            if ra != rb and ca != cb:
                continue
            mid = ((ra + rb) // 2, (ca + cb) // 2)
            if int(grid[mid]) == BLACK:
                pairs.append((ia, ib, mid))
    pairs.sort()
    return pairs


def _resolve_connection_middles(grid: np.ndarray, colors: list[int], legend_mask: np.ndarray) -> list[tuple[int, int]]:
    endpoints = []
    for color in colors:
        pts = _component_points(grid, color, legend_mask)
        endpoints.append(_path_endpoints(pts))

    all_pairs = []
    for i in range(len(colors) - 1):
        pairs = _candidate_pairs(grid, endpoints[i], endpoints[i + 1])
        assert pairs, f"consecutive colors {colors[i]} and {colors[i + 1]} must have a candidate endpoint pair"
        all_pairs.append(pairs)

    def dfs(edge_idx: int, used_left_endpoint: int | None) -> list[tuple[int, int]] | None:
        if edge_idx == len(all_pairs):
            return []
        for left_idx, right_idx, mid in all_pairs[edge_idx]:
            if used_left_endpoint is not None and left_idx == used_left_endpoint:
                continue
            rest = dfs(edge_idx + 1, right_idx)
            if rest is not None:
                return [mid] + rest
        return None

    mids = dfs(0, None)
    assert mids is not None, "legend order must admit a consistent chain of endpoint connections"
    return mids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect colored paths in the legend's order by filling the one-cell gaps between matching endpoints.

    Input structure:
    - Background is Black.
    - A legend strip touches the top edge or the left edge: a 1-cell-thick Grey frame encloses a single row or column of colored cells.
    - Outside the legend there is exactly one 1-pixel-wide 4-connected path for each legend color.
    - Every path has exactly two endpoints, and some endpoints of different-colored paths are separated by exactly one Black cell in a straight line.

    Rule:
    - Read the colors inside the Grey-framed legend in order; this order defines the path relay.
    - For each consecutive color pair in that order, choose the endpoint pair that is two cells apart horizontally or vertically with a Black cell between them.
    - Fill each such middle Black cell with Grey, and keep all original colored path pixels unchanged.
    - Remove the legend by turning all of its cells Black in the output.

    Key insight:
    - The Grey strip is not just a palette: its visible color order determines which candidate endpoint gaps are activated, while other geometrically possible gaps are ignored.
    """
    g = np.asarray(grid, dtype=int)
    colors, legend_mask = _legend_info(g)
    mids = _resolve_connection_middles(g, colors, legend_mask)

    out = g.copy()
    out[legend_mask] = BLACK
    for r, c in mids:
        out[r, c] = GREY
    return out


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("line_cells expects an axis-aligned segment")


def _port(box_r: int, box_c: int, side: str) -> tuple[int, int]:
    mid_r = box_r + BOX_H // 2
    mid_c = box_c + BOX_W // 2
    if side == "U":
        return box_r, mid_c
    if side == "D":
        return box_r + BOX_H - 1, mid_c
    if side == "L":
        return mid_r, box_c
    if side == "R":
        return mid_r, box_c + BOX_W - 1
    raise ValueError(side)


def _path_cells(box_r: int, box_c: int, side_a: str, side_b: str) -> set[tuple[int, int]]:
    center = (box_r + BOX_H // 2, box_c + BOX_W // 2)
    pa = _port(box_r, box_c, side_a)
    pb = _port(box_r, box_c, side_b)
    cells = set(_line_cells(pa, center))
    cells.update(_line_cells(center, pb))
    return cells


def _direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == -1 and dc == 0:
        return "U"
    if dr == 1 and dc == 0:
        return "D"
    if dr == 0 and dc == -1:
        return "L"
    if dr == 0 and dc == 1:
        return "R"
    raise ValueError((a, b))


def _opposite(side: str) -> str:
    return {"U": "D", "D": "U", "L": "R", "R": "L"}[side]


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return [(r - min_r, c - min_c) for r, c in coords]


def _symmetries(coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    def transform(pt: tuple[int, int], k: int) -> tuple[int, int]:
        r, c = pt
        if k == 0:
            return r, c
        if k == 1:
            return r, -c
        if k == 2:
            return -r, c
        if k == 3:
            return -r, -c
        if k == 4:
            return c, r
        if k == 5:
            return c, -r
        if k == 6:
            return -c, r
        return -c, -r

    out: list[list[tuple[int, int]]] = []
    seen = set()
    for k in range(8):
        cur = _normalize([transform(pt, k) for pt in coords])
        key = tuple(cur)
        if key not in seen:
            seen.add(key)
            out.append(cur)
    return out


def _has_nonconsecutive_adjacency(coords: list[tuple[int, int]]) -> bool:
    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            if j <= i + 1:
                continue
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                return True
    return False


BASE_NON_EXTRA = {
    3: [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1), (1, 1)],
    ],
    4: [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
    ],
}
BASE_EXTRA = [[(0, 0), (0, 1), (1, 1), (1, 0)]]


def _choose_template(extra_candidate: bool) -> list[tuple[int, int]]:
    if extra_candidate:
        base = random.choice(BASE_EXTRA)
        return random.choice(_symmetries(base))

    count = random.choice([3, 4])
    choices = []
    for base in BASE_NON_EXTRA[count]:
        for cur in _symmetries(base):
            if not _has_nonconsecutive_adjacency(cur):
                choices.append(cur)
    return random.choice(choices)


def _choose_end_sides(coords: list[tuple[int, int]], extra_candidate: bool) -> list[tuple[str, str]]:
    count = len(coords)
    dirs = [_direction(coords[i], coords[i + 1]) for i in range(count - 1)]
    sides: list[tuple[str, str]] = []
    for i in range(count):
        if i == 0:
            connect = dirs[0]
            if extra_candidate:
                free = _direction(coords[0], coords[-1])
            else:
                free = random.choice([s for s in "UDLR" if s != connect])
            sides.append((free, connect))
        elif i == count - 1:
            connect = _opposite(dirs[-1])
            if extra_candidate:
                free = _direction(coords[-1], coords[0])
            else:
                free = random.choice([s for s in "UDLR" if s != connect])
            sides.append((connect, free))
        else:
            sides.append((_opposite(dirs[i - 1]), dirs[i]))
    return sides


def _content_extent(coords: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    return max_r * BOX_STRIDE_R + BOX_H, max_c * BOX_STRIDE_C + BOX_W


def _place_legend(grid: np.ndarray, colors: list[int], orientation: str) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    if orientation == "top":
        start_c = random.randint(0, w - (len(colors) + 2))
        grid[0:3, start_c : start_c + len(colors) + 2] = GREY
        for i, color in enumerate(colors):
            grid[1, start_c + 1 + i] = color
        mask[0:3, start_c : start_c + len(colors) + 2] = True
    else:
        start_r = random.randint(0, h - (len(colors) + 2))
        grid[start_r : start_r + len(colors) + 2, 0:3] = GREY
        for i, color in enumerate(colors):
            grid[start_r + 1 + i, 1] = color
        mask[start_r : start_r + len(colors) + 2, 0:3] = True
    return mask


def _build_instance() -> np.ndarray:
    extra_candidate = random.random() < 0.5
    coords = _choose_template(extra_candidate)
    colors = random.sample(PATH_COLORS, len(coords))
    orientation = random.choice(["top", "left"])
    sides = _choose_end_sides(coords, extra_candidate)

    content_h, content_w = _content_extent(coords)
    if orientation == "top":
        max_base_r = min(7, 30 - content_h)
        max_base_c = min(4, 30 - content_w)
        if max_base_r < 4 or max_base_c < 1:
            raise AssertionError("top-oriented layout does not fit")
        base_r = random.randint(4, max_base_r)
        base_c = random.randint(1, max_base_c)
        max_bottom = min(3, 30 - (base_r + content_h))
        max_right = min(3, 30 - (base_c + content_w))
        h = base_r + content_h + random.randint(0, max_bottom)
        w = max(base_c + content_w + random.randint(0, max_right), len(colors) + 2)
    else:
        max_base_r = min(4, 30 - content_h)
        max_base_c = min(7, 30 - content_w)
        if max_base_r < 1 or max_base_c < 4:
            raise AssertionError("left-oriented layout does not fit")
        base_r = random.randint(1, max_base_r)
        base_c = random.randint(4, max_base_c)
        max_bottom = min(3, 30 - (base_r + content_h))
        max_right = min(3, 30 - (base_c + content_w))
        h = max(base_r + content_h + random.randint(0, max_bottom), len(colors) + 2)
        w = base_c + content_w + random.randint(0, max_right)

    assert h <= 30 and w <= 30
    grid = np.zeros((h, w), dtype=int)
    legend_mask = _place_legend(grid, colors, orientation)

    occupied = legend_mask.copy()
    for color, (cr, cc), (side_a, side_b) in zip(colors, coords, sides, strict=False):
        box_r = base_r + cr * BOX_STRIDE_R
        box_c = base_c + cc * BOX_STRIDE_C
        cells = _path_cells(box_r, box_c, side_a, side_b)
        for r, c in cells:
            assert not occupied[r, c], "generated path overlap"
            grid[r, c] = color
            occupied[r, c] = True

    return grid


def _candidate_edge_count(grid: np.ndarray) -> int:
    colors, legend_mask = _legend_info(grid)
    endpoints = []
    for color in colors:
        endpoints.append(_path_endpoints(_component_points(grid, color, legend_mask)))
    count = 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            count += len(_candidate_pairs(grid, endpoints[i], endpoints[j]))
    return count


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _build_instance()
            colors, legend_mask = _legend_info(grid)
            assert len(colors) in (3, 4)
            mids = _resolve_connection_middles(grid, colors, legend_mask)
            assert len(mids) == len(colors) - 1

            candidate_edges = _candidate_edge_count(grid)
            if candidate_edges < len(colors) - 1:
                continue

            out = main(grid)
            assert out.shape == grid.shape
            assert np.all(out[legend_mask] == BLACK)
            grey_added = {(int(r), int(c)) for r, c in np.argwhere(out == GREY)} - {
                (int(r), int(c)) for r, c in np.argwhere(grid == GREY)
            }
            assert grey_added == set(mids)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid ARC instance")
