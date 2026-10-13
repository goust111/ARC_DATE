import random
from collections import Counter

import numpy as np


BG_COLOR = 0
LINE_COLOR = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500
CELL_SIZE_MIN = 3
CELL_SIZE_MAX = 6


SEED_LIBRARY: dict[tuple[int, int], list[np.ndarray]] = {
    (2, 2): [
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
    ],
    (2, 3): [
        np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    ],
    (3, 2): [
        np.array([[1, 1], [1, 1], [1, 1]], dtype=int),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=int),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    ],
    (3, 3): [
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    ],
}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = map(int, mask.shape)
    labels = np.zeros((h, w), dtype=int)
    comps: list[np.ndarray] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            stack = [(r, c)]
            labels[r, c] = next_id
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return labels, comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rect_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _choose_line_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_score = None
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    for color in map(int, values):
        if color == bg:
            continue
        _labels, comps = _components(grid != color)
        if len(comps) < 2:
            continue
        rect_flags = [_is_rect_component(comp) for comp in comps]
        score = (
            all(rect_flags),
            sum(rect_flags),
            len(comps),
            count_map[color],
        )
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None and best_score is not None
    assert best_score[0], "expected line color to split grid into rectangular regions"
    return int(best_color)


def _region_adjacency(labels: np.ndarray, line_mask: np.ndarray) -> dict[int, set[int]]:
    h, w = map(int, labels.shape)
    region_ids = [int(x) for x in np.unique(labels) if int(x) != 0]
    adj = {rid: set() for rid in region_ids}
    for r in range(h):
        for c in range(w):
            if labels[r, c] == 0:
                continue
            rid = int(labels[r, c])
            if c + 2 < w and line_mask[r, c + 1] and labels[r, c + 2] not in (0, rid):
                adj[rid].add(int(labels[r, c + 2]))
            if r + 2 < h and line_mask[r + 1, c] and labels[r + 2, c] not in (0, rid):
                adj[rid].add(int(labels[r + 2, c]))
    for rid, nbrs in list(adj.items()):
        for other in nbrs:
            adj[other].add(rid)
    return adj


def _extract_seed(region: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    non_bg = [int(x) for x in np.unique(region) if int(x) != bg]
    assert len(non_bg) == 1, "each region must contain exactly one seed color"
    seed_color = int(non_bg[0])
    pts = np.argwhere(region == seed_color)
    r0, c0, r1, c1 = _bbox(pts)
    assert r0 == 0 and c0 == 0, "seed must start at region top-left"
    pattern = region[r0 : r1 + 1, c0 : c1 + 1].copy()
    return pattern, seed_color


def _stamp_pattern(
    out: np.ndarray,
    region_bbox: tuple[int, int, int, int],
    pattern: np.ndarray,
    copies: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = region_bbox
    region_h = int(r1 - r0 + 1)
    region_w = int(c1 - c0 + 1)
    ph, pw = map(int, pattern.shape)
    mask = pattern != bg
    cur_r = 0
    cur_c = 0
    for _ in range(int(copies)):
        if cur_c > 0 and cur_c + pw > region_w:
            cur_c = 0
            cur_r += ph
        rr0 = int(r0 + cur_r)
        cc0 = int(c0 + cur_c)
        rr1 = min(int(r0 + region_h), rr0 + ph)
        cc1 = min(int(c0 + region_w), cc0 + pw)
        if rr0 < int(r0 + region_h) and cc0 < int(c0 + region_w):
            submask = mask[: rr1 - rr0, : cc1 - cc0]
            subpat = pattern[: rr1 - rr0, : cc1 - cc0]
            window = out[rr0:rr1, cc0:cc1]
            window[submask] = subpat[submask]
        cur_c += pw


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    bg = _most_common_color(g)
    line = _choose_line_color(g, bg)
    line_mask = g == line
    labels, comps = _components(~line_mask)
    region_ids = [int(x) for x in np.unique(labels) if int(x) != 0]
    assert comps and region_ids

    region_boxes: dict[int, tuple[int, int, int, int]] = {}
    for rid, coords in zip(region_ids, comps):
        assert _is_rect_component(coords), "all non-line regions must be rectangles"
        region_boxes[int(rid)] = _bbox(coords)

    adjacency = _region_adjacency(labels, line_mask)
    out = np.where(line_mask, line, bg).astype(int)

    for rid in region_ids:
        r0, c0, r1, c1 = region_boxes[rid]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        pattern, _seed_color = _extract_seed(region, bg)
        copies = len(adjacency[rid])
        assert 1 <= copies <= 4, "adjacent region count must be between 1 and 4"
        _stamp_pattern(out, (r0, c0, r1, c1), pattern, copies, bg)

    return out.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a grey-partitioned grid, each rectangular region contains one small
    monochrome seed in its top-left corner. Count how many other regions share
    a boundary with that region, then tile the seed that many times inside the
    region from the top-left, wrapping to the next seed row when needed.
    """
    return _solve(grid)


def _line_owner_segments(owner: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    rows, cols = map(int, owner.shape)
    row_starts = [1 + sum(row_heights[:r]) + r for r in range(rows)]
    col_starts = [1 + sum(col_widths[:c]) + c for c in range(cols)]
    height = 2 + sum(row_heights) + (rows - 1)
    width = 2 + sum(col_widths) + (cols - 1)
    grid = np.full((height, width), BG_COLOR, dtype=int)

    grid[0, :] = LINE_COLOR
    grid[-1, :] = LINE_COLOR
    grid[:, 0] = LINE_COLOR
    grid[:, -1] = LINE_COLOR

    for r in range(rows - 1):
        br = row_starts[r] + row_heights[r]
        for c in range(cols):
            if int(owner[r, c]) != int(owner[r + 1, c]):
                c0 = col_starts[c]
                grid[br, c0 : c0 + col_widths[c]] = LINE_COLOR
    for c in range(cols - 1):
        bc = col_starts[c] + col_widths[c]
        for r in range(rows):
            if int(owner[r, c]) != int(owner[r, c + 1]):
                r0 = row_starts[r]
                grid[r0 : r0 + row_heights[r], bc] = LINE_COLOR
    for r in range(rows - 1):
        br = row_starts[r] + row_heights[r]
        for c in range(cols - 1):
            bc = col_starts[c] + col_widths[c]
            around = {
                int(owner[r, c]),
                int(owner[r + 1, c]),
                int(owner[r, c + 1]),
                int(owner[r + 1, c + 1]),
            }
            if len(around) > 1:
                grid[br, bc] = LINE_COLOR
    return grid


def _owner_rectangles(owner: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    rects: dict[int, tuple[int, int, int, int]] = {}
    for rid in sorted(int(x) for x in np.unique(owner)):
        pts = np.argwhere(owner == rid)
        r0, c0, r1, c1 = _bbox(pts)
        assert np.all(owner[r0 : r1 + 1, c0 : c1 + 1] == rid), "regions must be rectangles"
        rects[rid] = (int(r0), int(c0), int(r1), int(c1))
    return rects


def _owner_adjacency(owner: np.ndarray) -> dict[int, set[int]]:
    rows, cols = map(int, owner.shape)
    adj = {int(rid): set() for rid in np.unique(owner)}
    for r in range(rows):
        for c in range(cols - 1):
            a = int(owner[r, c])
            b = int(owner[r, c + 1])
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    for r in range(rows - 1):
        for c in range(cols):
            a = int(owner[r, c])
            b = int(owner[r + 1, c])
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    return adj


def _pixel_region_boxes(
    owner: np.ndarray, row_heights: list[int], col_widths: list[int]
) -> dict[int, tuple[int, int, int, int]]:
    row_starts = [1 + sum(row_heights[:r]) + r for r in range(owner.shape[0])]
    col_starts = [1 + sum(col_widths[:c]) + c for c in range(owner.shape[1])]
    coarse_rects = _owner_rectangles(owner)
    pixel_rects: dict[int, tuple[int, int, int, int]] = {}
    for rid, (r0, c0, r1, c1) in coarse_rects.items():
        top = row_starts[r0]
        left = col_starts[c0]
        bottom = row_starts[r1] + row_heights[r1] - 1
        right = col_starts[c1] + col_widths[c1] - 1
        pixel_rects[rid] = (int(top), int(left), int(bottom), int(right))
    return pixel_rects


def _choose_seed_pattern(region_h: int, region_w: int, rng: random.Random) -> np.ndarray:
    choices = [
        dims for dims in SEED_LIBRARY
        if dims[0] <= int(region_h) and dims[1] <= int(region_w)
    ]
    assert choices, "region must fit at least one seed pattern"
    dims = rng.choice(choices)
    mask = rng.choice(SEED_LIBRARY[dims])
    return mask.copy()


def _colorize_seed(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=int)
    out[mask != 0] = int(color)
    return out


def _build_strip_owner(rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        rows, cols = 1, rng.randint(3, 5)
        cuts = sorted(rng.sample(range(1, cols), rng.randint(1, cols - 1)))
        segments = [0] + cuts + [cols]
        owner = np.zeros((rows, cols), dtype=int)
        for rid, (a, b) in enumerate(zip(segments[:-1], segments[1:])):
            owner[:, a:b] = rid
        return owner
    rows, cols = rng.randint(3, 5), 1
    cuts = sorted(rng.sample(range(1, rows), rng.randint(1, rows - 1)))
    segments = [0] + cuts + [rows]
    owner = np.zeros((rows, cols), dtype=int)
    for rid, (a, b) in enumerate(zip(segments[:-1], segments[1:])):
        owner[a:b, :] = rid
    return owner


def _build_cross_owner(rng: random.Random) -> np.ndarray:
    owner = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [4, 4, 4],
        ],
        dtype=int,
    )
    if rng.random() < 0.5:
        owner = owner.T
    if rng.random() < 0.5:
        owner = owner[::-1, :]
    if rng.random() < 0.5:
        owner = owner[:, ::-1]
    ids = sorted(int(x) for x in np.unique(owner))
    remap = {rid: i for i, rid in enumerate(ids)}
    return np.vectorize(remap.get)(owner).astype(int)


def _random_rect_tiling(
    rows: int, cols: int, min_regions: int, max_regions: int, rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
    used = np.zeros((rows, cols), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []

    def search() -> bool:
        current = len(rects)
        remaining = int((~used).sum())
        if current > max_regions or current + remaining < min_regions:
            return False
        empty = np.argwhere(~used)
        if empty.size == 0:
            return min_regions <= current <= max_regions
        r0, c0 = map(int, empty[0])
        candidates: list[tuple[int, int, int, int]] = []
        for h in range(1, rows - r0 + 1):
            for w in range(1, cols - c0 + 1):
                block = used[r0 : r0 + h, c0 : c0 + w]
                if np.any(block):
                    break
                candidates.append((r0, r0 + h, c0, c0 + w))
        rng.shuffle(candidates)
        candidates.sort(key=lambda rect: (rect[1] - rect[0]) * (rect[3] - rect[2]), reverse=rng.random() < 0.5)
        for rect in candidates:
            a, b, c, d = rect
            used[a:b, c:d] = True
            rects.append(rect)
            if search():
                return True
            rects.pop()
            used[a:b, c:d] = False
        return False

    if search():
        return rects.copy()
    return None


def _build_mixed_owner(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(3, 4)
    cols = rng.randint(3, 4)
    rects = _random_rect_tiling(rows, cols, min_regions=4, max_regions=7, rng=rng)
    if rects is None:
        return None
    owner = -np.ones((rows, cols), dtype=int)
    for rid, (r0, r1, c0, c1) in enumerate(rects):
        owner[r0:r1, c0:c1] = rid
    assert np.all(owner >= 0)
    return owner


def _choose_owner(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(["strip", "cross", "mixed"])
    if mode == "strip":
        return _build_strip_owner(rng)
    if mode == "cross":
        return _build_cross_owner(rng)
    return _build_mixed_owner(rng)


def _make_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    owner = _choose_owner(rng)
    if owner is None:
        return None

    adjacency = _owner_adjacency(owner)
    counts = sorted(len(v) for v in adjacency.values())
    if not counts or min(counts) < 1 or max(counts) > 4:
        return None

    region_count = len(adjacency)
    if region_count > len(SEED_COLORS):
        return None

    row_heights = [rng.randint(CELL_SIZE_MIN, CELL_SIZE_MAX) for _ in range(owner.shape[0])]
    col_widths = [rng.randint(CELL_SIZE_MIN, CELL_SIZE_MAX) for _ in range(owner.shape[1])]
    grid = _line_owner_segments(owner, row_heights, col_widths)
    pixel_boxes = _pixel_region_boxes(owner, row_heights, col_widths)

    colors = SEED_COLORS.copy()
    rng.shuffle(colors)
    seeds: dict[int, np.ndarray] = {}
    region_specs: list[tuple[tuple[int, int, int, int], np.ndarray, int]] = []

    for rid, color in zip(sorted(pixel_boxes), colors):
        r0, c0, r1, c1 = pixel_boxes[rid]
        region_h = int(r1 - r0 + 1)
        region_w = int(c1 - c0 + 1)
        seed_mask = _choose_seed_pattern(region_h, region_w, rng)
        seed = _colorize_seed(seed_mask, color)
        seeds[rid] = seed
        sh, sw = map(int, seed.shape)
        grid[r0 : r0 + sh, c0 : c0 + sw] = seed
        region_specs.append(((r0, c0, r1, c1), seed, len(adjacency[rid])))

    bg_count = int(np.count_nonzero(grid == BG_COLOR))
    line_count = int(np.count_nonzero(grid == LINE_COLOR))
    if bg_count <= line_count:
        return None
    non_bg_counts = Counter(map(int, grid[grid != BG_COLOR].flatten()))
    if non_bg_counts[LINE_COLOR] <= max(v for k, v in non_bg_counts.items() if k != LINE_COLOR):
        return None

    expected = np.where(grid == LINE_COLOR, LINE_COLOR, BG_COLOR).astype(int)
    for bbox, seed, copies in region_specs:
        _stamp_pattern(expected, bbox, seed, copies, BG_COLOR)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    return grid.astype(int), expected.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        made = _make_instance(rng)
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate a valid input")
