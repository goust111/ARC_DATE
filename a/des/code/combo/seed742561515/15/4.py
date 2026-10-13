import random
from collections import deque

import numpy as np


BG = 0
PATH = 3
FRAME = 5
STRIPE_COLORS = (1, 2, 4, 6, 7, 8, 9)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

GRID_MIN = 22
GRID_MAX = 30
PAIR_COUNT_VALUES = (1, 2, 3)
PAIR_COUNT_WEIGHTS = (3, 4, 2)
SOURCE_H_RANGE = (2, 4)
SOURCE_W_RANGE = (3, 6)
TURN_OPTIONS = {0: (0, 2), 1: (1, 3)}
SEG_LEN_RANGE = (2, 5)
PAIR_SIZE_CAP = 16
PAIR_MARGIN = 1
PATH_ATTEMPTS = 200
PAIR_ATTEMPTS = 200
GRID_ATTEMPTS = 80
PLACE_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _iter_bbox(bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r, c


def _iter_border(bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = bbox
    for c in range(c0, c1 + 1):
        yield r0, c
        yield r1, c
    for r in range(r0 + 1, r1):
        yield r, c0
        yield r, c1


def _touching(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _find_sources(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != PATH) & (grid != FRAME)
    sources = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if not (2 <= h <= 4 and w >= 2):
            continue
        if coords.shape[0] != h * w:
            continue
        row_colors = []
        ok = True
        for rr in range(h):
            vals = np.unique(sub[rr])
            if len(vals) != 1 or int(vals[0]) in (BG, PATH, FRAME):
                ok = False
                break
            row_colors.append(int(vals[0]))
        if not ok or len(set(row_colors)) != h:
            continue
        sources.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (h, w),
                "row_colors": row_colors,
            }
        )
    return sources


def _find_pools(grid: np.ndarray) -> list[dict]:
    pools = []
    for coords in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h < 4 or w < 4:
            continue
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if not np.array_equal(sub == FRAME, border_mask):
            continue
        if not np.all(sub[1:-1, 1:-1] == BG):
            continue
        pools.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_shape": (h - 2, w - 2),
            }
        )
    return pools


def _trace_path(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, xy)) for xy in coords.tolist()}
    adjacency = {}
    for r, c in pts:
        nbrs = [p for p in _neighbors4(r, c) if p in pts]
        assert len(nbrs) <= 2
        adjacency[(r, c)] = nbrs
    endpoints = [p for p, nbrs in adjacency.items() if len(nbrs) == 1]
    assert len(endpoints) == 2
    order = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        nxts = [p for p in adjacency[cur] if p != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(pts)
    return order


def _count_turns(path: list[tuple[int, int]]) -> int:
    steps = []
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        steps.append((r1 - r0, c1 - c0))
    return sum(steps[i] != steps[i - 1] for i in range(1, len(steps)))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sources = _find_sources(g)
    pools = _find_pools(g)
    path_components = _components(g == PATH)

    assert sources
    assert len(sources) == len(pools) == len(path_components)

    source_id = -np.ones(g.shape, dtype=int)
    for idx, source in enumerate(sources):
        r0, c0, r1, c1 = source["bbox"]
        source_id[r0 : r1 + 1, c0 : c1 + 1] = idx

    pool_id = -np.ones(g.shape, dtype=int)
    for idx, pool in enumerate(pools):
        for r, c in _iter_border(pool["bbox"]):
            pool_id[r, c] = idx

    out = g.copy()
    out[g == PATH] = BG

    used_sources = set()
    used_pools = set()
    for coords in path_components:
        path = _trace_path(coords)
        turns = _count_turns(path)
        source_hits = set()
        pool_hits = set()
        for r, c in path:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    sid = int(source_id[nr, nc])
                    pid = int(pool_id[nr, nc])
                    if sid >= 0:
                        source_hits.add(sid)
                    if pid >= 0:
                        pool_hits.add(pid)
        assert len(source_hits) == 1
        assert len(pool_hits) == 1
        sid = next(iter(source_hits))
        pid = next(iter(pool_hits))
        used_sources.add(sid)
        used_pools.add(pid)

        source = sources[sid]
        pool = pools[pid]
        assert source["shape"] == pool["inner_shape"]
        colors = list(source["row_colors"])
        if turns % 2 == 1:
            colors.reverse()
        r0, c0, r1, c1 = pool["bbox"]
        for i, color in enumerate(colors):
            out[r0 + 1 + i, c0 + 1 : c1] = color

    assert len(used_sources) == len(sources)
    assert len(used_pools) == len(pools)
    return out


def _source_bbox(start: tuple[int, int], direction: int, h: int, w: int) -> tuple[int, int, int, int]:
    sr, sc = start
    if direction == 0:  # up, source is below
        top = sr + 1
        left = sc - w // 2
    elif direction == 1:  # right, source is left
        top = sr - h // 2
        left = sc - w
    elif direction == 2:  # down, source is above
        top = sr - h
        left = sc - w // 2
    else:  # left, source is right
        top = sr - h // 2
        left = sc + 1
    return top, left, top + h - 1, left + w - 1


def _pool_bbox(end: tuple[int, int], direction: int, h: int, w: int) -> tuple[int, int, int, int]:
    er, ec = end
    ph, pw = h + 2, w + 2
    if direction == 0:  # up, pool is above
        top = er - ph
        left = ec - (w // 2 + 1)
    elif direction == 1:  # right, pool is right
        top = er - (h // 2 + 1)
        left = ec + 1
    elif direction == 2:  # down, pool is below
        top = er + 1
        left = ec - (w // 2 + 1)
    else:  # left, pool is left
        top = er - (h // 2 + 1)
        left = ec - pw
    return top, left, top + ph - 1, left + pw - 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _sample_path(rng: random.Random, turns: int):
    for _ in range(PATH_ATTEMPTS):
        dirs = [rng.randrange(4)]
        for _ in range(turns):
            dirs.append((dirs[-1] + rng.choice((-1, 1))) % 4)
        lens = [rng.randint(*SEG_LEN_RANGE) for _ in dirs]

        cells = [(0, 0)]
        used = {(0, 0)}
        cur = (0, 0)
        ok = True
        for direction, seg_len in zip(dirs, lens):
            dr, dc = DIRS[direction]
            for _ in range(seg_len):
                prev = cur
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in used:
                    ok = False
                    break
                for nbr in _neighbors4(*nxt):
                    if nbr in used and nbr != prev:
                        ok = False
                        break
                if not ok:
                    break
                cells.append(nxt)
                used.add(nxt)
                cur = nxt
            if not ok:
                break
        if ok:
            return cells, dirs
    return None, None


def _make_pair(rng: random.Random):
    for _ in range(PAIR_ATTEMPTS):
        h = rng.randint(*SOURCE_H_RANGE)
        w = rng.randint(*SOURCE_W_RANGE)
        turns = rng.choice(TURN_OPTIONS[rng.randrange(2)])
        colors = rng.sample(STRIPE_COLORS, h)

        path, dirs = _sample_path(rng, turns)
        if path is None:
            continue

        source_bbox = _source_bbox(path[0], dirs[0], h, w)
        pool_bbox = _pool_bbox(path[-1], dirs[-1], h, w)
        if _boxes_overlap(source_bbox, pool_bbox):
            continue

        source_cells = set(_iter_bbox(source_bbox))
        pool_cells = set(_iter_bbox(pool_bbox))
        pool_border = set(_iter_border(pool_bbox))
        path_set = set(path)

        if path_set & source_cells:
            continue
        if path_set & pool_cells:
            continue
        if _touching(source_cells, pool_cells):
            continue

        source_touch = {
            cell for cell in path if any(nbr in source_cells for nbr in _neighbors4(*cell))
        }
        pool_touch = {
            cell for cell in path if any(nbr in pool_border for nbr in _neighbors4(*cell))
        }
        if source_touch != {path[0]}:
            continue
        if pool_touch != {path[-1]}:
            continue

        occupied = source_cells | pool_cells | path_set
        rows = [r for r, _ in occupied]
        cols = [c for _, c in occupied]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        pair_h = max_r - min_r + 1
        pair_w = max_c - min_c + 1
        if pair_h > PAIR_SIZE_CAP or pair_w > PAIR_SIZE_CAP:
            continue

        dr = -min_r
        dc = -min_c

        def shift_box(bbox):
            r0, c0, r1, c1 = bbox
            return r0 + dr, c0 + dc, r1 + dr, c1 + dc

        return {
            "height": pair_h,
            "width": pair_w,
            "source_bbox": shift_box(source_bbox),
            "pool_bbox": shift_box(pool_bbox),
            "path": [(r + dr, c + dc) for r, c in path],
            "colors": colors,
        }
    return None


def _draw_pair(grid: np.ndarray, pair: dict, top: int, left: int) -> None:
    sr0, sc0, sr1, sc1 = pair["source_bbox"]
    for i, color in enumerate(pair["colors"]):
        grid[top + sr0 + i, left + sc0 : left + sc1 + 1] = color

    pr0, pc0, pr1, pc1 = pair["pool_bbox"]
    grid[top + pr0, left + pc0 : left + pc1 + 1] = FRAME
    grid[top + pr1, left + pc0 : left + pc1 + 1] = FRAME
    grid[top + pr0 : top + pr1 + 1, left + pc0] = FRAME
    grid[top + pr0 : top + pr1 + 1, left + pc1] = FRAME

    for r, c in pair["path"]:
        grid[top + r, left + c] = PATH


def _sample_grid(rng: random.Random):
    pair_count = rng.choices(PAIR_COUNT_VALUES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    pairs = []
    for _ in range(pair_count):
        pair = _make_pair(rng)
        if pair is None:
            return None
        pairs.append(pair)
    pairs.sort(key=lambda pair: pair["height"] * pair["width"], reverse=True)

    area_need = sum(
        (pair["height"] + 2 * PAIR_MARGIN) * (pair["width"] + 2 * PAIR_MARGIN)
        for pair in pairs
    )
    min_side = max(
        GRID_MIN,
        max(pair["height"] for pair in pairs) + 2,
        max(pair["width"] for pair in pairs) + 2,
        18 + 3 * (pair_count - 1),
    )

    for _ in range(GRID_ATTEMPTS):
        h = rng.randint(min_side, GRID_MAX)
        w = rng.randint(min_side, GRID_MAX)
        if h * w < area_need * 2:
            continue
        blocked = np.zeros((h, w), dtype=bool)
        placements = []
        ok = True
        for pair in pairs:
            if pair["height"] + 2 > h or pair["width"] + 2 > w:
                ok = False
                break
            placed = False
            for _ in range(PLACE_ATTEMPTS):
                top = rng.randint(1, h - pair["height"] - 1)
                left = rng.randint(1, w - pair["width"] - 1)
                r0 = max(0, top - PAIR_MARGIN)
                c0 = max(0, left - PAIR_MARGIN)
                r1 = min(h, top + pair["height"] + PAIR_MARGIN)
                c1 = min(w, left + pair["width"] + PAIR_MARGIN)
                if blocked[r0:r1, c0:c1].any():
                    continue
                blocked[r0:r1, c0:c1] = True
                placements.append((pair, top, left))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        grid = np.zeros((h, w), dtype=int)
        for pair, top, left in placements:
            _draw_pair(grid, pair, top, left)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        if not np.any(grid == PATH):
            continue
        if np.any(out == PATH):
            continue
        if not np.any(out != grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
