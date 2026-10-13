import random
from collections import Counter, deque

import numpy as np


def _neighbors4(r: int, c: int):
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
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_palette(row: np.ndarray) -> list[int]:
    palette: list[int] = []
    c = 0
    while c < row.shape[0]:
        color = int(row[c])
        palette.append(color)
        c += 1
        while c < row.shape[0] and int(row[c]) == color:
            c += 1
    return palette


def _infer_region_colors(region: np.ndarray) -> tuple[int, int, int]:
    counts = Counter(map(int, region.ravel()))
    wall = int(Counter(map(int, region[0])).most_common(1)[0][0])
    non_wall = [color for color in counts if color != wall]
    assert len(non_wall) >= 2, "region must contain wall, background, and seeds"
    bg = max(non_wall, key=lambda color: counts[color])
    seed_candidates = [color for color in non_wall if color != bg]
    assert len(seed_candidates) == 1, "region must contain a unique seed color"
    return wall, int(bg), int(seed_candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room with the palette color indexed by its seed count.

    Input structure:
    - The first row is a palette legend: contiguous color blocks read left-to-right.
    - Below it sits a region enclosed and partitioned by one-cell-thick walls.
    - Each room contains black background pixels plus a small number of identical seed pixels.

    Rule:
    - Read the palette row as an ordered list of colors, one per contiguous block.
    - In the region below, treat each 4-connected non-wall area as one room.
    - Count the seeds in each room; if a room contains N seeds, replace that room's black background pixels with the Nth palette color.
    - Keep the palette row, the walls, and the seed pixels unchanged.

    Key insight: The top row is not part of the scene to repaint; it is an ordered legend, and each room's seed count is the index into that legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3

    palette = _parse_palette(g[0])
    region = g[1:]
    wall, bg, seed = _infer_region_colors(region)

    out = g.copy()
    room_mask = region != wall
    for comp in _components(room_mask):
        seed_count = sum(int(region[r, c]) == seed for r, c in comp)
        assert 1 <= seed_count <= len(palette), "room seed count must index the palette"
        fill_color = int(palette[seed_count - 1])
        for r, c in comp:
            if int(region[r, c]) == bg:
                out[r + 1, c] = fill_color
    return out


# --------------- generator ---------------

_BG = 0
_SEED = 2
_WALL = 5
_PALETTE_POOL = [1, 3, 4, 6, 7, 8, 9]

_COARSE_ROWS = (2, 4)
_COARSE_COLS = (2, 4)
_CELL_SIZE_MIN = 3
_CELL_SIZE_MAX = 5
_ROOMS_MAX = 6
_PALETTE_MIN = 3
_PALETTE_MAX = 6
_GEN_RETRIES = 500


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _cell_starts(sizes: list[int]) -> list[int]:
    starts: list[int] = []
    pos = 1
    for size in sizes:
        starts.append(pos)
        pos += size + 1
    return starts


def _random_sizes(rng: random.Random, count: int) -> list[int]:
    return [rng.randint(_CELL_SIZE_MIN, _CELL_SIZE_MAX) for _ in range(count)]


def _random_groups(rows: int, cols: int, rng: random.Random) -> _UnionFind:
    total = rows * cols
    target_rooms = rng.randint(2, min(_ROOMS_MAX, total - 1))
    uf = _UnionFind(total)
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if r + 1 < rows:
                edges.append((idx, (r + 1) * cols + c))
            if c + 1 < cols:
                edges.append((idx, r * cols + c + 1))
    rng.shuffle(edges)
    room_count = total
    for a, b in edges:
        if room_count == target_rooms:
            break
        if uf.union(a, b):
            room_count -= 1
    return uf


def _coarse_groups(rows: int, cols: int, uf: _UnionFind) -> list[list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            groups.setdefault(uf.find(idx), []).append((r, c))
    return list(groups.values())


def _render_region(
    row_sizes: list[int],
    col_sizes: list[int],
    uf: _UnionFind,
) -> np.ndarray:
    rows = len(row_sizes)
    cols = len(col_sizes)
    row_starts = _cell_starts(row_sizes)
    col_starts = _cell_starts(col_sizes)
    region_h = row_starts[-1] + row_sizes[-1] + 1
    region_w = col_starts[-1] + col_sizes[-1] + 1
    region = np.full((region_h, region_w), _WALL, dtype=int)

    for r in range(rows):
        for c in range(cols):
            r0 = row_starts[r]
            c0 = col_starts[c]
            region[r0 : r0 + row_sizes[r], c0 : c0 + col_sizes[c]] = _BG

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if r + 1 < rows and uf.find(idx) == uf.find((r + 1) * cols + c):
                wall_r = row_starts[r] + row_sizes[r]
                c0 = col_starts[c]
                region[wall_r, c0 : c0 + col_sizes[c]] = _BG
            if c + 1 < cols and uf.find(idx) == uf.find(r * cols + c + 1):
                wall_c = col_starts[c] + col_sizes[c]
                r0 = row_starts[r]
                region[r0 : r0 + row_sizes[r], wall_c] = _BG

    return region


def _room_buckets(region: np.ndarray) -> list[dict[tuple[int, int], list[tuple[int, int]]]]:
    buckets_all: list[dict[tuple[int, int], list[tuple[int, int]]]] = []
    for comp in _components(region == _BG):
        buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for r, c in comp:
            buckets.setdefault((r % 2, c % 2), []).append((r, c))
        buckets_all.append(buckets)
    return buckets_all


def _pick_seed_counts(
    buckets_all: list[dict[tuple[int, int], list[tuple[int, int]]]],
    rng: random.Random,
) -> tuple[int, list[int]] | None:
    capacities = [max(len(cells) for cells in buckets.values()) for buckets in buckets_all]
    max_capacity = max(capacities)
    if max_capacity < _PALETTE_MIN:
        return None

    palette_len = rng.randint(_PALETTE_MIN, min(_PALETTE_MAX, max_capacity))
    eligible_max = [i for i, cap in enumerate(capacities) if cap >= palette_len]
    if not eligible_max:
        return None

    counts = [rng.randint(1, min(palette_len, cap)) for cap in capacities]
    i_max = rng.choice(eligible_max)
    counts[i_max] = palette_len

    others = [i for i in range(len(counts)) if i != i_max]
    if others:
        counts[rng.choice(others)] = 1

    if len(set(counts)) < 2:
        adjustable = [i for i, cap in enumerate(capacities) if cap >= 2 and counts[i] != 2]
        if adjustable:
            counts[rng.choice(adjustable)] = 2

    if len(set(counts)) < 2:
        return None

    return palette_len, counts


def _paint_seeds(
    region: np.ndarray,
    buckets_all: list[dict[tuple[int, int], list[tuple[int, int]]]],
    counts: list[int],
    rng: random.Random,
) -> bool:
    for buckets, need in zip(buckets_all, counts):
        valid_classes = [cells for cells in buckets.values() if len(cells) >= need]
        if not valid_classes:
            return False
        cells = list(rng.choice(valid_classes))
        rng.shuffle(cells)
        for r, c in cells[:need]:
            region[r, c] = _SEED
    return True


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    prev = 0
    out: list[int] = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _render_palette(width: int, palette: list[int], rng: random.Random) -> np.ndarray:
    blocks = _random_composition(width, len(palette), rng)
    row = np.empty(width, dtype=int)
    c = 0
    for color, block in zip(palette, blocks):
        row[c : c + block] = int(color)
        c += block
    return row


def _expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    palette = _parse_palette(inp[0])
    region = inp[1:]
    for comp in _components(region != _WALL):
        seed_count = sum(int(region[r, c]) == _SEED for r, c in comp)
        fill = palette[seed_count - 1]
        for r, c in comp:
            if int(region[r, c]) == _BG:
                out[r + 1, c] = fill
    return out


def _is_irregular(groups: list[list[tuple[int, int]]]) -> bool:
    for cells in groups:
        if len(cells) == 1:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        if area != len(cells):
            return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    coarse_rows = rng.randint(*_COARSE_ROWS)
    coarse_cols = rng.randint(*_COARSE_COLS)
    row_sizes = _random_sizes(rng, coarse_rows)
    col_sizes = _random_sizes(rng, coarse_cols)
    uf = _random_groups(coarse_rows, coarse_cols, rng)
    groups = _coarse_groups(coarse_rows, coarse_cols, uf)
    if not _is_irregular(groups) and rng.random() < 0.65:
        return None

    region = _render_region(row_sizes, col_sizes, uf)
    if max(region.shape) >= 30:
        return None

    buckets_all = _room_buckets(region)
    if len(buckets_all) < 2:
        return None

    palette_info = _pick_seed_counts(buckets_all, rng)
    if palette_info is None:
        return None
    palette_len, seed_counts = palette_info

    palette = rng.sample(_PALETTE_POOL, palette_len)
    if len(region.shape) != 2 or region.shape[1] < palette_len:
        return None

    seeded = region.copy()
    if not _paint_seeds(seeded, buckets_all, seed_counts, rng):
        return None

    top_row = _render_palette(seeded.shape[1], palette, rng)
    inp = np.vstack([top_row[np.newaxis, :], seeded])
    if max(inp.shape) > 30:
        return None

    out = _expected_output(inp)
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid room-palette puzzle")
