import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows), min(cols), max(cols)


def _palette_mapping(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], tuple[int, int, int, int]]:
    mask = grid != bg
    comps = [comp for comp in _components(mask) if any(r == 0 for r, _ in comp)]
    assert comps, "expected a top-edge palette component"
    palette_comp = max(comps, key=len)
    r0, r1, c0, c1 = _bbox(palette_comp)
    assert r0 == 0
    palette = grid[r0 : r1 + 1, c0 : c1 + 1]
    ph, pw = palette.shape
    assert ph % 2 == 0 and ph >= 4

    top_row = palette[0]
    runs: list[tuple[int, int, int]] = []
    start = 0
    for idx in range(1, pw + 1):
        if idx == pw or int(top_row[idx]) != int(top_row[start]):
            runs.append((start, idx, int(top_row[start])))
            start = idx

    block_w = runs[0][1] - runs[0][0]
    assert block_w == ph, "palette blocks are square in this task family"
    assert all(end - begin == block_w for begin, end, _ in runs)

    split = ph // 2
    mapping: dict[int, np.ndarray] = {}
    for begin, end, key_color in runs:
        block = palette[:, begin:end]
        assert np.all(block[:split] == key_color)
        mapping[int(key_color)] = np.array(block[split:], dtype=int)

    assert mapping
    return mapping, (r0, r1, c0, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top palette strip to tile each closed region with the texture keyed by its marker color.

    Input structure:
    - The background color is dominant.
    - A connected horizontal palette strip touches the top edge; it is made of equal square blocks.
    - In each palette block, the upper half is a single index color and the lower half is a small two-color texture.
    - Below the palette, one boundary color draws thin closed walls around irregular regions, and each enclosed region contains exactly one marker pixel whose color matches a palette index color.

    Rule:
    - Read every palette block as an index-color to texture mapping.
    - In the area below the palette, find every non-boundary connected component that does not touch the area border; each such component is one enclosed region.
    - For each enclosed region, read its unique non-background marker color, look up the matching palette texture, and tile that texture across the whole region starting from the region's top-left cell.
    - Replace the marker and background cells inside the region with the tiled texture, while leaving the palette strip and the boundary walls unchanged.

    Key insight:
    - The interior marker is only a lookup key; the actual fill comes from the texture stored under the same color in the palette strip.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    mapping, palette_bbox = _palette_mapping(g, bg)
    area_start = palette_bbox[1] + 1
    assert area_start < g.shape[0]

    area = g[area_start:, :]
    non_bg_area = area[area != bg]
    assert non_bg_area.size > 0
    boundary = int(Counter(map(int, non_bg_area.tolist())).most_common(1)[0][0])

    out = g.copy()
    region_mask = area != boundary
    for comp in _components(region_mask):
        if any(r == 0 or c == 0 or r == area.shape[0] - 1 or c == area.shape[1] - 1 for r, c in comp):
            continue

        markers = [(r, c) for r, c in comp if int(area[r, c]) != bg]
        assert len(markers) == 1
        marker_r, marker_c = markers[0]
        key_color = int(area[marker_r, marker_c])
        assert key_color in mapping
        pattern = mapping[key_color]
        pr, pc = pattern.shape
        r0, _, c0, _ = _bbox(comp)

        for r, c in comp:
            out[area_start + r, c] = int(pattern[(r - r0) % pr, (c - c0) % pc])

    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 30
BLOCK_SIZES = (4, 6)
KEY_COUNT_RANGE = (2, 4)
REGION_COUNT_RANGE = (2, 5)
REGION_AREA_RANGE = (8, 32)
PALETTE_GAP = 1
OUTER_MARGIN = 2
BUFFER_RADIUS = 1
BLOB_RETRIES = 50
GEN_RETRIES = 400


def _random_pattern(height: int, width: int, color_a: int, color_b: int) -> np.ndarray | None:
    for _ in range(40):
        bits = np.random.randint(0, 2, size=(height, width), dtype=int)
        if bits.min() == bits.max():
            continue
        pattern = np.where(bits == 0, color_a, color_b).astype(int)
        if len(np.unique(pattern)) != 2:
            continue
        return pattern
    return None


def _build_palette(block_size: int, key_colors: list[int], patterns: dict[int, np.ndarray]) -> np.ndarray:
    split = block_size // 2
    palette = np.zeros((block_size, block_size * len(key_colors)), dtype=int)
    for idx, key_color in enumerate(key_colors):
        c0 = idx * block_size
        c1 = c0 + block_size
        palette[:split, c0:c1] = int(key_color)
        palette[split:, c0:c1] = patterns[int(key_color)]
    return palette


def _wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall: set[tuple[int, int]] = set()
    for r, c in blob:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((int(nr), int(nc)))
    return wall


def _expand8(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out = set(cells)
    frontier = set(cells)
    for _ in range(radius):
        new_frontier: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in out:
                        out.add((int(nr), int(nc)))
                        new_frontier.add((int(nr), int(nc)))
        frontier = new_frontier
    return out


def _grow_blob(canvas: np.ndarray, seed: tuple[int, int], target_size: int) -> set[tuple[int, int]] | None:
    h, w = canvas.shape
    if not canvas[seed]:
        return None

    blob = {seed}
    frontier = {pos for pos in _neighbors4(*seed) if 0 <= pos[0] < h and 0 <= pos[1] < w and canvas[pos]}
    while len(blob) < target_size and frontier:
        choices = list(frontier)
        random.shuffle(choices)
        chosen = None
        for r, c in choices:
            touch = sum((nr, nc) in blob for nr, nc in _neighbors4(r, c))
            if touch == 0:
                continue
            if touch <= 2 or random.random() < 0.3:
                chosen = (int(r), int(c))
                break
        if chosen is None:
            chosen = choices[0]
        frontier.discard(chosen)
        if chosen in blob or not canvas[chosen]:
            continue
        blob.add(chosen)
        for nr, nc in _neighbors4(*chosen):
            if 0 <= nr < h and 0 <= nc < w and canvas[nr, nc] and (nr, nc) not in blob:
                frontier.add((int(nr), int(nc)))
    return blob if len(blob) == target_size else None


def _place_one_region(
    area_h: int,
    area_w: int,
    forbidden: np.ndarray,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    canvas = ~forbidden
    free = np.argwhere(canvas)
    if len(free) < REGION_AREA_RANGE[0]:
        return None

    for _ in range(BLOB_RETRIES):
        target_size = random.randint(*REGION_AREA_RANGE)
        if len(free) < target_size:
            target_size = len(free)
        if target_size < REGION_AREA_RANGE[0]:
            return None

        sr, sc = free[random.randrange(len(free))]
        blob = _grow_blob(canvas, (int(sr), int(sc)), target_size)
        if blob is None:
            continue

        wall = _wall_cells(blob, area_h, area_w)
        if not wall:
            continue
        if any(forbidden[r, c] for r, c in wall):
            continue

        r0, r1, c0, c1 = _bbox(list(blob))
        if r1 - r0 < 1 or c1 - c0 < 1:
            continue
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(blob) / bbox_area < 0.35:
            continue

        return blob, wall
    return None


def _tile_region(
    out: np.ndarray,
    area_start: int,
    blob: set[tuple[int, int]],
    pattern: np.ndarray,
) -> None:
    pr, pc = pattern.shape
    r0, _, c0, _ = _bbox(list(blob))
    for r, c in blob:
        out[area_start + r, c] = int(pattern[(r - r0) % pr, (c - c0) % pc])


def _background_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flat))
    return counts[bg] == max(counts.values())


def _gen_attempt() -> np.ndarray | None:
    block_size = random.choice(BLOCK_SIZES)
    num_keys = random.randint(*KEY_COUNT_RANGE)
    min_w = max(GRID_MIN, block_size * num_keys + 2)
    min_h = max(GRID_MIN, block_size + PALETTE_GAP + 10)
    if min_w > GRID_MAX or min_h > GRID_MAX:
        return None

    height = random.randint(min_h, GRID_MAX)
    width = random.randint(min_w, GRID_MAX)

    bg = random.randrange(10)
    boundary = random.choice([c for c in range(10) if c != bg])
    remaining = [c for c in range(10) if c not in (bg, boundary)]
    key_colors = random.sample(remaining, num_keys)
    texture_pool = [c for c in remaining if c not in key_colors]
    if len(texture_pool) < 2:
        return None

    split = block_size // 2
    patterns: dict[int, np.ndarray] = {}
    signatures: set[tuple[int, ...]] = set()
    for key_color in key_colors:
        for _ in range(50):
            color_a, color_b = random.sample(texture_pool, 2)
            pattern = _random_pattern(split, block_size, color_a, color_b)
            if pattern is None:
                continue
            sig = tuple(map(int, pattern.flat))
            if sig in signatures:
                continue
            signatures.add(sig)
            patterns[int(key_color)] = pattern
            break
        else:
            return None

    palette = _build_palette(block_size, key_colors, patterns)
    palette_left = random.randint(0, width - palette.shape[1])
    area_start = block_size + PALETTE_GAP
    area_h = height - area_start
    area_w = width
    if area_h < 8:
        return None

    region_count = random.randint(*REGION_COUNT_RANGE)
    used_key_count = random.randint(2, min(num_keys, region_count))
    used_keys = random.sample(key_colors, used_key_count)
    region_keys = [random.choice(used_keys) for _ in range(region_count)]
    random.shuffle(region_keys)

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    inp[:block_size, palette_left : palette_left + palette.shape[1]] = palette
    out[:block_size, palette_left : palette_left + palette.shape[1]] = palette

    forbidden = np.zeros((area_h, area_w), dtype=bool)
    forbidden[:OUTER_MARGIN, :] = True
    forbidden[-OUTER_MARGIN:, :] = True
    forbidden[:, :OUTER_MARGIN] = True
    forbidden[:, -OUTER_MARGIN:] = True

    placed = 0
    for key_color in region_keys:
        region = _place_one_region(area_h, area_w, forbidden)
        if region is None:
            return None
        blob, wall = region
        reserve = _expand8(blob | wall, area_h, area_w, BUFFER_RADIUS)
        for r, c in reserve:
            forbidden[r, c] = True

        for r, c in wall:
            inp[area_start + r, c] = boundary
            out[area_start + r, c] = boundary

        marker = random.choice(list(blob))
        inp[area_start + marker[0], marker[1]] = int(key_color)
        _tile_region(out, area_start, blob, patterns[int(key_color)])
        placed += 1

    if placed < 2:
        return None
    if not _background_dominant(inp, bg):
        return None
    if Counter(map(int, inp[area_start:, :].flat))[boundary] <= placed:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
