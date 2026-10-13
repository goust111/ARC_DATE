import random
from collections import deque

import numpy as np


MAX_GEN_RETRIES = 300
GRID_SIZE_RANGE = (22, 31)
LEGEND_ENTRY_COUNTS = (2, 3, 4)
PATTERN_SIZES = (2, 3)
OBJECT_COUNT_RANGE = (2, 5)
COLOR_POOL = tuple(range(1, 10))
CORNERS = ("tl", "tr", "bl", "br")
ORIENTATIONS = ("horizontal", "vertical")
SHAPE_KINDS = ("rect", "cut_corner")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


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
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _flood_from_boundary(barrier: np.ndarray) -> np.ndarray:
    h, w = barrier.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not barrier[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _corner_start(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    starts = [(r, c) for r, c in corners if int(grid[r, c]) != 0]
    assert len(starts) == 1, "exactly one corner should contain the legend"
    return starts[0]


def _connected_component_from(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    q = deque([start])
    mask[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != 0 and not mask[nr, nc]:
                mask[nr, nc] = True
                q.append((nr, nc))
    return mask


def _extract_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    start = _corner_start(grid)
    mask = _connected_component_from(grid, start)
    pts = [(r, c) for r, c in zip(*np.where(mask))]
    r0, c0, r1, c1 = _bbox(pts)
    legend = grid[r0 : r1 + 1, c0 : c1 + 1]
    lh, lw = legend.shape
    side = min(lh, lw)
    assert side in (3, 4), f"legend entry side must be 3 or 4, got {side}"
    if lh == side:
        assert lw % side == 0, "horizontal legend width must be a multiple of entry size"
        blocks = [legend[:, i * side : (i + 1) * side] for i in range(lw // side)]
    else:
        assert lw == side and lh % side == 0, "vertical legend height must be a multiple of entry size"
        blocks = [legend[i * side : (i + 1) * side, :] for i in range(lh // side)]

    mapping: dict[int, np.ndarray] = {}
    for block in blocks:
        key = int(block[0, 0])
        pattern = np.asarray(block[1:, 1:], dtype=int)
        assert key != 0
        assert key not in mapping, "legend keys must be unique"
        mapping[key] = pattern
    return mapping, mask


def _enclosed_regions_for_component(
    grid: np.ndarray,
    color: int,
    comp: list[tuple[int, int]],
) -> list[tuple[int, list[tuple[int, int]]]]:
    r0, c0, r1, c1 = _bbox(comp)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    barrier = sub == color
    outside = _flood_from_boundary(barrier)
    enclosed = (~barrier) & (~outside)
    regions = []
    for pts in _components(enclosed):
        vals = [int(sub[r, c]) for r, c in pts]
        seed_colors = {v for v in vals if v not in (0, color)}
        if len(seed_colors) != 1:
            continue
        seed_color = seed_colors.pop()
        seed_count = sum(1 for v in vals if v == seed_color)
        if seed_count != 1:
            continue
        regions.append((seed_color, [(r0 + r, c0 + c) for r, c in pts]))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed region by tiling the legend pattern selected by its seed color.

    Input structure:
    - Background is black.
    - Several single-color closed outlines enclose black interior regions; each interior contains exactly one non-background seed pixel whose color is not the outline color.
    - One corner contains a compact legend strip made of small square entries; each entry's top-left pixel is a seed color key and the lower-right sub-square is the fill pattern for that key.

    Rule:
    - Find the corner legend and read it as a mapping from seed color to a 2x2 or 3x3 pattern.
    - For every single-color outline, locate each enclosed interior region and read its unique seed pixel color.
    - Replace the whole interior region, including the seed pixel, by tiling the matched pattern from the interior region's top-left cell, clipping the tile at the region boundary.
    - Keep the outlines and the legend unchanged.

    Key insight:
    - The seed pixel is only a lookup key: it points to one legend pattern, and that pattern must be tiled over the entire enclosed region rather than preserving the seed itself.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _extract_legend(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        mask = (g == color) & (~legend_mask)
        for comp in _components(mask):
            for seed_color, region in _enclosed_regions_for_component(g, color, comp):
                assert seed_color in mapping, f"seed color {seed_color} missing from legend"
                pattern = mapping[seed_color]
                pr, pc = pattern.shape
                top = min(r for r, _ in region)
                left = min(c for _, c in region)
                for r, c in region:
                    out[r, c] = int(pattern[(r - top) % pr, (c - left) % pc])
    return out


def _make_pattern(size: int, palette: list[int], rng: random.Random, used: set[bytes]) -> np.ndarray | None:
    if len(palette) < 2:
        return None
    for _ in range(50):
        color_count = 2 if len(palette) == 2 else rng.choice([2, 3])
        colors = rng.sample(palette, color_count)
        flat = colors[:]
        flat.extend(rng.choices(colors, k=size * size - color_count))
        rng.shuffle(flat)
        arr = np.array(flat, dtype=int).reshape(size, size)
        if len(set(arr.ravel().tolist())) < 2:
            continue
        if size >= 3 and len({tuple(row.tolist()) for row in arr}) == 1 and len({tuple(col.tolist()) for col in arr.T}) == 1:
            continue
        sig = arr.tobytes()
        if sig in used:
            continue
        used.add(sig)
        return arr
    return None


def _legend_entry(key: int, pattern: np.ndarray) -> np.ndarray:
    size = pattern.shape[0] + 1
    block = np.empty((size, size), dtype=int)
    block[1:, 1:] = pattern
    block[0, 1:] = pattern[0, :]
    block[1:, 0] = pattern[:, 0]
    block[0, 0] = key
    return block


def _make_interior_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 7)
    w = rng.randint(3, 7)
    mask = np.ones((h, w), dtype=bool)
    kind = rng.choice(SHAPE_KINDS)
    if kind == "cut_corner" and h >= 5 and w >= 5:
        cut_h = rng.randint(1, h - 2)
        cut_w = rng.randint(1, w - 2)
        if h - cut_h >= 2 and w - cut_w >= 2:
            corner = rng.choice(CORNERS)
            if corner == "tl":
                mask[:cut_h, :cut_w] = False
            elif corner == "tr":
                mask[:cut_h, w - cut_w :] = False
            elif corner == "bl":
                mask[h - cut_h :, :cut_w] = False
            else:
                mask[h - cut_h :, w - cut_w :] = False
            if not mask.any():
                mask[:, :] = True
    return mask


def _frame_patch(interior: np.ndarray, frame_color: int, seed_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = interior.shape
    inside = np.zeros((h + 2, w + 2), dtype=bool)
    inside[1:-1, 1:-1] = interior
    border = np.zeros_like(inside)
    for r, c in zip(*np.where(inside)):
        for nr, nc in _neighbors8(int(r), int(c)):
            if 0 <= nr < inside.shape[0] and 0 <= nc < inside.shape[1] and not inside[nr, nc]:
                border[nr, nc] = True
    patch = np.zeros_like(border, dtype=int)
    patch[border] = frame_color
    interior_cells = list(zip(*np.where(inside)))
    sr, sc = rng.choice(interior_cells)
    patch[sr, sc] = seed_color
    return patch, inside


def _valid_frame_patch(patch: np.ndarray, inside: np.ndarray, frame_color: int, seed_color: int) -> bool:
    comps = _components(patch == frame_color)
    if len(comps) != 1:
        return False
    regions = _enclosed_regions_for_component(patch, frame_color, comps[0])
    if len(regions) != 1:
        return False
    region_seed, region_pts = regions[0]
    if region_seed != seed_color:
        return False
    expected = {(int(r), int(c)) for r, c in zip(*np.where(inside))}
    found = set(region_pts)
    return expected == found


def _reserve(occ: np.ndarray, r0: int, c0: int, h: int, w: int):
    occ[max(0, r0 - 1) : min(occ.shape[0], r0 + h + 1), max(0, c0 - 1) : min(occ.shape[1], c0 + w + 1)] = True


def _place_patch(
    grid: np.ndarray,
    occ: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
    allow_edge_touch: bool = False,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    r_start = 0 if allow_edge_touch else 1
    c_start = 0 if allow_edge_touch else 1
    r_end = h - ph if allow_edge_touch else h - ph - 1
    c_end = w - pw if allow_edge_touch else w - pw - 1
    if r_end < r_start or c_end < c_start:
        return False
    positions = [(r, c) for r in range(r_start, r_end + 1) for c in range(c_start, c_end + 1)]
    rng.shuffle(positions)
    for r0, c0 in positions:
        rs = slice(max(0, r0 - 1), min(h, r0 + ph + 1))
        cs = slice(max(0, c0 - 1), min(w, c0 + pw + 1))
        if occ[rs, cs].any():
            continue
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any((patch != 0) & (sub != 0)):
            continue
        sub[patch != 0] = patch[patch != 0]
        _reserve(occ, r0, c0, ph, pw)
        return True
    return False


def _legend_patch(entry_count: int, size: int, key_colors: list[int], patterns: list[np.ndarray], orientation: str) -> np.ndarray:
    blocks = [_legend_entry(key, pattern) for key, pattern in zip(key_colors, patterns)]
    if orientation == "horizontal":
        return np.hstack(blocks)
    return np.vstack(blocks)


def _place_corner_legend(grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, corner: str):
    h, w = grid.shape
    ph, pw = patch.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        r0, c0 = h - ph, w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    _reserve(occ, r0, c0, ph, pw)


def _choose_seed_colors(key_colors: list[int], object_count: int, rng: random.Random) -> list[int]:
    seeds = []
    base = key_colors[:]
    rng.shuffle(base)
    while len(seeds) < object_count and base:
        seeds.append(base.pop())
    while len(seeds) < object_count:
        seeds.append(rng.choice(key_colors))
    rng.shuffle(seeds)
    return seeds


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    pattern_size = rng.choice(PATTERN_SIZES)
    entry_side = pattern_size + 1
    entry_count = rng.choice(LEGEND_ENTRY_COUNTS)
    orientation = rng.choice(ORIENTATIONS)
    legend_h = entry_side if orientation == "horizontal" else entry_side * entry_count
    legend_w = entry_side * entry_count if orientation == "horizontal" else entry_side
    if legend_h > h or legend_w > w:
        return None

    max_objects = min(OBJECT_COUNT_RANGE[1] - 1, 9 - entry_count)
    if max_objects < OBJECT_COUNT_RANGE[0]:
        return None
    object_count = rng.randint(OBJECT_COUNT_RANGE[0], max_objects)

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    key_colors = colors[:entry_count]
    frame_colors = colors[entry_count : entry_count + object_count]
    pattern_palette = [c for c in COLOR_POOL if c not in key_colors]
    used_patterns: set[bytes] = set()
    patterns: list[np.ndarray] = []
    for _ in range(entry_count):
        pattern = _make_pattern(pattern_size, pattern_palette, rng, used_patterns)
        if pattern is None:
            return None
        patterns.append(pattern)

    legend = _legend_patch(entry_count, entry_side, key_colors, patterns, orientation)
    corner = rng.choice(CORNERS)
    _place_corner_legend(grid, occ, legend, corner)

    seed_colors = _choose_seed_colors(key_colors, object_count, rng)
    patches: list[np.ndarray] = []
    for frame_color, seed_color in zip(frame_colors, seed_colors):
        ok = False
        for _ in range(40):
            interior = _make_interior_shape(rng)
            patch, inside = _frame_patch(interior, frame_color, seed_color, rng)
            if _valid_frame_patch(patch, inside, frame_color, seed_color):
                patches.append(patch)
                ok = True
                break
        if not ok:
            return None

    patches.sort(key=lambda p: p.shape[0] * p.shape[1], reverse=True)
    for patch in patches:
        if not _place_patch(grid, occ, patch, rng):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
