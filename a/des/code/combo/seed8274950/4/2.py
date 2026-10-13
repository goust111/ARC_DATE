import random
from itertools import combinations

import numpy as np


GENERATION_RETRIES = 300
HOUSE_MARGIN = 1
HOUSE_LEGEND_GAP = 1
LEGEND_WIDTH = 9
ROOM_SIZE_RANGE = (3, 6)
ROOM_AXIS_CHOICES = (2, 3)
LEGEND_ENTRY_CHOICES = (3, 4, 5)
MARKER_COLOR_COUNT = 4
PATTERN_COLOR_COUNT = 3

KEY_POSITIONS_SINGLE = ((1, 1),)
KEY_POSITIONS_PAIR = ((1, 0), (1, 2))

PATTERN_BASES = (
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=int),
    np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=int),
    np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=int),
    np.array([[0, 0, 1], [0, 1, 2], [1, 2, 2]], dtype=int),
    np.array([[0, 1, 1], [2, 0, 1], [2, 2, 0]], dtype=int),
    np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [2, 1, 2], [0, 1, 0]], dtype=int),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _flood_mask(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    stack = [start]
    seen[start] = True
    comp: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return comp


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                out.append(_flood_mask(mask, (r, c), seen))
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _inner_spans(size: int, separator_lines: list[int]) -> list[tuple[int, int]]:
    lines = sorted(separator_lines)
    assert lines and lines[0] == 0 and lines[-1] == size - 1
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _legend_data(grid: np.ndarray, bg: int):
    h, w = grid.shape
    corner_points = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))

    for cr, cc in corner_points:
        if int(grid[cr, cc]) == bg:
            continue

        mask = grid != bg
        seen = np.zeros_like(mask, dtype=bool)
        comp = _flood_mask(mask, (cr, cc), seen)
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        sep = int(grid[cr, cc])

        full_rows = [i for i in range(sub.shape[0]) if np.all(sub[i, :] == sep)]
        full_cols = [j for j in range(sub.shape[1]) if np.all(sub[:, j] == sep)]
        if not full_rows or not full_cols:
            continue

        row_spans = _inner_spans(sub.shape[0], full_rows)
        col_spans = _inner_spans(sub.shape[1], full_cols)
        if len(col_spans) != 2 or not row_spans:
            continue
        if any(r1i - r0i + 1 != 3 for r0i, r1i in row_spans):
            continue
        if any(c1i - c0i + 1 != 3 for c0i, c1i in col_spans):
            continue

        mapping: dict[frozenset[int], np.ndarray] = {}
        key_col_span, pattern_col_span = col_spans
        for rs0, rs1 in row_spans:
            key_patch = sub[rs0 : rs1 + 1, key_col_span[0] : key_col_span[1] + 1]
            pattern_patch = sub[rs0 : rs1 + 1, pattern_col_span[0] : pattern_col_span[1] + 1]
            key_colors = frozenset(
                int(v) for v in np.unique(key_patch) if int(v) not in {bg, sep}
            )
            assert 1 <= len(key_colors) <= 2
            assert key_colors not in mapping
            mapping[key_colors] = pattern_patch.copy()

        legend_bbox = (r0, c0, r1, c1)
        return mapping, legend_bbox

    raise AssertionError("legend not found")


def _wall_color(grid: np.ndarray, bg: int, legend_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = legend_bbox
    mask = np.ones_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    values = [int(v) for v in grid[mask].reshape(-1) if int(v) != bg]
    assert values
    unique, counts = np.unique(np.array(values, dtype=int), return_counts=True)
    return int(unique[np.argmax(counts)])


def _room_components(grid: np.ndarray, bg: int, wall: int, legend_bbox: tuple[int, int, int, int]):
    h, w = grid.shape
    mask = grid != wall
    r0, c0, r1, c1 = legend_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    rooms = []
    for comp in _components(mask):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        rr0, cc0, rr1, cc1 = _bbox(comp)
        area = (rr1 - rr0 + 1) * (cc1 - cc0 + 1)
        assert len(comp) == area
        room_colors = frozenset(
            int(grid[r, c]) for r, c in comp if int(grid[r, c]) != bg
        )
        assert 1 <= len(room_colors) <= 2
        rooms.append((comp, room_colors, (rr0, cc0, rr1, cc1)))
    assert rooms
    return rooms


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    mapping, legend_bbox = _legend_data(g, bg)
    wall = _wall_color(g, bg, legend_bbox)
    rooms = _room_components(g, bg, wall, legend_bbox)

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg

    for comp, key, (r0, c0, r1, c1) in rooms:
        assert key in mapping
        pattern = mapping[key]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if g[r, c] == bg:
                    out[r, c] = int(pattern[(r - r0) % 3, (c - c0) % 3])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room by matching its marker-color combination to a legend pattern.
    """
    return _solve(grid)


def _all_keys(marker_colors: list[int]) -> tuple[list[frozenset[int]], list[frozenset[int]]]:
    singles = [frozenset((color,)) for color in marker_colors]
    pairs = [frozenset(pair) for pair in combinations(marker_colors, 2)]
    return singles, pairs


def _choose_legend_keys(rng: random.Random, marker_colors: list[int], count: int) -> list[frozenset[int]]:
    singles, pairs = _all_keys(marker_colors)
    single_count = rng.randint(1, count - 1)
    pair_count = count - single_count
    keys = rng.sample(singles, single_count) + rng.sample(pairs, pair_count)
    rng.shuffle(keys)
    return keys


def _colorize_pattern(base: np.ndarray, rng: random.Random, pattern_colors: list[int]) -> np.ndarray:
    symbols = sorted(int(v) for v in np.unique(base))
    palette = rng.sample(pattern_colors, len(symbols))
    remap = {symbol: color for symbol, color in zip(symbols, palette)}
    out = np.zeros_like(base)
    for symbol, color in remap.items():
        out[base == symbol] = color
    k = rng.randrange(4)
    out = np.rot90(out, k)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out


def _choose_patterns(
    rng: random.Random, legend_keys: list[frozenset[int]], pattern_colors: list[int]
) -> dict[frozenset[int], np.ndarray]:
    bases = rng.sample(list(PATTERN_BASES), len(legend_keys))
    mapping: dict[frozenset[int], np.ndarray] = {}
    used: set[bytes] = set()
    for key, base in zip(legend_keys, bases):
        for _ in range(50):
            pattern = _colorize_pattern(base, rng, pattern_colors)
            blob = pattern.tobytes()
            if blob not in used:
                used.add(blob)
                mapping[key] = pattern
                break
        else:
            raise RuntimeError("failed to create distinct patterns")
    return mapping


def _legend_patch(
    bg: int,
    sep: int,
    legend_keys: list[frozenset[int]],
    patterns: dict[frozenset[int], np.ndarray],
) -> np.ndarray:
    height = 1 + 4 * len(legend_keys)
    patch = np.full((height, LEGEND_WIDTH), bg, dtype=int)
    patch[::4, :] = sep
    patch[:, 0] = sep
    patch[:, 4] = sep
    patch[:, 8] = sep

    for i, key in enumerate(legend_keys):
        rs = 1 + 4 * i
        key_patch = np.full((3, 3), bg, dtype=int)
        key_list = list(key)
        if len(key_list) == 1:
            kr, kc = KEY_POSITIONS_SINGLE[0]
            key_patch[kr, kc] = key_list[0]
        else:
            for (kr, kc), color in zip(KEY_POSITIONS_PAIR, key_list):
                key_patch[kr, kc] = color
        patch[rs : rs + 3, 1:4] = key_patch
        patch[rs : rs + 3, 5:8] = patterns[key]
    return patch


def _non_overlapping(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_legend_corner(shape: tuple[int, int], patch: np.ndarray, corner: str) -> tuple[int, int]:
    h, w = shape
    ph, pw = patch.shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - pw
    if corner == "bl":
        return h - ph, 0
    if corner == "br":
        return h - ph, w - pw
    raise ValueError(corner)


def _sample_room_sizes(rng: random.Random, count: int) -> list[int]:
    return [rng.randint(ROOM_SIZE_RANGE[0], ROOM_SIZE_RANGE[1]) for _ in range(count)]


def _room_slices(origin: int, sizes: list[int]) -> list[tuple[int, int]]:
    cur = origin + 1
    spans = []
    for size in sizes:
        spans.append((cur, cur + size - 1))
        cur += size + 1
    return spans


def _paint_house(
    grid: np.ndarray,
    wall: int,
    house_r0: int,
    house_c0: int,
    row_sizes: list[int],
    col_sizes: list[int],
):
    house_h = sum(row_sizes) + len(row_sizes) + 1
    house_w = sum(col_sizes) + len(col_sizes) + 1
    grid[house_r0 : house_r0 + house_h, house_c0] = wall
    grid[house_r0 : house_r0 + house_h, house_c0 + house_w - 1] = wall
    grid[house_r0, house_c0 : house_c0 + house_w] = wall
    grid[house_r0 + house_h - 1, house_c0 : house_c0 + house_w] = wall

    cur = house_r0
    for size in row_sizes[:-1]:
        cur += size + 1
        grid[cur, house_c0 : house_c0 + house_w] = wall

    cur = house_c0
    for size in col_sizes[:-1]:
        cur += size + 1
        grid[house_r0 : house_r0 + house_h, cur] = wall


def _assign_room_keys(
    rng: random.Random, room_count: int, legend_keys: list[frozenset[int]]
) -> list[frozenset[int]]:
    singles = [key for key in legend_keys if len(key) == 1]
    pairs = [key for key in legend_keys if len(key) == 2]
    assert singles and pairs

    picks = [rng.choice(singles), rng.choice(pairs)]
    while len(picks) < room_count:
        picks.append(rng.choice(legend_keys))
    rng.shuffle(picks)
    return picks


def _paint_room_markers(
    grid: np.ndarray,
    rng: random.Random,
    room_span: tuple[int, int, int, int],
    key: frozenset[int],
):
    r0, c0, r1, c1 = room_span
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    chosen = rng.sample(cells, len(key))
    for (r, c), color in zip(chosen, list(key)):
        grid[r, c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors[0]
    wall = colors[1]
    sep = colors[2]
    marker_colors = colors[3 : 3 + MARKER_COLOR_COUNT]
    pattern_colors = colors[3 + MARKER_COLOR_COUNT : 3 + MARKER_COLOR_COUNT + PATTERN_COLOR_COUNT]

    legend_count = rng.choice(LEGEND_ENTRY_CHOICES)
    legend_keys = _choose_legend_keys(rng, marker_colors, legend_count)
    patterns = _choose_patterns(rng, legend_keys, pattern_colors)
    legend = _legend_patch(bg, sep, legend_keys, patterns)
    legend_h, legend_w = legend.shape

    room_rows = rng.choice(ROOM_AXIS_CHOICES)
    room_cols = rng.choice(ROOM_AXIS_CHOICES)
    row_sizes = _sample_room_sizes(rng, room_rows)
    col_sizes = _sample_room_sizes(rng, room_cols)
    house_h = sum(row_sizes) + room_rows + 1
    house_w = sum(col_sizes) + room_cols + 1

    min_h = max(legend_h + 2, house_h + 2)
    min_w = max(legend_w + 2, house_w + 2)
    max_h = min(30, min_h + 7)
    max_w = min(30, min_w + 9)
    if min_h + 1 > max_h or min_w + 1 > max_w:
        return None
    grid_h = rng.randint(min_h + 1, max_h)
    grid_w = rng.randint(min_w + 1, max_w)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    legend_r0, legend_c0 = _place_legend_corner((grid_h, grid_w), legend, corner)
    legend_bbox = (legend_r0, legend_c0, legend_r0 + legend_h - 1, legend_c0 + legend_w - 1)

    house_bbox = None
    for _ in range(200):
        house_r0 = rng.randint(HOUSE_MARGIN, grid_h - house_h - HOUSE_MARGIN)
        house_c0 = rng.randint(HOUSE_MARGIN, grid_w - house_w - HOUSE_MARGIN)
        cand = (house_r0, house_c0, house_r0 + house_h - 1, house_c0 + house_w - 1)
        if _non_overlapping(cand, legend_bbox, HOUSE_LEGEND_GAP):
            house_bbox = cand
            break
    if house_bbox is None:
        return None

    grid[legend_r0 : legend_r0 + legend_h, legend_c0 : legend_c0 + legend_w] = legend
    _paint_house(grid, wall, house_bbox[0], house_bbox[1], row_sizes, col_sizes)

    row_spans = _room_slices(house_bbox[0], row_sizes)
    col_spans = _room_slices(house_bbox[1], col_sizes)
    room_keys = _assign_room_keys(rng, len(row_spans) * len(col_spans), legend_keys)

    idx = 0
    for rs0, rs1 in row_spans:
        for cs0, cs1 in col_spans:
            _paint_room_markers(grid, rng, (rs0, cs0, rs1, cs1), room_keys[idx])
            idx += 1

    try:
        _solve(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
