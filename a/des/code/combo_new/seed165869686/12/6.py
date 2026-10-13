import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[np.ndarray] = []
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
            next_id += 1
    return labels, comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(cols == 0)
        or np.any(rows == h - 1)
        or np.any(cols == w - 1)
    )


def _row_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    r = 0
    while r < len(active):
        if not active[r]:
            r += 1
            continue
        start = r
        while r + 1 < len(active) and active[r + 1]:
            r += 1
        runs.append((start, r))
        r += 1
    return runs


def _find_separator(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
    c = 0
    while c < w:
        col_color = int(g[0, c])
        if col_color != bg and np.all(g[:, c] == col_color):
            start = c
            while c + 1 < w and int(g[0, c + 1]) == col_color and np.all(g[:, c + 1] == col_color):
                c += 1
            end = c
            if start > 0 and end < w - 1 and np.any(g[:, end + 1 :] != bg):
                candidates.append((start, end))
        c += 1
    assert candidates, "expected a full-height separator before the right-edge palette"
    return max(candidates, key=lambda item: item[0])


def _parse_palette(palette: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    bands = _row_runs(np.any(palette != bg, axis=1))
    mapping: dict[int, np.ndarray] = {}
    for r0, r1 in bands:
        band = palette[r0 : r1 + 1]
        comps = _components(band != bg)
        assert len(comps) >= 2, "each palette band needs a sample plus one or more dots"
        sample = max(comps, key=len)
        count = len(comps) - 1
        assert count not in mapping, "dot counts must uniquely identify palette samples"
        sr0, sc0, sr1, sc1 = _bbox(sample)
        pattern = band[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        assert pattern.size > count and np.all(pattern != bg), "sample must be the largest solid block in its band"
        mapping[count] = pattern
    assert mapping, "expected a non-empty palette"
    return mapping


def _choose_outline_color(scene: np.ndarray, bg: int) -> int:
    best: tuple[int, int, int] | None = None
    for color in [int(c) for c in np.unique(scene) if int(c) != bg]:
        labels, comps = _label_components(scene != color)
        interior_count = 0
        interior_area = 0
        for comp_id, coords in enumerate(comps, start=1):
            if _touches_border(coords, scene.shape):
                continue
            rr, cc = coords[:, 0], coords[:, 1]
            if not np.any(scene[rr, cc] == bg):
                continue
            if np.all(labels[rr, cc] == comp_id):
                interior_count += 1
                interior_area += len(coords)
        score = (interior_count, interior_area, color)
        if best is None or score > best:
            best = score
    assert best is not None and best[0] > 0, "failed to identify an outline color that encloses regions"
    return int(best[2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the tiled palette pattern indexed by its island count.

    Input structure:
    - Black is the background around several same-colored closed outline rings on the left side of the grid.
    - Inside each ring is one enclosed region; that region may contain one or more small solid islands of another color.
    - A full-height vertical separator near the right edge splits off a palette area whose stacked bands show one sample block plus a matching number of isolated marker dots.

    Rule:
    - Ignore the separator and palette while solving the left scene.
    - For each enclosed region, count how many island components lie inside it.
    - In the palette, find the band with the same number of dots, crop its sample block, and tile that sample across the whole enclosed region from the region's top-left corner, overwriting any islands.

    Key insight:
    - The islands are only counters: their component count selects a palette sample, and the chosen sample is repeated to cover the entire enclosed interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    sep0, sep1 = _find_separator(g, bg)
    scene = g[:, :sep0]
    palette = g[:, sep1 + 1 :]
    assert scene.size and palette.size, "expected content on both sides of the separator"

    patterns = _parse_palette(palette, bg)
    outline = _choose_outline_color(scene, bg)

    passable = scene != outline
    region_labels, region_comps = _label_components(passable)
    interior_ids: list[int] = []
    for comp_id, coords in enumerate(region_comps, start=1):
        if _touches_border(coords, scene.shape):
            continue
        rr, cc = coords[:, 0], coords[:, 1]
        if np.any(scene[rr, cc] == bg):
            interior_ids.append(comp_id)
    assert interior_ids, "expected at least one enclosed region"

    island_mask = (scene != bg) & (scene != outline)
    island_counts = Counter()
    for island in _components(island_mask):
        label = int(region_labels[int(island[0, 0]), int(island[0, 1])])
        assert label in interior_ids, "every island must sit inside an enclosed region"
        island_counts[label] += 1

    out = g.copy()
    for comp_id in interior_ids:
        coords = region_comps[comp_id - 1]
        count = int(island_counts[comp_id])
        assert count in patterns, f"missing palette sample for island count {count}"
        pattern = patterns[count]
        ph, pw = pattern.shape
        r0, c0, _r1, _c1 = _bbox(coords)
        for r, c in coords:
            out[int(r), int(c)] = int(pattern[(int(r) - r0) % ph, (int(c) - c0) % pw])

    return out


PATTERN_STYLES = ("checker", "vstripe", "hstripe", "diag", "bands")
GEN_ATTEMPTS = 400
ISLAND_SHAPES_SMALL = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
]
ISLAND_SHAPES_LARGE = ISLAND_SHAPES_SMALL + [
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (1, 0), (1, 1)],
]


def _pattern_bytes(arr: np.ndarray) -> bytes:
    return arr.tobytes() + str(arr.shape).encode()


def _make_pattern(color_pool: list[int], used: set[bytes]) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(2, 3)
        w = random.randint(3, 4)
        colors = random.sample(color_pool, random.randint(2, min(3, len(color_pool))))
        style = random.choice(PATTERN_STYLES)
        offset = random.randint(0, len(colors) - 1)
        arr = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                if style == "checker":
                    idx = (r + c + offset) % len(colors)
                elif style == "vstripe":
                    idx = (c + offset) % len(colors)
                elif style == "hstripe":
                    idx = (r + offset) % len(colors)
                elif style == "diag":
                    idx = (2 * r + c + offset) % len(colors)
                else:
                    idx = ((r // 1) + (c // 2) + offset) % len(colors)
                arr[r, c] = colors[idx]
        if len(np.unique(arr)) < 2:
            arr[0, -1] = colors[(offset + 1) % len(colors)]
        key = _pattern_bytes(arr)
        if key in used:
            continue
        used.add(key)
        return arr
    return None


def _cells_with_padding(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                padded.add((nr, nc))
        padded.add((r, c))
    return padded


def _shape_dimensions(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _place_islands(inner_h: int, inner_w: int, count: int) -> list[set[tuple[int, int]]] | None:
    shape_pool = ISLAND_SHAPES_SMALL if count >= 3 else ISLAND_SHAPES_LARGE
    for _ in range(80):
        blocked: set[tuple[int, int]] = set()
        placed: list[set[tuple[int, int]]] = []
        ok = True
        for _i in range(count):
            candidates: list[set[tuple[int, int]]] = []
            for shape in shape_pool:
                sh, sw = _shape_dimensions(shape)
                if inner_h - sh <= 1 or inner_w - sw <= 1:
                    continue
                for r0 in range(1, inner_h - sh):
                    for c0 in range(1, inner_w - sw):
                        cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
                        if cells & blocked:
                            continue
                        candidates.append(cells)
            if not candidates:
                ok = False
                break
            cells = random.choice(candidates)
            placed.append(cells)
            blocked |= _cells_with_padding(cells, inner_h, inner_w)
        if ok:
            return placed
    return None


def _reserve_clear(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(mask.shape[0], r0 + h + 1)
    cc1 = min(mask.shape[1], c0 + w + 1)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(mask.shape[0], r0 + h + 1)
    cc1 = min(mask.shape[1], c0 + w + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _place_ring(scene_h: int, scene_w: int, reserved: np.ndarray, count: int) -> dict | None:
    min_h = 4 if count <= 2 else 5
    min_w = 4 if count == 1 else 5
    for _ in range(120):
        inner_h = random.randint(min_h, min(7, scene_h - 4))
        inner_w = random.randint(min_w, min(8, scene_w - 4))
        islands = _place_islands(inner_h, inner_w, count)
        if islands is None:
            continue
        top_t = random.randint(1, 2)
        bottom_t = random.randint(1, 2)
        left_t = random.randint(1, 2)
        right_t = random.randint(1, 2)
        outer_h = inner_h + top_t + bottom_t
        outer_w = inner_w + left_t + right_t
        if outer_h + 2 > scene_h or outer_w + 2 > scene_w:
            continue
        r0 = random.randint(1, scene_h - outer_h - 1)
        c0 = random.randint(1, scene_w - outer_w - 1)
        if not _reserve_clear(reserved, r0, c0, outer_h, outer_w):
            continue
        _reserve(reserved, r0, c0, outer_h, outer_w)
        inner_r0 = r0 + top_t
        inner_c0 = c0 + left_t
        return {
            "count": count,
            "outer": (r0, c0, outer_h, outer_w),
            "inner": (inner_r0, inner_c0, inner_h, inner_w),
            "islands": islands,
        }
    return None


def _draw_ring(grid: np.ndarray, ring: dict, outline_color: int, island_color: int, bg: int) -> None:
    r0, c0, outer_h, outer_w = ring["outer"]
    ir0, ic0, inner_h, inner_w = ring["inner"]
    grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = outline_color
    grid[ir0 : ir0 + inner_h, ic0 : ic0 + inner_w] = bg
    for island in ring["islands"]:
        for rr, cc in island:
            grid[ir0 + rr, ic0 + cc] = island_color


def _tile_region(out: np.ndarray, inner: tuple[int, int, int, int], pattern: np.ndarray) -> None:
    ir0, ic0, inner_h, inner_w = inner
    ph, pw = pattern.shape
    for r in range(inner_h):
        for c in range(inner_w):
            out[ir0 + r, ic0 + c] = int(pattern[r % ph, c % pw])


def _build_palette(patterns: dict[int, np.ndarray], dot_color: int, h: int, w: int) -> np.ndarray | None:
    palette = np.zeros((h, w), dtype=int)
    counts = sorted(patterns)
    heights = [patterns[count].shape[0] for count in counts]
    gaps = [random.randint(1, 2) for _ in range(max(0, len(counts) - 1))]
    required_h = sum(heights) + sum(gaps)
    if required_h > h:
        return None
    row = random.randint(0, h - required_h)
    for idx, count in enumerate(counts):
        pattern = patterns[count]
        ph, pw = pattern.shape
        dot_w = 2 * count - 1
        entry_w = dot_w + 1 + pw
        if entry_w > w:
            return None
        col = random.randint(0, w - entry_w)
        dot_row = row + random.randint(0, ph - 1)
        for i in range(count):
            palette[dot_row, col + 2 * i] = dot_color
        palette[row : row + ph, col + dot_w + 1 : col + dot_w + 1 + pw] = pattern
        row += ph
        if idx < len(gaps):
            row += gaps[idx]
    return palette


def _generate_patterns(counts: list[int], color_pool: list[int]) -> dict[int, np.ndarray] | None:
    used: set[bytes] = set()
    patterns: dict[int, np.ndarray] = {}
    for count in counts:
        pattern = _make_pattern(color_pool, used)
        if pattern is None or pattern.size <= count:
            return None
        patterns[count] = pattern
    return patterns


def _expected_output(inp: np.ndarray, rings: list[dict], patterns: dict[int, np.ndarray]) -> np.ndarray:
    out = inp.copy()
    for ring in rings:
        _tile_region(out, ring["inner"], patterns[ring["count"]])
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    num_regions = random.randint(2, 4)
    counts = list(range(1, num_regions + 1))
    random.shuffle(counts)

    palette_colors = list(range(1, 10))
    random.shuffle(palette_colors)
    outline_color, island_color, separator_color, dot_color = palette_colors[:4]
    pattern_pool = [c for c in range(1, 10) if c not in {outline_color, island_color, separator_color, dot_color}]
    patterns = _generate_patterns(list(range(1, num_regions + 1)), pattern_pool)
    if patterns is None:
        return None

    sep_w = random.randint(1, 2)
    palette_w = max((2 * n - 1) + 1 + patterns[n].shape[1] for n in patterns) + random.randint(0, 2)
    max_scene_w = 30 - sep_w - palette_w
    if max_scene_w < 14:
        return None

    palette_h_needed = sum(patterns[n].shape[0] for n in sorted(patterns)) + (num_regions - 1)
    grid_h = random.randint(max(18, palette_h_needed), 28)
    scene_w = random.randint(14, max_scene_w)
    grid_w = scene_w + sep_w + palette_w

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    reserved = np.zeros((grid_h, scene_w), dtype=bool)

    rings: list[dict] = []
    for count in counts:
        ring = _place_ring(grid_h, scene_w, reserved, count)
        if ring is None:
            return None
        rings.append(ring)
        _draw_ring(grid[:, :scene_w], ring, outline_color, island_color, bg)

    palette = _build_palette(patterns, dot_color, grid_h, palette_w)
    if palette is None:
        return None

    grid[:, scene_w : scene_w + sep_w] = separator_color
    grid[:, scene_w + sep_w :] = palette

    expected = _expected_output(grid, rings, patterns)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
