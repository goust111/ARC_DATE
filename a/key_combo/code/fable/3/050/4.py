from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors(r: int, c: int, connectivity: int = 4):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1
    if connectivity == 8:
        yield r - 1, c - 1
        yield r - 1, c + 1
        yield r + 1, c - 1
        yield r + 1, c + 1


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in _neighbors(r, c, connectivity):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _get_bg(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    dr = 0
    if a[2] < b[0]:
        dr = b[0] - a[2]
    elif b[2] < a[0]:
        dr = a[0] - b[2]
    dc = 0
    if a[3] < b[1]:
        dc = b[1] - a[3]
    elif b[3] < a[1]:
        dc = a[1] - b[3]
    return int(dr + dc)


def _bbox_touches_corner(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _enclosed_regions(wall_coords: np.ndarray, box: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = map(int, box)
    h, w = r1 - r0 + 1, c1 - c0 + 1

    wall = np.zeros((h, w), dtype=bool)
    local = wall_coords - np.array([r0, c0])
    wall[local[:, 0], local[:, 1]] = True
    passable = ~wall

    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for rr in range(h):
        for cc in (0, w - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors(rr, cc, 4):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & ~outside
    return [coords + np.array([r0, c0]) for coords in _components_mask(enclosed, connectivity=4)]


def _parse_legend(g: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, int], list[int]]:
    best = None
    for coords in _components_mask(g == BLACK, connectivity=8):
        box = _bbox_coords(coords)
        if not _bbox_touches_corner(box, g.shape):
            continue
        r0, c0, r1, c1 = box
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any(sub != BLACK):
            continue

        key_cols = [
            int(cc)
            for cc in range(sub.shape[1])
            if np.any((sub[:, cc] != BLACK) & (sub[:, cc] != GREY))
        ]
        if not key_cols:
            continue
        key_col = min(key_cols)

        rows: list[tuple[int, int, int]] = []
        for rr in range(sub.shape[0]):
            color = int(sub[rr, key_col])
            if color in (BLACK, GREY):
                continue
            dots = int(np.count_nonzero(sub[rr, key_col + 1 :] == GREY))
            if dots <= 0:
                continue
            rows.append((rr, color, dots))
        counts = [dots for _rr, _color, dots in rows]
        colors = [color for _rr, color, _dots in rows]
        if len(rows) < 2 or len(set(counts)) != len(counts) or len(set(colors)) != len(colors):
            continue

        score = (len(rows), coords.shape[0], -(r1 - r0 + c1 - c0))
        if best is None or score > best[0]:
            best = (score, box, {dots: color for _rr, color, dots in rows}, colors)

    assert best is not None, "expected a corner legend box"
    _score, box, count_to_color, key_colors = best
    return box, count_to_color, key_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each addressed enclosed outline interior with the color of the outline that points to it.

    Input structure:
    - A black legend box sits in one corner. Each key row has one key-color pixel
      in the left key column and grey tally dots every other cell to its right;
      the row dot counts are distinct.
    - Outside the legend, there is one closed outline for every key color, and
      grey dots appear both inside outlines and as outside distractors.

    Rule:
    - Count the grey dots inside each outline. Look up that count in the legend
      and read the key color on the matching row as an address.
    - The addressed outline's interior is filled with the source outline's own
      color. Every outline is addressed once, so these writes form a permutation.
    - Preserve the legend and outline borders. Remove all non-legend grey dots;
      interior grey dots are covered by the fills.

    Key insight:
    - The legend color selected by the dot count is not paint. It is the address
      of the target outline; the paint is the color of the source outline that
      made the lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    legend_box, count_to_color, key_colors = _parse_legend(g)
    lr0, lc0, lr1, lc1 = legend_box
    outside_legend = np.ones(g.shape, dtype=bool)
    outside_legend[lr0 : lr1 + 1, lc0 : lc1 + 1] = False

    records: dict[int, dict[str, np.ndarray | tuple[int, int, int, int]]] = {}
    for color in key_colors:
        comps = _components_mask((g == int(color)) & outside_legend, connectivity=4)
        comps = [coords for coords in comps if coords.shape[0] > 1]
        assert len(comps) == 1, "expected one closed outline per key color"
        coords = comps[0]
        box = _bbox_coords(coords)
        regions = _enclosed_regions(coords, box)
        regions = [region for region in regions if region.shape[0] > 0]
        assert len(regions) == 1, "expected each outline to enclose one region"
        records[int(color)] = {"outline": coords, "box": box, "region": regions[0]}

    source_to_target: dict[int, int] = {}
    for source_color in key_colors:
        region = records[int(source_color)]["region"]
        rr = region[:, 0]
        cc = region[:, 1]
        dot_count = int(np.count_nonzero(g[rr, cc] == GREY))
        assert dot_count in count_to_color, "outline dot count must be keyed by the legend"
        source_to_target[int(source_color)] = int(count_to_color[dot_count])

    assert set(source_to_target.values()) == set(map(int, key_colors)), "pointers must form a permutation"

    out = g.copy()
    out[(g == GREY) & outside_legend] = int(bg)

    for source_color, target_color in source_to_target.items():
        target_region = records[int(target_color)]["region"]
        rr = target_region[:, 0]
        cc = target_region[:, 1]
        out[rr, cc] = int(source_color)

    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    return out


# --------------- generator ---------------

GRID_SIZE_RANGE = (24, 30)
OBJECT_COUNT_CHOICES = (3, 3, 4)
COUNT_POOL = (1, 2, 3, 4, 5)
RECT_SIZE_RANGE = (5, 7)
BBOX_GAP = 3
STRAY_DOT_RANGE = (3, 8)
GEN_ATTEMPTS = 500


def _rect_outline_patch(color: int, h: int, w: int, bg: int) -> np.ndarray:
    patch = np.full((h, w), int(bg), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _build_legend(key_colors: list[int], counts_by_color: dict[int, int]) -> np.ndarray:
    max_count = max(counts_by_color.values())
    height = 2 * len(key_colors) + 1
    width = 2 * max_count + 3 + 2 * random.randint(0, 1)
    legend = np.full((height, width), BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        row = 1 + 2 * i
        legend[row, 1] = int(color)
        for j in range(counts_by_color[int(color)]):
            legend[row, 3 + 2 * j] = GREY
    return legend


def _legend_origin(shape: tuple[int, int], legend: np.ndarray) -> tuple[int, int]:
    h, w = shape
    lh, lw = legend.shape
    return random.choice(((0, 0), (0, w - lw), (h - lh, 0), (h - lh, w - lw)))


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _place_outline(
    grid: np.ndarray,
    color: int,
    placed_boxes: list[tuple[int, int, int, int]],
    bg: int,
) -> dict | None:
    for _ in range(80):
        h = random.randint(*RECT_SIZE_RANGE)
        w = random.randint(*RECT_SIZE_RANGE)
        patch = _rect_outline_patch(color, h, w, bg)
        candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for r in range(grid.shape[0] - h + 1):
            for c in range(grid.shape[1] - w + 1):
                box = (r, c, r + h - 1, c + w - 1)
                if any(_bbox_distance(box, old) < BBOX_GAP for old in placed_boxes):
                    continue
                view = grid[r : r + h, c : c + w]
                if np.any((patch != int(bg)) & (view != int(bg))):
                    continue
                candidates.append((r, c, box))
        if not candidates:
            continue
        r, c, box = random.choice(candidates)
        _paste(grid, patch, r, c, transparent=bg)
        local = np.argwhere(patch == int(color))
        coords = local + np.array([r, c])
        region = np.argwhere(np.ones((h - 2, w - 2), dtype=bool)) + np.array([r + 1, c + 1])
        placed_boxes.append(box)
        return {"color": int(color), "box": box, "outline": coords, "region": region}
    return None


def _sample_permutation(colors: list[int]) -> dict[int, int]:
    want_fixed = random.random() < 0.45
    for _ in range(100):
        perm = colors[:]
        random.shuffle(perm)
        if all(a == b for a, b in zip(colors, perm)):
            continue
        if want_fixed and not any(a == b for a, b in zip(colors, perm)):
            continue
        return {int(src): int(dst) for src, dst in zip(colors, perm)}
    perm = colors[1:] + colors[:1]
    return {int(src): int(dst) for src, dst in zip(colors, perm)}


def _choose_spaced_cells(cells: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    coords = [tuple(map(int, p)) for p in cells]
    random.shuffle(coords)
    chosen: list[tuple[int, int]] = []
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        if (r, c) in blocked:
            continue
        chosen.append((r, c))
        blocked.add((r, c))
        for nr, nc in _neighbors(r, c, 4):
            blocked.add((nr, nc))
        if len(chosen) == int(count):
            return chosen
    return None


def _place_internal_dots(
    grid: np.ndarray,
    records: dict[int, dict],
    source_to_target: dict[int, int],
    counts_by_color: dict[int, int],
) -> set[tuple[int, int]]:
    used: set[tuple[int, int]] = set()
    for source_color, target_color in source_to_target.items():
        count = int(counts_by_color[int(target_color)])
        cells = np.asarray(records[int(source_color)]["region"], dtype=int)
        chosen = _choose_spaced_cells(cells, count)
        if chosen is None:
            raise ValueError("not enough interior cells for dots")
        for r, c in chosen:
            grid[r, c] = GREY
            used.add((r, c))
    return used


def _place_stray_dots(
    grid: np.ndarray,
    legend_box: tuple[int, int, int, int],
    records: dict[int, dict],
    bg: int,
    reserved: set[tuple[int, int]],
) -> bool:
    lr0, lc0, lr1, lc1 = legend_box
    interior = {tuple(map(int, p)) for rec in records.values() for p in rec["region"]}
    candidates = []
    for r, c in np.argwhere(grid == int(bg)):
        r = int(r)
        c = int(c)
        if lr0 <= r <= lr1 and lc0 <= c <= lc1:
            continue
        if (r, c) in interior:
            continue
        if (r, c) in reserved:
            continue
        if any((nr, nc) in reserved for nr, nc in _neighbors(r, c, 4)):
            continue
        candidates.append((r, c))
    random.shuffle(candidates)

    target = random.randint(*STRAY_DOT_RANGE)
    placed = 0
    for r, c in candidates:
        if (r, c) in reserved or any((nr, nc) in reserved for nr, nc in _neighbors(r, c, 4)):
            continue
        grid[r, c] = GREY
        reserved.add((r, c))
        for nr, nc in _neighbors(r, c, 4):
            reserved.add((nr, nc))
        placed += 1
        if placed == target:
            return True
    return placed >= STRAY_DOT_RANGE[0]


def _expected_output(
    inp: np.ndarray,
    bg: int,
    legend_box: tuple[int, int, int, int],
    records: dict[int, dict],
    source_to_target: dict[int, int],
) -> np.ndarray:
    out = inp.copy()
    lr0, lc0, lr1, lc1 = legend_box
    outside_legend = np.ones(inp.shape, dtype=bool)
    outside_legend[lr0 : lr1 + 1, lc0 : lc1 + 1] = False
    out[(inp == GREY) & outside_legend] = int(bg)
    for source_color, target_color in source_to_target.items():
        region = np.asarray(records[int(target_color)]["region"], dtype=int)
        out[region[:, 0], region[:, 1]] = int(source_color)
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = inp[lr0 : lr1 + 1, lc0 : lc1 + 1]
    return out


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    legend_box: tuple[int, int, int, int],
    records: dict[int, dict],
    source_to_target: dict[int, int],
    counts_by_color: dict[int, int],
) -> bool:
    colors = set(records)
    if set(source_to_target) != colors or set(source_to_target.values()) != colors:
        return False
    if all(src == dst for src, dst in source_to_target.items()):
        return False
    if len(set(counts_by_color.values())) != len(counts_by_color):
        return False

    lr0, lc0, lr1, lc1 = legend_box
    if not _bbox_touches_corner(legend_box, inp.shape):
        return False
    if not np.any(inp[lr0 : lr1 + 1, lc0 : lc1 + 1] == GREY):
        return False

    nonself_sources = 0
    for source_color, target_color in source_to_target.items():
        region = np.asarray(records[int(source_color)]["region"], dtype=int)
        actual_count = int(np.count_nonzero(inp[region[:, 0], region[:, 1]] == GREY))
        if actual_count != int(counts_by_color[int(target_color)]):
            return False
        target_region = np.asarray(records[int(target_color)]["region"], dtype=int)
        if not np.all(out[target_region[:, 0], target_region[:, 1]] == int(source_color)):
            return False
        nonself_sources += int(source_color != target_color)
    if nonself_sources == 0:
        return False

    outside_legend = np.ones(inp.shape, dtype=bool)
    outside_legend[lr0 : lr1 + 1, lc0 : lc1 + 1] = False
    return bool(np.any((inp == GREY) & outside_legend))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    obj_count = random.choice(OBJECT_COUNT_CHOICES)

    bg = random.choice(COLOR_POOL)
    available = [c for c in COLOR_POOL if c != bg]
    key_colors = random.sample(available, obj_count)
    counts = random.sample(COUNT_POOL, obj_count)
    counts_by_color = {int(color): int(count) for color, count in zip(key_colors, counts)}
    source_to_target = _sample_permutation(key_colors)

    legend = _build_legend(key_colors, counts_by_color)
    if legend.shape[0] > h or legend.shape[1] > w:
        return None
    lr, lc = _legend_origin((h, w), legend)
    legend_box = (lr, lc, lr + legend.shape[0] - 1, lc + legend.shape[1] - 1)

    grid = np.full((h, w), int(bg), dtype=int)
    _paste(grid, legend, lr, lc)

    placed_boxes = [legend_box]
    records: dict[int, dict] = {}
    for color in key_colors:
        rec = _place_outline(grid, int(color), placed_boxes, int(bg))
        if rec is None:
            return None
        records[int(color)] = rec

    try:
        reserved = _place_internal_dots(grid, records, source_to_target, counts_by_color)
    except ValueError:
        return None

    legend_greys = {
        tuple(map(int, p))
        for p in np.argwhere(grid == GREY)
        if legend_box[0] <= int(p[0]) <= legend_box[2] and legend_box[1] <= int(p[1]) <= legend_box[3]
    }
    reserved |= legend_greys
    if not _place_stray_dots(grid, legend_box, records, int(bg), reserved):
        return None

    expected = _expected_output(grid, int(bg), legend_box, records, source_to_target)
    if not _semantic_checks(grid, expected, legend_box, records, source_to_target, counts_by_color):
        return None

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
    raise RuntimeError("failed to generate a valid pointer legend input")
