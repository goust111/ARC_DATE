import random
from collections import Counter, deque

import numpy as np


GREY = 5
ALL_COLORS = tuple(range(10))
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
MAX_LEGEND_AREA = 7
MAX_LEGEND_ROWS = 4
GEN_RETRIES = 300


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _in_bbox(r: int, c: int, bbox: tuple[int, int, int, int] | None) -> bool:
    if bbox is None:
        return False
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
            }
            if set(comp) == border:
                candidates.append((r1 + c1, r1, c1, r0, c0, color))
    assert candidates, "expected a bottom-right legend frame"
    _, _, _, r0, c0, color = max(candidates)
    comp = _components(grid == color)
    for part in comp:
        if _bbox(part)[:2] == (r0, c0):
            return _bbox(part), color
    raise AssertionError("legend frame lookup failed")


def _parse_legend(
    grid: np.ndarray, legend_bbox: tuple[int, int, int, int], frame_color: int, bg: int
) -> dict[int, int]:
    r0, c0, r1, c1 = legend_bbox
    mapping: dict[int, int] = {}
    for r in range(r0 + 1, r1):
        row = grid[r, c0 + 1 : c1]
        nz = np.where((row != bg) & (row != frame_color))[0]
        if nz.size == 0:
            continue
        start = int(nz[0])
        bar_color = int(row[start])
        end = start
        while end < row.size and int(row[end]) == bar_color:
            end += 1
        assert end < row.size, "legend row must contain a label cell"
        area = end - start
        label_color = int(row[end])
        assert label_color not in (bg, frame_color), "legend label must be non-bg"
        assert area not in mapping, "legend areas must be unique"
        mapping[area] = label_color
    assert mapping, "expected at least one legend entry"
    return mapping


def _pair_cells(
    grid: np.ndarray, bg: int, legend_bbox: tuple[int, int, int, int] | None
) -> dict[int, list[tuple[int, int]]]:
    h, w = grid.shape
    pairs: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if _in_bbox(r, c, legend_bbox):
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            pairs.setdefault(color, []).append((r, c))
    for color, coords in pairs.items():
        assert len(coords) == 2, f"color {color} must appear exactly twice outside legend"
        (r0, c0), (r1, c1) = coords
        assert r0 == r1 or c0 == c1, "paired pixels must share a row or column"
    return pairs


def _draw_bridges(
    grid: np.ndarray, bg: int, legend_bbox: tuple[int, int, int, int] | None
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for color, coords in _pair_cells(grid, bg, legend_bbox).items():
        (r0, c0), (r1, c1) = coords
        if r0 == r1:
            c_lo, c_hi = sorted((c0, c1))
            out[r0, c_lo : c_hi + 1] = color
        else:
            r_lo, r_hi = sorted((r0, r1))
            out[r_lo : r_hi + 1, c0] = color
    return out


def _bg_regions(
    bridged: np.ndarray, bg: int, legend_bbox: tuple[int, int, int, int] | None
) -> list[list[tuple[int, int]]]:
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(bridged == bg):
        if legend_bbox is not None and all(_in_bbox(r, c, legend_bbox) for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    legend_bbox, frame_color = _find_legend(g, bg)
    mapping = _parse_legend(g, legend_bbox, frame_color, bg)
    bridged = _draw_bridges(g, bg, legend_bbox)
    out = np.array(bridged, copy=True)
    for region in _bg_regions(bridged, bg, legend_bbox):
        fill = mapping.get(len(region))
        if fill is None:
            continue
        for r, c in region:
            out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw bridge segments between same-color endpoint pairs, then fill each
    resulting background region by looking up its area in the bottom-right legend.
    """
    return _solve(np.asarray(grid, dtype=int))


def _side_paths(bbox: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    return [
        [(r0, c) for c in range(c0, c1)],
        [(r, c1) for r in range(r0, r1)],
        [(r1, c) for c in range(c0 + 1, c1 + 1)],
        [(r, c0) for r in range(r0 + 1, r1 + 1)],
    ]


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_single_box(
    rng: random.Random, h: int, w: int
) -> list[tuple[int, int, int, int]] | None:
    inner_h = rng.randint(1, 3)
    inner_w = rng.randint(1, 3)
    box_h = inner_h + 2
    box_w = inner_w + 2
    arena_bottom = h - 8
    arena_right = w - 11
    if arena_bottom - box_h < 1 or arena_right - box_w < 1:
        return None
    r0 = rng.randint(1, arena_bottom - box_h)
    c0 = rng.randint(1, arena_right - box_w)
    return [(r0, c0, r0 + box_h - 1, c0 + box_w - 1)]


def _sample_disjoint_boxes(
    rng: random.Random, h: int, w: int
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    areas: set[int] = set()
    arena_bottom = h - 8
    arena_right = w - 11
    for _ in range(80):
        inner_h = rng.randint(1, 3)
        inner_w = rng.randint(1, 3)
        area = inner_h * inner_w
        if area in areas:
            continue
        box_h = inner_h + 2
        box_w = inner_w + 2
        if arena_bottom - box_h < 1 or arena_right - box_w < 1:
            return None
        r0 = rng.randint(1, arena_bottom - box_h)
        c0 = rng.randint(1, arena_right - box_w)
        bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if any(_boxes_too_close(bbox, other, pad=1) for other in boxes):
            continue
        boxes.append(bbox)
        areas.add(area)
        if len(boxes) == 2:
            return boxes
    return None


def _sample_nested_boxes(
    rng: random.Random, h: int, w: int
) -> list[tuple[int, int, int, int]] | None:
    arena_bottom = h - 8
    arena_right = w - 11
    outer_h = rng.randint(7, 8)
    outer_w = rng.randint(7, 8)
    if arena_bottom - outer_h < 1 or arena_right - outer_w < 1:
        return None
    r0 = rng.randint(1, arena_bottom - outer_h)
    c0 = rng.randint(1, arena_right - outer_w)
    outer = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    inner_h = rng.choice((3, 4))
    inner_w = rng.choice((3, 4))
    min_r = outer[0] + 2
    max_r = outer[2] - inner_h - 1
    min_c = outer[1] + 2
    max_c = outer[3] - inner_w - 1
    if min_r > max_r or min_c > max_c:
        return None
    ir0 = rng.randint(min_r, max_r)
    ic0 = rng.randint(min_c, max_c)
    inner = (ir0, ic0, ir0 + inner_h - 1, ic0 + inner_w - 1)
    return [outer, inner]


def _sample_boxes(rng: random.Random, h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    mode = rng.choice(("single", "disjoint", "nested"))
    if mode == "single":
        return _sample_single_box(rng, h, w)
    if mode == "disjoint":
        return _sample_disjoint_boxes(rng, h, w)
    return _sample_nested_boxes(rng, h, w)


def _small_region_areas(grid: np.ndarray, bg: int) -> list[int]:
    bridged = _draw_bridges(grid, bg, None)
    areas = sorted(len(comp) for comp in _bg_regions(bridged, bg, None))
    if not areas:
        return []
    outside = max(areas)
    return sorted({area for area in areas if area != outside and area <= MAX_LEGEND_AREA})


def _add_legend(
    grid: np.ndarray, bg: int, rng: random.Random, areas_to_fill: list[int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    entries = list(areas_to_fill)
    unmatched_pool = [a for a in range(1, MAX_LEGEND_AREA + 1) if a not in entries]
    rng.shuffle(unmatched_pool)
    if unmatched_pool and rng.random() < 0.7:
        entries.append(unmatched_pool.pop())
    if unmatched_pool and len(entries) < MAX_LEGEND_ROWS and rng.random() < 0.35:
        entries.append(unmatched_pool.pop())
    entries = entries[:MAX_LEGEND_ROWS]
    rng.shuffle(entries)

    colors = [c for c in ALL_COLORS if c not in (bg, GREY)]
    rng.shuffle(colors)
    fill_colors = colors[: len(entries)]
    bar_color = next(c for c in colors if c not in fill_colors)

    h, w = grid.shape
    inner_h = len(entries)
    inner_w = max(entries) + 1
    total_h = inner_h + 2
    total_w = inner_w + 2
    r0 = h - total_h
    c0 = w - total_w
    legend_bbox = (r0, c0, h - 1, w - 1)

    out = np.array(grid, copy=True)
    out[r0 : h, c0 : w] = bg
    out[r0, c0:w] = GREY
    out[h - 1, c0:w] = GREY
    out[r0:h, c0] = GREY
    out[r0:h, w - 1] = GREY

    for idx, area in enumerate(entries):
        rr = r0 + 1 + idx
        out[rr, c0 + 1 : c0 + 1 + area] = bar_color
        out[rr, c0 + 1 + area] = fill_colors[idx]

    return out, legend_bbox


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg_choices = [c for c in ALL_COLORS if c != GREY]
    bg = rng.choice(bg_choices)
    boxes = _sample_boxes(rng, h, w)
    if boxes is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    line_colors = [c for c in ALL_COLORS if c not in (bg, GREY)]
    rng.shuffle(line_colors)

    needed = 4 * len(boxes)
    if len(line_colors) < needed:
        return None

    color_idx = 0
    for bbox in boxes:
        for path in _side_paths(bbox):
            color = line_colors[color_idx]
            color_idx += 1
            (r0, c0), (r1, c1) = path[0], path[-1]
            grid[r0, c0] = color
            grid[r1, c1] = color

    candidate_areas = _small_region_areas(grid, bg)
    if not candidate_areas:
        return None
    areas_to_fill = [area for area in candidate_areas if rng.random() < 0.65]
    if not areas_to_fill:
        areas_to_fill = [rng.choice(candidate_areas)]
    areas_to_fill = areas_to_fill[:MAX_LEGEND_ROWS]

    grid, legend_bbox = _add_legend(grid, bg, rng, areas_to_fill)
    if any(_boxes_too_close(bbox, legend_bbox, pad=1) for bbox in boxes):
        return None

    try:
        bridged = _draw_bridges(grid, bg, legend_bbox)
        solved = _solve(grid)
    except AssertionError:
        return None

    if np.array_equal(bridged, solved):
        return None

    pair_counts = Counter(
        int(grid[r, c])
        for r in range(h)
        for c in range(w)
        if not _in_bbox(r, c, legend_bbox) and int(grid[r, c]) != bg
    )
    if not pair_counts or any(count != 2 for count in pair_counts.values()):
        return None

    if _bg_color(grid) != bg:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
