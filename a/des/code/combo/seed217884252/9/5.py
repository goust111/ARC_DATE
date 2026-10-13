from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
ORANGE = 7
MAX_LAYERS = 3
LEGEND_DEPTH = 4
MARKER_COLORS = (0, 1, 2, 3, 4, 6, 8, 9)

GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
OBJECT_COUNT_SMALL = (3, 4)
OBJECT_COUNT_LARGE = (4, 5)
PLACEMENT_ATTEMPTS = 250
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _parse_legend_entry(line: np.ndarray, bg: int) -> tuple[int, int] | None:
    if line.shape != (LEGEND_DEPTH,):
        return None

    key = int(line[0])
    if key in {int(bg), GREY}:
        return None

    layers = 0
    idx = 1
    while idx < LEGEND_DEPTH and int(line[idx]) == GREY:
        layers += 1
        idx += 1
    if not (1 <= layers <= MAX_LAYERS):
        return None

    while idx < LEGEND_DEPTH:
        if int(line[idx]) != int(bg):
            return None
        idx += 1

    return key, int(layers)


def _legend_segments(parsed: list[tuple[int, int] | None]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    out: list[tuple[int, int, list[tuple[int, int]]]] = []
    i = 0
    n = len(parsed)
    while i < n:
        if parsed[i] is None:
            i += 1
            continue

        j = i
        seen_keys: set[int] = set()
        entries: list[tuple[int, int]] = []
        while j < n and parsed[j] is not None and int(parsed[j][0]) not in seen_keys:
            entries.append((int(parsed[j][0]), int(parsed[j][1])))
            seen_keys.add(int(parsed[j][0]))
            j += 1

        if entries:
            out.append((int(i), int(j), entries))
        i = max(j, i + 1)
    return out


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[tuple[int, int], str, int, int, list[tuple[int, int]]]] = []

    if w >= LEGEND_DEPTH:
        left = [_parse_legend_entry(g[r, :LEGEND_DEPTH], bg) for r in range(h)]
        for start, end, entries in _legend_segments(left):
            score = (len(entries), len({layers for _, layers in entries}))
            candidates.append((score, "left", start, end, entries))

        right = [_parse_legend_entry(g[r, w - LEGEND_DEPTH : w][::-1], bg) for r in range(h)]
        for start, end, entries in _legend_segments(right):
            score = (len(entries), len({layers for _, layers in entries}))
            candidates.append((score, "right", start, end, entries))

    if h >= LEGEND_DEPTH:
        top = [_parse_legend_entry(g[:LEGEND_DEPTH, c], bg) for c in range(w)]
        for start, end, entries in _legend_segments(top):
            score = (len(entries), len({layers for _, layers in entries}))
            candidates.append((score, "top", start, end, entries))

        bottom = [_parse_legend_entry(g[h - LEGEND_DEPTH : h, c][::-1], bg) for c in range(w)]
        for start, end, entries in _legend_segments(bottom):
            score = (len(entries), len({layers for _, layers in entries}))
            candidates.append((score, "bottom", start, end, entries))

    assert candidates, "expected an edge legend"
    score, side, start, end, entries = max(candidates, key=lambda item: item[0])
    assert score[0] >= 1

    legend_mask = np.zeros_like(g, dtype=bool)
    mapping: dict[int, int] = {}

    if side == "left":
        for offset, (color, layers) in enumerate(entries):
            r = int(start + offset)
            legend_mask[r, 0] = True
            legend_mask[r, 1 : 1 + layers] = True
            mapping[int(color)] = int(layers)
    elif side == "right":
        for offset, (color, layers) in enumerate(entries):
            r = int(start + offset)
            legend_mask[r, w - 1] = True
            legend_mask[r, w - 1 - layers : w - 1] = True
            mapping[int(color)] = int(layers)
    elif side == "top":
        for offset, (color, layers) in enumerate(entries):
            c = int(start + offset)
            legend_mask[0, c] = True
            legend_mask[1 : 1 + layers, c] = True
            mapping[int(color)] = int(layers)
    else:
        for offset, (color, layers) in enumerate(entries):
            c = int(start + offset)
            legend_mask[h - 1, c] = True
            legend_mask[h - 1 - layers : h - 1, c] = True
            mapping[int(color)] = int(layers)

    assert mapping, "legend must map at least one marker color"
    return mapping, legend_mask


def _exterior_background(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not bool(exterior[r, c]) and int(grid[r, c]) == int(bg):
            exterior[r, c] = True
            q.append((int(r), int(c)))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            push(nr, nc)

    return exterior


def _expand_cells(seed_cells: list[tuple[int, int]] | set[tuple[int, int]], exterior_bg: np.ndarray, layers: int) -> set[tuple[int, int]]:
    h, w = exterior_bg.shape
    grown = {(int(r), int(c)) for r, c in seed_cells}
    frontier = set(grown)

    for _ in range(int(layers)):
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and bool(exterior_bg[nr, nc]) and (int(nr), int(nc)) not in grown:
                    nxt.add((int(nr), int(nc)))
        if not nxt:
            break
        grown |= nxt
        frontier = nxt

    return grown


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each grey object's outer contour by the legend-selected number of layers.

    The border legend maps marker color -> layer count using one edge-adjacent key
    cell followed by 1..3 inward grey cells. Each object is a 4-connected mostly-grey
    component with exactly one embedded colored marker pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    mapping, legend_mask = _detect_legend(g, bg)
    exterior_bg = _exterior_background(g, bg)

    object_mask = (g != int(bg)) & (~legend_mask)
    comps = _components(object_mask)
    out = g.copy()

    for comp in comps:
        marker_cells = [(r, c) for r, c in comp if int(g[r, c]) != GREY]
        assert len(marker_cells) == 1, "each object must contain exactly one colored marker pixel"
        marker_r, marker_c = marker_cells[0]
        marker_color = int(g[marker_r, marker_c])
        assert marker_color in mapping, "marker color must appear in the legend"

        grown = _expand_cells(comp, exterior_bg, mapping[marker_color])
        for r, c in grown:
            out[r, c] = GREY

    return out


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols].copy()


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not bool(mask[r, c]):
                continue
            if bool(mask[r - 1, c]) and bool(mask[r + 1, c]) and bool(mask[r, c - 1]) and bool(mask[r, c + 1]):
                out.append((int(r), int(c)))
    return out


def _random_orient(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return out.copy()


def _make_solid_shape() -> np.ndarray:
    for _ in range(100):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        mask = np.ones((h, w), dtype=bool)
        variant = random.choice(("rect", "corner_bite", "edge_bite"))

        if variant == "corner_bite" and h >= 4 and w >= 4:
            rr = 0 if random.random() < 0.5 else h - 1
            cc = 0 if random.random() < 0.5 else w - 1
            mask[rr, cc] = False
        elif variant == "edge_bite" and h >= 4 and w >= 4:
            side = random.choice(("top", "bottom", "left", "right"))
            if side == "top":
                mask[0, random.randint(1, w - 2)] = False
            elif side == "bottom":
                mask[h - 1, random.randint(1, w - 2)] = False
            elif side == "left":
                mask[random.randint(1, h - 2), 0] = False
            else:
                mask[random.randint(1, h - 2), w - 1] = False

        mask = _trim_mask(_random_orient(mask))
        if _is_connected(mask) and _interior_cells(mask):
            return mask
    raise AssertionError("failed to build a solid shape")


def _make_hole_shape() -> np.ndarray:
    for _ in range(120):
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        mask = np.ones((h, w), dtype=bool)

        hole_h = random.choice((1, 1, 2))
        hole_w = random.choice((1, 1, 2))
        hole_r = random.randint(1, h - hole_h - 1)
        hole_c = random.randint(1, w - hole_w - 1)
        mask[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = False

        mask = _trim_mask(_random_orient(mask))
        if _is_connected(mask) and len(_interior_cells(mask)) >= 2:
            return mask
    raise AssertionError("failed to build a hole shape")


def _make_blob_shape() -> np.ndarray:
    for _ in range(180):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        target_area = random.randint(max(8, (h * w) // 2), h * w - 1)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randint(0, h - 1), random.randint(0, w - 1))
        mask[start] = True
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < h and 0 <= nc < w
        }

        while int(mask.sum()) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            rr, cc = cell
            mask[rr, cc] = True
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((int(nr), int(nc)))

        mask = _trim_mask(_random_orient(mask))
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if _is_connected(mask) and _interior_cells(mask):
            return mask
    raise AssertionError("failed to build a blob shape")


def _make_shape(kind: str) -> np.ndarray:
    if kind == "solid":
        return _make_solid_shape()
    if kind == "hole":
        return _make_hole_shape()
    assert kind == "blob"
    return _make_blob_shape()


def _object_patch(mask: np.ndarray, marker_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.full(mask.shape, ORANGE, dtype=int)
    patch[mask] = GREY
    marker_cell = random.choice(_interior_cells(mask))
    patch[marker_cell] = int(marker_color)
    return patch, (int(marker_cell[0]), int(marker_cell[1]))


def _legend_box(shape: tuple[int, int], side: str, start: int, count: int) -> tuple[int, int, int, int]:
    h, w = shape
    if side == "left":
        return int(start), 0, int(start + count - 1), LEGEND_DEPTH - 1
    if side == "right":
        return int(start), int(w - LEGEND_DEPTH), int(start + count - 1), int(w - 1)
    if side == "top":
        return 0, int(start), LEGEND_DEPTH - 1, int(start + count - 1)
    return int(h - LEGEND_DEPTH), int(start), int(h - 1), int(start + count - 1)


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paint_legend(grid: np.ndarray, side: str, start: int, entries: list[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    legend_mask = np.zeros((h, w), dtype=bool)

    if side == "left":
        for offset, (color, layers) in enumerate(entries):
            r = int(start + offset)
            grid[r, 0] = int(color)
            grid[r, 1 : 1 + layers] = GREY
            legend_mask[r, 0] = True
            legend_mask[r, 1 : 1 + layers] = True
    elif side == "right":
        for offset, (color, layers) in enumerate(entries):
            r = int(start + offset)
            grid[r, w - 1] = int(color)
            grid[r, w - 1 - layers : w - 1] = GREY
            legend_mask[r, w - 1] = True
            legend_mask[r, w - 1 - layers : w - 1] = True
    elif side == "top":
        for offset, (color, layers) in enumerate(entries):
            c = int(start + offset)
            grid[0, c] = int(color)
            grid[1 : 1 + layers, c] = GREY
            legend_mask[0, c] = True
            legend_mask[1 : 1 + layers, c] = True
    else:
        for offset, (color, layers) in enumerate(entries):
            c = int(start + offset)
            grid[h - 1, c] = int(color)
            grid[h - 1 - layers : h - 1, c] = GREY
            legend_mask[h - 1, c] = True
            legend_mask[h - 1 - layers : h - 1, c] = True

    return legend_mask


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != ORANGE
    region[mask] = patch[mask]
    cells = {
        (int(top + r), int(left + c))
        for r, c in np.argwhere(mask)
    }
    return cells


def _expected_output(inp: np.ndarray, objects: list[tuple[set[tuple[int, int]], int]]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    exterior = _exterior_background(inp, ORANGE)
    for cells, layers in objects:
        grown = _expand_cells(cells, exterior, layers)
        for r, c in grown:
            out[r, c] = GREY
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), ORANGE, dtype=int)

    marker_colors = random.sample(MARKER_COLORS, 3)
    layer_values = [1, 2, 3]
    random.shuffle(layer_values)
    mapping = {int(color): int(layer) for color, layer in zip(marker_colors, layer_values, strict=False)}

    entries = list(mapping.items())
    random.shuffle(entries)
    side = random.choice(("left", "right", "top", "bottom"))
    count = len(entries)

    if side in {"left", "right"}:
        if h < count + 2:
            return None
        start = random.randint(1, h - count - 1)
    else:
        if w < count + 2:
            return None
        start = random.randint(1, w - count - 1)

    legend_mask = _paint_legend(grid, side, start, entries)
    legend_box = _legend_box(grid.shape, side, start, count)
    reserved_boxes = [_expand_box(legend_box, MAX_LAYERS, h, w)]

    count_hi = OBJECT_COUNT_LARGE[1] if min(h, w) >= 23 else OBJECT_COUNT_SMALL[1]
    object_count = random.randint(3, count_hi)
    jobs = [("solid", marker_colors[0]), ("hole", marker_colors[1]), ("blob", marker_colors[2])]
    for _ in range(object_count - 3):
        jobs.append((random.choice(("solid", "hole", "blob")), random.choice(marker_colors)))
    random.shuffle(jobs)

    specs = []
    for kind, color in jobs:
        mask = _make_shape(kind)
        patch, _marker = _object_patch(mask, color)
        specs.append((int(patch.shape[0] * patch.shape[1]), patch, int(color), int(mapping[color]), kind))
    specs.sort(key=lambda item: item[0], reverse=True)

    placed_objects: list[tuple[set[tuple[int, int]], int]] = []
    seen_kinds: set[str] = set()
    seen_colors: set[int] = set()

    for _area, patch, color, layers, kind in specs:
        ph, pw = patch.shape
        if h - ph - 2 * MAX_LAYERS < 0 or w - pw - 2 * MAX_LAYERS < 0:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(MAX_LAYERS, h - ph - MAX_LAYERS)
            left = random.randint(MAX_LAYERS, w - pw - MAX_LAYERS)
            box = (int(top), int(left), int(top + ph - 1), int(left + pw - 1))
            safe_box = _expand_box(box, MAX_LAYERS, h, w)
            if any(_boxes_overlap(safe_box, other) for other in reserved_boxes):
                continue
            if np.any(grid[top : top + ph, left : left + pw] != ORANGE):
                continue

            cells = _place_patch(grid, patch, top, left)
            reserved_boxes.append(safe_box)
            placed_objects.append((cells, layers))
            seen_kinds.add(kind)
            seen_colors.add(color)
            placed = True
            break
        if not placed:
            return None

    if seen_kinds != {"solid", "hole", "blob"}:
        return None
    if seen_colors != set(marker_colors):
        return None

    expected = _expected_output(grid, placed_objects)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    mapping2, legend_mask2 = _detect_legend(grid, ORANGE)
    if mapping2 != mapping or not np.array_equal(legend_mask2, legend_mask):
        return None

    obj_mask = (grid != ORANGE) & (~legend_mask)
    comps = _components(obj_mask)
    if len(comps) != len(placed_objects):
        return None
    if any(sum(int(grid[r, c]) != GREY for r, c in comp) != 1 for comp in comps):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid ARC sample")
