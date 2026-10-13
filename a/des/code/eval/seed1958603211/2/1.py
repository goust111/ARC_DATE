from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_bbox(coords: list[Coord]) -> BBox | None:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    if set(coords) != border:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _extract_top_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    row0 = grid[0]
    best: tuple[int, int, int, int] | None = None

    c = 0
    while c < w:
        if int(row0[c]) == bg:
            c += 1
            continue
        start = c
        while c < w and int(row0[c]) != bg:
            c += 1
        end = c
        seq = [int(v) for v in row0[start:end]]
        if len(seq) < 2 or len(set(seq)) != len(seq):
            continue

        height = 0
        while height < h:
            if not np.array_equal(grid[height, start:end], row0[start:end]):
                break
            if start > 0 and np.any(grid[height, :start] != bg):
                break
            if end < w and np.any(grid[height, end:] != bg):
                break
            height += 1

        score = (end - start, height, -start)
        if best is None or score > best[:3]:
            best = score + (start, end)

    assert best is not None, "expected a top legend made of distinct vertical stripes"
    _width, height, _neg_start, start, end = best
    legend = [int(v) for v in grid[0, start:end]]
    mask = np.zeros_like(grid, dtype=bool)
    mask[:height, start:end] = True
    return legend, mask


def _find_ring_family(
    grid: np.ndarray, bg: int, legend_colors: set[int], legend_mask: np.ndarray
) -> tuple[int, list[tuple[BBox, list[Coord]]]]:
    candidates: list[tuple[int, list[tuple[BBox, list[Coord]]]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg} - legend_colors):
        mask = (grid == color) & (~legend_mask)
        if not np.any(mask):
            continue
        rings: list[tuple[BBox, list[Coord]]] = []
        ok = True
        for comp in _components(mask):
            bbox = _ring_bbox(comp)
            if bbox is None:
                ok = False
                break
            rings.append((bbox, comp))
        if ok and rings:
            candidates.append((int(color), rings))

    assert len(candidates) == 1, "expected exactly one non-legend ring color family"
    return candidates[0]


def _is_outer_of(outer: BBox, inner: BBox) -> bool:
    inset_top = inner[0] - outer[0]
    inset_left = inner[1] - outer[1]
    inset_bottom = outer[2] - inner[2]
    inset_right = outer[3] - inner[3]
    return (
        inset_top == inset_left == inset_bottom == inset_right
        and inset_top >= 2
        and inset_top % 2 == 0
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested rectangular ring from outer to inner using the top legend order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    legend, legend_mask = _extract_top_legend(g, bg)
    _, rings = _find_ring_family(g, bg, set(legend), legend_mask)

    bboxes = [bbox for bbox, _coords in rings]
    out = g.copy()
    for bbox, coords in rings:
        depth = sum(_is_outer_of(other, bbox) for other in bboxes if other != bbox)
        color = legend[depth % len(legend)]
        for r, c in coords:
            out[r, c] = color
    return out


# --------------- generator ---------------

GRID_HEIGHT_BOUNDS = (18, 30)
GRID_WIDTH_BOUNDS = (18, 30)
LEGEND_HEIGHT_BOUNDS = (1, 3)
LEGEND_LEN_BOUNDS = (2, 6)
TOP_GAP_CHOICES = (1, 2)
FRAMEWORK_COUNT_BOUNDS = (1, 4)
EXTRA_SIDE_BOUNDS = (0, 6)
PLACEMENT_PADDING = 1
PLACEMENT_DIM_ATTEMPTS = 40
GENERATION_RETRIES = 400


def _max_layers_for_side(side: int) -> int:
    return max(0, (side + 1) // 4)


def _draw_framework(
    grid: np.ndarray, top: int, left: int, height: int, width: int, layers: int, color: int
) -> None:
    for depth in range(layers):
        r0 = top + 2 * depth
        c0 = left + 2 * depth
        r1 = top + height - 1 - 2 * depth
        c1 = left + width - 1 - 2 * depth
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color


def _fits(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(h, top + height + PLACEMENT_PADDING)
    c1 = min(w, left + width + PLACEMENT_PADDING)
    return not np.any(occupied[r0:r1, c0:c1])


def _occupy(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(h, top + height + PLACEMENT_PADDING)
    c1 = min(w, left + width + PLACEMENT_PADDING)
    occupied[r0:r1, c0:c1] = True


def _choose_mode(max_layers: int) -> str:
    modes = ["short"]
    if max_layers >= 3:
        modes.extend(["long", "mixed"])
    return random.choice(modes)


def _choose_legend_len(width: int, max_layers: int, mode: str) -> int | None:
    upper = min(LEGEND_LEN_BOUNDS[1], width)
    if mode in {"long", "mixed"}:
        upper = min(upper, max_layers - 1)
    lower = LEGEND_LEN_BOUNDS[0]
    if upper < lower:
        return None
    return random.randint(lower, upper)


def _choose_layer_counts(legend_len: int, max_layers: int, mode: str) -> list[int] | None:
    short_hi = min(max_layers, max(1, legend_len - 1))
    long_lo = legend_len + 1

    if mode == "short":
        count = random.randint(*FRAMEWORK_COUNT_BOUNDS)
        return [random.randint(1, short_hi) for _ in range(count)]

    if long_lo > max_layers:
        return None

    if mode == "long":
        count = random.randint(FRAMEWORK_COUNT_BOUNDS[0], min(3, FRAMEWORK_COUNT_BOUNDS[1]))
        return [random.randint(long_lo, max_layers) for _ in range(count)]

    assert mode == "mixed"
    count = random.randint(2, FRAMEWORK_COUNT_BOUNDS[1])
    layers = [random.randint(1, short_hi), random.randint(long_lo, max_layers)]
    while len(layers) < count:
        branch = random.choice(("short", "long", "equal"))
        if branch == "short":
            layers.append(random.randint(1, short_hi))
        elif branch == "long":
            layers.append(random.randint(long_lo, max_layers))
        else:
            layers.append(legend_len)
    random.shuffle(layers)
    return layers


def _place_frameworks(
    grid_h: int, grid_w: int, row_start: int, layer_counts: list[int]
) -> list[tuple[int, int, int, int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int, int, int, int]] = []
    available_h = grid_h - row_start

    for layers in sorted(layer_counts, reverse=True):
        min_side = 4 * layers - 1
        if min_side > available_h or min_side > grid_w:
            return None

        placed = False
        for _ in range(PLACEMENT_DIM_ATTEMPTS):
            max_h = min(available_h, min_side + EXTRA_SIDE_BOUNDS[1])
            max_w = min(grid_w, min_side + EXTRA_SIDE_BOUNDS[1])
            height = random.randint(min_side + EXTRA_SIDE_BOUNDS[0], max_h)
            width = random.randint(min_side + EXTRA_SIDE_BOUNDS[0], max_w)

            valid_positions: list[tuple[int, int]] = []
            for top in range(row_start, grid_h - height + 1):
                for left in range(0, grid_w - width + 1):
                    if _fits(occupied, top, left, height, width):
                        valid_positions.append((top, left))
            if not valid_positions:
                continue

            top, left = random.choice(valid_positions)
            _occupy(occupied, top, left, height, width)
            placements.append((top, left, height, width, layers))
            placed = True
            break

        if not placed:
            return None

    random.shuffle(placements)
    return placements


def _attempt_generate() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_BOUNDS)
    grid_w = random.randint(*GRID_WIDTH_BOUNDS)
    legend_h = random.randint(*LEGEND_HEIGHT_BOUNDS)
    top_gap = random.choice(TOP_GAP_CHOICES)
    row_start = legend_h + top_gap
    if row_start >= grid_h:
        return None

    max_layers = min(_max_layers_for_side(grid_h - row_start), _max_layers_for_side(grid_w))
    if max_layers < 1:
        return None

    mode = _choose_mode(max_layers)
    legend_len = _choose_legend_len(grid_w, max_layers, mode)
    if legend_len is None:
        return None

    layer_counts = _choose_layer_counts(legend_len, max_layers, mode)
    if not layer_counts:
        return None

    placements = _place_frameworks(grid_h, grid_w, row_start, layer_counts)
    if placements is None:
        return None

    palette = random.sample(list(range(10)), legend_len + 2)
    bg = palette[0]
    legend_colors = palette[1 : 1 + legend_len]
    ring_color = palette[-1]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    legend_start = random.randint(0, grid_w - legend_len)
    for offset, color in enumerate(legend_colors):
        grid[:legend_h, legend_start + offset] = color

    for top, left, height, width, layers in placements:
        _draw_framework(grid, top, left, height, width, layers, ring_color)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring recoloring task")
