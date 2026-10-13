import random

import numpy as np


MAROON = 9
BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)

MAX_CANVAS = 30
OUTER_MARGIN = 2
PLACEMENT_GAP = 1
PAIR_COUNT_CHOICES = (2, 3, 4)
SHAPE_HEIGHT_RANGE = (3, 5)
SHAPE_WIDTH_RANGE = (3, 6)
SHAPE_AREA_RANGE = (5, 10)
RECT_COUNT_CHOICES = (1, 2)
PADDING_MAX = 4
GEN_RETRIES = 400


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords) -> tuple[int, int, int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def normalize_coords(coords) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = bbox_coords(coords)
    norm = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))
    return norm, r1 - r0 + 1, c1 - c0 + 1


def components_mask(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: set[tuple[int, int]] = set()
        while stack:
            cr, cc = stack.pop()
            comp.add((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def is_filled_rectangle(coords) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def translate_coords(coords, dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _find_occluded_copy(
    template_rel: tuple[tuple[int, int], ...],
    visible_residual: set[tuple[int, int]],
    occlusion_mask: np.ndarray,
) -> set[tuple[int, int]]:
    h, w = occlusion_mask.shape
    offsets: set[tuple[int, int]] = set()

    # Match the full template against the visible same-color fragments; any missing template cells
    # must sit under maroon pixels.
    for vr, vc in visible_residual:
        for tr, tc in template_rel:
            dr = vr - tr
            dc = vc - tc
            translated = translate_coords(template_rel, dr, dc)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in translated):
                continue
            if not visible_residual.issubset(translated):
                continue
            if not any(occlusion_mask[r, c] for r, c in translated):
                continue
            if any((r, c) not in visible_residual and not occlusion_mask[r, c] for r, c in translated):
                continue
            offsets.add((dr, dc))

    assert len(offsets) == 1, f"expected one template alignment, found {len(offsets)}"
    dr, dc = next(iter(offsets))
    return translate_coords(template_rel, dr, dc)


def main(grid: np.ndarray) -> np.ndarray:
    """Restore each partially occluded same-color copy from its intact twin and erase the maroon blockers.

    Input structure:
    - The grid has one dominant background color plus several foreground colors.
    - For each foreground color there are exactly two identical copies of one connected sprite shape.
    - One copy of each color is fully visible, while the other is partially covered by one or more maroon filled rectangles.
    - Maroon pixels are an occlusion layer, so a covered copy may appear as several smaller visible fragments.

    Rule:
    - Treat every maroon connected component as a rectangular blocker that should be removed in the output.
    - For each non-maroon foreground color, take the unique largest visible component as the intact template copy.
    - Align that template to the other same-color fragments so that all visible pixels match and every missing template pixel falls under maroon.
    - Paint the recovered copy, then replace all maroon blocker pixels with the background color.

    Key insight:
    - The maroon rectangles are not objects to preserve; they hide pixels of one member of each same-color pair, and the uncovered twin supplies the exact shape needed to reconstruct the hidden copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    occlusion_mask = g == MAROON
    occluders = components_mask(occlusion_mask)
    assert occluders, "expected at least one maroon occluder"
    for rect in occluders:
        assert is_filled_rectangle(rect), "maroon occluders must be filled rectangles"

    out = g.copy()
    out[occlusion_mask] = bg

    colors = [int(v) for v in np.unique(g) if int(v) not in {bg, MAROON}]
    assert colors, "expected at least one foreground color"

    for color in sorted(colors):
        comps = components_mask(g == color)
        assert len(comps) >= 2, f"color {color} must have an intact copy and visible occluded fragments"

        sizes = [len(comp) for comp in comps]
        max_size = max(sizes)
        template_candidates = [comp for comp, size in zip(comps, sizes) if size == max_size]
        assert len(template_candidates) == 1, f"color {color} must have one unique intact template"
        template_abs = template_candidates[0]
        template_rel, _shape_h, _shape_w = normalize_coords(template_abs)

        all_visible = {(int(r), int(c)) for r, c in np.argwhere(g == color)}
        visible_residual = all_visible - template_abs
        assert visible_residual, f"color {color} is missing its occluded copy fragments"

        restored = _find_occluded_copy(template_rel, visible_residual, occlusion_mask)
        for r, c in restored:
            out[r, c] = color

    return out


def _random_shape(rng: random.Random) -> tuple[tuple[tuple[int, int], ...], int, int]:
    for _ in range(200):
        h = rng.randint(*SHAPE_HEIGHT_RANGE)
        w = rng.randint(*SHAPE_WIDTH_RANGE)
        max_area = min(SHAPE_AREA_RANGE[1], h * w - 1)
        min_area = min(SHAPE_AREA_RANGE[0], max_area)
        if min_area < 4 or max_area < min_area:
            continue
        target = rng.randint(min_area, max_area)

        seed = (rng.randrange(h), rng.randrange(w))
        cells = {seed}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != target:
            continue

        norm, nh, nw = normalize_coords(cells)
        rows = {r for r, _ in norm}
        cols = {c for _, c in norm}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len(norm) == nh * nw:
            continue
        return norm, nh, nw
    raise RuntimeError("failed to sample a non-degenerate connected shape")


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _rectangles_touch(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    cells_a = _rect_cells(rect_a)
    cells_b = _rect_cells(rect_b)
    if cells_a & cells_b:
        return True
    return any(abs(ra - rb) + abs(ca - cb) == 1 for ra, ca in cells_a for rb, cb in cells_b)


def _pick_occluders(
    shape: tuple[tuple[int, int], ...],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int, int, int]], set[tuple[int, int]]] | None:
    shape_set = set(shape)
    area = len(shape)
    min_hidden = max(2, int(np.ceil(area * 0.25)))
    max_hidden = min(area - 2, int(np.floor(area * 0.7)))
    if max_hidden < min_hidden:
        return None

    candidates: list[tuple[tuple[int, int, int, int], set[tuple[int, int]]]] = []
    for r0 in range(h):
        for c0 in range(w):
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    rect = (r0, c0, r1, c1)
                    hidden = {(r, c) for r, c in shape_set if r0 <= r <= r1 and c0 <= c <= c1}
                    if min_hidden <= len(hidden) <= max_hidden:
                        candidates.append((rect, hidden))
    if not candidates:
        return None

    rect_count = rng.choice(RECT_COUNT_CHOICES)
    order = list(candidates)
    rng.shuffle(order)

    if rect_count == 1:
        for rect, hidden in order:
            visible = shape_set - hidden
            if len(visible) >= 2:
                return [rect], hidden
        return None

    for i, (rect_a, hidden_a) in enumerate(order):
        for rect_b, hidden_b in order[i + 1 :]:
            if _rectangles_touch(rect_a, rect_b):
                continue
            hidden = hidden_a | hidden_b
            if not (min_hidden <= len(hidden) <= max_hidden):
                continue
            visible = shape_set - hidden
            if len(visible) < 2:
                continue
            if len(hidden_a - hidden_b) == 0 or len(hidden_b - hidden_a) == 0:
                continue
            return [rect_a, rect_b], hidden
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    pr0 = max(0, r0 - PLACEMENT_GAP)
    pc0 = max(0, c0 - PLACEMENT_GAP)
    pr1 = min(occupied.shape[0] - 1, r1 + PLACEMENT_GAP)
    pc1 = min(occupied.shape[1] - 1, c1 + PLACEMENT_GAP)
    return not occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any()


def _mark_placed(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    pr0 = max(0, r0 - PLACEMENT_GAP)
    pc0 = max(0, c0 - PLACEMENT_GAP)
    pr1 = min(occupied.shape[0] - 1, r1 + PLACEMENT_GAP)
    pc1 = min(occupied.shape[1] - 1, c1 + PLACEMENT_GAP)
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _place_box(occupied: np.ndarray, h: int, w: int, rng: random.Random) -> tuple[int, int] | None:
    row_lo = OUTER_MARGIN
    row_hi = occupied.shape[0] - OUTER_MARGIN - h
    col_lo = OUTER_MARGIN
    col_hi = occupied.shape[1] - OUTER_MARGIN - w
    if row_hi < row_lo or col_hi < col_lo:
        return None

    for _ in range(200):
        r0 = rng.randint(row_lo, row_hi)
        c0 = rng.randint(col_lo, col_hi)
        if _can_place(occupied, r0, c0, h, w):
            _mark_placed(occupied, r0, c0, h, w)
            return r0, c0
    return None


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], origin: tuple[int, int], color: int) -> None:
    r0, c0 = origin
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = color


def _random_padding(limit: int, rng: random.Random) -> int:
    if limit <= 0:
        return 0
    low = 1 if limit >= 1 else 0
    return rng.randint(low, min(PADDING_MAX, limit))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    colors = rng.sample(PALETTE, pair_count)
    occupied = np.zeros((MAX_CANVAS, MAX_CANVAS), dtype=bool)
    specs = []

    for color in colors:
        shape, h, w = _random_shape(rng)
        picked = _pick_occluders(shape, h, w, rng)
        if picked is None:
            return None
        rects, _hidden = picked

        full_origin = _place_box(occupied, h, w, rng)
        occluded_origin = _place_box(occupied, h, w, rng)
        if full_origin is None or occluded_origin is None:
            return None

        specs.append(
            {
                "color": color,
                "shape": shape,
                "bbox": (h, w),
                "rects": rects,
                "full_origin": full_origin,
                "occluded_origin": occluded_origin,
            }
        )

    clean = np.full((MAX_CANVAS, MAX_CANVAS), BACKGROUND, dtype=int)
    inp = clean.copy()

    for spec in specs:
        color = spec["color"]
        shape = spec["shape"]
        _paint_shape(clean, shape, spec["full_origin"], color)
        _paint_shape(clean, shape, spec["occluded_origin"], color)
        _paint_shape(inp, shape, spec["full_origin"], color)
        _paint_shape(inp, shape, spec["occluded_origin"], color)

        r_base, c_base = spec["occluded_origin"]
        for r0, c0, r1, c1 in spec["rects"]:
            inp[r_base + r0 : r_base + r1 + 1, c_base + c0 : c_base + c1 + 1] = MAROON

    if get_bg(inp) != BACKGROUND or not np.any(inp == MAROON):
        return None

    pts = np.argwhere(inp != BACKGROUND)
    r0, c0, r1, c1 = bbox_coords(pts)
    r0 -= _random_padding(r0, rng)
    c0 -= _random_padding(c0, rng)
    r1 += _random_padding(MAX_CANVAS - 1 - r1, rng)
    c1 += _random_padding(MAX_CANVAS - 1 - c1, rng)

    inp = inp[r0 : r1 + 1, c0 : c1 + 1]
    clean = clean[r0 : r1 + 1, c0 : c1 + 1]

    if max(inp.shape) > 30 or max(clean.shape) > 30:
        return None
    if get_bg(inp) != BACKGROUND:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, clean):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-repair puzzle")
