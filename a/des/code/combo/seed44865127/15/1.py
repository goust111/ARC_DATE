from __future__ import annotations

import numpy as np


NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Generator configuration.
GRID_HEIGHT_RANGE = (12, 28)
GRID_WIDTH_RANGE = (12, 28)
SPRITE_COUNT_RANGE = (2, 5)
BBOX_HEIGHT_RANGE = (4, 7)
BBOX_WIDTH_RANGE = (4, 7)
MASK_DENSITY_RANGE = (0.55, 0.8)
MIN_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
VARIANT_ATTEMPTS = 200
HOLE_SET_ATTEMPTS = 120
LAYOUT_EXTRA_SLACK = 5


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS_8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            parts.append(np.asarray(coords, dtype=int))

    return parts


def _is_connected(mask: np.ndarray) -> bool:
    return np.any(mask) and len(_components(mask)) == 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    region = grid[top : top + h, left : left + w]
    region[mask] = color
    grid[top : top + h, left : left + w] = region


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [total]
    cuts = np.sort(rng.integers(0, total + 1, size=parts - 1))
    values = np.diff(np.concatenate(([0], cuts, [total])))
    return [int(v) for v in values]


def _random_path(
    start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator
) -> list[tuple[int, int]]:
    r, c = start
    cells = [(r, c)]
    while (r, c) != end:
        dr = 0 if end[0] == r else (1 if end[0] > r else -1)
        dc = 0 if end[1] == c else (1 if end[1] > c else -1)
        options: list[tuple[int, int]] = []
        if dr != 0 and dc != 0:
            options.append((dr, dc))
        if dr != 0:
            options.append((dr, 0))
        if dc != 0:
            options.append((0, dc))
        step = options[int(rng.integers(len(options)))]
        r += step[0]
        c += step[1]
        cells.append((r, c))
    return cells


def _grow_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    total = height * width
    min_area = max(int(np.ceil(MASK_DENSITY_RANGE[0] * total)), height + width)
    max_area = max(min_area, int(np.floor(MASK_DENSITY_RANGE[1] * total)))

    for _ in range(SHAPE_ATTEMPTS):
        target_area = int(rng.integers(min_area, max_area + 1))
        anchors = [
            (0, int(rng.integers(width))),
            (height - 1, int(rng.integers(width))),
            (int(rng.integers(height)), 0),
            (int(rng.integers(height)), width - 1),
        ]
        order = [int(i) for i in rng.permutation(len(anchors))]
        current = anchors[order[0]]
        cells = {current}

        for idx in order[1:]:
            target = anchors[idx]
            for cell in _random_path(current, target, rng):
                cells.add(cell)
            current = target

        frontier = set()
        for r, c in list(cells):
            for dr, dc in NEIGHBORS_8:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        while len(cells) < target_area and frontier:
            frontier_list = list(frontier)
            interior = [(r, c) for r, c in frontier_list if 0 < r < height - 1 and 0 < c < width - 1]
            pool = interior if interior and rng.random() < 0.7 else frontier_list
            chosen = pool[int(rng.integers(len(pool)))]
            frontier.discard(chosen)
            cells.add(chosen)
            r, c = chosen
            for dr, dc in NEIGHBORS_8:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) < target_area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not _is_connected(mask):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        interior_count = int(np.count_nonzero(mask[1:-1, 1:-1]))
        if interior_count < 4:
            continue

        row_sums = tuple(map(int, mask.sum(axis=1)))
        col_sums = tuple(map(int, mask.sum(axis=0)))
        if len(set(row_sums)) == 1 and len(set(col_sums)) == 1 and rng.random() < 0.8:
            continue

        return mask

    return None


def _sample_variants(
    base_mask: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[frozenset[tuple[int, int]]]] | None:
    holes = [(int(r), int(c)) for r, c in np.argwhere(base_mask) if 0 < r < base_mask.shape[0] - 1 and 0 < c < base_mask.shape[1] - 1]
    if len(holes) < 4:
        return None

    for _ in range(VARIANT_ATTEMPTS):
        masks: list[np.ndarray] = []
        hole_sets: list[frozenset[tuple[int, int]]] = []

        hole_counts = rng.integers(1, 4, size=count)
        for hole_count in hole_counts:
            found = False
            for _ in range(HOLE_SET_ATTEMPTS):
                idxs = rng.choice(len(holes), size=int(hole_count), replace=False)
                chosen = frozenset(holes[int(i)] for i in np.atleast_1d(idxs))
                if chosen in hole_sets:
                    continue

                variant = base_mask.copy()
                for r, c in chosen:
                    variant[r, c] = False
                if _is_connected(variant):
                    masks.append(variant)
                    hole_sets.append(chosen)
                    found = True
                    break

            if not found:
                break

        if len(masks) != count:
            continue
        if len(set(hole_sets)) != count:
            continue

        common_missing = set(hole_sets[0])
        for hole_set in hole_sets[1:]:
            common_missing &= set(hole_set)
        if common_missing:
            continue

        consensus = np.zeros_like(base_mask, dtype=bool)
        for mask in masks:
            consensus |= mask
        if not np.array_equal(consensus, base_mask):
            continue

        return masks, hole_sets

    return None


def _slot_positions(
    slots: int, cell_size: int, total_size: int, rng: np.random.Generator
) -> list[int] | None:
    minimum = slots * cell_size + (slots + 1) * MIN_GAP
    extra = total_size - minimum
    if extra < 0:
        return None

    slack = _random_composition(extra, slots + 1, rng)
    positions: list[int] = []
    cursor = MIN_GAP + slack[0]
    for idx in range(slots):
        positions.append(cursor)
        cursor += cell_size + MIN_GAP + slack[idx + 1]
    return positions


def _sample_layout(
    sprite_count: int, bbox_h: int, bbox_w: int, rng: np.random.Generator
) -> tuple[int, int, list[tuple[int, int]]] | None:
    options = []
    for row_slots in range(1, sprite_count + 1):
        col_slots = int(np.ceil(sprite_count / row_slots))
        min_h = row_slots * bbox_h + (row_slots + 1) * MIN_GAP
        min_w = col_slots * bbox_w + (col_slots + 1) * MIN_GAP
        if min_h <= GRID_HEIGHT_RANGE[1] and min_w <= GRID_WIDTH_RANGE[1]:
            options.append((row_slots, col_slots, min_h, min_w))
    if not options:
        return None

    row_slots, col_slots, min_h, min_w = options[int(rng.integers(len(options)))]
    height_hi = min(GRID_HEIGHT_RANGE[1], min_h + LAYOUT_EXTRA_SLACK)
    width_hi = min(GRID_WIDTH_RANGE[1], min_w + LAYOUT_EXTRA_SLACK)
    height_lo = max(GRID_HEIGHT_RANGE[0], min_h)
    width_lo = max(GRID_WIDTH_RANGE[0], min_w)
    height_hi = max(height_hi, height_lo)
    width_hi = max(width_hi, width_lo)
    height = int(rng.integers(height_lo, height_hi + 1))
    width = int(rng.integers(width_lo, width_hi + 1))

    rows = _slot_positions(row_slots, bbox_h, height, rng)
    cols = _slot_positions(col_slots, bbox_w, width, rng)
    if rows is None or cols is None:
        return None

    all_positions = [(r, c) for r in rows for c in cols]
    perm = [int(i) for i in rng.permutation(len(all_positions))]
    positions = [all_positions[i] for i in perm[:sprite_count]]
    return height, width, positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the common sprite by overlaying damaged copies, then repaint every copy with the full mask.

    Input structure:
    - Background is the most common color.
    - One foreground color forms several separated sprite copies.
    - Each copy has the same bounding-box size and is missing 1-3 pixels.

    Rule:
    - Find every sprite copy.
    - Align their local masks by top-left corner and take the union.
    - Repaint the complete mask back into every original bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    components = _components(g == fg)
    assert len(components) >= 2, "expected multiple sprite copies"

    local_masks: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    bbox_shape: tuple[int, int] | None = None
    for comp in components:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        local = g[r0 : r1 + 1, c0 : c1 + 1] == fg
        if bbox_shape is None:
            bbox_shape = local.shape
        else:
            assert local.shape == bbox_shape, "all sprite boxes must share one size"
        boxes.append(box)
        local_masks.append(local)

    assert bbox_shape is not None
    consensus = np.zeros(bbox_shape, dtype=bool)
    for local in local_masks:
        consensus |= local
    assert np.any(consensus), "consensus sprite cannot be empty"

    out = g.copy()
    for r0, c0, r1, c1 in boxes:
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        _paint_mask(out, r0, c0, consensus, fg)

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sprite_count = int(rng.integers(SPRITE_COUNT_RANGE[0], SPRITE_COUNT_RANGE[1] + 1))
    bbox_h = int(rng.integers(BBOX_HEIGHT_RANGE[0], BBOX_HEIGHT_RANGE[1] + 1))
    bbox_w = int(rng.integers(BBOX_WIDTH_RANGE[0], BBOX_WIDTH_RANGE[1] + 1))

    base_mask = _grow_mask(bbox_h, bbox_w, rng)
    if base_mask is None:
        return None

    sampled = _sample_variants(base_mask, sprite_count, rng)
    if sampled is None:
        return None
    variants, _hole_sets = sampled

    layout = _sample_layout(sprite_count, bbox_h, bbox_w, rng)
    if layout is None:
        return None
    height, width, positions = layout

    palette = [int(c) for c in rng.permutation(10)]
    bg = palette[0]
    fg = palette[1]

    grid = np.full((height, width), bg, dtype=int)
    expected = np.full((height, width), bg, dtype=int)

    order = [int(i) for i in rng.permutation(sprite_count)]
    for idx, pos_idx in enumerate(order):
        top, left = positions[pos_idx]
        _paint_mask(grid, top, left, variants[idx], fg)
        _paint_mask(expected, top, left, base_mask, fg)

    if int(np.count_nonzero(grid != bg)) * 2 >= grid.size:
        return None
    if np.array_equal(grid, expected):
        return None

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid consensus-sprite input")
