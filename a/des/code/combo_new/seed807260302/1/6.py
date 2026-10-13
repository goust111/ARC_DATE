from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 500

MIN_STAMPS = 2
MAX_STAMPS = 4
MAX_REGIONS = 5
MIN_REGION_AREA = 4
MAX_REGION_AREA = 9
MAX_SHAPE_SIDE = 5
MAX_PUZZLE_SIDE = 12

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    height, width = map(int, mask.shape)

    for sr, sc in map(tuple, np.argwhere(mask)):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(comp)
    return comps


def _normalize_cells(cells) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in cells]
    assert pts, "shape must contain at least one cell"
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _is_connected_cells(cells) -> bool:
    cells_set = set((int(r), int(c)) for r, c in cells)
    if not cells_set:
        return False
    start = next(iter(cells_set))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nb = (r + dr, c + dc)
            if nb in cells_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells_set)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor matching regions in a large puzzle from same-shape stamp sprites, then erase the stamps.

    Input structure:
    - Black background contains exactly one large 4-connected multicolor puzzle component.
    - Each color inside that puzzle forms one irregular region, and different puzzle regions have different shapes.
    - Away from the puzzle are several small monochrome stamp sprites, each with a unique color not used by the puzzle.
    - Every stamp shape matches exactly one puzzle color region by translation only, with no rotation or reflection.

    Rule:
    - Extract each stamp's relative foreground-cell shape.
    - Find the puzzle color region whose full set of cells has the same relative shape.
    - Recolor that entire puzzle region to the stamp's color.
    - Clear the stamp's original cells to black.
    - Puzzle regions with no matching stamp keep their original colors.

    Key insight:
    - The stamp color is not a label for a target color; its geometry selects the puzzle region, and its color supplies the repaint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g != BLACK)
    assert comps, "expected at least one foreground component"

    comp_colors = [set(int(g[r, c]) for r, c in comp) for comp in comps]
    puzzle_indices = [i for i, colors in enumerate(comp_colors) if len(colors) > 1]
    assert len(puzzle_indices) == 1, "expected exactly one multicolor puzzle component"
    puzzle_idx = puzzle_indices[0]

    puzzle_mask = np.zeros(g.shape, dtype=bool)
    for r, c in comps[puzzle_idx]:
        puzzle_mask[r, c] = True

    shape_to_regions: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = {}
    for color in set(int(g[r, c]) for r, c in comps[puzzle_idx]):
        pts = [(int(r), int(c)) for r, c in np.argwhere((g == color) & puzzle_mask)]
        assert pts, "puzzle color region cannot be empty"
        assert _is_connected_cells(pts), "each puzzle color region must be connected"
        shape = _normalize_cells(pts)
        shape_to_regions.setdefault(shape, []).append({"color": color, "pts": pts})

    out = g.copy()
    used_puzzle_colors: set[int] = set()

    for i, comp in enumerate(comps):
        if i == puzzle_idx:
            continue

        colors = comp_colors[i]
        assert len(colors) == 1, "stamp components must be monochrome"
        stamp_color = next(iter(colors))
        stamp_shape = _normalize_cells(comp)

        candidates = [
            region
            for region in shape_to_regions.get(stamp_shape, [])
            if int(region["color"]) not in used_puzzle_colors
        ]
        assert len(candidates) == 1, "each stamp shape must match exactly one unused puzzle region"

        for r, c in comp:
            out[r, c] = BLACK

        region = candidates[0]
        used_puzzle_colors.add(int(region["color"]))
        for r, c in region["pts"]:
            out[int(r), int(c)] = int(stamp_color)

    return out


def _random_polyomino() -> tuple[tuple[int, int], ...] | None:
    for _ in range(120):
        target = random.randint(MIN_REGION_AREA, MAX_REGION_AREA)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nb = (r + dr, c + dc)
                    if nb not in cells:
                        frontier.add(nb)
            choices = list(frontier)
            random.shuffle(choices)
            added = False
            for nb in choices:
                candidate = _normalize_cells(cells | {nb})
                h, w = _shape_dims(candidate)
                if h <= MAX_SHAPE_SIDE and w <= MAX_SHAPE_SIDE:
                    cells = set(candidate)
                    added = True
                    break
            if not added:
                break
        if len(cells) != target:
            continue

        shape = _normalize_cells(cells)
        h, w = _shape_dims(shape)
        if h < 2 or w < 2:
            continue
        if h * w == len(shape):
            continue
        if not _is_connected_cells(shape):
            continue
        return shape
    return None


def _make_region_shapes(count: int) -> list[tuple[tuple[int, int], ...]] | None:
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(600):
        shape = _random_polyomino()
        if shape is None or shape in seen:
            continue
        shapes.append(shape)
        seen.add(shape)
        if len(shapes) == count:
            return shapes
    return None


def _bbox(cells) -> tuple[int, int, int, int]:
    pts = [(int(r), int(c)) for r, c in cells]
    return (
        min(r for r, _ in pts),
        min(c for _, c in pts),
        max(r for r, _ in pts),
        max(c for _, c in pts),
    )


def _touches(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _union_has_hole(cells: set[tuple[int, int]]) -> bool:
    shape = _normalize_cells(cells)
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    for comp in _components(~mask):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            return True
    return False


def _pack_regions(
    shapes: list[tuple[tuple[int, int], ...]],
) -> tuple[list[tuple[tuple[int, int], ...]], tuple[int, int]] | None:
    placed: list[set[tuple[int, int]]] = [set(shapes[0])]
    occupied = set(shapes[0])

    for shape in shapes[1:]:
        candidates: list[tuple[int, int, float, set[tuple[int, int]]]] = []
        for roff in range(-MAX_PUZZLE_SIDE, MAX_PUZZLE_SIDE + 1):
            for coff in range(-MAX_PUZZLE_SIDE, MAX_PUZZLE_SIDE + 1):
                cells = {(r + roff, c + coff) for r, c in shape}
                if cells & occupied:
                    continue
                if not _touches(cells, occupied):
                    continue
                union = occupied | cells
                r0, c0, r1, c1 = _bbox(union)
                height = r1 - r0 + 1
                width = c1 - c0 + 1
                if height > MAX_PUZZLE_SIDE or width > MAX_PUZZLE_SIDE:
                    continue
                candidates.append((height * width, height + width, random.random(), cells))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        cells = set(random.choice(candidates[: min(25, len(candidates))])[3])
        placed.append(cells)
        occupied |= cells

    if not _is_connected_cells(occupied):
        return None
    if _union_has_hole(occupied):
        return None

    r0, c0, r1, c1 = _bbox(occupied)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return None

    normalized = [
        tuple(sorted((r - r0, c - c0) for r, c in cells))
        for cells in placed
    ]
    return normalized, (height, width)


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    sub[patch != BLACK] = patch[patch != BLACK]


def _place_stamps(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    stamp_specs: list[tuple[int, int, tuple[tuple[int, int], ...]]],
) -> list[tuple[int, int, list[tuple[int, int]]]] | None:
    r0, r1, c0, c1 = bounds
    blocked = np.zeros(grid.shape, dtype=bool)
    placed: list[tuple[int, int, list[tuple[int, int]]]] = []

    for region_id, color, shape in random.sample(stamp_specs, len(stamp_specs)):
        mask = _shape_mask(shape)
        mh, mw = mask.shape
        if r1 - r0 < mh or c1 - c0 < mw:
            return None

        for _ in range(300):
            top = random.randint(r0, r1 - mh)
            left = random.randint(c0, c1 - mw)
            if blocked[top : top + mh, left : left + mw].any():
                continue

            patch = np.zeros(mask.shape, dtype=int)
            patch[mask] = color
            _paste_patch(grid, patch, top, left)
            br0 = max(r0, top - 1)
            br1 = min(r1, top + mh + 1)
            bc0 = max(c0, left - 1)
            bc1 = min(c1, left + mw + 1)
            blocked[br0:br1, bc0:bc1] = True
            cells = [(top + r, left + c) for r, c in shape]
            placed.append((region_id, color, cells))
            break
        else:
            return None
    return placed


def _render_input(
    regions: list[tuple[tuple[int, int], ...]],
    puzzle_dims: tuple[int, int],
    region_colors: list[int],
    stamp_specs: list[tuple[int, int, tuple[tuple[int, int], ...]]],
) -> tuple[np.ndarray, np.ndarray] | None:
    puzzle_h, puzzle_w = puzzle_dims
    puzzle_patch = np.zeros(puzzle_dims, dtype=int)
    for region_id, cells in enumerate(regions):
        for r, c in cells:
            puzzle_patch[r, c] = region_colors[region_id]

    max_stamp_h = 0
    max_stamp_w = 0
    for _, _, shape in stamp_specs:
        h, w = _shape_dims(shape)
        max_stamp_h = max(max_stamp_h, h)
        max_stamp_w = max(max_stamp_w, w)

    for _ in range(80):
        gap = random.randint(2, 4)
        min_h = max(16, puzzle_h + 2, max_stamp_h + 2)
        if min_h > 30:
            return None
        height = random.randint(min_h, 30)

        min_panel_w = max(8, max_stamp_w + 2)
        min_w = puzzle_w + gap + min_panel_w + 2
        if min_w > 30:
            return None
        width = random.randint(min_w, 30)

        puzzle_side = random.choice(("left", "right"))
        grid = np.zeros((height, width), dtype=int)
        puzzle_r0 = random.randint(1, height - puzzle_h - 1)
        if puzzle_side == "left":
            puzzle_c0 = 1
            bounds = (1, height - 1, puzzle_c0 + puzzle_w + gap, width - 1)
        else:
            puzzle_c0 = width - puzzle_w - 1
            bounds = (1, height - 1, 1, puzzle_c0 - gap)
        if bounds[2] >= bounds[3]:
            continue

        _paste_patch(grid, puzzle_patch, puzzle_r0, puzzle_c0)
        placed = _place_stamps(grid, bounds, stamp_specs)
        if placed is None:
            continue

        expected = grid.copy()
        for _, _, cells in placed:
            for r, c in cells:
                expected[r, c] = BLACK
        for region_id, color, _ in stamp_specs:
            for dr, dc in regions[region_id]:
                expected[puzzle_r0 + dr, puzzle_c0 + dc] = color

        return grid, expected
    return None


def _gen_attempt() -> np.ndarray | None:
    stamp_count = random.randint(MIN_STAMPS, MAX_STAMPS)
    max_extra = min(2, MAX_REGIONS - stamp_count, len(COLORS) - 2 * stamp_count)
    if max_extra < 1:
        return None
    region_count = stamp_count + random.randint(1, max_extra)

    shapes = _make_region_shapes(region_count)
    if shapes is None:
        return None
    packed = _pack_regions(shapes)
    if packed is None:
        return None
    regions, puzzle_dims = packed

    signatures = [_normalize_cells(cells) for cells in regions]
    if len(set(signatures)) != len(signatures):
        return None

    colors = random.sample(COLORS, region_count + stamp_count)
    region_colors = colors[:region_count]
    stamp_colors = colors[region_count:]

    stamp_ids = random.sample(range(region_count), stamp_count)
    stamp_specs = [
        (region_id, stamp_colors[i], _normalize_cells(regions[region_id]))
        for i, region_id in enumerate(stamp_ids)
    ]

    rendered = _render_input(regions, puzzle_dims, region_colors, stamp_specs)
    if rendered is None:
        return None
    grid, expected = rendered

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed807260302/1/6 input")
