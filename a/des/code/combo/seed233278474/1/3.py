from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PATTERN_COLORS = (1, 3, 4, 6, 7, 8, 9)

DIRECTIONS = ("up", "down", "left", "right")
DIR_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_OFFSETS = {
    "right": ((0, 0), (1, 1), (2, 0)),
    "left": ((0, 1), (1, 0), (2, 1)),
    "up": ((0, 1), (1, 0), (1, 2)),
    "down": ((0, 0), (0, 2), (1, 1)),
}
ARROW_PATTERNS = {
    frozenset(offsets): direction for direction, offsets in ARROW_OFFSETS.items()
}

GEN_RETRIES = 400
PLACEMENT_RETRIES = 200

BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
)


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_once(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _r, c in cells)
    return {(c, max_c - r) for r, c in cells}


def _flip_h(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _r, c in cells)
    return {(r, max_c - c) for r, c in cells}


def _all_shape_variants() -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for base in BASE_SHAPES:
        cur = set(base)
        for _ in range(4):
            variants.add(_normalize(cur))
            variants.add(_normalize(_flip_h(cur)))
            cur = _rotate_once(cur)
    return sorted(variants)


SHAPE_VARIANTS = _all_shape_variants()


def _spans(size: int, line_indices: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(line_indices):
        if idx > start:
            spans.append((start, idx - 1))
        start = idx + 1
    if start < size:
        spans.append((start, size - 1))
    return spans


def _connected_4(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbr = (r + dr, c + dc)
            if nbr in cells and nbr not in seen:
                stack.append(nbr)
    return seen == cells


def _decode_arrow(red_coords: np.ndarray) -> str:
    assert red_coords.shape == (3, 2), "expected exactly one 3-pixel red arrow per region"
    r0 = int(red_coords[:, 0].min())
    c0 = int(red_coords[:, 1].min())
    norm = frozenset((int(r) - r0, int(c) - c0) for r, c in red_coords)
    assert norm in ARROW_PATTERNS, "unrecognized arrow shape"
    return ARROW_PATTERNS[norm]


def _parse_regions(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    grey_rows = [r for r in range(h) if bool(np.all(g[r, :] == GREY))]
    grey_cols = [c for c in range(w) if bool(np.all(g[:, c] == GREY))]
    assert grey_rows and grey_cols, "expected full-width and full-height grey divider lines"
    assert all(b - a > 1 for a, b in zip(grey_rows, grey_rows[1:])), "grey rows must be 1 cell thick"
    assert all(b - a > 1 for a, b in zip(grey_cols, grey_cols[1:])), "grey cols must be 1 cell thick"

    row_spans = _spans(h, grey_rows)
    col_spans = _spans(w, grey_cols)
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected a rectangular partition"

    regions: list[dict] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            red_coords = np.argwhere(sub == RED)
            direction = _decode_arrow(red_coords)

            content_mask = (sub != BLACK) & (sub != RED) & (sub != GREY)
            content_coords = np.argwhere(content_mask)
            assert 2 <= int(content_coords.shape[0]) <= 4, "each region must contain one small pattern"
            content_cells = {(int(r), int(c)) for r, c in content_coords}
            assert _connected_4(content_cells), "pattern must be 4-connected"

            pixels = [
                (int(r), int(c), int(sub[int(r), int(c)]))
                for r, c in content_coords
            ]
            regions.append(
                {
                    "ri": ri,
                    "ci": ci,
                    "r0": r0,
                    "c0": c0,
                    "direction": direction,
                    "pixels": pixels,
                }
            )
    return row_spans, col_spans, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify grey-divided rectangular regions, read each region's red V-shaped arrow,
    and push the non-arrow pattern into the adjacent region while preserving local offset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans, col_spans, regions = _parse_regions(g)
    out = np.full_like(g, BLACK)
    out[g == GREY] = GREY

    placements: list[tuple[tuple[int, int, int], int, int, list[tuple[int, int, int]]]] = []
    max_ri = len(row_spans) - 1
    max_ci = len(col_spans) - 1

    for region in regions:
        ri = int(region["ri"])
        ci = int(region["ci"])
        dr, dc = DIR_STEP[region["direction"]]
        tri = ri + dr
        tci = ci + dc

        if not (0 <= tri <= max_ri and 0 <= tci <= max_ci):
            tri, tci = ri, ci
            priority = (-1, ri, ci)
        else:
            if ri < tri:
                cls = 0
            elif ci < tci:
                cls = 1
            elif ci > tci:
                cls = 2
            else:
                cls = 3
            priority = (cls, ri, ci)

        target_r0 = row_spans[tri][0]
        target_c0 = col_spans[tci][0]
        placements.append((priority, target_r0, target_c0, list(region["pixels"])))

    placements.sort(key=lambda item: item[0])
    for _priority, target_r0, target_c0, pixels in placements:
        for local_r, local_c, color in pixels:
            rr = target_r0 + local_r
            cc = target_c0 + local_c
            assert out[rr, cc] != GREY, "translated pattern may not cross divider lines"
            if out[rr, cc] == BLACK:
                out[rr, cc] = color

    return out


def _target_of(pos: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIR_STEP[direction]
    return pos[0] + dr, pos[1] + dc


def _transformed_shape() -> tuple[tuple[int, int], ...]:
    return random.choice(SHAPE_VARIANTS)


def _positioned_shape(
    cell_h: int,
    cell_w: int,
    *,
    required: tuple[tuple[int, int], ...] | None = None,
) -> set[tuple[int, int]]:
    shape = required if required is not None else _transformed_shape()
    height = max(r for r, _c in shape) + 1
    width = max(c for _r, c in shape) + 1
    top = random.randint(0, cell_h - height)
    left = random.randint(0, cell_w - width)
    return {(top + r, left + c) for r, c in shape}


def _separated(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _place_arrow(
    direction: str,
    cell_h: int,
    cell_w: int,
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    offsets = ARROW_OFFSETS[direction]
    height = max(r for r, _c in offsets) + 1
    width = max(c for _r, c in offsets) + 1
    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(0, cell_h - height)
        left = random.randint(0, cell_w - width)
        cells = {(top + r, left + c) for r, c in offsets}
        if cells & forbidden:
            continue
        if not _separated(cells, forbidden):
            continue
        return cells
    return None


def _place_region_content(
    direction: str,
    cell_h: int,
    cell_w: int,
    *,
    forced_shape: set[tuple[int, int]] | None = None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_RETRIES):
        shape_cells = forced_shape if forced_shape is not None else _positioned_shape(cell_h, cell_w)
        arrow_cells = _place_arrow(direction, cell_h, cell_w, shape_cells)
        if arrow_cells is not None:
            return set(shape_cells), arrow_cells
    return None


def _pick_outward_region(
    rows: int,
    cols: int,
    banned: set[tuple[int, int]],
) -> tuple[tuple[int, int], str] | None:
    options: list[tuple[tuple[int, int], str]] = []
    for ri in range(rows):
        for ci in range(cols):
            pos = (ri, ci)
            if pos in banned:
                continue
            if ri == 0:
                options.append((pos, "up"))
            if ri == rows - 1:
                options.append((pos, "down"))
            if ci == 0:
                options.append((pos, "left"))
            if ci == cols - 1:
                options.append((pos, "right"))
    return random.choice(options) if options else None


def _build_directions(rows: int, cols: int) -> tuple[list[list[str]], list[tuple[int, int]]]:
    directions = [
        [random.choice(DIRECTIONS) for _ in range(cols)]
        for _ in range(rows)
    ]
    collision_sources: list[tuple[int, int]] = []
    reserved: set[tuple[int, int]] = set()

    if random.random() < 0.5:
        target = (random.randrange(1, rows), random.randrange(1, cols))
        top_src = (target[0] - 1, target[1])
        left_src = (target[0], target[1] - 1)
        directions[top_src[0]][top_src[1]] = "down"
        directions[left_src[0]][left_src[1]] = "right"
        collision_sources = [top_src, left_src]
        reserved.update(collision_sources)

    outward = _pick_outward_region(rows, cols, reserved)
    if outward is not None and random.random() < 0.5:
        (ri, ci), outward_dir = outward
        directions[ri][ci] = outward_dir
        reserved.add((ri, ci))
        for nr, nc in ((ri - 1, ci), (ri + 1, ci), (ri, ci - 1), (ri, ci + 1)):
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in reserved:
                continue
            if _target_of((nr, nc), directions[nr][nc]) != (ri, ci):
                continue
            alternatives = [
                d
                for d in DIRECTIONS
                if _target_of((nr, nc), d) != (ri, ci)
            ]
            directions[nr][nc] = random.choice(alternatives)

    return directions, collision_sources


def _render_attempt() -> np.ndarray | None:
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    cell_h = random.randint(6, 8)
    cell_w = random.randint(6, 8)
    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)

    grid = np.full((height, width), BLACK, dtype=int)
    row_spans = []
    col_spans = []
    for ri in range(rows):
        start = ri * (cell_h + 1)
        row_spans.append((start, start + cell_h - 1))
    for ci in range(cols):
        start = ci * (cell_w + 1)
        col_spans.append((start, start + cell_w - 1))

    line_rows = [r1 + 1 for r0, r1 in row_spans[:-1]]
    line_cols = [c1 + 1 for c0, c1 in col_spans[:-1]]
    for r in line_rows:
        grid[r, :] = GREY
    for c in line_cols:
        grid[:, c] = GREY

    directions, collision_sources = _build_directions(rows, cols)
    shared_shape: set[tuple[int, int]] | None = None
    if len(collision_sources) == 2:
        for _ in range(PLACEMENT_RETRIES):
            candidate = _positioned_shape(cell_h, cell_w)
            ok = True
            for src in collision_sources:
                placed = _place_region_content(
                    directions[src[0]][src[1]],
                    cell_h,
                    cell_w,
                    forced_shape=candidate,
                )
                if placed is None:
                    ok = False
                    break
            if ok:
                shared_shape = candidate
                break
        if shared_shape is None:
            return None

    collision_colors: dict[tuple[int, int], int] = {}
    if len(collision_sources) == 2:
        c1, c2 = random.sample(PATTERN_COLORS, 2)
        collision_colors[collision_sources[0]] = c1
        collision_colors[collision_sources[1]] = c2

    for ri in range(rows):
        for ci in range(cols):
            direction = directions[ri][ci]
            forced_shape = shared_shape if (ri, ci) in collision_sources else None
            placed = _place_region_content(direction, cell_h, cell_w, forced_shape=forced_shape)
            if placed is None:
                return None
            shape_cells, arrow_cells = placed
            color = collision_colors.get((ri, ci), random.choice(PATTERN_COLORS))
            r0, _r1 = row_spans[ri]
            c0, _c1 = col_spans[ci]
            for lr, lc in shape_cells:
                grid[r0 + lr, c0 + lc] = color
            for lr, lc in arrow_cells:
                grid[r0 + lr, c0 + lc] = RED

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _render_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
