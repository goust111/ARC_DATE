from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
TEAL = 8

GEN_ATTEMPTS = 400


def _cell_ranges(length: int, wall_indices: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(int(x) for x in wall_indices), int(length)]
    ranges: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if a + 1 <= b - 1:
            ranges.append((a + 1, b - 1))
    return ranges


def _find_teal_lattice(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    wall_rows = [r for r in range(h) if np.all(g[r, :] == TEAL)]
    wall_cols = [c for c in range(w) if np.all(g[:, c] == TEAL)]
    row_ranges = _cell_ranges(h, wall_rows)
    col_ranges = _cell_ranges(w, wall_cols)
    assert len(row_ranges) >= 2 and len(col_ranges) >= 2, "expected a teal lattice"
    return row_ranges, col_ranges


def _cell_patch(
    g: np.ndarray, row_ranges: list[tuple[int, int]], col_ranges: list[tuple[int, int]], rr: int, cc: int
) -> np.ndarray:
    r0, r1 = row_ranges[rr]
    c0, c1 = col_ranges[cc]
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _paint_cell(
    out: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    rr: int,
    cc: int,
    color: int,
) -> None:
    r0, r1 = row_ranges[rr]
    c0, c1 = col_ranges[cc]
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_coords(coords)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _canonical_shape_key(coords: list[tuple[int, int]]) -> tuple[bytes, tuple[int, int]]:
    mask = _coords_to_mask(coords)
    variants: list[tuple[bytes, tuple[int, int]]] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            variants.append((candidate.tobytes(), tuple(map(int, candidate.shape))))
    return min(variants)


def _filled_cells_from_markers(markers_by_color: dict[int, list[tuple[int, int]]]) -> dict[tuple[int, int], set[int]]:
    filled: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color, cells in markers_by_color.items():
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for rr, cc in cells:
            by_row[int(rr)].append(int(cc))
            by_col[int(cc)].append(int(rr))

        for rr, cols in by_row.items():
            if len(cols) >= 2:
                for cc in range(min(cols), max(cols) + 1):
                    filled[(int(rr), int(cc))].add(int(color))
        for cc, rows in by_col.items():
            if len(rows) >= 2:
                for rr in range(min(rows), max(rows) + 1):
                    filled[(int(rr), int(cc))].add(int(color))
    return filled


def _extract_scene(g: np.ndarray):
    row_ranges, col_ranges = _find_teal_lattice(g)
    markers_by_color: dict[int, list[tuple[int, int]]] = {RED: [], GREEN: []}
    sprites = []

    for rr in range(len(row_ranges)):
        for cc in range(len(col_ranges)):
            patch = _cell_patch(g, row_ranges, col_ranges, rr, cc)
            if np.any(patch == RED):
                markers_by_color[RED].append((rr, cc))
            if np.any(patch == GREEN):
                markers_by_color[GREEN].append((rr, cc))

            blue_mask = patch == BLUE
            if blue_mask.any():
                comps = _components4(blue_mask)
                assert len(comps) == 1, "each lattice cell should contain at most one blue sprite"
                rel_coords = [(int(r), int(c)) for r, c in comps[0]]
                r0, c0 = row_ranges[rr][0], col_ranges[cc][0]
                abs_coords = [(r0 + r, c0 + c) for r, c in rel_coords]
                sprites.append(
                    {
                        "cell": (rr, cc),
                        "coords": abs_coords,
                        "shape_key": _canonical_shape_key(rel_coords),
                    }
                )

    return row_ranges, col_ranges, markers_by_color, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill lattice cells from red/green marker alignments, erase unique blue sprites by shape, then repaint filled cells yellow where shared-sprite rows and columns intersect.

    Input structure:
    - A teal one-pixel lattice partitions the canvas into regular rectangular cells.
    - Some cells contain a single red or green marker pixel near the cell center.
    - Other cells contain one blue 4-connected sprite of 3-5 pixels; sprite shapes are compared up to rotation and reflection.

    Rule:
    - Detect the lattice cells from the full teal rows and columns.
    - For each color separately, if a lattice row has at least two same-color marker cells, fill every cell between the leftmost and rightmost markers; do the analogous fill for lattice columns.
    - A cell reached by both red and green fills becomes pink.
    - Group blue sprites by canonical shape under the 8 rotation/reflection transforms; keep shapes seen at least twice and erase unique shapes.
    - Any filled cell whose lattice row contains a surviving shared sprite and whose lattice column contains a surviving shared sprite is repainted yellow; surviving shared sprites stay blue and teal lattice lines stay unchanged.

    Key insight:
    - The task is a convergence of two independent coarse-grid analyses: red/green line filling works only in lattice coordinates, while the final yellow repaint depends on which sprite rows and columns survive the shape-frequency filter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_ranges, col_ranges, markers_by_color, sprites = _extract_scene(g)
    filled = _filled_cells_from_markers(markers_by_color)

    freq = Counter(sprite["shape_key"] for sprite in sprites)
    shared_rows = {int(sprite["cell"][0]) for sprite in sprites if freq[sprite["shape_key"]] >= 2}
    shared_cols = {int(sprite["cell"][1]) for sprite in sprites if freq[sprite["shape_key"]] >= 2}
    shared_sprites = [sprite for sprite in sprites if freq[sprite["shape_key"]] >= 2]

    out = np.full_like(g, BLACK)
    out[g == TEAL] = TEAL

    for (rr, cc), colors in filled.items():
        color = PINK if colors == {RED, GREEN} else next(iter(colors))
        _paint_cell(out, row_ranges, col_ranges, rr, cc, color)

    for rr, cc in list(filled):
        if rr in shared_rows and cc in shared_cols:
            _paint_cell(out, row_ranges, col_ranges, rr, cc, YELLOW)

    for sprite in shared_sprites:
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = BLUE

    return out


BASE_SHAPES: list[list[tuple[int, int]]] = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
]


def _unique_base_shapes() -> list[list[tuple[int, int]]]:
    unique: dict[tuple[bytes, tuple[int, int]], list[tuple[int, int]]] = {}
    for shape in BASE_SHAPES:
        unique.setdefault(_canonical_shape_key(shape), shape)
    return list(unique.values())


UNIQUE_BASE_SHAPES = _unique_base_shapes()


def _transform_shape(coords: list[tuple[int, int]], rot_k: int, flip_lr: bool) -> list[tuple[int, int]]:
    mask = _coords_to_mask(coords)
    arr = np.rot90(mask, int(rot_k))
    if flip_lr:
        arr = np.fliplr(arr)
    return [(int(r), int(c)) for r, c in np.argwhere(arr)]


def _choose_dims() -> tuple[int, int, int, int]:
    for _ in range(100):
        n_rows = random.choice([5, 6])
        n_cols = random.choice([5, 6])
        cell_h = random.randint(3, 4 if n_rows == 5 else 3)
        cell_w = random.randint(3, 4 if n_cols == 5 else 3)
        h = n_rows * cell_h + n_rows + 1
        w = n_cols * cell_w + n_cols + 1
        if h <= 30 and w <= 30:
            return n_rows, n_cols, cell_h, cell_w
    raise RuntimeError("failed to choose valid lattice dimensions")


def _coarse_layout(n_rows: int, n_cols: int):
    y0 = random.randint(1, n_rows - 2)
    x_left = 1
    x_right = n_cols - 2

    red = {
        (y0, 0),
        (y0, x_left),
        (y0 - 1, x_right),
        (y0 + 1, x_right),
    }
    green = {
        (y0, x_right),
        (y0, n_cols - 1),
        (y0 - 1, x_left),
        (y0 + 1, x_left),
    }

    shared_row_pos = (y0, random.choice([c for c in range(n_cols) if c not in {0, x_left, x_right, n_cols - 1}]))
    col_choices = [r for r in range(n_rows) if r not in {y0 - 1, y0, y0 + 1}]
    shared_col_pos = (random.choice(col_choices), x_left)

    flip_v = bool(random.getrandbits(1))
    flip_h = bool(random.getrandbits(1))

    def xf(cell: tuple[int, int]) -> tuple[int, int]:
        rr, cc = cell
        if flip_v:
            rr = n_rows - 1 - rr
        if flip_h:
            cc = n_cols - 1 - cc
        return int(rr), int(cc)

    red = {xf(cell) for cell in red}
    green = {xf(cell) for cell in green}
    shared_row_pos = xf(shared_row_pos)
    shared_col_pos = xf(shared_col_pos)

    return red, green, shared_row_pos, shared_col_pos


def _build_sprite_plan(available_cells: list[tuple[int, int]], row_pos: tuple[int, int], col_pos: tuple[int, int]):
    total = random.randint(5, 8)
    if total >= 6 and random.random() < 0.45:
        shared_counts = [random.choice([2, 3]), 2]
        if sum(shared_counts) >= total:
            shared_counts = [2, 2]
    else:
        shared_counts = [random.choice([2, 3])]
    while sum(shared_counts) >= total:
        shared_counts[-1] -= 1
    assert min(shared_counts) >= 2
    unique_count = total - sum(shared_counts)
    assert unique_count >= 1

    template_count = len(shared_counts) + unique_count
    templates = random.sample(UNIQUE_BASE_SHAPES, template_count)
    shared_templates = templates[: len(shared_counts)]
    unique_templates = templates[len(shared_counts) :]

    shared_allowed = [cell for cell in available_cells if cell[1] != col_pos[1] + (0 if col_pos[1] != row_pos[1] else 1)]
    required = [row_pos, col_pos]
    assert row_pos in available_cells and col_pos in available_cells and row_pos != col_pos

    used = set(required)
    plan: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []

    first_count = shared_counts[0]
    extras = first_count - 2
    first_pool = [cell for cell in shared_allowed if cell not in used and cell[1] != row_pos[1]]
    if len(first_pool) < extras:
        first_pool = [cell for cell in available_cells if cell not in used]
    first_positions = required + random.sample(first_pool, extras)
    for pos in first_positions:
        used.add(pos)
        plan.append((pos, shared_templates[0]))

    for group_index in range(1, len(shared_counts)):
        count = shared_counts[group_index]
        pool = [cell for cell in available_cells if cell not in used and cell[1] != row_pos[1]]
        if len(pool) < count:
            pool = [cell for cell in available_cells if cell not in used]
        positions = random.sample(pool, count)
        for pos in positions:
            used.add(pos)
            plan.append((pos, shared_templates[group_index]))

    remaining = [cell for cell in available_cells if cell not in used]
    for pos, shape in zip(random.sample(remaining, unique_count), unique_templates, strict=True):
        used.add(pos)
        plan.append((pos, shape))

    return plan


def _render_grid(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    red_cells: set[tuple[int, int]],
    green_cells: set[tuple[int, int]],
    sprite_plan: list[tuple[tuple[int, int], list[tuple[int, int]]]],
) -> np.ndarray:
    h = n_rows * cell_h + n_rows + 1
    w = n_cols * cell_w + n_cols + 1
    g = np.full((h, w), BLACK, dtype=int)

    wall_rows = [i * (cell_h + 1) for i in range(n_rows + 1)]
    wall_cols = [i * (cell_w + 1) for i in range(n_cols + 1)]
    g[wall_rows, :] = TEAL
    g[:, wall_cols] = TEAL
    row_ranges = _cell_ranges(h, wall_rows)
    col_ranges = _cell_ranges(w, wall_cols)

    def cell_center(rr: int, cc: int) -> tuple[int, int]:
        r0, r1 = row_ranges[rr]
        c0, c1 = col_ranges[cc]
        return r0 + (r1 - r0) // 2, c0 + (c1 - c0) // 2

    for rr, cc in red_cells:
        r, c = cell_center(rr, cc)
        g[r, c] = RED
    for rr, cc in green_cells:
        r, c = cell_center(rr, cc)
        g[r, c] = GREEN

    for (rr, cc), base_shape in sprite_plan:
        shape = _transform_shape(base_shape, random.randint(0, 3), bool(random.getrandbits(1)))
        shape = _normalize_coords(shape)
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        off_r = random.randint(0, cell_h - shape_h)
        off_c = random.randint(0, cell_w - shape_w)
        cell_r0, _ = row_ranges[rr]
        cell_c0, _ = col_ranges[cc]
        for sr, sc in shape:
            g[cell_r0 + off_r + sr, cell_c0 + off_c + sc] = BLUE

    return g


def _accept(inp: np.ndarray) -> bool:
    out = main(inp)
    if inp.shape != out.shape:
        return False
    if np.count_nonzero(inp == BLUE) < 5:
        return False
    if not np.any(out == YELLOW):
        return False
    if not np.any(out == PINK):
        return False
    if not np.any(out == BLUE):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_rows, n_cols, cell_h, cell_w = _choose_dims()
        red_cells, green_cells, shared_row_pos, shared_col_pos = _coarse_layout(n_rows, n_cols)
        markers_by_color = {RED: list(red_cells), GREEN: list(green_cells)}
        filled = _filled_cells_from_markers(markers_by_color)
        unavailable = set(filled)
        unavailable |= red_cells | green_cells

        available = [
            (rr, cc)
            for rr in range(n_rows)
            for cc in range(n_cols)
            if (rr, cc) not in unavailable
        ]
        if shared_row_pos not in available or shared_col_pos not in available:
            continue
        if len(available) < 8:
            continue

        try:
            sprite_plan = _build_sprite_plan(available, shared_row_pos, shared_col_pos)
        except Exception:
            continue

        inp = _render_grid(n_rows, n_cols, cell_h, cell_w, red_cells, green_cells, sprite_plan)
        if not _accept(inp):
            continue
        return inp

    raise RuntimeError("failed to generate a valid ARC input")
