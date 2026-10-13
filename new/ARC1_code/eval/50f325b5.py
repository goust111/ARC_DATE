from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords, components_mask
from itertools import combinations


Shape = tuple[tuple[int, int], ...]


def _normalize(coords: np.ndarray) -> Shape:
    coords = np.asarray(coords, dtype=int)
    coords = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _dihedral_variants(shape: Shape) -> list[Shape]:
    pts = np.asarray(shape, dtype=int)
    variants: list[Shape] = []
    seen: set[Shape] = set()

    for rotation in range(4):
        transformed = pts.copy()
        for _ in range(rotation):
            transformed = np.column_stack((-transformed[:, 1], transformed[:, 0]))

        for reflected in (False, True):
            candidate = transformed.copy()
            if reflected:
                candidate[:, 1] = -candidate[:, 1]
            normalized = _normalize(candidate)
            if normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)

    return variants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every non-overlapping green copy of the teal template motif to teal.

    Input structure:
    - A noisy same-size grid contains one 8-connected teal motif.
    - Elsewhere, green pixels contain exact copies of that motif, possibly rotated or reflected, embedded among distractor pixels.

    Rule:
    - Use the teal motif's bbox-normalized mask as the template.
    - Enumerate all unique rotations and reflections of that template.
    - Scan the grid in row-major order for placements whose template cells are all green.
    - Recolor selected placements to teal, skipping any later placement that overlaps an already selected one.

    Key insight:
    The teal pixels are not noise; they are the shape key for identifying matching green masks hidden in the field.
    """
    g = np.asarray(grid, dtype=int)

    template_components = components_mask(g == Color.TEAL, connectivity=8)
    assert len(template_components) == 1, "expected one 8-connected teal template"
    template_coords = template_components[0]
    assert len(template_coords) > 0, "teal template must be nonempty"

    r0, c0, _r1, _c1 = bbox_coords(template_coords)
    template_shape = _normalize(template_coords - np.array([r0, c0]))
    variants = _dihedral_variants(template_shape)
    assert variants, "expected at least one template variant"

    H, W = g.shape
    candidates: list[tuple[int, int, frozenset[tuple[int, int]]]] = []
    for variant in variants:
        h, w = _shape_size(variant)
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                cells = frozenset((r + dr, c + dc) for dr, dc in variant)
                if all(g[rr, cc] == Color.GREEN for rr, cc in cells):
                    candidates.append((r, c, cells))

    assert candidates, "expected at least one green template copy"
    by_anchor: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for r, c, cells in candidates:
        anchor = (r, c)
        if anchor in by_anchor:
            assert by_anchor[anchor] == cells, "top-left placement must be unambiguous"
        else:
            by_anchor[anchor] = cells

    candidates.sort(key=lambda item: (item[0], item[1]))

    selected: set[tuple[int, int]] = set()
    for _r, _c, cells in candidates:
        if selected.isdisjoint(cells):
            selected.update(cells)

    assert selected, "expected selected green template cells"
    out = g.copy()
    for r, c in selected:
        out[r, c] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------

Coord = tuple[int, int]
Placement = frozenset[Coord]
PlacementIndex = dict[Coord, list[Placement]]

# Canvas dimensions use np.random.randint-exclusive upper bounds.
GRID_H_MIN, GRID_H_MAX_EXCL = 11, 19
GRID_W_MIN, GRID_W_MAX_EXCL = 17, 19

# Template catalog: 4-5 cells inside a 3x3 box, one 8-connected component.
TEMPLATE_SIZES = (4, 5)
TEMPLATE_BBOX_MAX = 3
TEMPLATE_FLAT_CELLS = range(TEMPLATE_BBOX_MAX * TEMPLATE_BBOX_MAX)
TEMPLATE_CONNECTIVITY = 8

# Copy placement and local green-noise requirements.
COPY_COUNT_MIN, COPY_COUNT_MAX_EXCL = 1, 4
MIN_EXTRA_GREEN_NOISE = 1
UNFILLED = -1
NEIGHBOR_DELTAS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))

# Color density targets; red absorbs rounding after the sampled colors.
COLOR_COUNT_RANGES = {
    Color.BLACK: (0.27, 0.33),
    Color.GREEN: (0.15, 0.29),
    Color.YELLOW: (0.08, 0.13),
    Color.ORANGE: (0.13, 0.17),
    Color.RED: (0.16, 0.30),
}
FILL_COLORS = (Color.BLACK, Color.RED, Color.YELLOW, Color.ORANGE)
REQUIRED_COLORS = {Color.BLACK, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.TEAL}

# Rejection-sampling limits.
COUNT_RETRY_LIMIT = 80
GEN_RETRY_LIMIT = 300


def _gen_template_shapes() -> dict[int, dict[int, list[Shape]]]:
    by_size: dict[int, dict[int, list[Shape]]] = {}
    seen: set[Shape] = set()
    for size in TEMPLATE_SIZES:
        for flat_cells in combinations(TEMPLATE_FLAT_CELLS, size):
            coords = np.array(
                [(p // TEMPLATE_BBOX_MAX, p % TEMPLATE_BBOX_MAX) for p in flat_cells],
                dtype=int,
            )
            coords -= coords.min(axis=0)
            h = int(coords[:, 0].max()) + 1
            w = int(coords[:, 1].max()) + 1
            if h > TEMPLATE_BBOX_MAX or w > TEMPLATE_BBOX_MAX:
                continue
            mask = np.zeros((h, w), dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            if len(components_mask(mask, connectivity=TEMPLATE_CONNECTIVITY)) != 1:
                continue
            shape = tuple(sorted((int(r), int(c)) for r, c in coords))
            if shape in seen:
                continue
            seen.add(shape)
            variant_count = len(_dihedral_variants(shape))
            by_size.setdefault(size, {}).setdefault(variant_count, []).append(shape)
    return by_size


def _gen_choice(seq):
    return seq[int(np.random.randint(0, len(seq)))]


def _gen_template_shape() -> Shape:
    shapes = _gen_template_shapes()
    size = int(np.random.choice(TEMPLATE_SIZES))
    variant_counts = sorted(shapes[size])
    variant_count = _gen_choice(variant_counts)
    return _gen_choice(shapes[size][variant_count])


def _gen_cells_at(shape: Shape, r0: int, c0: int) -> Placement:
    return frozenset((r0 + dr, c0 + dc) for dr, dc in shape)


def _gen_variant_placements(variants: list[Shape], H: int, W: int):
    for variant in variants:
        h, w = _shape_size(variant)
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                yield _gen_cells_at(variant, r, c)


def _gen_all_placements(shape: Shape, H: int, W: int) -> tuple[list[Placement], PlacementIndex]:
    placements: list[Placement] = []
    by_cell: PlacementIndex = {}
    seen: set[Placement] = set()
    for cells in _gen_variant_placements(_dihedral_variants(shape), H, W):
        if cells in seen:
            continue
        seen.add(cells)
        placements.append(cells)
        for cell in cells:
            by_cell.setdefault(cell, []).append(cells)
    return placements, by_cell


def _gen_has_bad_match(green_cells: set[Coord], allowed: set[Placement], placements: list[Placement]) -> bool:
    for cells in placements:
        if cells not in allowed and cells.issubset(green_cells):
            return True
    return False


def _gen_adds_bad_match(cell: Coord, green_cells: set[Coord], allowed: set[Placement], by_cell: PlacementIndex) -> bool:
    trial_green = green_cells | {cell}
    for cells in by_cell.get(cell, []):
        if cells not in allowed and cells.issubset(trial_green):
            return True
    return False


def _gen_counts(N: int, template_size: int, fixed_green: int) -> dict[int, int] | None:
    black_lo, black_hi = _gen_count_bounds(Color.BLACK, N)
    green_lo, green_hi = _gen_count_bounds(Color.GREEN, N)
    green_lo = max(fixed_green + MIN_EXTRA_GREEN_NOISE, green_lo)
    yellow_lo, yellow_hi = _gen_count_bounds(Color.YELLOW, N)
    orange_lo, orange_hi = _gen_count_bounds(Color.ORANGE, N)
    red_lo, red_hi = _gen_count_bounds(Color.RED, N)

    for _ in range(COUNT_RETRY_LIMIT):
        black = int(np.random.randint(black_lo, black_hi + 1))
        green = int(np.random.randint(green_lo, green_hi + 1))
        yellow = int(np.random.randint(yellow_lo, yellow_hi + 1))
        orange = int(np.random.randint(orange_lo, orange_hi + 1))
        red = N - black - green - yellow - orange - template_size
        if red_lo <= red <= red_hi:
            return {
                Color.BLACK: black,
                Color.RED: red,
                Color.GREEN: green,
                Color.YELLOW: yellow,
                Color.ORANGE: orange,
                Color.TEAL: template_size,
            }
    return None


def _gen_count_bounds(color: int, N: int) -> tuple[int, int]:
    lo, hi = COLOR_COUNT_RANGES[color]
    return int(np.ceil(lo * N)), int(np.floor(hi * N))


def _gen_neighbor_cells(cells: Placement, H: int, W: int) -> list[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr, dc in NEIGHBOR_DELTAS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in cells:
                out.add((rr, cc))
    return list(out)


def _gen_place_copies(
    shape: Shape, H: int, W: int, K: int, occupied: set[Coord], placements: list[Placement]
) -> tuple[list[Placement], set[Coord]] | None:
    variants = _dihedral_variants(shape)
    copy_sets: list[Placement] = []
    green_cells: set[Coord] = set()
    allowed: set[Placement] = set()

    for _ in range(K):
        candidates = [
            cells
            for cells in _gen_variant_placements(variants, H, W)
            if cells.isdisjoint(occupied)
        ]
        order = np.random.permutation(len(candidates))
        placed = False
        for idx in order:
            cells = candidates[int(idx)]
            trial_allowed = allowed | {cells}
            trial_green = green_cells | set(cells)
            if _gen_has_bad_match(trial_green, trial_allowed, placements):
                continue
            copy_sets.append(cells)
            green_cells = trial_green
            allowed = trial_allowed
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return None
    return copy_sets, green_cells


def _gen_embed_copy_clutter(
    inp: np.ndarray, copy_sets: list[Placement], green_cells: set[Coord], allowed: set[Placement], by_cell: PlacementIndex
) -> bool:
    """Add one adjacent green noise cell around each meaningful copy."""
    H, W = inp.shape

    for cells in copy_sets:
        added = False
        neighbors = _gen_neighbor_cells(cells, H, W)
        for idx in np.random.permutation(len(neighbors)):
            cell = neighbors[int(idx)]
            if inp[cell] != UNFILLED or _gen_adds_bad_match(cell, green_cells, allowed, by_cell):
                continue
            green_cells.add(cell)
            inp[cell] = Color.GREEN
            added = True
            break
        if not added:
            return False
    return True


def _gen_fill_green_noise(
    inp: np.ndarray, target_green: int, green_cells: set[Coord], allowed: set[Placement], by_cell: PlacementIndex
) -> bool:
    """Fill random green noise without creating an extra template match."""
    H, W = inp.shape
    if len(green_cells) > target_green:
        return False

    free_cells = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == UNFILLED]
    for idx in np.random.permutation(len(free_cells)):
        if len(green_cells) >= target_green:
            break
        cell = free_cells[int(idx)]
        if inp[cell] != UNFILLED or _gen_adds_bad_match(cell, green_cells, allowed, by_cell):
            continue
        green_cells.add(cell)
        inp[cell] = Color.GREEN
    return len(green_cells) == target_green


def _gen_fill_remaining_colors(inp: np.ndarray, counts: dict[int, int]) -> bool:
    """Fill all non-green, non-template cells according to sampled color counts."""
    H, W = inp.shape
    remaining = [(r, c) for r in range(H) for c in range(W) if inp[r, c] == UNFILLED]
    fill_colors: list[int] = []
    for color in FILL_COLORS:
        fill_colors.extend([color] * counts[color])
    if len(fill_colors) != len(remaining):
        return False
    fill_colors = [fill_colors[int(i)] for i in np.random.permutation(len(fill_colors))]
    for (r, c), color in zip(remaining, fill_colors):
        inp[r, c] = color
    return True


def _gen_oracle_accepts(inp: np.ndarray, out: np.ndarray) -> bool:
    """Accept only grids whose solver output and structural checks match."""
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False
    if set(np.unique(inp).tolist()) != REQUIRED_COLORS:
        return False
    if len(components_mask(inp == Color.TEAL, connectivity=TEMPLATE_CONNECTIVITY)) != 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = int(np.random.randint(GRID_H_MIN, GRID_H_MAX_EXCL))
    W = int(np.random.randint(GRID_W_MIN, GRID_W_MAX_EXCL))
    shape = _gen_template_shape()
    template_size = len(shape)
    placements, by_cell = _gen_all_placements(shape, H, W)

    h0, w0 = _shape_size(shape)
    template_r = int(np.random.randint(0, H - h0 + 1))
    template_c = int(np.random.randint(0, W - w0 + 1))
    template_cells = _gen_cells_at(shape, template_r, template_c)
    occupied = set(template_cells)

    K = int(np.random.randint(COPY_COUNT_MIN, COPY_COUNT_MAX_EXCL))
    placed = _gen_place_copies(shape, H, W, K, occupied, placements)
    if placed is None:
        return None
    copy_sets, green_cells = placed
    allowed = set(copy_sets)

    counts = _gen_counts(H * W, template_size, len(green_cells))
    if counts is None:
        return None

    inp = np.full((H, W), UNFILLED, dtype=int)
    for r, c in template_cells:
        inp[r, c] = Color.TEAL
    for r, c in green_cells:
        inp[r, c] = Color.GREEN

    if not _gen_embed_copy_clutter(inp, copy_sets, green_cells, allowed, by_cell):
        return None
    if not _gen_fill_green_noise(inp, counts[Color.GREEN], green_cells, allowed, by_cell):
        return None
    if not _gen_fill_remaining_colors(inp, counts):
        return None

    out = inp.copy()
    for cells in copy_sets:
        for r, c in cells:
            out[r, c] = Color.TEAL
    if not _gen_oracle_accepts(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 50f325b5 input")
