import random
from itertools import combinations

import numpy as np


Coord = tuple[int, int]
Shape = tuple[Coord, ...]
Placement = tuple[Coord, ...]

BACKGROUND = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_HEIGHT_RANGE = (16, 22)
GRID_WIDTH_RANGE = (16, 22)
COUNT_PATTERNS = (
    (2, 1, 1),
    (1, 2, 1),
    (1, 1, 2),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
)
GENERATION_RETRIES = 300
SOLVER_MAX_SOLUTIONS = 2


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(cells: list[Coord] | tuple[Coord, ...]) -> Shape:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _dihedral_variants(shape: Shape) -> list[Shape]:
    arr = np.zeros(_shape_dims(shape), dtype=bool)
    for r, c in shape:
        arr[r, c] = True
    variants: set[Shape] = set()
    for k in range(4):
        rotated = np.rot90(arr, k)
        for candidate in (rotated, np.fliplr(rotated)):
            coords = [tuple(map(int, rc)) for rc in np.argwhere(candidate)]
            variants.add(_normalize(coords))
    return sorted(variants, key=lambda s: (_shape_dims(s), s))


def _all_template_shapes() -> list[Shape]:
    shapes: set[Shape] = set()
    cells = [(r, c) for r in range(3) for c in range(3)]
    for size in (3, 4, 5):
        for combo in combinations(cells, size):
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            if len(_components(mask, connectivity=4)) != 1:
                continue
            shape = _normalize(combo)
            h, w = _shape_dims(shape)
            if h <= 3 and w <= 3:
                shapes.add(shape)
    return sorted(shapes, key=lambda s: (len(s), len(_dihedral_variants(s)), s))


TEMPLATE_SHAPES = _all_template_shapes()


def _legend_order(grid: np.ndarray) -> tuple[list[int], set[Coord]]:
    cols = [c for c in range(grid.shape[1]) if int(grid[0, c]) != BACKGROUND]
    assert len(cols) == 3, "expected exactly three legend pixels"
    assert cols == list(range(cols[0], cols[0] + 3)), "legend must be contiguous"
    order = [int(grid[0, c]) for c in cols]
    assert len(set(order)) == 3 and GREY not in order
    return order, {(0, c) for c in cols}


def _template_shape(grid: np.ndarray) -> Shape:
    comps = _components(grid == GREY, connectivity=4)
    assert len(comps) == 1, "expected one grey template"
    shape = _normalize(comps[0])
    h, w = _shape_dims(shape)
    assert 3 <= len(shape) <= 5 and h <= 3 and w <= 3
    return shape


def _placement_anchor(placement: Placement) -> Coord:
    return min(placement)


def _place_shape(shape: Shape, r0: int, c0: int) -> Placement:
    return tuple(sorted((r0 + dr, c0 + dc) for dr, dc in shape))


def _bbox(cells: Placement | list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches(a: Placement, b: Placement) -> bool:
    aset = set(a)
    for r, c in b:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in aset:
                    return True
    return False


def _candidate_placements(grid: np.ndarray, order: list[int], variants: list[Shape]) -> dict[int, list[Placement]]:
    h, w = grid.shape
    rank = {color: i for i, color in enumerate(order)}
    allowed = set(order)
    candidates: dict[int, list[Placement]] = {}
    for color in order:
        color_rank = rank[color]
        seen: set[Placement] = set()
        choices: list[Placement] = []
        for variant in variants:
            vh, vw = _shape_dims(variant)
            for r0 in range(1, h - vh + 1):
                for c0 in range(w - vw + 1):
                    placement = _place_shape(variant, r0, c0)
                    if placement in seen:
                        continue
                    visible_ranks: list[int] = []
                    for r, c in placement:
                        value = int(grid[r, c])
                        if value not in allowed:
                            break
                        visible_ranks.append(rank[value])
                    else:
                        if all(vr <= color_rank for vr in visible_ranks) and any(vr == color_rank for vr in visible_ranks):
                            seen.add(placement)
                            choices.append(placement)
        choices.sort(key=lambda p: (_placement_anchor(p), len(p), p))
        candidates[color] = choices
    return candidates


def _subset_options(candidates: list[Placement], required: set[Coord]) -> list[tuple[Placement, ...]]:
    options: list[tuple[Placement, ...]] = []
    for count in (1, 2):
        for combo in combinations(candidates, count):
            union: set[Coord] = set()
            valid = True
            for placement in combo:
                placement_set = set(placement)
                if union & placement_set:
                    valid = False
                    break
                union |= placement_set
            if valid and required <= union:
                options.append(combo)
    options.sort(
        key=lambda combo: (
            sum(len(p) for p in combo),
            len(combo),
            tuple(_placement_anchor(p) for p in combo),
        )
    )
    return options


def _render_rank_maps(order: list[int], selected: dict[int, tuple[Placement, ...]]) -> tuple[dict[Coord, int], dict[Coord, int]]:
    rank = {color: i for i, color in enumerate(order)}
    min_rank: dict[Coord, int] = {}
    max_rank: dict[Coord, int] = {}
    for color in order:
        color_rank = rank[color]
        for placement in selected[color]:
            for cell in placement:
                if cell not in min_rank or color_rank < min_rank[cell]:
                    min_rank[cell] = color_rank
                if cell not in max_rank or color_rank > max_rank[cell]:
                    max_rank[cell] = color_rank
    return min_rank, max_rank


def _solve_placements(grid: np.ndarray, require_unique: bool = False) -> dict[int, tuple[Placement, ...]]:
    order, _legend_cells = _legend_order(grid)
    template = _template_shape(grid)
    variants = _dihedral_variants(template)
    candidates = _candidate_placements(grid, order, variants)
    visible_by_color = {
        color: {(int(r), int(c)) for r, c in np.argwhere(grid == color) if r > 0}
        for color in order
    }
    options_by_color = {
        color: _subset_options(candidates[color], visible_by_color[color])
        for color in order
    }
    assert all(options_by_color[color] for color in order), "every color needs at least one covering subset"
    rank = {color: i for i, color in enumerate(order)}
    colored_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(np.isin(grid, order))
        if r > 0
    }
    colors_by_difficulty = sorted(order, key=lambda color: (len(options_by_color[color]), rank[color]))
    solutions: list[dict[int, tuple[Placement, ...]]] = []
    partial: dict[int, tuple[Placement, ...]] = {}

    def backtrack(index: int) -> None:
        if require_unique and len(solutions) >= SOLVER_MAX_SOLUTIONS:
            return
        if index == len(colors_by_difficulty):
            selected = {color: partial[color] for color in order}
            min_rank, _max_rank = _render_rank_maps(order, selected)
            if set(min_rank) != colored_cells:
                return
            for cell, low_rank in min_rank.items():
                if low_rank != rank[int(grid[cell])]:
                    return
            solutions.append(selected)
            return
        color = colors_by_difficulty[index]
        for option in options_by_color[color]:
            partial[color] = option
            backtrack(index + 1)
            if require_unique and len(solutions) >= SOLVER_MAX_SOLUTIONS:
                return
        partial.pop(color, None)

    backtrack(0)
    assert solutions, "failed to recover template copies"
    if require_unique:
        assert len(solutions) == 1, "expected a unique reconstruction"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve overlapping colored template copies using the top-row legend priority order.

    Input structure:
    - The top row contains one contiguous three-pixel color legend on a black background.
    - Elsewhere there is one small grey template and several colored placements built from that template under rotation or reflection.
    - Different colors may claim some of the same template cells, while the legend colors give the only ordering cue.

    Rule:
    - Read the legend colors from left to right as low-to-high priority.
    - Extract the grey template mask, enumerate its dihedral variants, and reconstruct every 1-2 copy for each legend color.
    - For each claimed cell, keep the highest-priority color among all reconstructed copies that cover it.
    - Leave the legend, the grey template, and the black background unchanged.

    Key insight:
    - The legend does not name positions; it names the overwrite order needed to resolve conflicts between reconstructed template instances.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    order, _legend_cells = _legend_order(g)
    selected = _solve_placements(g, require_unique=False)
    _min_rank, max_rank = _render_rank_maps(order, selected)
    out = g.copy()
    for cell, high_rank in max_rank.items():
        out[cell] = order[high_rank]
    return out


def _all_placements(variants: list[Shape], h: int, w: int) -> list[Placement]:
    seen: set[Placement] = set()
    placements: list[Placement] = []
    for variant in variants:
        vh, vw = _shape_dims(variant)
        for r0 in range(1, h - vh + 1):
            for c0 in range(w - vw + 1):
                placement = _place_shape(variant, r0, c0)
                if placement not in seen:
                    seen.add(placement)
                    placements.append(placement)
    placements.sort(key=lambda p: (_placement_anchor(p), len(p), p))
    return placements


def _valid_nonoverlap(placement: Placement, existing: list[tuple[int, Placement]], template_cells: set[Coord]) -> bool:
    placement_set = set(placement)
    if placement_set & template_cells:
        return False
    for _rank, other in existing:
        if placement_set & set(other):
            return False
        if _touches(placement, other):
            return False
    return True


def _valid_overlap(
    placement: Placement,
    base: Placement,
    base_rank: int,
    new_rank: int,
    existing: list[tuple[int, Placement]],
    template_cells: set[Coord],
    overlap_size: int,
) -> bool:
    placement_set = set(placement)
    if placement_set & template_cells:
        return False
    base_set = set(base)
    shared = placement_set & base_set
    if len(shared) != overlap_size or len(shared) >= len(placement):
        return False
    for rank, other in existing:
        other_set = set(other)
        other_shared = placement_set & other_set
        if other == base:
            continue
        if other_shared:
            return False
        if _touches(placement, other):
            return False
    if new_rank <= base_rank:
        return False
    return True


def _render_input_output(
    shape: Shape,
    order: list[int],
    template_origin: Coord,
    legend_start: int,
    placements_by_rank: dict[int, list[Placement]],
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid_in = np.zeros((h, w), dtype=int)
    grid_out = np.zeros((h, w), dtype=int)
    for i, color in enumerate(order):
        grid_in[0, legend_start + i] = color
        grid_out[0, legend_start + i] = color
    tr, tc = template_origin
    for dr, dc in shape:
        grid_in[tr + dr, tc + dc] = GREY
        grid_out[tr + dr, tc + dc] = GREY
    all_selected = {order[rank]: tuple(placements_by_rank[rank]) for rank in range(3)}
    min_rank, max_rank = _render_rank_maps(order, all_selected)
    for cell, low_rank in min_rank.items():
        grid_in[cell] = order[low_rank]
    for cell, high_rank in max_rank.items():
        grid_out[cell] = order[high_rank]
    return grid_in, grid_out


def _overlap_cells(placements_by_rank: dict[int, list[Placement]]) -> set[Coord]:
    counts: dict[Coord, int] = {}
    for placements in placements_by_rank.values():
        for placement in placements:
            for cell in placement:
                counts[cell] = counts.get(cell, 0) + 1
    return {cell for cell, count in counts.items() if count > 1}


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    order = random.sample(PALETTE, 3)
    legend_start = random.randint(0, w - 3)
    shape = random.choice(TEMPLATE_SHAPES)
    variants = _dihedral_variants(shape)
    all_placements = _all_placements(variants, h, w)
    counts = random.choice(COUNT_PATTERNS)
    template_h, template_w = _shape_dims(shape)

    for _ in range(80):
        tr = random.randint(2, h - template_h)
        tc = random.randint(0, w - template_w)
        template = _place_shape(shape, tr, tc)
        template_cells = set(template)
        if any((0, c) in template_cells for c in range(legend_start, legend_start + 3)):
            continue
        break
    else:
        return None

    placements_by_rank: dict[int, list[Placement]] = {0: [], 1: [], 2: []}
    existing: list[tuple[int, Placement]] = []

    for _ in range(counts[0]):
        choices = [p for p in all_placements if _valid_nonoverlap(p, existing, template_cells)]
        if not choices:
            return None
        placement = random.choice(choices)
        placements_by_rank[0].append(placement)
        existing.append((0, placement))

    planned_pairs: dict[int, list[tuple[int, int]]] = {1: [], 2: []}
    possible_pairs = []
    if counts[1] > 0 and counts[0] > 0:
        possible_pairs.append((1, 0))
    if counts[2] > 0 and counts[0] > 0:
        possible_pairs.append((2, 0))
    if counts[2] > 0 and counts[1] > 0:
        possible_pairs.append((2, 1))
    if not possible_pairs:
        return None
    pair_count = 1
    if sum(counts) >= 5 and len(possible_pairs) >= 2 and random.random() < 0.35:
        pair_count = 2
    random.shuffle(possible_pairs)
    higher_used: set[tuple[int, int]] = set()
    overlap_budget = random.randint(pair_count, 3)
    for pair_index in range(pair_count):
        higher_rank, lower_rank = possible_pairs[pair_index]
        copy_slot = random.randrange(counts[higher_rank])
        if (higher_rank, copy_slot) in higher_used:
            continue
        higher_used.add((higher_rank, copy_slot))
        size = 1 if pair_index == pair_count - 1 else random.randint(1, overlap_budget - (pair_count - pair_index - 1))
        overlap_budget -= size
        planned_pairs[higher_rank].append((copy_slot, lower_rank, size))

    for rank in (1, 2):
        overlap_by_slot = {slot: (lower_rank, size) for slot, lower_rank, size in planned_pairs[rank]}
        for slot in range(counts[rank]):
            if slot in overlap_by_slot:
                lower_rank, overlap_size = overlap_by_slot[slot]
                lower_base = random.choice(placements_by_rank[lower_rank])
                choices = [
                    p
                    for p in all_placements
                    if _valid_overlap(p, lower_base, lower_rank, rank, existing, template_cells, overlap_size)
                    and all(set(p) & set(other) == set() for other in placements_by_rank[rank])
                ]
                if not choices:
                    return None
                placement = random.choice(choices)
            else:
                choices = [
                    p
                    for p in all_placements
                    if _valid_nonoverlap(p, existing, template_cells)
                    and all(set(p) & set(other) == set() for other in placements_by_rank[rank])
                ]
                if not choices:
                    return None
                placement = random.choice(choices)
            placements_by_rank[rank].append(placement)
            existing.append((rank, placement))

    overlap_cells = _overlap_cells(placements_by_rank)
    if not (1 <= len(overlap_cells) <= 3):
        return None

    grid_in, grid_out = _render_input_output(shape, order, (tr, tc), legend_start, placements_by_rank, h, w)
    if np.array_equal(grid_in, grid_out):
        return None

    for rank, color in enumerate(order):
        visible = {(int(r), int(c)) for r, c in np.argwhere(grid_in == color)}
        if any(not (set(placement) & visible) for placement in placements_by_rank[rank]):
            return None

    try:
        solved = _solve_placements(grid_in, require_unique=True)
    except AssertionError:
        return None
    predicted = grid_in.copy()
    _min_rank, max_rank = _render_rank_maps(order, solved)
    for cell, high_rank in max_rank.items():
        predicted[cell] = order[high_rank]
    if not np.array_equal(predicted, grid_out):
        return None
    return grid_in


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
