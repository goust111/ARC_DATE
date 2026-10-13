from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location
import random
from itertools import combinations


Coord = tuple[int, int]
Copy = tuple[int, frozenset[Coord]]


def _template_offsets(g: np.ndarray, color: int) -> tuple[Coord, ...]:
    coords = np.argwhere(g == color)
    assert coords.size > 0, "template color must be present"
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _candidate_copies(
    g: np.ndarray, bg: int, color: int, template: tuple[Coord, ...]
) -> tuple[Copy, ...]:
    h, w = g.shape
    pixels = tuple(map(tuple, np.argwhere(g == color)))
    pixel_index = {pixel: i for i, pixel in enumerate(pixels)}
    seen_anchors: set[Coord] = set()
    copies: list[Copy] = []

    for pr, pc in pixels:
        for tr, tc in template:
            ar, ac = int(pr - tr), int(pc - tc)
            if (ar, ac) in seen_anchors:
                continue
            seen_anchors.add((ar, ac))

            cells = frozenset((ar + r, ac + c) for r, c in template)
            if not all(0 <= r < h and 0 <= c < w for r, c in cells):
                continue
            if not all(int(g[r, c]) in (bg, color) for r, c in cells):
                continue

            cover = 0
            for pixel in pixels:
                if pixel in cells:
                    cover |= 1 << pixel_index[pixel]
            if cover.bit_count() >= 2:
                copies.append((cover, cells))

    assert copies, f"expected at least one anchored template copy for color {color}"
    return tuple(copies)


def _completed_cells(g: np.ndarray, bg: int, color: int, template: tuple[Coord, ...]) -> frozenset[Coord]:
    pixels = tuple(map(tuple, np.argwhere(g == color)))
    assert len(pixels) >= 2, "each incomplete fragment color must provide at least two anchor pixels"

    copies = _candidate_copies(g, bg, color, template)
    by_first_pixel: list[list[Copy]] = [[] for _ in pixels]
    for copy in copies:
        cover, _cells = copy
        for i in range(len(pixels)):
            if cover & (1 << i):
                by_first_pixel[i].append(copy)

    full_cover = (1 << len(pixels)) - 1
    best_len: int | None = None
    solutions: list[tuple[Copy, ...]] = []

    def search(remaining: int, used_cells: frozenset[Coord], chosen: list[Copy]) -> None:
        nonlocal best_len, solutions
        if remaining == 0:
            solution = tuple(chosen)
            if best_len is None or len(solution) < best_len:
                best_len = len(solution)
                solutions = [solution]
            elif len(solution) == best_len:
                solutions.append(solution)
            return

        if best_len is not None and len(chosen) >= best_len:
            return

        first = (remaining & -remaining).bit_length() - 1
        for cover, cells in by_first_pixel[first]:
            if cover & ~remaining:
                continue
            if cells & used_cells:
                continue
            search(remaining & ~cover, used_cells | cells, chosen + [(cover, cells)])

    search(full_cover, frozenset(), [])
    assert solutions, f"expected a unique template-cover assignment for color {color}"

    completed: set[frozenset[Coord]] = set()
    for solution in solutions:
        cells: set[Coord] = set()
        for _cover, copy_cells in solution:
            cells.update(copy_cells)
        completed.add(frozenset(cells))

    assert len(completed) == 1, f"template-cover assignment must be unique for color {color}"
    return next(iter(completed))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every colored fragment into a translated copy of the unique largest foreground template.

    Input structure
    - A uniform background contains one complete monochromatic template mask and smaller same-orientation fragments in other colors.
    - A color may contain more than one separated fragment; each fragment has at least two pixels.
    - All completed copies fit inside the grid and do not overlap other foreground pixels.

    Rule
    - Use the unique foreground color with the most pixels as the template mask.
    - For each other color, find the unique minimum set of translated template placements whose existing template cells cover that color's input pixels.
    - Paint every cell of those placements with that color, leaving the template and background otherwise unchanged.

    Key insight
    The incomplete colors are partial masks of the same template; their existing pixels act as anchors that determine where the missing cells must be filled.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected foreground colors"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    template_area = max(counts.values())
    template_colors = [color for color, area in counts.items() if area == template_area]
    assert len(template_colors) == 1, "largest foreground template color must be unique"
    template_color = template_colors[0]
    template = _template_offsets(g, template_color)

    out = g.copy()
    painted: frozenset[Coord] = frozenset()
    for color in colors:
        if color == template_color:
            continue
        cells = _completed_cells(g, bg, color, template)
        assert not (cells & painted), "completed copies of different colors must not overlap"
        for r, c in cells:
            out[r, c] = color
        painted = painted | cells

    return out

# --------------- generator (o5x) ---------------

# Canvas/template/density bounds.
GRID_SIDE_RANGE, TEMPLATE_AREA_RANGE = (22, 30), (5, 8)
TEMPLATE_HEIGHT_CHOICES, TEMPLATE_WIDTH_CHOICES = (3, 4), (3, 4, 5)
ROTATED_HEIGHT_RANGE, ROTATED_WIDTH_RANGE = (3, 4), (3, 5)
OUTPUT_DENSITY_LIMIT, INPUT_DENSITY_LIMIT, TEMPLATE_BUILD_RETRIES = 0.10, 0.07, 80
# Template validation.
PROTECTED_TEMPLATE_CELL, TEMPLATE_SEED_CELL = (0, 0), (1, 1)
ROTATED_TEMPLATE_KS = (1, 3)
TEMPLATE_CONNECTIVITY, DIAGONAL_DISCONNECTIVITY = 8, 4
# Copy/color structure.
PALETTE_COLORS, TOTAL_COPY_RANGE, FRAGMENT_COUNT_RANGE = tuple(range(10)), range(5, 10), (4, 7)
FORCE_SINGLE_COPY_TOTAL, FORCE_DUPLICATE_TOTAL, DUPLICATE_COLOR_PROB = 5, 9, 0.20

# Fragment anchor sampling.
ANCHOR_COUNTS = (2, 3)
SINGLE_COPY_THREE_ANCHOR_PROB, MULTI_COPY_THREE_ANCHOR_PROB = 0.45, 0.50
MULTI_COPY_THREE_ANCHOR_AREA_THRESHOLD = 5

# Placement spacing and retry budget.
PLACEMENT_BORDER, PLACEMENT_PADDING, PLACEMENT_PADDING_CONNECTIVITY = 1, 2, 8
GENERATION_RETRIES = 300


def _gen_normalized(coords):
    coord_array = np.array(coords, dtype=int)
    r0, c0, _r1, _c1 = bbox_coords(coord_array)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coord_array))


def _gen_is_asymmetric(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            return False
        if np.array_equal(mask, np.rot90(mask)):
            return False
        if np.array_equal(mask, np.fliplr(mask).T):
            return False
    return True


def _gen_template_variants(mask: np.ndarray):
    variants = [mask, np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)]
    for k in ROTATED_TEMPLATE_KS:
        rotated = np.rot90(mask, k)
        h_ok = ROTATED_HEIGHT_RANGE[0] <= rotated.shape[0] <= ROTATED_HEIGHT_RANGE[1]
        w_ok = ROTATED_WIDTH_RANGE[0] <= rotated.shape[1] <= ROTATED_WIDTH_RANGE[1]
        if h_ok and w_ok:
            variants.append(rotated)
    random.shuffle(variants)
    return variants


def _gen_valid_template_variant(mask: np.ndarray) -> bool:
    """Accept masks that match the solver-facing template connectivity contract."""
    if len(components_mask(mask, connectivity=TEMPLATE_CONNECTIVITY)) != 1:
        return False
    if len(components_mask(mask, connectivity=DIAGONAL_DISCONNECTIVITY)) <= 1:
        return False
    return _gen_is_asymmetric(mask)


def _gen_template_mask(area: int) -> np.ndarray | None:
    for _ in range(TEMPLATE_BUILD_RETRIES):
        h = random.choice(TEMPLATE_HEIGHT_CHOICES)
        w = random.choice(TEMPLATE_WIDTH_CHOICES)
        cells = [PROTECTED_TEMPLATE_CELL, TEMPLATE_SEED_CELL]
        r, c = TEMPLATE_SEED_CELL
        while r < h - 1 or c < w - 1:
            moves = []
            if r < h - 1 and c < w - 1:
                moves.append((1, 1))
            if r < h - 1:
                moves.append((1, 0))
            if c < w - 1:
                moves.append((0, 1))
            dr, dc = random.choice(moves)
            r += dr
            c += dc
            cells.append((r, c))

        cell_set = set(cells)
        if len(cell_set) > area:
            continue

        pr, pc = PROTECTED_TEMPLATE_CELL
        while len(cell_set) < area:
            candidates = []
            for rr in range(h):
                for cc in range(w):
                    if (rr, cc) in cell_set:
                        continue
                    if abs(rr - pr) + abs(cc - pc) == 1:
                        continue
                    if any(max(abs(rr - r0), abs(cc - c0)) == 1 for r0, c0 in cell_set):
                        candidates.append((rr, cc))
            if not candidates:
                break
            cell_set.add(random.choice(candidates))

        if len(cell_set) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        rows, cols = zip(*cell_set)
        mask[rows, cols] = True
        spans_bbox = mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        if not spans_bbox:
            continue
        for variant in _gen_template_variants(mask):
            if _gen_valid_template_variant(variant):
                return variant.copy()
    return None


def _gen_unique_anchor_choices(mask: np.ndarray, count: int):
    offsets = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    sig_counts = {}
    combos = []
    for combo in combinations(offsets, count):
        sig = _gen_normalized(combo)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
        combos.append((combo, sig))
    return [combo for combo, sig in combos if sig_counts[sig] == 1]


def _gen_pick_dimensions_and_counts():
    side = random.randint(*GRID_SIDE_RANGE)
    area = random.randint(*TEMPLATE_AREA_RANGE)
    valid_totals = [n for n in TOTAL_COPY_RANGE if n * area <= int(OUTPUT_DENSITY_LIMIT * side * side)]
    if not valid_totals:
        return None
    total_copies = random.choice(valid_totals)
    if total_copies == FORCE_SINGLE_COPY_TOTAL:
        duplicate_color = False
    elif total_copies == FORCE_DUPLICATE_TOTAL:
        duplicate_color = True
    else:
        duplicate_color = random.random() < DUPLICATE_COLOR_PROB
    fragment_count = total_copies - 1 - int(duplicate_color)
    if not FRAGMENT_COUNT_RANGE[0] <= fragment_count <= FRAGMENT_COUNT_RANGE[1]:
        return None
    return side, area, total_copies, fragment_count, duplicate_color


def _gen_pick_colors(fragment_count: int, duplicate_color: bool):
    colors = list(PALETTE_COLORS)
    bg = random.choice(colors)
    fg_colors = [c for c in colors if c != bg]
    random.shuffle(fg_colors)
    template_color = fg_colors[0]
    fragment_colors = fg_colors[1 : 1 + fragment_count]
    fragment_copy_colors = list(fragment_colors)
    if duplicate_color:
        fragment_copy_colors.append(random.choice(fragment_colors))
    random.shuffle(fragment_copy_colors)
    copy_colors = [template_color] + fragment_copy_colors
    return bg, template_color, fragment_colors, copy_colors


def _gen_place_copies(side: int, bg: int, template_mask: np.ndarray, copy_colors: list[int]):
    out = np.full((side, side), bg, dtype=int)
    placements = []
    for color in copy_colors:
        patch = np.full(template_mask.shape, bg, dtype=int)
        patch[template_mask] = color
        try:
            r0, c0 = random_free_location(
                out,
                patch,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(out, patch, r0, c0, transparent=bg)
        placements.append((color, r0, c0))
    return out, placements


def _gen_apply_fragment_anchors(inp, bg, area, template_mask, placements, fragment_colors, anchor_choices):
    offsets = [tuple(map(int, rc)) for rc in np.argwhere(template_mask)]
    placements_by_color = {color: [] for color in fragment_colors}
    for color, r0, c0 in placements[1:]:
        placements_by_color[color].append((r0, c0))

    for color_placements in placements_by_color.values():
        if len(color_placements) == 1:
            count_list = [3 if anchor_choices[3] and random.random() < SINGLE_COPY_THREE_ANCHOR_PROB else 2]
        elif area > MULTI_COPY_THREE_ANCHOR_AREA_THRESHOLD and anchor_choices[3] and random.random() < MULTI_COPY_THREE_ANCHOR_PROB:
            count_list = [2, 3]
            random.shuffle(count_list)
        else:
            count_list = [2, 2]
            random.shuffle(count_list)
        for (r0, c0), anchor_count in zip(color_placements, count_list):
            choices = anchor_choices[anchor_count] or anchor_choices[2]
            chosen = set(random.choice(choices))
            for dr, dc in offsets:
                if (dr, dc) not in chosen:
                    inp[r0 + dr, c0 + dc] = bg


def _gen_attempt() -> np.ndarray | None:
    dims = _gen_pick_dimensions_and_counts()
    if dims is None:
        return None
    side, area, _total_copies, fragment_count, duplicate_color = dims

    template_mask = _gen_template_mask(area)
    if template_mask is None:
        return None

    anchor_choices = {count: _gen_unique_anchor_choices(template_mask, count) for count in ANCHOR_COUNTS}
    if not anchor_choices[2]:
        return None

    bg, template_color, fragment_colors, copy_colors = _gen_pick_colors(fragment_count, duplicate_color)

    placed = _gen_place_copies(side, bg, template_mask, copy_colors)
    if placed is None:
        return None
    out, placements = placed

    inp = out.copy()
    _gen_apply_fragment_anchors(inp, bg, area, template_mask, placements, fragment_colors, anchor_choices)

    if np.count_nonzero(inp != bg) >= INPUT_DENSITY_LIMIT * side * side:
        return None
    counts = {color: int(np.count_nonzero(inp == color)) for color in copy_colors}
    if counts[template_color] <= max(counts[color] for color in fragment_colors):
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate fd096ab6 input")
