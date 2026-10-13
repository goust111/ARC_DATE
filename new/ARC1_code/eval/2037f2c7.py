import numpy as np

from arc_utils import Color, bbox, crop_bbox, find_components, get_bg, paste
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Compare the two same-sized motifs and output the cropped mask of cells missing from the damaged copy.

    Input structure
    - The black background contains exactly two 4-connected multicolor foreground objects.
    - The two objects have the same rectangular bbox shape and represent aligned copies of the same motif.
    - One object is complete; the other is a damaged copy with only foreground cells removed.

    Rule
    - Align the two object crops by their bbox top-left corners.
    - Select the cells that are foreground in the complete crop and background in the damaged crop.
    - Crop that missing-cell mask to its own bbox, recolor selected cells teal, and leave other cells black.

    Key insight
    - The output is not one of the colored motifs; it is the shape of the damage after subtracting the smaller motif mask from the larger matching motif mask.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) == 2, "expected exactly two motif objects"

    crops = [crop_bbox(g, component["bbox"]) for component in components]
    assert crops[0].shape == crops[1].shape, "motif bboxes must have the same shape"

    candidates = []
    for complete, damaged in ((crops[0], crops[1]), (crops[1], crops[0])):
        complete_mask = complete != bg
        damaged_mask = damaged != bg
        missing = complete_mask & ~damaged_mask
        extra = damaged_mask & ~complete_mask
        overlap = complete_mask & damaged_mask
        if missing.any() and not extra.any() and np.array_equal(complete[overlap], damaged[overlap]):
            candidates.append(missing)

    assert len(candidates) == 1, "expected a unique complete-to-damaged mask difference"
    missing = candidates[0]
    missing_bbox = bbox(missing)
    assert missing_bbox is not None, "expected at least one missing cell"

    cropped_missing = crop_bbox(missing, missing_bbox)
    out = np.zeros(cropped_missing.shape, dtype=int)
    out[cropped_missing] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------

# Missing-mask dimensions and occupancy.
MISSING_H_RANGE = (2, 4)
MISSING_W_RANGE = (6, 10)
MISSING_CELL_RANGE = (6, 13)
SINGLE_MISSING_MAX_W = 8
SINGLE_MISSING_PROB = 0.35

# Two-component missing-mask split geometry.
TWO_MISSING_GAP_CHOICES = (1, 2)
TWO_MISSING_WIDE_GAP_MIN_W = 8
TWO_MISSING_MIN_SIDE_W = 2

# Motif bbox, foreground density, and missing-cell fraction constraints.
MOTIF_H_RANGE = (8, 15)
MOTIF_W_RANGE = (6, 10)
MOTIF_DENSITY_RANGE = (0.67, 0.92)
MISSING_FRACTION_RANGE = (0.08, 0.16)
NEAR_GEOMETRY_LIMIT = 20

# Layered motif coloring.
MOTIF_PALETTE = (1, 2, 3, 4, 6, 7)
MOTIF_COLOR_COUNT_RANGE = (4, 6)
MOTIF_BAND_HEIGHTS = (2, 3)

# Canvas placement and retry limits.
CANVAS_H_RANGE = (22, 25)
CANVAS_W_RANGE = (22, 25)
PLACEMENT_GAP_RANGE = (3, 6)
GENERATION_ATTEMPTS = 300

FOUR_NEIGHBOR_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HORIZONTAL = "h"
VERTICAL = "v"


def _gen_mask_from_cells(h: int, w: int, cells: set[tuple[int, int]], col_offset: int = 0) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c + col_offset] = True
    return mask


def _gen_grow_cells(h: int, w: int, size: int, seeds: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    cells = set(seeds)
    if len(cells) > size:
        return None
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for dr, dc in FOUR_NEIGHBOR_DELTAS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _gen_single_missing(h: int, w: int, count: int) -> np.ndarray | None:
    base_r = random.randrange(h)
    branch_c = random.randrange(w)
    seeds = {(base_r, c) for c in range(w)}
    seeds.update((r, branch_c) for r in range(h))
    cells = _gen_grow_cells(h, w, count, seeds)
    if cells is None:
        return None
    return _gen_mask_from_cells(h, w, cells)


def _gen_two_missing(h: int, w: int, count: int) -> np.ndarray | None:
    gap_w = random.choice(TWO_MISSING_GAP_CHOICES) if w >= TWO_MISSING_WIDE_GAP_MIN_W else 1
    lo_gap = TWO_MISSING_MIN_SIDE_W
    hi_gap = w - gap_w - TWO_MISSING_MIN_SIDE_W
    if lo_gap > hi_gap:
        return None
    gap_c = random.randint(lo_gap, hi_gap)
    left_w = gap_c
    right_c0 = gap_c + gap_w
    right_w = w - right_c0

    max_left = h * left_w
    max_right = h * right_w
    lo_left = max(1, count - max_right)
    hi_left = min(max_left, count - 1)
    if lo_left > hi_left:
        return None
    left_count = random.randint(lo_left, hi_left)
    right_count = count - left_count

    top_on_left = random.random() < 0.5
    left_seed = (0 if top_on_left else h - 1, 0)
    right_seed = (h - 1 if top_on_left else 0, right_w - 1)
    left_cells = _gen_grow_cells(h, left_w, left_count, {left_seed})
    right_cells = _gen_grow_cells(h, right_w, right_count, {right_seed})
    if left_cells is None or right_cells is None:
        return None

    mask = _gen_mask_from_cells(h, w, left_cells)
    mask |= _gen_mask_from_cells(h, w, right_cells, col_offset=right_c0)
    return mask


def _gen_missing_mask() -> tuple[np.ndarray, int] | None:
    h = random.randint(*MISSING_H_RANGE)
    w = random.randint(*MISSING_W_RANGE)
    use_single = w <= SINGLE_MISSING_MAX_W and random.random() < SINGLE_MISSING_PROB
    if use_single:
        lo = max(MISSING_CELL_RANGE[0], w + h - 1)
        hi = min(MISSING_CELL_RANGE[1], h * w)
        if lo <= hi:
            count = random.randint(lo, hi)
            mask = _gen_single_missing(h, w, count)
            if mask is not None:
                return mask, 1

    count = random.randint(MISSING_CELL_RANGE[0], min(MISSING_CELL_RANGE[1], h * w))
    mask = _gen_two_missing(h, w, count)
    if mask is None:
        return None
    return mask, 2


def _gen_component_count(mask: np.ndarray) -> int:
    return len(find_components(mask.astype(int), bg=Color.BLACK, connectivity=4, monochromatic=False))


def _gen_margin_options(
    motif_h: int,
    motif_w: int,
    missing: np.ndarray,
    row_offset: int,
    col_offset: int,
    n_components: int,
) -> list[list[int]]:
    mh, mw = missing.shape
    missing_by_row = {row_offset + r: [] for r in range(mh)}
    for r, c in zip(*np.nonzero(missing)):
        missing_by_row[row_offset + r].append(col_offset + c)

    options = []
    max_margin = (motif_w - 1) // 2
    for r in range(motif_h):
        allowed = max_margin
        if r in missing_by_row and missing_by_row[r]:
            allowed = min(allowed, min(min(c, motif_w - 1 - c) for c in missing_by_row[r]))
        if n_components == 1 and row_offset <= r < row_offset + mh:
            allowed = 0
        options.append(list(range(allowed + 1)))
    return options


def _gen_possible_margin_sums(options: list[list[int]], motif_w: int) -> set[int]:
    sums = {0}
    for row_options in options:
        sums = {total + motif_w - 2 * margin for total in sums for margin in row_options}
    return sums


def _gen_choose_margins(options: list[list[int]], motif_w: int, target_fg: int) -> list[int] | None:
    states: dict[int, list[int]] = {0: []}
    for row_options in options:
        shuffled = list(row_options)
        random.shuffle(shuffled)
        next_states: dict[int, list[int]] = {}
        sums = list(states.items())
        random.shuffle(sums)
        for total, margins in sums:
            for margin in shuffled:
                new_total = total + motif_w - 2 * margin
                if new_total <= target_fg and new_total not in next_states:
                    next_states[new_total] = margins + [margin]
        states = next_states
        if not states:
            return None
    return states.get(target_fg)


def _gen_width_options(missing_w: int, n_components: int) -> list[int]:
    if n_components == 1:
        return list(range(missing_w + 2, MOTIF_W_RANGE[1] + 1))
    return [missing_w]


def _gen_foreground_bounds(area: int, missing_count: int) -> tuple[int, int]:
    min_density, max_density = MOTIF_DENSITY_RANGE
    min_missing_frac, max_missing_frac = MISSING_FRACTION_RANGE
    min_fg = max(math.ceil(min_density * area), math.ceil(missing_count / max_missing_frac), missing_count + 1)
    max_fg = min(math.floor(max_density * area), math.floor(missing_count / min_missing_frac), area)
    return min_fg, max_fg


def _gen_pick_geometry(missing: np.ndarray, n_components: int) -> tuple[int, int, int, int, list[int]] | None:
    mh, mw = missing.shape
    missing_count = int(missing.sum())
    width_options = _gen_width_options(mw, n_components)
    random.shuffle(width_options)

    configs = []
    target_density = random.uniform(*MOTIF_DENSITY_RANGE)
    for motif_w in width_options:
        if not (MOTIF_W_RANGE[0] <= motif_w <= MOTIF_W_RANGE[1]):
            continue
        col_offsets = [0] if n_components == 2 else list(range(1, motif_w - mw))
        for motif_h in range(MOTIF_H_RANGE[0], MOTIF_H_RANGE[1] + 1):
            if motif_h < mh + 2:
                continue
            area = motif_h * motif_w
            min_fg, max_fg = _gen_foreground_bounds(area, missing_count)
            if min_fg > max_fg:
                continue
            for row_offset in range(1, motif_h - mh):
                for col_offset in col_offsets:
                    margin_options = _gen_margin_options(motif_h, motif_w, missing, row_offset, col_offset, n_components)
                    possible_fg = _gen_possible_margin_sums(margin_options, motif_w)
                    for fg in range(min_fg, max_fg + 1):
                        if fg in possible_fg:
                            density = fg / area
                            configs.append((abs(density - target_density), motif_h, motif_w, row_offset, col_offset, margin_options, fg))

    if not configs:
        return None
    configs.sort(key=lambda item: item[0])
    near = configs[: max(1, min(NEAR_GEOMETRY_LIMIT, len(configs)))]
    _score, motif_h, motif_w, row_offset, col_offset, margin_options, target_fg = random.choice(near)
    margins = _gen_choose_margins(margin_options, motif_w, target_fg)
    if margins is None:
        return None
    return motif_h, motif_w, row_offset, col_offset, margins


def _gen_color_motif(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    colors = random.sample(MOTIF_PALETTE, random.randint(*MOTIF_COLOR_COUNT_RANGE))
    phase = random.randrange(len(colors))
    band_h = random.choice(MOTIF_BAND_HEIGHTS)
    motif = np.zeros((h, w), dtype=int)
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            continue
        left = int(cols[0])
        right = int(cols[-1])
        row_band = r // band_h
        for c in range(w):
            if mask[r, c]:
                d = min(c - left, right - c)
                motif[r, c] = colors[(d + row_band + phase) % len(colors)]
    return motif


def _gen_foreground_connected(sprite: np.ndarray) -> bool:
    comps = find_components(sprite, bg=Color.BLACK, connectivity=4, monochromatic=False)
    return len(comps) == 1


def _gen_build_pair(missing: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray] | None:
    geometry = _gen_pick_geometry(missing, n_components)
    if geometry is None:
        return None
    motif_h, motif_w, row_offset, col_offset, margins = geometry

    complete_mask = np.zeros((motif_h, motif_w), dtype=bool)
    for r, margin in enumerate(margins):
        complete_mask[r, margin : motif_w - margin] = True
    missing_abs = np.zeros((motif_h, motif_w), dtype=bool)
    missing_abs[row_offset : row_offset + missing.shape[0], col_offset : col_offset + missing.shape[1]] = missing
    if np.any(~complete_mask & missing_abs):
        return None

    complete = _gen_color_motif(complete_mask)
    damaged = complete.copy()
    damaged[missing_abs] = 0

    if not _gen_foreground_connected(complete) or not _gen_foreground_connected(damaged):
        return None
    if complete.shape != damaged.shape:
        return None
    return complete, damaged


def _gen_placement_options(obj_h: int, obj_w: int) -> list[tuple[str, int, int, int]]:
    grids = [(r, c) for r in range(CANVAS_H_RANGE[0], CANVAS_H_RANGE[1] + 1) for c in range(CANVAS_W_RANGE[0], CANVAS_W_RANGE[1] + 1)]
    random.shuffle(grids)
    options = []
    for grid_h, grid_w in grids:
        for gap in range(PLACEMENT_GAP_RANGE[0], PLACEMENT_GAP_RANGE[1] + 1):
            if 2 * obj_w + gap <= grid_w:
                options.append((HORIZONTAL, grid_h, grid_w, gap))
            if 2 * obj_h + gap <= grid_h:
                options.append((VERTICAL, grid_h, grid_w, gap))
    return options


def _gen_place_pair(complete: np.ndarray, damaged: np.ndarray) -> np.ndarray | None:
    h, w = complete.shape
    options = _gen_placement_options(h, w)
    if not options:
        return None

    orient, grid_h, grid_w, gap = random.choice(options)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    first, second = (complete, damaged) if random.random() < 0.5 else (damaged, complete)
    if orient == HORIZONTAL:
        c0 = random.randint(0, grid_w - (2 * w + gap))
        c1 = c0 + w + gap
        r0 = random.randint(0, grid_h - h)
        r1 = random.randint(0, grid_h - h)
    else:
        r0 = random.randint(0, grid_h - (2 * h + gap))
        r1 = r0 + h + gap
        c0 = random.randint(0, grid_w - w)
        c1 = random.randint(0, grid_w - w)
    paste(grid, first, r0, c0, transparent=Color.BLACK)
    paste(grid, second, r1, c1, transparent=Color.BLACK)
    return grid


def _gen_expected_output(missing: np.ndarray) -> np.ndarray:
    expected = np.zeros(missing.shape, dtype=int)
    expected[missing] = Color.TEAL
    return expected


def _gen_oracle_accepts(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    return get_bg(inp) == Color.BLACK


def _gen_attempt() -> np.ndarray | None:
    missing_info = _gen_missing_mask()
    if missing_info is None:
        return None
    missing, n_components = missing_info
    if _gen_component_count(missing) != n_components:
        return None

    pair = _gen_build_pair(missing, n_components)
    if pair is None:
        return None
    complete, damaged = pair
    inp = _gen_place_pair(complete, damaged)
    if inp is None:
        return None

    if not _gen_oracle_accepts(inp, _gen_expected_output(missing)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 2037f2c7 input")
