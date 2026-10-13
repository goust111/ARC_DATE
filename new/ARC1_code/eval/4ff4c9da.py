import numpy as np

from arc_utils import Color, bbox_coords, colors_present, find_components
import random
from collections import defaultdict


def _normalized_mask(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    shifted = coords - np.array([r0, c0])
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every object whose shape matches a teal template object to teal.

    Input structure
    - A black-background lattice contains many monochrome connected objects in blue or red.
    - One or more teal objects are already drawn in places where such objects have been recolored.
    - Shape identity is translation-normalized and uses 8-connectivity, so diagonal contact can join one template.

    Rule
    - Treat each teal connected object as a shape template.
    - Among non-black, non-teal objects, find the unique source color whose object masks match those teal templates.
    - Recolor every object of that source color with a matching normalized mask to teal; preserve all other pixels.

    Key insight
    - The teal pixels are examples of completed objects, not a path or paint source; their masks select all same-shaped objects elsewhere in the lattice.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    marker = Color.TEAL

    template_components = find_components(g, bg=bg, connectivity=8, colors=[marker])
    assert template_components
    template_shapes = {_normalized_mask(comp["coords"]) for comp in template_components}
    assert template_shapes

    candidate_colors = colors_present(g, exclude={bg, marker})
    candidate_components = find_components(
        g,
        bg=bg,
        connectivity=8,
        colors=candidate_colors,
    )

    matches = [
        comp
        for comp in candidate_components
        if _normalized_mask(comp["coords"]) in template_shapes
    ]
    assert matches

    matched_shapes = {_normalized_mask(comp["coords"]) for comp in matches}
    assert template_shapes <= matched_shapes

    source_colors = {int(comp["color"]) for comp in matches}
    assert len(source_colors) == 1
    source_color = next(iter(source_colors))

    out = g.copy()
    for comp in candidate_components:
        assert comp["color"] is not None
        if int(comp["color"]) != source_color:
            continue
        coords = comp["coords"]
        if _normalized_mask(coords) in template_shapes:
            out[coords[:, 0], coords[:, 1]] = marker

    return out

# --------------- generator (o5x) ---------------

# Canvas and lattice structure.
GEN_GRID_SIZE_MIN, GEN_GRID_SIZE_MAX = 17, 27
GEN_CELL_COUNT_MIN, GEN_CELL_COUNT_MAX = 5, 8
GEN_CELL_WIDTH_MIN, GEN_CELL_WIDTH_MAX = 2, 3
GEN_LATTICE_GAP_WIDTH = 1

# Cell symbols and weighted within-cell patterns.
GEN_LATTICE_TYPE = "L"
GEN_MIXED_TYPE = "m"
GEN_SOURCE_TYPE = "s"
GEN_PATTERN_WEIGHTS = {
    2: ((("m", "m"), 3), (("m", "s"), 4), (("s", "m"), 4), (("s", "s"), 1)),
    3: ((("m", "m", "m"), 3), (("m", "m", "s"), 5), (("m", "s", "m"), 4),
        (("s", "m", "m"), 5), (("m", "s", "s"), 2), (("s", "m", "s"), 1),
        (("s", "s", "m"), 2)),
}
GEN_PATTERN_WIDTHS = (GEN_CELL_WIDTH_MIN, GEN_CELL_WIDTH_MAX)
GEN_GROUP3_PROB, GEN_GROUP2_PROB = 0.25, 0.65

# Optional source pixels inside source/source blocks.
GEN_SOURCE_OBJECT_AREA_MAX = 8
GEN_EMPTY_SS_PROB = 0.35
GEN_EMPTY_SS_TARGET_MAX = 5

# Source components, templates, and matches.
GEN_SOURCE_COUNT_MIN, GEN_SOURCE_COUNT_MAX = 19, 56
GEN_TARGET_COUNT_TOLERANCE = 2
GEN_OBJECT_BBOX_MAX = 3
GEN_TEMPLATE_AREA_MIN, GEN_TEMPLATE_AREA_MAX = 2, 5
GEN_TEMPLATE_COMP_MIN = 3
GEN_SINGLE_COMP_MIN, GEN_SINGLE_COMP_MAX = 4, 7
GEN_MATCH_COUNT_MIN, GEN_MATCH_COUNT_MAX = 3, 6
GEN_PAIR_PROB = 0.45

# Color proportions and retry policy.
GEN_LATTICE_FRAC_MIN, GEN_LATTICE_FRAC_MAX = 0.42, 0.66
GEN_SOURCE_FRAC_MIN, GEN_SOURCE_FRAC_MAX = 0.18, 0.35
GEN_MAX_ATTEMPTS = 300
GEN_TARGET_REFRESH_ATTEMPTS = {75, 150, 225}
GEN_PALETTE = (Color.BLUE, Color.RED)

GEN_K_TARGET_PREFERENCES = ((28, (5, 6)), (38, (6, 5, 7)), (48, (7, 6, 8)))
GEN_K_HIGH_TARGET_PREFERENCE = (8, 7)


def _gen_widths(n: int, k: int) -> list[int] | None:
    total_cell_width = n - GEN_LATTICE_GAP_WIDTH * (k - 1)
    n3 = total_cell_width - GEN_CELL_WIDTH_MIN * k
    if n3 < 0 or n3 > k:
        return None
    widths = [GEN_CELL_WIDTH_MAX] * n3 + [GEN_CELL_WIDTH_MIN] * (k - n3)
    random.shuffle(widths)
    return widths


def _gen_weighted_pattern(width: int) -> tuple[str, ...]:
    choices = GEN_PATTERN_WEIGHTS[width]
    total = sum(w for _pat, w in choices)
    pick = random.randrange(total)
    for pat, weight in choices:
        if pick < weight:
            return pat
        pick -= weight
    return choices[-1][0]


def _gen_cell_patterns(widths: list[int]) -> list[tuple[str, ...]]:
    patterns: list[tuple[str, ...] | None] = [None] * len(widths)
    for width in GEN_PATTERN_WIDTHS:
        idxs = [i for i, w in enumerate(widths) if w == width]
        random.shuffle(idxs)
        pos = 0
        while pos < len(idxs):
            remaining = len(idxs) - pos
            if remaining >= 3 and random.random() < GEN_GROUP3_PROB:
                group_size = GEN_CELL_WIDTH_MAX
            elif remaining >= 2 and random.random() < GEN_GROUP2_PROB:
                group_size = GEN_CELL_WIDTH_MIN
            else:
                group_size = 1
            pat = _gen_weighted_pattern(width)
            for idx in idxs[pos : pos + group_size]:
                patterns[idx] = pat
            pos += group_size
    return [pat for pat in patterns if pat is not None]


def _gen_sequence(widths: list[int], patterns: list[tuple[str, ...]]):
    seq: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for i, (width, pat) in enumerate(zip(widths, patterns)):
        spans.append((pos, pos + width))
        seq.extend(pat)
        pos += width
        if i != len(widths) - 1:
            seq.append(GEN_LATTICE_TYPE)
            pos += GEN_LATTICE_GAP_WIDTH
    return seq, spans


def _gen_adjacent_to(mask: np.ndarray, r: int, c: int) -> bool:
    r0, r1 = max(0, r - 1), min(mask.shape[0], r + 2)
    c0, c1 = max(0, c - 1), min(mask.shape[1], c + 2)
    return bool(mask[r0:r1, c0:c1].any())


def _gen_ss_groups(ss_mask: np.ndarray, diagonal: bool) -> list[list[tuple[int, int]]]:
    if not diagonal:
        return [[(int(r), int(c))] for r, c in np.argwhere(ss_mask)]

    groups: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(ss_mask):
        r, c = int(r), int(c)
        if r > c:
            continue
        group = [(r, c)] if r == c else [(r, c), (c, r)]
        if all(ss_mask[rr, cc] for rr, cc in group):
            groups.append(group)
    return groups


def _gen_optional_ss(base_source: np.ndarray, ss_mask: np.ndarray, diagonal: bool):
    extra = np.zeros_like(base_source, dtype=bool)
    max_add = min(
        int(ss_mask.sum()), GEN_SOURCE_OBJECT_AREA_MAX - int(base_source.sum())
    )
    if max_add <= 0:
        return extra

    if base_source.any():
        target = random.randint(0, max_add)
        current = base_source.copy()
    else:
        if random.random() > GEN_EMPTY_SS_PROB:
            return extra
        target = random.randint(1, min(max_add, GEN_EMPTY_SS_TARGET_MAX))
        current = np.zeros_like(base_source, dtype=bool)

    groups = _gen_ss_groups(ss_mask, diagonal)
    random.shuffle(groups)
    if not base_source.any():
        group = random.choice([g for g in groups if len(g) <= target])
        for r, c in group:
            extra[r, c] = True
            current[r, c] = True

    changed = True
    while int(extra.sum()) < target and changed:
        changed = False
        random.shuffle(groups)
        for group in groups:
            if any(extra[r, c] for r, c in group):
                continue
            if int(extra.sum()) + len(group) > target:
                continue
            if any(_gen_adjacent_to(current, r, c) for r, c in group):
                for r, c in group:
                    extra[r, c] = True
                    current[r, c] = True
                changed = True
                break
    return extra


def _gen_make_base_grid(n, seq, spans, lattice_color, source_color):
    grid = np.zeros((n, n), dtype=int)
    seq_arr = np.asarray(seq)
    is_lattice = seq_arr == GEN_LATTICE_TYPE
    is_mixed = seq_arr == GEN_MIXED_TYPE
    is_source = seq_arr == GEN_SOURCE_TYPE
    lattice_mask = is_lattice[:, None] | is_lattice[None, :] | (
        is_source[:, None] & is_source[None, :]
    )
    source_mask = (is_mixed[:, None] & is_source[None, :]) | (
        is_source[:, None] & is_mixed[None, :]
    )
    grid[lattice_mask] = lattice_color
    grid[source_mask] = source_color

    for i, (r0, r1) in enumerate(spans):
        row_pat = np.asarray(seq[r0:r1])
        for j in range(i, len(spans)):
            c0, c1 = spans[j]
            col_pat = np.asarray(seq[c0:c1])
            base_source = row_pat[:, None] != col_pat[None, :]
            ss_mask = (row_pat[:, None] == GEN_SOURCE_TYPE) & (
                col_pat[None, :] == GEN_SOURCE_TYPE
            )
            extra = _gen_optional_ss(base_source, ss_mask, i == j)
            if extra.any():
                block = grid[r0:r1, c0:c1]
                block[extra] = source_color
                if i != j:
                    transpose_block = grid[c0:c1, r0:r1]
                    transpose_block[extra.T] = source_color
    return grid


def _gen_components_by_shape(grid: np.ndarray, source_color: int) -> dict:
    groups = defaultdict(list)
    for comp in find_components(
        grid, bg=Color.BLACK, connectivity=8, colors=[source_color]
    ):
        groups[_normalized_mask(comp["coords"])].append(comp)
    return groups


def _gen_select_template_groups(groups: dict) -> list:
    eligible = [
        (shape, comps)
        for shape, comps in groups.items()
        if GEN_TEMPLATE_AREA_MIN <= len(shape) <= GEN_TEMPLATE_AREA_MAX
        and len(comps) >= GEN_TEMPLATE_COMP_MIN
    ]
    random.shuffle(eligible)

    singles = [
        (shape, comps)
        for shape, comps in eligible
        if GEN_SINGLE_COMP_MIN <= len(comps) <= GEN_SINGLE_COMP_MAX
    ]
    pairs = []
    for a_i, a in enumerate(eligible):
        for b in eligible[a_i + 1 :]:
            additional = len(a[1]) + len(b[1]) - 2
            if GEN_MATCH_COUNT_MIN <= additional <= GEN_MATCH_COUNT_MAX:
                pairs.append((a, b))

    if pairs and (not singles or random.random() < GEN_PAIR_PROB):
        return list(random.choice(pairs))
    if singles:
        return [random.choice(singles)]
    if pairs:
        return list(random.choice(pairs))
    return []


def _gen_k_choices_for_target(n: int, target_count: int | None) -> list[int]:
    """Return feasible cell counts while preserving existing random consumption."""
    possible = [
        k
        for k in range(GEN_CELL_COUNT_MIN, GEN_CELL_COUNT_MAX + 1)
        if _gen_widths(n, k) is not None
    ]
    if target_count is None:
        return possible

    preferred = GEN_K_HIGH_TARGET_PREFERENCE
    for threshold, bucket in GEN_K_TARGET_PREFERENCES:
        if target_count <= threshold:
            preferred = bucket
            break
    choices = [k for k in preferred if k in possible]
    return choices or possible


def _gen_design_layout(target_count: int | None):
    n = random.randint(GEN_GRID_SIZE_MIN, GEN_GRID_SIZE_MAX)
    possible_k = _gen_k_choices_for_target(n, target_count)
    if not possible_k:
        return None
    k = random.choice(possible_k)
    widths = _gen_widths(n, k)
    if widths is None:
        return None
    patterns = _gen_cell_patterns(widths)
    seq, spans = _gen_sequence(widths, patterns)
    return n, seq, spans


def _gen_coords_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(map(tuple, coords.tolist()))


def _gen_recolor_templates(base: np.ndarray, selected: list, source_color: int):
    out = base.copy()
    template_coords: list[np.ndarray] = []
    matching_count = 0
    for _shape, comps in selected:
        random.shuffle(comps)
        template_coords.append(comps[0]["coords"])
        matching_count += len(comps) - 1
        for comp in comps:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = Color.TEAL
    if not GEN_MATCH_COUNT_MIN <= matching_count <= GEN_MATCH_COUNT_MAX:
        return None

    inp = out.copy()
    selected_template_sets = {_gen_coords_key(coords) for coords in template_coords}
    for _shape, comps in selected:
        for comp in comps:
            coords = comp["coords"]
            if _gen_coords_key(coords) not in selected_template_sets:
                inp[coords[:, 0], coords[:, 1]] = source_color
    return inp, out


def _gen_output_is_valid(inp, out, selected_count, lattice_color, source_color, target_count):
    source_comps = find_components(
        inp, bg=Color.BLACK, connectivity=8, colors=[source_color]
    )
    if (
        target_count is not None
        and abs(len(source_comps) - target_count) > GEN_TARGET_COUNT_TOLERANCE
    ):
        return False
    if not GEN_SOURCE_COUNT_MIN <= len(source_comps) <= GEN_SOURCE_COUNT_MAX:
        return False
    for comp in source_comps:
        r0, c0, r1, c1 = comp["bbox"]
        if (
            comp["area"] > GEN_SOURCE_OBJECT_AREA_MAX
            or r1 - r0 + 1 > GEN_OBJECT_BBOX_MAX
            or c1 - c0 + 1 > GEN_OBJECT_BBOX_MAX
        ):
            return False

    total = inp.size
    lattice_frac = float(np.count_nonzero(inp == lattice_color)) / total
    source_frac = float(np.count_nonzero(inp == source_color)) / total
    if not GEN_LATTICE_FRAC_MIN <= lattice_frac <= GEN_LATTICE_FRAC_MAX:
        return False
    if not GEN_SOURCE_FRAC_MIN <= source_frac <= GEN_SOURCE_FRAC_MAX:
        return False

    teal_comps = find_components(
        inp, bg=Color.BLACK, connectivity=8, colors=[Color.TEAL]
    )
    if len(teal_comps) != selected_count:
        return False
    for comp in teal_comps:
        r0, c0, r1, c1 = comp["bbox"]
        if (
            not GEN_TEMPLATE_AREA_MIN <= comp["area"] <= GEN_TEMPLATE_AREA_MAX
            or r1 - r0 + 1 > GEN_OBJECT_BBOX_MAX
            or c1 - c0 + 1 > GEN_OBJECT_BBOX_MAX
        ):
            return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt(target_count: int | None = None) -> np.ndarray | None:
    layout = _gen_design_layout(target_count)
    if layout is None:
        return None
    n, seq, spans = layout

    lattice_color, source_color = random.sample(GEN_PALETTE, 2)
    base = _gen_make_base_grid(n, seq, spans, lattice_color, source_color)
    if not np.array_equal(base, base.T):
        return None

    lattice_comps = find_components(
        base, bg=Color.BLACK, connectivity=8, colors=[lattice_color]
    )
    if len(lattice_comps) != 1 or lattice_comps[0]["bbox"] != (0, 0, n - 1, n - 1):
        return None

    groups = _gen_components_by_shape(base, source_color)
    selected = _gen_select_template_groups(groups)
    if not selected:
        return None

    recolored = _gen_recolor_templates(base, selected, source_color)
    if recolored is None:
        return None
    inp, out = recolored

    if not _gen_output_is_valid(
        inp, out, len(selected), lattice_color, source_color, target_count
    ):
        return None

    return inp


def generate_input() -> np.ndarray:
    target_count = random.randint(GEN_SOURCE_COUNT_MIN, GEN_SOURCE_COUNT_MAX)
    for attempt_i in range(GEN_MAX_ATTEMPTS):
        if attempt_i in GEN_TARGET_REFRESH_ATTEMPTS:
            target_count = random.randint(GEN_SOURCE_COUNT_MIN, GEN_SOURCE_COUNT_MAX)
        grid = _gen_attempt(target_count)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_MAX_ATTEMPTS} attempts")
