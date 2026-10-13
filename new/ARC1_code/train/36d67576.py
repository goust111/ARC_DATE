from collections import Counter

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def _component_patch(g: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(g[int(r), int(c)])
    return patch


def _dihedral_variants(patch: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(patch, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _color_set(g: np.ndarray, coords: np.ndarray) -> set[int]:
    return {int(g[int(r), int(c)]) for r, c in coords}


def _infer_role_colors(g: np.ndarray, comps: list[dict], bg: int) -> tuple[int, int, set[int]]:
    counts = [
        Counter(int(g[int(r), int(c)]) for r, c in comp["coords"])
        for comp in comps
    ]
    fg_colors = {int(v) for v in np.unique(g) if int(v) != int(bg)}
    common = [color for color in fg_colors if all(cnt.get(color, 0) > 0 for cnt in counts)]
    assert len(common) >= 2, "motifs must share a skeleton color and an anchor color"

    totals = {color: sum(cnt.get(color, 0) for cnt in counts) for color in common}
    max_total = max(totals.values())
    skeleton_candidates = [color for color, total in totals.items() if total == max_total]
    assert len(skeleton_candidates) == 1, "skeleton color must be uniquely most frequent among shared colors"
    skeleton = skeleton_candidates[0]

    anchor_candidates = [
        color
        for color in common
        if color != skeleton and all(cnt.get(color, 0) == 1 for cnt in counts)
    ]
    assert len(anchor_candidates) == 1, "anchor color must appear once in every motif"
    anchor = anchor_candidates[0]
    return skeleton, anchor, fg_colors - {skeleton, anchor}


def _translated_cells(
    variant: np.ndarray,
    target_skeleton: set[tuple[int, int]],
    target_anchor: set[tuple[int, int]],
    bg: int,
    skeleton: int,
    anchor: int,
) -> tuple[tuple[int, int, int], ...] | None:
    variant_skeleton = set(map(tuple, np.argwhere(variant == int(skeleton)).tolist()))
    variant_anchor = set(map(tuple, np.argwhere(variant == int(anchor)).tolist()))
    if len(variant_skeleton) != len(target_skeleton) or len(variant_anchor) != len(target_anchor):
        return None

    dr = min(r for r, _c in target_skeleton) - min(r for r, _c in variant_skeleton)
    dc = min(c for _r, c in target_skeleton) - min(c for _r, c in variant_skeleton)
    if {(r + dr, c + dc) for r, c in variant_skeleton} != target_skeleton:
        return None
    if {(r + dr, c + dc) for r, c in variant_anchor} != target_anchor:
        return None

    marker_mask = (variant != int(bg)) & (variant != int(skeleton))
    cells = [
        (int(r + dr), int(c + dc), int(variant[int(r), int(c)]))
        for r, c in np.argwhere(marker_mask)
    ]
    return tuple(sorted(cells))


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the complete colored annotation from the fully marked motif onto each red-anchored copy.

    Input structure
    - The background surrounds several 8-connected foreground motifs.
    - Every motif contains the same structural skeleton color and exactly one shared anchor-color pixel.
    - Exactly one motif also contains the other marker colors; the remaining motifs contain only the skeleton and anchor.

    Rule
    - Use the fully marked motif as the source template.
    - For each incomplete motif, find the unique rotation/reflection and translation that maps the source skeleton and anchor pixel onto the target skeleton and anchor pixel.
    - Stamp all non-skeleton source marker pixels through that alignment, leaving the skeleton cells unchanged.

    Key insight
    - The anchor pixel fixes the orientation of an otherwise reusable skeleton, so the same local marker pattern can be transferred by a dihedral transform.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    assert len(comps) >= 2, "expected a reference motif and at least one target motif"

    skeleton, anchor, extra_colors = _infer_role_colors(g, comps, bg)
    ref_indices = [
        i
        for i, comp in enumerate(comps)
        if any(int(g[int(r), int(c)]) in extra_colors for r, c in comp["coords"])
    ]
    assert len(ref_indices) == 1, "expected exactly one fully marked reference motif"
    ref_index = ref_indices[0]
    reference = _component_patch(g, comps[ref_index], bg)
    variants = _dihedral_variants(reference)

    out = g.copy()
    for i, comp in enumerate(comps):
        if i == ref_index:
            continue
        assert _color_set(g, comp["coords"]) <= {skeleton, anchor}, "target motif already has non-anchor markers"
        target_skeleton = {
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(g[int(r), int(c)]) == int(skeleton)
        }
        target_anchor = {
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(g[int(r), int(c)]) == int(anchor)
        }
        assert len(target_anchor) == 1, "target motif must have one anchor pixel"

        candidates = [
            cells
            for variant in variants
            if (cells := _translated_cells(variant, target_skeleton, target_anchor, bg, skeleton, anchor)) is not None
        ]
        unique_candidates = sorted(set(candidates))
        assert len(unique_candidates) == 1, "target motif must have a unique source alignment"

        for r, c, color in unique_candidates[0]:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "stamped marker leaves the grid"
            assert int(out[r, c]) in (int(bg), int(color)), "stamped marker conflicts with existing content"
            out[r, c] = int(color)

    return out

# --------------- generator (o5x) ---------------

from arc_utils import bbox_coords, neighbors4, neighbors8


# Canvas bounds.
_GEN_GRID_ROWS_RANGE = (13, 15)
_GEN_GRID_COLS_RANGE = (13, 15)

# Motif structure.
_GEN_MOTIF_COUNT_RANGE = (3, 4)
_GEN_SKELETON_SIZE_RANGE = (5, 6)
_GEN_SKELETON_BBOX_MAX = (5, 5)
_GEN_MARKER_COUNT_RANGE = (3, 5)
_GEN_MIN_MOTIF_GAP = 4
_GEN_REQUIRED_COMBO_VARIANTS = 8

# Palette.
_GEN_BG = 0
_GEN_MARKER_COLORS = (1, 3)
_GEN_ANCHOR_COLOR = 2
_GEN_SKELETON_COLOR = 4

# Dihedral transforms and retry budget.
_GEN_ROTATION_COUNT = 4
_GEN_MAX_ATTEMPTS = 300


def _gen_normalize_shape(shape):
    r0 = min(r for r, _c in shape)
    c0 = min(c for _r, c in shape)
    return frozenset((r - r0, c - c0) for r, c in shape)


_GEN_SKELETON_CACHE = {}


def _gen_skeleton_pool(size):
    size = int(size)
    if size in _GEN_SKELETON_CACHE:
        return _GEN_SKELETON_CACHE[size]

    shapes = {frozenset({(0, 0)})}
    for _ in range(size - 1):
        next_shapes = set()
        for shape in shapes:
            frontier = set()
            for cell in shape:
                frontier.update(neighbors4(*cell))
            for cell in frontier - set(shape):
                next_shapes.add(_gen_normalize_shape(set(shape) | {cell}))
        shapes = next_shapes

    max_h, max_w = _GEN_SKELETON_BBOX_MAX
    pool = []
    for shape in shapes:
        rows = [r for r, _c in shape]
        cols = [c for _r, c in shape]
        h = max(rows) - min(rows) + 1
        w = max(cols) - min(cols) + 1
        if h <= max_h and w <= max_w:
            pool.append(tuple(sorted(shape)))

    _GEN_SKELETON_CACHE[size] = pool
    return pool


def _gen_patch(cells):
    coords = np.array([(r, c) for r, c, _color in cells], dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c, color in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def _gen_dihedral_variants(patch):
    variants = []
    seen = set()
    for k in range(_GEN_ROTATION_COUNT):
        rotated = np.rot90(patch, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _gen_colored_key(patch, colors):
    coords = [
        (int(r), int(c), int(patch[r, c]))
        for r, c in np.argwhere(np.isin(patch, list(colors)))
    ]
    r0 = min(r for r, _c, _color in coords)
    c0 = min(c for _r, c, _color in coords)
    return tuple(sorted((r - r0, c - c0, color) for r, c, color in coords))


def _gen_combo_variant_count(skeleton, anchor):
    cells = [(r, c, _GEN_SKELETON_COLOR) for r, c in skeleton] + [
        (anchor[0], anchor[1], _GEN_ANCHOR_COLOR)
    ]
    patch = _gen_patch(cells)
    combo_colors = {_GEN_ANCHOR_COLOR, _GEN_SKELETON_COLOR}
    return len({_gen_colored_key(v, combo_colors) for v in _gen_dihedral_variants(patch)})


def _gen_adjacent_open_cells(skeleton, blocked=()):
    blocked = set(blocked)
    return sorted(
        {
            nb
            for cell in skeleton
            for nb in neighbors8(*cell)
            if nb not in skeleton and nb not in blocked
        }
    )


def _gen_choose_anchor(skeleton):
    candidates = _gen_adjacent_open_cells(skeleton)
    random.shuffle(candidates)
    valid = [
        cell
        for cell in candidates
        if _gen_combo_variant_count(skeleton, cell) == _GEN_REQUIRED_COMBO_VARIANTS
    ]
    if not valid:
        return None
    return random.choice(valid)


def _gen_choose_markers(skeleton, anchor, count):
    candidates = _gen_adjacent_open_cells(skeleton, blocked=(anchor,))
    if len(candidates) < count:
        return None
    random.shuffle(candidates)
    coords = candidates[:count]
    colors = list(_GEN_MARKER_COLORS) + [
        random.choice(_GEN_MARKER_COLORS)
        for _ in range(count - len(_GEN_MARKER_COLORS))
    ]
    random.shuffle(colors)
    return [(r, c, color) for (r, c), color in zip(coords, colors)]


def _gen_variant_records(full_patch, motif_count):
    """Select distinct skeleton+anchor variants and mark one as the reference."""
    records = []
    seen_combo = set()
    for variant in _gen_dihedral_variants(full_patch):
        combo_key = _gen_colored_key(variant, {_GEN_ANCHOR_COLOR, _GEN_SKELETON_COLOR})
        if combo_key in seen_combo:
            continue
        seen_combo.add(combo_key)
        records.append({"full": variant})

    if len(records) < motif_count:
        return None

    selected = random.sample(records, motif_count)
    reference_i = random.randrange(motif_count)
    for i, record in enumerate(selected):
        record["is_reference"] = (i == reference_i)
        record["marker_coords"] = [
            (int(r), int(c))
            for r, c in np.argwhere(np.isin(record["full"], _GEN_MARKER_COLORS))
        ]
    return selected


def _gen_patch_offsets(patch):
    return [
        (int(r), int(c))
        for r, c in np.argwhere(patch != _GEN_BG)
    ]


def _gen_offset_cells(offsets, r0, c0):
    return [(int(r0 + r), int(c0 + c)) for r, c in offsets]


def _gen_valid_locations(out, patch, occupied):
    """Return top-left placements where patch cells are clear and separated."""
    patch_h, patch_w = patch.shape
    rows, cols = out.shape
    offsets = _gen_patch_offsets(patch)
    candidates = []
    for r0 in range(rows - patch_h + 1):
        for c0 in range(cols - patch_w + 1):
            cells = _gen_offset_cells(offsets, r0, c0)
            if any(out[r, c] != _GEN_BG for r, c in cells):
                continue
            if all(
                abs(r - er) + abs(c - ec) >= _GEN_MIN_MOTIF_GAP
                for r, c in cells
                for er, ec in occupied
            ):
                candidates.append((r0, c0))
    return candidates


def _gen_place_records(records, rows, cols):
    """Render every full motif variant and return the output grid plus placements."""
    out = np.zeros((rows, cols), dtype=int)
    occupied = []
    placements = []
    order = list(range(len(records)))
    random.shuffle(order)

    for index in order:
        patch = records[index]["full"]
        candidates = _gen_valid_locations(out, patch, occupied)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        paste(out, patch, r0, c0, transparent=_GEN_BG)
        occupied.extend(_gen_offset_cells(_gen_patch_offsets(patch), r0, c0))
        placements.append((index, r0, c0))

    return out, placements


def _gen_hide_target_markers(inp, records, placements):
    for index, r0, c0 in placements:
        if records[index]["is_reference"]:
            continue
        for r, c in records[index]["marker_coords"]:
            inp[r0 + r, c0 + c] = _GEN_BG


def _gen_design_reference_patch():
    """Build one full skeleton+anchor+marker patch before dihedral expansion."""
    skeleton_size = random.randint(*_GEN_SKELETON_SIZE_RANGE)
    skeleton = set(random.choice(_gen_skeleton_pool(skeleton_size)))

    anchor = _gen_choose_anchor(skeleton)
    if anchor is None:
        return None

    marker_count = random.randint(*_GEN_MARKER_COUNT_RANGE)
    markers = _gen_choose_markers(skeleton, anchor, marker_count)
    if markers is None:
        return None

    full_cells = (
        [(r, c, _GEN_SKELETON_COLOR) for r, c in skeleton]
        + [(anchor[0], anchor[1], _GEN_ANCHOR_COLOR)]
        + markers
    )
    return _gen_patch(full_cells)


def _gen_oracle_accepts(inp, out):
    try:
        return np.array_equal(main(inp), out)
    except Exception:
        return False


def _gen_attempt():
    motif_count = random.randint(*_GEN_MOTIF_COUNT_RANGE)
    full_patch = _gen_design_reference_patch()
    if full_patch is None:
        return None

    records = _gen_variant_records(full_patch, motif_count)
    if records is None:
        return None

    rows = random.randint(*_GEN_GRID_ROWS_RANGE)
    cols = random.randint(*_GEN_GRID_COLS_RANGE)
    placed = _gen_place_records(records, rows, cols)
    if placed is None:
        return None

    out, placements = placed
    inp = out.copy()
    _gen_hide_target_markers(inp, records, placements)
    if _gen_oracle_accepts(inp, out):
        return inp
    return None


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 36d67576 input")
