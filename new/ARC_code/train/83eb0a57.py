from __future__ import annotations

import numpy as np

from arc_utils import components_mask, crop_bbox, find_components, get_bg, paste, random_free_location
import itertools
import random


CoordSet = frozenset[tuple[int, int]]


def _component_data(grid: np.ndarray, comp: dict) -> tuple[np.ndarray, np.ndarray, set[int]]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    mask = np.zeros(patch.shape, dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    colors = {int(grid[int(r), int(c)]) for r, c in coords}
    return patch, mask, colors


def _coord_set(coords: np.ndarray) -> CoordSet:
    return frozenset((int(r), int(c)) for r, c in coords)


def _component_sets(mask: np.ndarray) -> list[CoordSet]:
    sets = [_coord_set(coords) for coords in components_mask(mask, connectivity=4)]
    assert sets, "marker mask must contain at least one connected component"
    return sets


def _normalized(coords: CoordSet) -> CoordSet:
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _unique_marker_offset(src_marker: np.ndarray, dst_marker: np.ndarray) -> tuple[int, int]:
    src_components = _component_sets(src_marker)
    dst_components = _component_sets(dst_marker)
    dst_component_set = set(dst_components)

    candidate_offsets: set[tuple[int, int]] = set()
    for src_comp in src_components:
        src_norm = _normalized(src_comp)
        sr0 = min(r for r, _c in src_comp)
        sc0 = min(c for _r, c in src_comp)
        for dst_comp in dst_components:
            if len(dst_comp) != len(src_comp) or _normalized(dst_comp) != src_norm:
                continue
            dr0 = min(r for r, _c in dst_comp)
            dc0 = min(c for _r, c in dst_comp)
            candidate_offsets.add((dr0 - sr0, dc0 - sc0))

    valid_offsets = []
    for dr, dc in candidate_offsets:
        all_components_match = True
        for src_comp in src_components:
            shifted = frozenset((r + dr, c + dc) for r, c in src_comp)
            if shifted not in dst_component_set:
                all_components_match = False
                break
        if all_components_match:
            valid_offsets.append((dr, dc))

    assert len(valid_offsets) == 1, "marker components must determine one translation"
    return valid_offsets[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the largest object and overlay each smaller sprite where its shared marker pattern matches.

    Input structure
    - The background is the most common color, and the foreground has one largest destination object plus smaller source sprites.
    - Each source sprite shares exactly one marker color with the destination object.

    Rule
    - Output the tight bbox crop of the largest object.
    - Translate each source sprite so every connected marker component of the shared color exactly matches destination marker components.
    - Paste source sprites in decreasing area order, so smaller sprites can sit on top of larger ones when they overlap.

    Key insight
    - The shared-color marker components are alignment keys; matching whole marker components distinguishes singleton slots from larger same-color marker blocks.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert len(comps) >= 2, "must have a destination object and at least one source sprite"

    areas = [int(comp["area"]) for comp in comps]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "largest destination object must be unique"
    dst_index = areas.index(max_area)
    dst_comp = comps[dst_index]
    dst_patch, dst_mask, dst_colors = _component_data(g, dst_comp)

    sources = [comp for idx, comp in enumerate(comps) if idx != dst_index]
    source_areas = [int(comp["area"]) for comp in sources]
    assert len(set(source_areas)) == len(source_areas), "source sprites must have distinct areas for layering"
    sources.sort(key=lambda comp: -int(comp["area"]))

    out = dst_patch.copy()
    out_h, out_w = out.shape

    for src_comp in sources:
        src_patch, src_mask, src_colors = _component_data(g, src_comp)
        common_colors = src_colors & dst_colors
        assert len(common_colors) == 1, "each source must share exactly one marker color with the destination"
        marker_color = next(iter(common_colors))

        src_marker = (src_patch == marker_color) & src_mask
        dst_marker = (dst_patch == marker_color) & dst_mask
        assert bool(src_marker.any()) and bool(dst_marker.any()), "marker pixels must exist in source and destination"

        top, left = _unique_marker_offset(src_marker, dst_marker)
        src_h, src_w = src_patch.shape
        assert 0 <= top and 0 <= left, "translated sprite must fit inside destination crop"
        assert top + src_h <= out_h and left + src_w <= out_w, "translated sprite must fit inside destination crop"
        paste(out, src_patch, top, left, mask=src_mask, overwrite=True)

    return out

# --------------- generator (att2) ---------------

from arc_utils import bbox_coords


# Canvas, destination, and source bounds.
INPUT_SHAPE = (30, 30)
DST_HEIGHT_RANGE, DST_WIDTH_RANGE, DST_MAX_AREA = (9, 13), (5, 16), 195
SOURCE_COUNT_RANGE, SRC_HEIGHT_RANGE, SRC_WIDTH_RANGE = (1, 2), (3, 8), (3, 9)
SRC_DST_MARGIN = 1
DST_SIZE_OPTIONS = tuple(
    (h, w) for h in range(DST_HEIGHT_RANGE[0], DST_HEIGHT_RANGE[1] + 1)
    for w in range(DST_WIDTH_RANGE[0], DST_WIDTH_RANGE[1] + 1) if h * w <= DST_MAX_AREA
)

# Marker component counts, shape library, and spacing.
MARKER_COUNT_RANGE, MARKER_AREA_RANGE = (2, 4), (1, 8)
MARKER_SHAPE_BOX, MARKER_BORDER, MARKER_CLEARANCE = 3, 1, 1
MIN_GROUPS_SINGLE_SOURCE, MIN_GROUPS_TWO_SOURCES = 1, 2

# Palette layout: background, destination fill, marker, then source-own colors.
COLOR_COUNT, FIXED_COLOR_COUNT = 10, 3

# Input placement spacing and generation retry budget.
PIECE_BORDER, PIECE_PADDING, PIECE_PADDING_CONNECTIVITY = 1, 4, 8
MAX_GENERATION_ATTEMPTS = 300

Shape = tuple[tuple[int, int], ...]


def _norm_shape(cells) -> Shape:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _connected(shape: Shape) -> bool:
    cells = set(shape)
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _shape_library() -> dict[int, list[Shape]]:
    base = [(r, c) for r in range(MARKER_SHAPE_BOX) for c in range(MARKER_SHAPE_BOX)]
    by_area: dict[int, list[Shape]] = {
        area: [] for area in range(MARKER_AREA_RANGE[0], MARKER_AREA_RANGE[1] + 1)
    }
    seen: set[Shape] = set()
    for area in range(MARKER_AREA_RANGE[0], MARKER_AREA_RANGE[1] + 1):
        for combo in itertools.combinations(base, area):
            shape = _norm_shape(combo)
            if shape not in seen and _connected(shape):
                seen.add(shape)
                by_area[area].append(shape)
    return by_area


_SHAPES_BY_AREA = _shape_library()


def _partition_count(total: int, groups: int) -> list[int]:
    parts = [1] * groups
    for _ in range(total - groups):
        parts[random.randrange(groups)] += 1
    random.shuffle(parts)
    return parts


def _choose_group_shapes(n_groups: int) -> list[Shape]:
    used: set[Shape] = set()
    shapes: list[Shape] = []
    for _ in range(n_groups):
        target_area = random.randint(*MARKER_AREA_RANGE)
        options = [s for s in _SHAPES_BY_AREA[target_area] if s not in used]
        if not options:
            options = [s for group in _SHAPES_BY_AREA.values() for s in group if s not in used]
        shape = random.choice(options)
        used.add(shape)
        shapes.append(shape)
    return shapes


def _place_marker_components(
    dst: np.ndarray, group_sizes: list[int], group_shapes: list[Shape], marker: int
) -> list[dict] | None:
    h, w = dst.shape
    blocked = np.zeros(dst.shape, dtype=bool)
    comps: list[dict] = []

    for group, (count, shape) in enumerate(zip(group_sizes, group_shapes)):
        sh = max(r for r, _c in shape) + 1
        sw = max(c for _r, c in shape) + 1
        for _ in range(count):
            candidates: list[tuple[int, int]] = []
            for r0 in range(MARKER_BORDER, h - sh):
                for c0 in range(MARKER_BORDER, w - sw):
                    coords = [(r0 + r, c0 + c) for r, c in shape]
                    if not any(blocked[r, c] for r, c in coords):
                        candidates.append((r0, c0))
            if not candidates:
                return None

            r0, c0 = random.choice(candidates)
            coords = tuple((r0 + r, c0 + c) for r, c in shape)
            rows, cols = zip(*coords)
            dst[rows, cols] = int(marker)
            for r, c in coords:
                blocked[
                    max(0, r - MARKER_CLEARANCE) : min(h, r + MARKER_CLEARANCE + 1),
                    max(0, c - MARKER_CLEARANCE) : min(w, c + MARKER_CLEARANCE + 1),
                ] = True
            comps.append({"group": group, "shape": shape, "coords": coords})
    return comps


def _axis_starts(lo: int, hi: int, size: int, limit: int) -> range:
    start_min = max(SRC_DST_MARGIN, hi - size + 1)
    start_max = min(lo, limit - size - SRC_DST_MARGIN)
    if start_min > start_max:
        return range(0)
    return range(start_min, start_max + 1)


def _candidate_rects(comp_indices: list[int], comps: list[dict], dst_h: int, dst_w: int) -> list[tuple[int, int, int, int]]:
    coords = np.array([pt for idx in comp_indices for pt in comps[idx]["coords"]], dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    min_h = max(SRC_HEIGHT_RANGE[0], r1 - r0 + 1)
    min_w = max(SRC_WIDTH_RANGE[0], c1 - c0 + 1)
    max_h = min(SRC_HEIGHT_RANGE[1], dst_h - 2 * SRC_DST_MARGIN)
    max_w = min(SRC_WIDTH_RANGE[1], dst_w - 2 * SRC_DST_MARGIN)
    dst_area = dst_h * dst_w

    candidates: list[tuple[int, int, int, int]] = []
    for src_h in range(min_h, max_h + 1):
        top_starts = _axis_starts(r0, r1, src_h, dst_h)
        if not top_starts:
            continue
        for src_w in range(min_w, max_w + 1):
            if src_h * src_w >= dst_area:
                continue
            left_starts = _axis_starts(c0, c1, src_w, dst_w)
            if not left_starts:
                continue
            for top in top_starts:
                for left in left_starts:
                    candidates.append((top, left, src_h, src_w))
    return candidates


def _make_source(
    rect: tuple[int, int, int, int], comp_indices: list[int], comps: list[dict], marker: int, own: int
) -> dict:
    top, left, src_h, src_w = rect
    patch = np.full((src_h, src_w), int(own), dtype=int)
    for idx in comp_indices:
        coords = comps[idx]["coords"]
        rows = [r - top for r, _c in coords]
        cols = [c - left for _r, c in coords]
        patch[rows, cols] = int(marker)
    return {"top": top, "left": left, "patch": patch}


def _source_specs(comps: list[dict], n_groups: int, dst_h: int, dst_w: int, n_src: int):
    by_subset = {}
    for mask in range(1, 1 << n_groups):
        groups = tuple(i for i in range(n_groups) if mask & (1 << i))
        comp_indices = [idx for idx, comp in enumerate(comps) if comp["group"] in groups]
        rects = _candidate_rects(comp_indices, comps, dst_h, dst_w)
        if rects:
            by_subset[groups] = (comp_indices, rects)
    if n_src == 1:
        if not by_subset:
            return None
        groups = random.choice(list(by_subset))
        comp_indices, rects = by_subset[groups]
        return [(comp_indices, random.choice(rects))]

    plans = []
    subsets = list(by_subset)
    for a, b in itertools.combinations(subsets, 2):
        if set(a) & set(b):
            continue
        a_indices, a_rects = by_subset[a]
        b_indices, b_rects = by_subset[b]
        rect_pairs = [
            (ra, rb) for ra in a_rects for rb in b_rects if ra[2] * ra[3] != rb[2] * rb[3]
        ]
        if rect_pairs:
            plans.append((a_indices, b_indices, rect_pairs))
    if not plans:
        return None
    a_indices, b_indices, rect_pairs = random.choice(plans)
    ra, rb = random.choice(rect_pairs)
    return [(a_indices, ra), (b_indices, rb)]


def _compose_input(bg: int, dst: np.ndarray, sources: list[dict]) -> np.ndarray | None:
    inp = np.full(INPUT_SHAPE, int(bg), dtype=int)
    pieces = [dst] + [src["patch"] for src in sources]
    try:
        for patch in sorted(pieces, key=lambda p: -p.size):
            r0, c0 = random_free_location(
                inp, patch, transparent=bg, border=PIECE_BORDER, padding=PIECE_PADDING,
                padding_connectivity=PIECE_PADDING_CONNECTIVITY,
            )
            paste(inp, patch, r0, c0)
    except ValueError:
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    n_src = random.randint(*SOURCE_COUNT_RANGE)
    dst_h, dst_w = random.choice(DST_SIZE_OPTIONS)
    n_marker = random.randint(*MARKER_COUNT_RANGE)
    min_groups = MIN_GROUPS_TWO_SOURCES if n_src == 2 else MIN_GROUPS_SINGLE_SOURCE
    n_groups = random.randint(min_groups, n_marker)

    colors = random.sample(range(COLOR_COUNT), FIXED_COLOR_COUNT + n_src)
    bg, dst_primary, marker = colors[:FIXED_COLOR_COUNT]
    own_colors = colors[FIXED_COLOR_COUNT:]

    dst = np.full((dst_h, dst_w), int(dst_primary), dtype=int)
    group_sizes = _partition_count(n_marker, n_groups)
    group_shapes = _choose_group_shapes(n_groups)
    comps = _place_marker_components(dst, group_sizes, group_shapes, marker)
    if comps is None:
        return None

    specs = _source_specs(comps, n_groups, dst_h, dst_w, n_src)
    if specs is None:
        return None

    sources = [
        _make_source(rect, comp_indices, comps, marker, own)
        for own, (comp_indices, rect) in zip(own_colors, specs)
    ]
    out = dst.copy()
    for src in sorted(sources, key=lambda s: -s["patch"].size):
        paste(out, src["patch"], src["top"], src["left"])
    inp = _compose_input(bg, dst, sources)
    if inp is None:
        return None

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 83eb0a57 input")
