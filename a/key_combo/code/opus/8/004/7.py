from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, find_components, neighbors4, neighbors8, paste, random_free_location


BG = 0
SKELETON = 1
ANCHOR = 2
LABEL_COLORS = (3, 4, 6)

GEN_GRID_ROWS_RANGE = (20, 29)
GEN_GRID_COLS_RANGE = (18, 29)
GEN_MOTIF_COUNT_RANGE = (5, 7)
GEN_ANNOTATED_COUNT_RANGE = (2, 3)
GEN_SKELETON_SIZE_RANGE = (4, 8)
GEN_SKELETON_BBOX_MAX = (5, 5)
GEN_SOURCE_MARKER_COUNT_RANGE = (4, 7)
GEN_MAX_ATTEMPTS = 400
GEN_REQUIRED_ROTATIONS = 4

_SKELETON_CACHE: dict[int, list[tuple[tuple[int, int], ...]]] = {}


def _component_patch(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _patch_cells(patch: np.ndarray, colors: tuple[int, ...] | list[int] | set[int] | None = None) -> list[tuple[int, int]]:
    if colors is None:
        return [tuple(map(int, rc)) for rc in np.argwhere(patch != BG)]
    return [tuple(map(int, rc)) for rc in np.argwhere(np.isin(patch, list(colors)))]


def _normalize_shape(shape: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in shape)
    c0 = min(c for _, c in shape)
    return frozenset((r - r0, c - c0) for r, c in shape)


def _skeleton_pool(size: int) -> list[tuple[tuple[int, int], ...]]:
    size = int(size)
    if size in _SKELETON_CACHE:
        return _SKELETON_CACHE[size]

    shapes = {frozenset({(0, 0)})}
    for _ in range(size - 1):
        next_shapes = set()
        for shape in shapes:
            frontier = set()
            for cell in shape:
                frontier.update(neighbors4(*cell))
            for cell in frontier - set(shape):
                next_shapes.add(_normalize_shape(set(shape) | {cell}))
        shapes = next_shapes

    max_h, max_w = GEN_SKELETON_BBOX_MAX
    pool: list[tuple[tuple[int, int], ...]] = []
    for shape in shapes:
        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        h = max(rows) - min(rows) + 1
        w = max(cols) - min(cols) + 1
        if h <= max_h and w <= max_w:
            pool.append(tuple(sorted(shape)))
    _SKELETON_CACHE[size] = pool
    return pool


def _patch_from_cells(cells: list[tuple[int, int, int]]) -> np.ndarray:
    coords = np.array([(r, c) for r, c, _ in cells], dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c, color in cells:
        patch[int(r) - r0, int(c) - c0] = int(color)
    return patch


def _rotation_count(skeleton: set[tuple[int, int]], anchor: tuple[int, int]) -> int:
    patch = _patch_from_cells(
        [(r, c, SKELETON) for r, c in skeleton] + [(anchor[0], anchor[1], ANCHOR)]
    )
    seen = set()
    for k in range(4):
        rotated = np.rot90(patch, k)
        coords = tuple(
            sorted((int(r), int(c), int(rotated[r, c])) for r, c in np.argwhere(rotated != BG))
        )
        seen.add(coords)
    return len(seen)


def _adjacent_open_cells(
    skeleton: set[tuple[int, int]], blocked: set[tuple[int, int]] | None = None
) -> list[tuple[int, int]]:
    blocked = set() if blocked is None else set(blocked)
    candidates = set()
    for cell in skeleton:
        for nb in neighbors8(*cell):
            if nb not in skeleton and nb not in blocked:
                candidates.add((int(nb[0]), int(nb[1])))
    return sorted(candidates)


def _choose_anchor(skeleton: set[tuple[int, int]]) -> tuple[int, int] | None:
    candidates = _adjacent_open_cells(skeleton)
    random.shuffle(candidates)
    valid = [cell for cell in candidates if _rotation_count(skeleton, cell) == GEN_REQUIRED_ROTATIONS]
    if not valid:
        return None
    return random.choice(valid)


def _choose_source_markers(
    skeleton: set[tuple[int, int]], anchor: tuple[int, int]
) -> list[tuple[int, int, int]] | None:
    candidates = _adjacent_open_cells(skeleton, blocked={anchor})
    if len(candidates) < GEN_SOURCE_MARKER_COUNT_RANGE[0]:
        return None

    source_color_count = random.choice([2, 3])
    count = random.randint(max(source_color_count, GEN_SOURCE_MARKER_COUNT_RANGE[0]), GEN_SOURCE_MARKER_COUNT_RANGE[1])
    if len(candidates) < count:
        return None

    coords = random.sample(candidates, count)
    active_colors = random.sample(list(LABEL_COLORS), source_color_count)
    colors = active_colors.copy()
    while len(colors) < count:
        colors.append(random.choice(active_colors))
    random.shuffle(colors)
    markers = [(r, c, color) for (r, c), color in zip(coords, colors)]
    used = {color for _, _, color in markers}
    if used != set(active_colors):
        return None
    return markers


def _rotate_local_points(
    patch: np.ndarray, points: list[tuple[int, int, int]], rotation: int
) -> list[tuple[int, int, int]]:
    rotated = np.rot90(patch, rotation)
    transformed: list[tuple[int, int, int]] = []
    for color in set(color for _, _, color in points):
        src = np.full_like(patch, BG)
        for r, c, point_color in points:
            if point_color == color:
                src[int(r), int(c)] = int(color)
        rot = np.rot90(src, rotation)
        for rr, cc in np.argwhere(rot == int(color)):
            transformed.append((int(rr), int(cc), int(color)))
    transformed.sort()
    return transformed


def _label_kind_count(patch: np.ndarray) -> int:
    return len({int(v) for v in np.unique(patch) if int(v) not in (BG, SKELETON, ANCHOR)})


def _aligned_marker_cells(
    source_patch: np.ndarray,
    target_comp: dict,
    rotation: int,
) -> list[tuple[int, int, int]] | None:
    rotated = np.rot90(source_patch, rotation)
    source_anchor = np.argwhere(rotated == ANCHOR)
    target_anchor = [
        (int(r), int(c))
        for r, c in target_comp["coords"]
        if int(target_comp["grid"][int(r), int(c)]) == ANCHOR
    ]
    if len(source_anchor) != 1 or len(target_anchor) != 1:
        return None

    sr, sc = map(int, source_anchor[0])
    tr, tc = target_anchor[0]
    dr = tr - sr
    dc = tc - sc

    source_skeleton = {
        (int(r) + dr, int(c) + dc)
        for r, c in np.argwhere(rotated == SKELETON)
    }
    target_skeleton = {
        (int(r), int(c))
        for r, c in target_comp["coords"]
        if int(target_comp["grid"][int(r), int(c)]) == SKELETON
    }
    if source_skeleton != target_skeleton:
        return None

    cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(np.isin(rotated, LABEL_COLORS)):
        cells.append((int(r) + dr, int(c) + dc, int(rotated[int(r), int(c)])))
    return sorted(cells)


def _choose_unique_rotation(source_patch: np.ndarray, target_comp: dict) -> tuple[int, list[tuple[int, int, int]]]:
    matches: list[tuple[int, list[tuple[int, int, int]]]] = []
    for rotation in range(4):
        cells = _aligned_marker_cells(source_patch, target_comp, rotation)
        if cells is not None:
            matches.append((rotation, cells))
    assert len(matches) == 1, "source and target must match under exactly one rotation"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Choose the most color-diverse annotated motif as the source, erase other annotations, and rotate-stamp its labels onto every copy.

    Input structure:
    - The grid has a black background and several separated 8-connected motifs.
    - Every motif contains a blue 4-connected skeleton shape and exactly one red anchor pixel.
    - Two or three motifs also contain extra label pixels in green, yellow, or magenta, with exactly one motif using the largest number of distinct label colors.

    Rule:
    - Count distinct label colors in each motif while ignoring blue skeleton cells and the red anchor.
    - Keep the uniquely most color-diverse motif as the transfer source and erase every other motif's label pixels back to black.
    - For each non-source motif, find the unique rotation that aligns the source skeleton and anchor to the target skeleton and anchor, then stamp the rotated source labels into that motif without changing skeleton or anchor cells.

    Key insight:
    - Several motifs look partially annotated, but only the one with the richest label-color vocabulary is the true template whose labels must be propagated after rotational alignment.
    """
    g = np.asarray(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=8, monochromatic=False, return_format="dict")
    assert 5 <= len(comps) <= 7

    for comp in comps:
        comp["grid"] = g
        patch = _component_patch(g, comp)
        assert np.count_nonzero(patch == ANCHOR) == 1
        assert np.count_nonzero(patch == SKELETON) >= 4
        comp["label_kind_count"] = _label_kind_count(patch)

    max_kinds = max(comp["label_kind_count"] for comp in comps)
    assert max_kinds > 0
    sources = [comp for comp in comps if comp["label_kind_count"] == max_kinds]
    assert len(sources) == 1
    source = sources[0]
    source_patch = _component_patch(g, source)

    out = g.copy()
    for comp in comps:
        if comp is source:
            continue
        for r, c in comp["coords"]:
            if int(g[int(r), int(c)]) in LABEL_COLORS:
                out[int(r), int(c)] = BG

    for comp in comps:
        if comp is source:
            continue
        _rotation, cells = _choose_unique_rotation(source_patch, comp)
        for r, c, color in cells:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            assert int(out[r, c]) == BG
            out[r, c] = int(color)

    return out


def _subset_markers(
    rotated_full_patch: np.ndarray,
    source_kind_count: int,
    allow_empty: bool,
) -> list[tuple[int, int, int]]:
    full_markers = [
        (int(r), int(c), int(rotated_full_patch[int(r), int(c)]))
        for r, c in np.argwhere(np.isin(rotated_full_patch, LABEL_COLORS))
    ]
    by_color: dict[int, list[tuple[int, int, int]]] = {}
    for item in full_markers:
        by_color.setdefault(item[2], []).append(item)
    colors = list(by_color)

    if allow_empty and random.random() < 0.5:
        return []

    max_subset_kinds = source_kind_count - 1
    subset_kind_count = random.randint(1, max_subset_kinds)
    chosen_colors = random.sample(colors, subset_kind_count)
    kept: list[tuple[int, int, int]] = []
    for color in chosen_colors:
        cells = by_color[color]
        take = random.randint(1, len(cells))
        kept.extend(random.sample(cells, take))
    kept.sort()
    return kept


def _design_reference_patch() -> np.ndarray | None:
    skeleton_size = random.randint(*GEN_SKELETON_SIZE_RANGE)
    skeleton = set(random.choice(_skeleton_pool(skeleton_size)))
    anchor = _choose_anchor(skeleton)
    if anchor is None:
        return None
    markers = _choose_source_markers(skeleton, anchor)
    if markers is None:
        return None
    cells = [(r, c, SKELETON) for r, c in skeleton]
    cells.append((anchor[0], anchor[1], ANCHOR))
    cells.extend(markers)
    patch = _patch_from_cells(cells)
    if _label_kind_count(patch) < 2:
        return None
    return patch


def _place_patch(
    canvas: np.ndarray, occupancy: np.ndarray, patch: np.ndarray, reserve_patch: np.ndarray | None = None
) -> tuple[int, int] | None:
    reserve = patch if reserve_patch is None else reserve_patch
    mask_patch = np.where(reserve != BG, 9, BG)
    try:
        r0, c0 = random_free_location(
            occupancy,
            mask_patch,
            transparent=BG,
            border=1,
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    paste(canvas, patch, r0, c0, transparent=BG)
    paste(occupancy, mask_patch, r0, c0, transparent=BG)
    return r0, c0


def _attempt() -> np.ndarray | None:
    rows = random.randint(*GEN_GRID_ROWS_RANGE)
    cols = random.randint(*GEN_GRID_COLS_RANGE)
    motif_count = random.randint(*GEN_MOTIF_COUNT_RANGE)
    annotated_count = random.randint(*GEN_ANNOTATED_COUNT_RANGE)
    if annotated_count >= motif_count:
        return None

    reference_patch = _design_reference_patch()
    if reference_patch is None:
        return None

    source_kind_count = _label_kind_count(reference_patch)
    full_variants = [np.rot90(reference_patch, k).copy() for k in range(4)]

    source_index = random.randrange(motif_count)
    annotated_others = set(random.sample([i for i in range(motif_count) if i != source_index], annotated_count - 1))

    out = np.full((rows, cols), BG, dtype=int)
    inp = np.full((rows, cols), BG, dtype=int)
    occupancy = np.full((rows, cols), BG, dtype=int)

    for motif_i in range(motif_count):
        rotation = random.randrange(4)
        full_patch = full_variants[rotation]

        if motif_i == source_index:
            input_patch = full_patch.copy()
        else:
            input_patch = np.where(np.isin(full_patch, LABEL_COLORS), BG, full_patch)
            if motif_i in annotated_others:
                subset = _subset_markers(full_patch, source_kind_count, allow_empty=False)
                for r, c, color in subset:
                    input_patch[int(r), int(c)] = int(color)

        placed_input = _place_patch(inp, occupancy, input_patch, reserve_patch=full_patch)
        if placed_input is None:
            return None
        r0, c0 = placed_input
        paste(out, full_patch, r0, c0, transparent=BG)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
