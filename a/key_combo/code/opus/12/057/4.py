from __future__ import annotations

from collections import Counter
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox_coords = _ARC_UTILS.bbox_coords
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
neighbors4 = _ARC_UTILS.neighbors4
neighbors8 = _ARC_UTILS.neighbors8
paste = _ARC_UTILS.paste


BG = 0
SKELETON = 5
SEPARATOR = 4
ANCHOR = 7
MARKER_POOL = (1, 2, 3, 6, 8, 9)

GRID_ROWS = (18, 25)
GRID_COLS = (18, 25)
MOTIF_COUNTS = (3, 4)
CORE_SIZES = (6, 10)  # total core cells = grey skeleton + anchor
SEP_BOTTOM_MARGIN = (3, 4)
TOP_CLEARANCE = 2
PLACEMENT_RETRIES = 120
GEN_RETRIES = 400
FULL_VARIANT_TARGET = 8

COUNT_PROFILES = (
    (1, 3),
    (2, 3),
    (1, 4),
    (2, 4),
    (1, 2, 3),
    (1, 2, 4),
)


def _component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _dihedral_variants(patch: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(patch, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _core_key(patch: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    cells = [
        (int(r), int(c), int(patch[r, c]))
        for r, c in np.argwhere((patch == SKELETON) | (patch == ANCHOR))
    ]
    assert cells
    r0 = min(r for r, _c, _color in cells)
    c0 = min(c for _r, c, _color in cells)
    return tuple(sorted((r - r0, c - c0, color) for r, c, color in cells))


def _patch_from_cells(cells: list[tuple[int, int, int]]) -> np.ndarray:
    coords = np.array([(r, c) for r, c, _color in cells], dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c, color in cells:
        patch[int(r) - r0, int(c) - c0] = int(color)
    return patch


def _variant_groups(full_patch: np.ndarray) -> list[np.ndarray]:
    groups: dict[tuple[tuple[int, int, int], ...], np.ndarray] = {}
    for variant in _dihedral_variants(full_patch):
        groups.setdefault(_core_key(variant), variant)
    return list(groups.values())


def _aligned_active_cells(
    variant: np.ndarray,
    target_skeleton: set[tuple[int, int]],
    target_anchor: tuple[int, int],
    active_colors: set[int],
) -> tuple[tuple[int, int, int], ...] | None:
    variant_skeleton = {(int(r), int(c)) for r, c in np.argwhere(variant == SKELETON)}
    variant_anchor_coords = [(int(r), int(c)) for r, c in np.argwhere(variant == ANCHOR)]
    if len(variant_anchor_coords) != 1:
        return None
    variant_anchor = variant_anchor_coords[0]
    dr = int(target_anchor[0]) - int(variant_anchor[0])
    dc = int(target_anchor[1]) - int(variant_anchor[1])
    if {(r + dr, c + dc) for r, c in variant_skeleton} != target_skeleton:
        return None
    if (variant_anchor[0] + dr, variant_anchor[1] + dc) != tuple(target_anchor):
        return None

    cells = []
    for r, c in np.argwhere(np.isin(variant, list(active_colors))):
        cells.append((int(r) + dr, int(c) + dc, int(variant[int(r), int(c)])))
    return tuple(sorted(cells))


def main(grid: np.ndarray) -> np.ndarray:
    """Filter template annotations by code-row frequency, then stamp only the surviving colors onto aligned skeleton copies.

    Input structure:
    - Black background contains 3-4 separated 8-connected motifs above one full-width yellow separator row.
    - Every motif shares the same grey skeleton shape and contains exactly one orange anchor pixel embedded in that shape.
    - Exactly one motif also has extra colored annotation pixels around the skeleton; the row immediately below the separator is a sparse code row using a subset of those annotation colors.

    Rule:
    - Find the unique fully annotated template motif and count all of its annotation pixels, excluding black, grey skeleton, orange anchor, and the separator/code area; call this total T.
    - In the code row, collect every non-black color whose frequency is exactly T; these are the active colors.
    - Erase non-active annotation colors from the template, then for each incomplete motif find the unique rotation/reflection plus translation that maps the template's grey skeleton and orange anchor onto that copy.
    - Stamp only the active-color annotation pixels through that alignment onto each copy; keep the grey skeleton, orange anchor, yellow separator row, and code row unchanged.

    Key insight:
    - The template's own annotation count becomes an external frequency key: the code row selects which annotation colors survive, and only those filtered pixels are transferred through the skeleton-anchor alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert int(bg) == BG, "background must be black"

    separator_rows = [
        r
        for r in range(g.shape[0] - 1)
        if np.all(g[r, :] == SEPARATOR)
    ]
    assert len(separator_rows) == 1, "expected exactly one full-width separator row"
    separator_row = int(separator_rows[0])
    code_row = g[separator_row + 1, :]

    motifs_only = g.copy()
    motifs_only[separator_row:, :] = BG
    comps = find_components(
        motifs_only,
        bg=BG,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert 3 <= len(comps) <= 4, "expected 3-4 motifs"

    template_indices = []
    for index, comp in enumerate(comps):
        colors = {int(g[int(r), int(c)]) for r, c in comp["coords"]}
        assert colors <= {BG, SKELETON, ANCHOR, *MARKER_POOL}, "unexpected motif colors"
        assert sum(int(g[int(r), int(c)] == ANCHOR) for r, c in comp["coords"]) == 1
        if any(color not in {BG, SKELETON, ANCHOR} for color in colors):
            template_indices.append(index)
    assert len(template_indices) == 1, "expected exactly one annotated template motif"
    template_index = template_indices[0]

    template_comp = comps[template_index]
    template_patch = _component_patch(g, template_comp, BG)
    template_marker_mask = ~np.isin(template_patch, [BG, SKELETON, ANCHOR])
    marker_colors = {int(color) for color in np.unique(template_patch[template_marker_mask])}
    total_markers = int(template_marker_mask.sum())
    assert total_markers > 0

    code_counts = Counter(int(color) for color in code_row if int(color) != BG)
    active_colors = {int(color) for color, count in code_counts.items() if int(count) == total_markers}
    assert active_colors, "expected at least one active annotation color"
    assert active_colors <= marker_colors, "active colors must come from template annotation colors"

    variants = _variant_groups(template_patch)
    out = g.copy()
    for r, c in template_comp["coords"]:
        color = int(g[int(r), int(c)])
        if color not in {BG, SKELETON, ANCHOR} and color not in active_colors:
            out[int(r), int(c)] = BG

    for index, comp in enumerate(comps):
        if index == template_index:
            continue
        colors = {int(g[int(r), int(c)]) for r, c in comp["coords"]}
        assert colors <= {SKELETON, ANCHOR}, "non-template motifs must be bare skeleton copies"

        target_skeleton = {
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(g[int(r), int(c)]) == SKELETON
        }
        target_anchor_list = [
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(g[int(r), int(c)]) == ANCHOR
        ]
        assert len(target_anchor_list) == 1
        target_anchor = target_anchor_list[0]

        candidates = [
            cells
            for variant in variants
            if (cells := _aligned_active_cells(variant, target_skeleton, target_anchor, active_colors)) is not None
        ]
        unique = sorted(set(candidates))
        assert len(unique) == 1, "each target motif must have a unique skeleton-anchor alignment"
        for r, c, color in unique[0]:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                assert int(out[r, c]) in {BG, color}, "stamped annotation conflicts with existing content"
                out[r, c] = int(color)

    return out


def _normalize_shape(shape: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _c in shape)
    c0 = min(c for _r, c in shape)
    return frozenset((r - r0, c - c0) for r, c in shape)


@lru_cache(maxsize=None)
def _shape_pool(size: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    shapes = {frozenset({(0, 0)})}
    for _ in range(int(size) - 1):
        next_shapes: set[frozenset[tuple[int, int]]] = set()
        for shape in shapes:
            frontier = set()
            for cell in shape:
                frontier.update(neighbors4(*cell))
            for cell in frontier - set(shape):
                grown = set(shape)
                grown.add(cell)
                next_shapes.add(_normalize_shape(grown))
        shapes = next_shapes

    pool = []
    for shape in shapes:
        rows = [r for r, _c in shape]
        cols = [c for _r, c in shape]
        h = max(rows) - min(rows) + 1
        w = max(cols) - min(cols) + 1
        if h <= 5 and w <= 5:
            pool.append(tuple(sorted(shape)))
    return tuple(pool)


def _anchor_candidates(shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    cells = set(shape)
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    candidates = []
    for r, c in shape:
        deg4 = sum((nr, nc) in cells for nr, nc in neighbors4(r, c))
        deg8 = sum((nr, nc) in cells for nr, nc in neighbors8(r, c))
        if deg4 >= 2 and deg8 >= 3 and (r0 < r < r1 or c0 < c < c1):
            candidates.append((int(r), int(c)))
    return candidates


def _core_patch(shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> np.ndarray:
    cells = []
    for r, c in shape:
        color = ANCHOR if (int(r), int(c)) == tuple(anchor) else SKELETON
        cells.append((int(r), int(c), int(color)))
    return _patch_from_cells(cells)


def _ring_positions(shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    shape_set = set(shape)
    ring = set()
    for r, c in shape:
        for nr, nc in neighbors8(r, c):
            if (int(nr), int(nc)) not in shape_set:
                ring.add((int(nr), int(nc)))
    return sorted(ring)


def _build_template_patch() -> tuple[np.ndarray, dict[int, int], int, int]:
    motif_count = random.choice(MOTIF_COUNTS)
    count_profile = random.choice(COUNT_PROFILES)
    total_markers = sum(int(x) for x in count_profile)
    active_indices = [i for i, count in enumerate(count_profile) if int(count) >= 2]
    active_index = random.choice(active_indices)

    for size in random.sample(range(*CORE_SIZES), CORE_SIZES[1] - CORE_SIZES[0]):
        for shape in random.sample(list(_shape_pool(size)), len(_shape_pool(size))):
            anchors = _anchor_candidates(shape)
            if not anchors:
                continue
            for anchor in random.sample(anchors, len(anchors)):
                core_patch = _core_patch(shape, anchor)
                variants = _variant_groups(core_patch)
                if len(variants) < motif_count:
                    continue

                ring = _ring_positions(shape)
                if len(ring) < total_markers:
                    continue

                chosen_cells = random.sample(ring, total_markers)
                marker_colors = random.sample(MARKER_POOL, len(count_profile))
                color_counts = {int(color): int(count) for color, count in zip(marker_colors, count_profile)}
                full_cells = [
                    (int(r), int(c), ANCHOR if (int(r), int(c)) == tuple(anchor) else SKELETON)
                    for r, c in shape
                ]
                cursor = 0
                for color, count in color_counts.items():
                    for r, c in chosen_cells[cursor : cursor + count]:
                        full_cells.append((int(r), int(c), int(color)))
                    cursor += count

                patch = _patch_from_cells(full_cells)
                if len(_variant_groups(patch)) < motif_count:
                    continue
                active_color = int(marker_colors[active_index])
                return patch, color_counts, active_color, motif_count
    raise RuntimeError("failed to build a valid template patch")


def _active_patch(full_patch: np.ndarray, active_colors: set[int]) -> np.ndarray:
    patch = full_patch.copy()
    mask = ~np.isin(patch, [BG, SKELETON, ANCHOR, *active_colors])
    patch[mask] = BG
    return patch


def _core_only_patch(full_patch: np.ndarray) -> np.ndarray:
    patch = full_patch.copy()
    patch[~np.isin(patch, [BG, SKELETON, ANCHOR])] = BG
    return patch


def _non_bg_coords(patch: np.ndarray, top: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = map(int, top)
    return {
        (int(r) + r0, int(c) + c0)
        for r, c in np.argwhere(patch != BG)
    }


def _blocked_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set(coords)
    for r, c in coords:
        for nr, nc in neighbors8(r, c):
            blocked.add((int(nr), int(nc)))
    return blocked


def _place_patches(
    patches: list[np.ndarray],
    top_height: int,
    width: int,
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: -int(np.count_nonzero(patches[i] != BG)))
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    blocked: set[tuple[int, int]] = set()
    for index in order:
        patch = patches[index]
        h, w = map(int, patch.shape)
        if h > top_height or w > width:
            return None
        candidates = [
            (r0, c0)
            for r0 in range(0, top_height - h + 1)
            for c0 in range(0, width - w + 1)
        ]
        random.shuffle(candidates)
        placed = None
        for r0, c0 in candidates:
            coords = _non_bg_coords(patch, (r0, c0))
            if coords & blocked:
                continue
            placed = (int(r0), int(c0))
            blocked |= _blocked_coords(coords)
            break
        if placed is None:
            return None
        positions[index] = placed
    assert all(pos is not None for pos in positions)
    return [(int(r), int(c)) for r, c in positions if r is not None]


def _build_code_row(width: int, marker_counts: dict[int, int], active_color: int, total_markers: int) -> np.ndarray:
    row = np.full(width, BG, dtype=int)
    inactive_colors = [color for color in marker_counts if int(color) != int(active_color)]
    distractor_count = 1 if len(inactive_colors) == 1 else random.choice([1, 2])
    chosen_distractors = random.sample(inactive_colors, distractor_count)

    counts = {int(active_color): int(total_markers)}
    used = int(total_markers)
    remaining = width - used
    for color in chosen_distractors:
        options = [count for count in range(1, min(remaining, total_markers + 1) + 1) if count != total_markers]
        assert options
        count = random.choice(options)
        counts[int(color)] = int(count)
        remaining -= int(count)

    total_colored = sum(counts.values())
    positions = random.sample(range(width), total_colored)
    random.shuffle(positions)
    cursor = 0
    for color, count in counts.items():
        for pos in positions[cursor : cursor + count]:
            row[int(pos)] = int(color)
        cursor += int(count)
    return row


def _expected_output(
    shape: tuple[int, int],
    placements: list[tuple[int, int]],
    full_variants: list[np.ndarray],
    active_variants: list[np.ndarray],
    core_variants: list[np.ndarray],
    template_index: int,
    separator_row: int,
    code_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    for index, (full_patch, active_patch, core_patch, top) in enumerate(
        zip(full_variants, active_variants, core_variants, placements)
    ):
        r0, c0 = top
        if index == template_index:
            paste(inp, full_patch, r0, c0, transparent=BG)
            paste(out, active_patch, r0, c0, transparent=BG)
        else:
            paste(inp, core_patch, r0, c0, transparent=BG)
            paste(out, active_patch, r0, c0, transparent=BG)
    inp[separator_row, :] = SEPARATOR
    out[separator_row, :] = SEPARATOR
    inp[separator_row + 1, :] = code_row
    out[separator_row + 1, :] = code_row
    return inp, out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    active_color: int,
    total_markers: int,
    template_index: int,
    placements: list[tuple[int, int]],
    full_variants: list[np.ndarray],
    active_variants: list[np.ndarray],
    core_variants: list[np.ndarray],
    separator_row: int,
) -> bool:
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    code_counts = Counter(int(color) for color in inp[separator_row + 1, :] if int(color) != BG)
    if code_counts.get(int(active_color), 0) != int(total_markers):
        return False
    if sum(int(count == total_markers) for count in code_counts.values()) != 1:
        return False

    removed = int(np.count_nonzero((inp != BG) & (expected == BG)))
    if removed <= 0:
        return False

    for index, top in enumerate(placements):
        r0, c0 = top
        active_patch = active_variants[index]
        core_patch = core_variants[index]
        full_patch = full_variants[index]
        h, w = full_patch.shape
        if index == template_index:
            before = inp[r0 : r0 + h, c0 : c0 + w]
            after = expected[r0 : r0 + h, c0 : c0 + w]
            if np.count_nonzero((before != BG) & (after == BG)) <= 0:
                return False
        else:
            before = inp[r0 : r0 + h, c0 : c0 + w]
            after = expected[r0 : r0 + h, c0 : c0 + w]
            if np.count_nonzero((before == BG) & np.isin(after, [active_color])) <= 0:
                return False
            if not np.array_equal(np.where(core_patch != BG, before, BG), core_patch):
                return False
            if not np.array_equal(np.where(active_patch != BG, after, BG), active_patch):
                return False

    motifs_only = inp.copy()
    motifs_only[separator_row:, :] = BG
    comps = find_components(motifs_only, bg=BG, connectivity=8, monochromatic=False, return_format="dict")
    template_candidates = [
        comp
        for comp in comps
        if any(int(inp[int(r), int(c)]) not in {BG, SKELETON, ANCHOR} for r, c in comp["coords"])
    ]
    if len(template_candidates) != 1:
        return False
    template_patch = _component_patch(inp, template_candidates[0], BG)
    variants = _variant_groups(template_patch)
    for index, comp in enumerate(comps):
        colors = {int(inp[int(r), int(c)]) for r, c in comp["coords"]}
        if any(color not in {BG, SKELETON, ANCHOR} for color in colors):
            continue
        target_skeleton = {
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(inp[int(r), int(c)]) == SKELETON
        }
        target_anchor = [
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(inp[int(r), int(c)]) == ANCHOR
        ][0]
        candidates = [
            cells
            for variant in variants
            if (cells := _aligned_active_cells(variant, target_skeleton, target_anchor, {active_color})) is not None
        ]
        if len(set(candidates)) != 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    full_patch, marker_counts, active_color, motif_count = _build_template_patch()
    total_markers = sum(marker_counts.values())
    active_colors = {int(active_color)}

    oriented_full = _variant_groups(full_patch)
    if len(oriented_full) < motif_count:
        return None
    chosen_full = random.sample(oriented_full, motif_count)
    random.shuffle(chosen_full)
    template_index = random.randrange(motif_count)

    active_variants = [_active_patch(patch, active_colors) for patch in chosen_full]
    core_variants = [_core_only_patch(patch) for patch in chosen_full]

    height = random.randrange(*GRID_ROWS)
    width = random.randrange(*GRID_COLS)
    separator_row = height - random.choice(SEP_BOTTOM_MARGIN)
    if separator_row < 8 or separator_row + 1 >= height:
        return None
    top_height = separator_row - TOP_CLEARANCE
    if top_height <= 0:
        return None

    safety_patches = chosen_full
    placements = None
    for _ in range(PLACEMENT_RETRIES):
        placements = _place_patches(safety_patches, top_height, width)
        if placements is not None:
            break
    if placements is None:
        return None

    code_row = _build_code_row(width, marker_counts, active_color, total_markers)
    inp, expected = _expected_output(
        (height, width),
        placements,
        chosen_full,
        active_variants,
        core_variants,
        template_index,
        separator_row,
        code_row,
    )

    if not _semantic_checks(
        inp,
        expected,
        active_color,
        total_markers,
        template_index,
        placements,
        chosen_full,
        active_variants,
        core_variants,
        separator_row,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid frequency-filtered stamping puzzle")
