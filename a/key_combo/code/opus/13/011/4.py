from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
ANCHOR = 1
RED = 2
GREEN = 3
YELLOW = 4
SKELETON = 5
MARKER_COLORS = (RED, YELLOW)
RING_PALETTE = np.array([GREEN, YELLOW, RED], dtype=int)

MAX_GENERATION_ATTEMPTS = 400
GRID_ROWS_RANGE = (18, 25)
GRID_COLS_RANGE = (18, 25)
MOTIF_COUNT_OPTIONS = (4, 5)
SKELETON_SIZE_RANGE = (3, 7)
MIN_COMPONENT_GAP = 3


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 8,
    monochromatic: bool = False,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    neighbor_fn = neighbors4 if connectivity == 4 else neighbors8
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbor_fn(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    if monochromatic:
                        ok = int(grid[nr, nc]) == color
                    else:
                        ok = int(grid[nr, nc]) != int(bg)
                    if ok:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            comps.append({"coords": coord_arr, "bbox": bbox_coords(coord_arr), "area": len(coords)})
    return comps


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
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _colored_key(patch: np.ndarray, colors: set[int]) -> tuple[tuple[int, int, int], ...]:
    coords = [
        (int(r), int(c), int(patch[r, c]))
        for r, c in np.argwhere(np.isin(patch, list(colors)))
    ]
    r0 = min(r for r, _c, _color in coords)
    c0 = min(c for _r, c, _color in coords)
    return tuple(sorted((r - r0, c - c0, color) for r, c, color in coords))


def _translated_marker_cells(
    variant: np.ndarray,
    target_skeleton: set[tuple[int, int]],
    target_anchor: tuple[int, int],
) -> tuple[tuple[int, int, int], ...] | None:
    variant_skeleton = {tuple(map(int, rc)) for rc in np.argwhere(variant == SKELETON)}
    variant_anchor_cells = [tuple(map(int, rc)) for rc in np.argwhere(variant == ANCHOR)]
    if len(variant_anchor_cells) != 1:
        return None
    anchor_r, anchor_c = variant_anchor_cells[0]
    dr = int(target_anchor[0]) - anchor_r
    dc = int(target_anchor[1]) - anchor_c
    if {(r + dr, c + dc) for r, c in variant_skeleton} != target_skeleton:
        return None
    cells = [
        (int(r + dr), int(c + dc), int(variant[r, c]))
        for r, c in np.argwhere(np.isin(variant, MARKER_COLORS))
    ]
    return tuple(sorted(cells))


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the complete source annotations onto every anchor-aligned skeleton copy, then paint nearest-anchor Manhattan rings on the background.

    Input structure:
    - A black background contains 4-5 separated 8-connected foreground motifs.
    - Every motif shares the same grey skeleton shape and contains exactly one blue anchor pixel.
    - Exactly one motif is fully annotated with extra red and yellow marker pixels, while the others contain only the skeleton and anchor.

    Rule:
    - Identify the fully annotated motif as the source template.
    - For each incomplete motif, find the unique rotation/reflection and translation that maps the source skeleton and blue anchor onto the target skeleton and blue anchor, then stamp only the red/yellow marker pixels.
    - After all stamps are copied, color every remaining background cell by its Manhattan distance to the nearest blue anchor using the repeating palette green, yellow, red for distances 1, 2, 3, 4, ...

    Key insight:
    - The blue anchor has a dual role: it disambiguates the dihedral alignment of the skeleton copy and simultaneously seeds the Manhattan-distance ring field.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    assert len(comps) >= 2, "expected at least one source motif and one target motif"

    source_indices = [
        idx
        for idx, comp in enumerate(comps)
        if any(int(g[int(r), int(c)]) in MARKER_COLORS for r, c in comp["coords"])
    ]
    assert len(source_indices) == 1, "expected exactly one fully marked source motif"
    source_comp = comps[source_indices[0]]
    source_patch = _component_patch(g, source_comp, bg)
    variants = _dihedral_variants(source_patch)

    out = g.copy()
    anchors: list[tuple[int, int]] = []
    for idx, comp in enumerate(comps):
        anchor_cells = [(int(r), int(c)) for r, c in comp["coords"] if int(g[int(r), int(c)]) == ANCHOR]
        assert len(anchor_cells) == 1, "each motif must contain exactly one blue anchor"
        anchor = anchor_cells[0]
        anchors.append(anchor)

        if idx == source_indices[0]:
            continue

        allowed = {SKELETON, ANCHOR}
        assert all(int(g[int(r), int(c)]) in allowed for r, c in comp["coords"]), "target motif already has copied markers"
        target_skeleton = {
            (int(r), int(c))
            for r, c in comp["coords"]
            if int(g[int(r), int(c)]) == SKELETON
        }
        candidates = {
            cells
            for variant in variants
            if (cells := _translated_marker_cells(variant, target_skeleton, anchor)) is not None
        }
        assert len(candidates) == 1, "each target motif must admit a unique dihedral alignment"
        for r, c, color in next(iter(candidates)):
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            assert int(out[r, c]) in (int(bg), int(color)), "stamped markers must only land on background"
            out[r, c] = int(color)

    rows, cols = np.indices(out.shape)
    dist_stack = np.stack([np.abs(rows - ar) + np.abs(cols - ac) for ar, ac in anchors], axis=0)
    distances = dist_stack.min(axis=0)
    ring_colors = RING_PALETTE[(distances - 1) % len(RING_PALETTE)]
    bg_mask = out == int(bg)
    out[bg_mask] = ring_colors[bg_mask]
    return out


def _normalize_points(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in points)
    c0 = min(c for _r, c in points)
    return {(r - r0, c - c0) for r, c in points}


def _random_connected_shape(rng: np.random.Generator, size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = sorted({nbr for cell in cells for nbr in neighbors4(*cell)} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return _normalize_points(cells)


def _adjacent_empty_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    around = set()
    for r, c in cells:
        for nr, nc in neighbors8(r, c):
            if (nr, nc) not in cells:
                around.add((nr, nc))
    return sorted(around)


def _make_patch(
    skeleton: set[tuple[int, int]],
    anchor: tuple[int, int],
    red_markers: set[tuple[int, int]],
    yellow_markers: set[tuple[int, int]],
) -> np.ndarray:
    color_map = {cell: SKELETON for cell in skeleton}
    color_map[anchor] = ANCHOR
    for cell in red_markers:
        color_map[cell] = RED
    for cell in yellow_markers:
        color_map[cell] = YELLOW
    coords = np.array(list(color_map.keys()), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for (r, c), color in color_map.items():
        patch[r - r0, c - c0] = int(color)
    return patch


def _strip_markers(patch: np.ndarray) -> np.ndarray:
    stripped = patch.copy()
    stripped[np.isin(stripped, MARKER_COLORS)] = BG
    return stripped


def _valid_target_variants(source_patch: np.ndarray) -> list[np.ndarray]:
    groups: dict[tuple[tuple[int, int, int], ...], dict[tuple[tuple[int, int], bytes], np.ndarray]] = {}
    for variant in _dihedral_variants(source_patch):
        combo_key = _colored_key(variant, {SKELETON, ANCHOR})
        full_key = (variant.shape, variant.tobytes())
        groups.setdefault(combo_key, {})[full_key] = variant
    valid = [next(iter(full_variants.values())).copy() for full_variants in groups.values() if len(full_variants) == 1]
    random.shuffle(valid)
    return valid


def _patch_cells(patch: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(patch != BG)}


def _absolute_cells(cells: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in cells}


def _min_manhattan(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _paint_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = dst[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_ROWS_RANGE))
    w = int(rng.integers(*GRID_COLS_RANGE))
    motif_count = int(rng.choice(MOTIF_COUNT_OPTIONS))
    target_count = motif_count - 1

    source_patch = None
    candidate_targets: list[np.ndarray] = []
    for _ in range(120):
        skeleton_size = int(rng.integers(*SKELETON_SIZE_RANGE))
        skeleton = _random_connected_shape(rng, skeleton_size)
        anchor_candidates = _adjacent_empty_cells(skeleton)
        rng.shuffle(anchor_candidates)
        for anchor in anchor_candidates:
            occupied = set(skeleton) | {anchor}
            marker_pool = _adjacent_empty_cells(occupied)
            if len(marker_pool) < 4:
                continue
            rng.shuffle(marker_pool)
            red_count = 1 + int(rng.integers(2))
            yellow_count = 1 + int(rng.integers(2))
            total_markers = red_count + yellow_count
            if len(marker_pool) < total_markers:
                continue
            chosen = marker_pool[:total_markers]
            red_markers = set(chosen[:red_count])
            yellow_markers = set(chosen[red_count:red_count + yellow_count])
            patch = _make_patch(skeleton, anchor, red_markers, yellow_markers)
            variants = _dihedral_variants(patch)
            patch = variants[int(rng.integers(len(variants)))]
            valid_targets = _valid_target_variants(patch)
            if len(valid_targets) >= target_count:
                source_patch = patch
                candidate_targets = valid_targets
                break
        if source_patch is not None:
            break
    if source_patch is None:
        return None

    chosen_indices = rng.choice(len(candidate_targets), size=target_count, replace=False)
    full_patches = [source_patch] + [candidate_targets[int(i)] for i in chosen_indices]
    input_patches = [source_patch] + [_strip_markers(patch) for patch in full_patches[1:]]

    placements: list[tuple[int, int]] = []
    placed_cells: list[set[tuple[int, int]]] = []
    order = sorted(range(motif_count), key=lambda i: input_patches[i].size, reverse=True)
    top_lefts: list[tuple[int, int] | None] = [None] * motif_count
    for idx in order:
        input_patch = input_patches[idx]
        full_patch = full_patches[idx]
        input_cells = _patch_cells(input_patch)
        full_cells = _patch_cells(full_patch)
        fh, fw = full_patch.shape
        placed = False
        for _ in range(250):
            top = int(rng.integers(0, h - fh + 1))
            left = int(rng.integers(0, w - fw + 1))
            abs_input = _absolute_cells(input_cells, top, left)
            abs_full = _absolute_cells(full_cells, top, left)
            if any(abs_full & other for other in placed_cells):
                continue
            if any(_min_manhattan(abs_input, other) < MIN_COMPONENT_GAP for other in placements):
                continue
            top_lefts[idx] = (top, left)
            placements.append(abs_input)
            placed_cells.append(abs_full)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), BG, dtype=int)
    for patch, loc in zip(input_patches, top_lefts):
        assert loc is not None
        _paint_patch(grid, patch, loc[0], loc[1])

    comps = find_components(grid, bg=BG, connectivity=8, monochromatic=False)
    if len(comps) != motif_count:
        return None
    if sum(any(int(grid[int(r), int(c)]) in MARKER_COLORS for r, c in comp["coords"]) for comp in comps) != 1:
        return None

    try:
        expected = _expected_output(grid)
    except Exception:
        return None
    if expected.shape != grid.shape or np.any(expected == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
