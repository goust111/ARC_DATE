from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, neighbors4, neighbors8


BG = Color.ORANGE
SKELETON = Color.GREY
ANCHOR = Color.MAROON
MARKER = Color.RED
BLOCK = Color.BLACK

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
MOTIF_COUNT_RANGE = (3, 4)
SKELETON_SIZE_RANGE = (6, 9)
RED_COUNT_RANGE = (2, 3)
MAX_SHAPE_ATTEMPTS = 300
MAX_PLACE_ATTEMPTS = 250
MAX_GEN_ATTEMPTS = 500


def _transform_coord(coord: tuple[int, int], rotation: int, flip: bool) -> tuple[int, int]:
    r, c = map(int, coord)
    if flip:
        c = -c
    for _ in range(int(rotation) % 4):
        r, c = c, -r
    return int(r), int(c)


def _dihedral_variants(
    skeleton_rel: frozenset[tuple[int, int]],
    red_rel: frozenset[tuple[int, int]],
) -> list[dict]:
    variants = []
    seen = set()
    for flip in (False, True):
        for rotation in range(4):
            skel = frozenset(_transform_coord(cell, rotation, flip) for cell in skeleton_rel)
            reds = frozenset(_transform_coord(cell, rotation, flip) for cell in red_rel)
            key = tuple(sorted(skel))
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "rotation": int(rotation),
                    "flip": bool(flip),
                    "skeleton_rel": skel,
                    "red_rel": reds,
                }
            )
    return variants


def _normalize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _frontier8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    frontier = set()
    for r, c in cells:
        for nr, nc in neighbors8(r, c):
            if (nr, nc) not in cells:
                frontier.add((int(nr), int(nc)))
    return frontier


def _random_shape(size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < int(size):
        frontier = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((int(nr), int(nc)))
        cells.add(random.choice(tuple(frontier)))
    return _normalize_shape(cells)


def _build_base_motif(motif_count: int) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]], list[dict]] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        shape = _random_shape(random.randint(*SKELETON_SIZE_RANGE))
        r0, c0, r1, c1 = _shape_bbox(shape)
        if r1 - r0 + 1 > 6 or c1 - c0 + 1 > 6:
            continue
        frontier = sorted(_frontier8(shape))
        if len(frontier) < 3:
            continue

        anchors = list(shape)
        random.shuffle(anchors)
        for anchor_cell in anchors:
            ar, ac = anchor_cell
            skeleton_rel = frozenset((r - ar, c - ac) for r, c in shape)
            variants = _dihedral_variants(skeleton_rel, frozenset())
            if len(variants) != 8 or len(variants) < int(motif_count):
                continue

            red_count = random.randint(*RED_COUNT_RANGE)
            for _sample_try in range(20):
                red_cells = random.sample(frontier, red_count)
                red_rel = frozenset((r - ar, c - ac) for r, c in red_cells)
                return skeleton_rel, red_rel, _dihedral_variants(skeleton_rel, red_rel)
    return None


def _rel_bbox(cells: frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _absolute_cells(anchor: tuple[int, int], rels: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in rels}


def _place_motifs(
    grid_shape: tuple[int, int],
    chosen_variants: list[dict],
    reference_index: int,
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full(grid_shape, int(BG), dtype=int)
    forbidden: set[tuple[int, int]] = set()
    motifs = []

    order = list(range(len(chosen_variants)))
    random.shuffle(order)

    placed_by_index: dict[int, dict] = {}
    for idx in order:
        variant = chosen_variants[idx]
        all_rel = variant["skeleton_rel"] | variant["red_rel"]
        min_r, min_c, max_r, max_c = _rel_bbox(all_rel)
        row_range = range(-min_r, grid.shape[0] - max_r)
        col_range = range(-min_c, grid.shape[1] - max_c)
        candidates = [(r, c) for r in row_range for c in col_range]
        random.shuffle(candidates)

        placed = None
        for anchor in candidates[:MAX_PLACE_ATTEMPTS]:
            skeleton_abs = _absolute_cells(anchor, variant["skeleton_rel"])
            red_abs = _absolute_cells(anchor, variant["red_rel"])
            all_abs = skeleton_abs | red_abs
            if any(cell in forbidden for cell in all_abs):
                continue
            if not all(_in_bounds(grid.shape, cell) for cell in all_abs):
                continue
            placed = {
                "anchor": tuple(map(int, anchor)),
                "skeleton_abs": skeleton_abs,
                "grey_abs": skeleton_abs - {tuple(map(int, anchor))},
                "full_red_abs": red_abs,
                "is_reference": idx == int(reference_index),
                "input_red_abs": red_abs if idx == int(reference_index) else set(),
                "skeleton_rel": variant["skeleton_rel"],
            }
            break

        if placed is None:
            return None

        placed_by_index[idx] = placed
        for r, c in placed["grey_abs"]:
            grid[int(r), int(c)] = int(SKELETON)
        ar, ac = placed["anchor"]
        grid[int(ar), int(ac)] = int(ANCHOR)
        for r, c in placed["input_red_abs"]:
            grid[int(r), int(c)] = int(MARKER)

        for r, c in placed["skeleton_abs"] | placed["full_red_abs"]:
            forbidden.add((int(r), int(c)))
            for nr, nc in neighbors8(r, c):
                if _in_bounds(grid.shape, (int(nr), int(nc))):
                    forbidden.add((int(nr), int(nc)))

    motifs = [placed_by_index[i] for i in range(len(chosen_variants))]
    return grid, motifs


def _find_block(input_grid: np.ndarray) -> tuple[int, int]:
    black = np.argwhere(input_grid == int(BLOCK))
    assert len(black) == 4, "expected exactly one 2x2 black block"
    r0, c0 = black.min(axis=0)
    r1, c1 = black.max(axis=0)
    assert int(r1) == int(r0) + 1 and int(c1) == int(c0) + 1, "black marker must be 2x2"
    assert np.all(input_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(BLOCK))
    return int(r0), int(c0)


def _extract_motifs(input_grid: np.ndarray) -> list[dict]:
    mask = np.isin(input_grid, [int(SKELETON), int(ANCHOR), int(MARKER)])
    comps = components_mask(mask, connectivity=8)
    motifs = []
    for coords in comps:
        coords = [tuple(map(int, cell)) for cell in coords.tolist()]
        anchors = [cell for cell in coords if int(input_grid[cell]) == int(ANCHOR)]
        assert len(anchors) == 1, "each motif must contain exactly one anchor"
        anchor = anchors[0]
        ar, ac = anchor
        skeleton_abs = {cell for cell in coords if int(input_grid[cell]) in (int(SKELETON), int(ANCHOR))}
        red_abs = {cell for cell in coords if int(input_grid[cell]) == int(MARKER)}
        skeleton_rel = frozenset((r - ar, c - ac) for r, c in skeleton_abs)
        red_rel = frozenset((r - ar, c - ac) for r, c in red_abs)
        motifs.append(
            {
                "anchor": anchor,
                "skeleton_abs": skeleton_abs,
                "red_abs": red_abs,
                "skeleton_rel": skeleton_rel,
                "red_rel": red_rel,
            }
        )
    return motifs


def _propagate_markers(input_grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    motifs = _extract_motifs(input_grid)
    assert 3 <= len(motifs) <= 4, "expected 3 or 4 motifs"

    references = [motif for motif in motifs if motif["red_abs"]]
    assert len(references) == 1, "expected exactly one fully marked motif"
    reference = references[0]

    variants = []
    for flip in (False, True):
        for rotation in range(4):
            skel = frozenset(_transform_coord(cell, rotation, flip) for cell in reference["skeleton_rel"])
            reds = frozenset(_transform_coord(cell, rotation, flip) for cell in reference["red_rel"])
            variants.append((skel, reds))

    out = np.array(input_grid, dtype=int, copy=True)
    added_reds: set[tuple[int, int]] = set()
    for motif in motifs:
        if motif is reference:
            continue
        candidates = {reds for skel, reds in variants if skel == motif["skeleton_rel"]}
        assert len(candidates) == 1, "target motif must admit a unique dihedral alignment"
        red_rel = next(iter(candidates))
        ar, ac = motif["anchor"]
        for dr, dc in red_rel:
            r, c = int(ar + dr), int(ac + dc)
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            assert int(out[r, c]) in (int(BG), int(MARKER))
            out[r, c] = int(MARKER)
            added_reds.add((r, c))
    return out, added_reds


def _apply_cross_erase(grid: np.ndarray, block_top_left: tuple[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r0, c0 = map(int, block_top_left)
    rows = {int(r0), int(r0 + 1)}
    cols = {int(c0), int(c0 + 1)}
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if (r in rows or c in cols) and int(out[r, c]) != int(MARKER):
                out[r, c] = int(BLOCK)
    return out


def _render_expected_from_blueprint(base_grid: np.ndarray, motifs: list[dict], block_top_left: tuple[int, int]) -> np.ndarray:
    propagated = np.array(base_grid, dtype=int, copy=True)
    for motif in motifs:
        if motif["is_reference"]:
            continue
        for r, c in motif["full_red_abs"]:
            propagated[int(r), int(c)] = int(MARKER)
    return _apply_cross_erase(propagated, block_top_left)


def _choose_block(base_grid: np.ndarray, motifs: list[dict]) -> tuple[tuple[int, int], np.ndarray] | None:
    candidates = []
    for r0 in range(base_grid.shape[0] - 1):
        for c0 in range(base_grid.shape[1] - 1):
            if not np.all(base_grid[r0 : r0 + 2, c0 : c0 + 2] == int(BG)):
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(base_grid.shape[0], r0 + 3)
            cc1 = min(base_grid.shape[1], c0 + 3)
            if not np.all(base_grid[rr0:rr1, cc0:cc1] == int(BG)):
                continue
            candidates.append((int(r0), int(c0)))
    random.shuffle(candidates)

    target_reds = set().union(*(motif["full_red_abs"] for motif in motifs if not motif["is_reference"]))
    non_red_motif_cells = set().union(*(motif["skeleton_abs"] for motif in motifs))

    for block_top_left in candidates:
        r0, c0 = block_top_left
        rows = {int(r0), int(r0 + 1)}
        cols = {int(c0), int(c0 + 1)}
        protected_target_reds = {cell for cell in target_reds if cell[0] in rows or cell[1] in cols}
        if not protected_target_reds:
            continue
        erased_motif = {cell for cell in non_red_motif_cells if cell[0] in rows or cell[1] in cols}
        if not erased_motif:
            continue

        input_grid = np.array(base_grid, dtype=int, copy=True)
        input_grid[r0 : r0 + 2, c0 : c0 + 2] = int(BLOCK)
        expected = _render_expected_from_blueprint(input_grid, motifs, block_top_left)
        if np.array_equal(expected, input_grid):
            continue
        if np.array_equal(expected, _apply_cross_erase(input_grid, block_top_left)):
            continue
        propagated_only = np.array(input_grid, dtype=int, copy=True)
        for cell in target_reds:
            propagated_only[cell] = int(MARKER)
        if np.array_equal(expected, propagated_only):
            continue
        return block_top_left, expected
    return None


def main(input_grid: np.ndarray) -> np.ndarray:
    """Transfer the red side-markers through motif alignment, then erase the marker-block cross except for red cells.

    Input structure:
    - The grid is mostly orange background and contains 3 or 4 separated motifs plus one isolated 2x2 black block.
    - Each motif is an 8-connected grey skeleton with exactly one maroon anchor cell; exactly one motif also has 2 or 3 adjacent red cells.
    - All motifs share the same grey+maroon structure up to a rotation or reflection, while the black block contributes only its two rows and two columns.

    Rule:
    - Identify the unique fully marked motif and every incomplete copy.
    - For each incomplete motif, find the unique dihedral transform that aligns the reference grey skeleton and maroon anchor to the target grey skeleton and maroon anchor, then stamp the transformed red cells onto that copy.
    - After all red cells are propagated, erase the two rows and two columns defined by the 2x2 black block by repainting every non-red cell on those lines to black, while preserving red cells and leaving all other cells unchanged.

    Key insight:
    - The red pattern is not read directly from the incomplete motifs: it must first be transferred by skeleton+anchor alignment, and those transferred red cells are exactly the pixels that survive the later cross erasure.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert int(get_bg(grid)) == int(BG)

    block_top_left = _find_block(grid)
    propagated, _added_reds = _propagate_markers(grid)
    return _apply_cross_erase(propagated, block_top_left)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid_shape = (random.randint(*GRID_H_RANGE), random.randint(*GRID_W_RANGE))
        motif_count = random.randint(*MOTIF_COUNT_RANGE)
        base = _build_base_motif(motif_count)
        if base is None:
            continue
        skeleton_rel, red_rel, variants = base
        del skeleton_rel, red_rel

        chosen_variants = random.sample(variants, motif_count)
        reference_index = random.randrange(motif_count)
        placed = _place_motifs(grid_shape, chosen_variants, reference_index)
        if placed is None:
            continue
        base_grid, motifs = placed

        block_choice = _choose_block(base_grid, motifs)
        if block_choice is None:
            continue
        block_top_left, expected = block_choice

        input_grid = np.array(base_grid, dtype=int, copy=True)
        r0, c0 = block_top_left
        input_grid[r0 : r0 + 2, c0 : c0 + 2] = int(BLOCK)

        solved = main(input_grid)
        if not np.array_equal(solved, expected):
            continue

        propagated, added_reds = _propagate_markers(input_grid)
        rows = {int(r0), int(r0 + 1)}
        cols = {int(c0), int(c0 + 1)}
        if not any(r in rows or c in cols for r, c in added_reds):
            continue
        if not any(
            (r in rows or c in cols) and int(propagated[r, c]) in (int(SKELETON), int(ANCHOR))
            for r in range(propagated.shape[0])
            for c in range(propagated.shape[1])
        ):
            continue
        if np.array_equal(solved, _apply_cross_erase(input_grid, block_top_left)):
            continue

        return input_grid

    raise RuntimeError("failed to generate a valid ARC instance")
