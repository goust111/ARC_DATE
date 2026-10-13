from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

TOP_COLORS = (BLUE, RED, GREEN)
L_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
TOP_ROW_MOTIFS = (
    (RED, BLUE, GREEN),
    (RED, RED, BLUE, GREEN),
    (BLUE, GREEN, RED, BLUE, RED),
    (GREEN, BLUE, BLUE, RED),
    (BLUE, RED, GREEN, RED),
    (GREEN, RED, BLUE, GREEN, BLUE),
)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _first_separator_row(grid: np.ndarray) -> int:
    grey_rows = np.where((grid == GREY).all(axis=1))[0]
    assert grey_rows.size > 0, "expected a full grey separator row"
    return int(grey_rows[0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _missing_index_from_box(
    box: tuple[int, int, int, int], missing: tuple[int, int]
) -> int:
    r0, c0, _r1, _c1 = box
    dr = missing[0] - r0
    dc = missing[1] - c0
    return int(L_OFFSETS.index((dr, dc)))


def _teal_l_objects(grid: np.ndarray, start_row: int) -> list[dict[str, object]]:
    mask = np.zeros_like(grid, dtype=bool)
    mask[start_row:, :] = grid[start_row:, :] == TEAL
    objects: list[dict[str, object]] = []
    for comp in _components4(mask):
        assert len(comp) == 3, "each teal object must contain exactly three cells"
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        assert r1 - r0 == 1 and c1 - c0 == 1, "each teal object must fit exactly in a 2x2 box"

        present = set(comp)
        box_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        missing_cells = box_cells - present
        assert len(missing_cells) == 1, "each teal object must miss exactly one 2x2 corner"

        missing = next(iter(missing_cells))
        assert grid[missing] == BLACK, "the missing corner must be black in the input"
        objects.append({"cells": comp, "box": box, "missing": missing})
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each teal L with the top-row color of its missing-corner column, then fill the remaining lower background by column.

    Input structure:
    - Row 0 is a left-to-right repeating color sequence made from red, blue, and green.
    - A full grey separator row appears below the top row.
    - Under the separator, black background contains several teal 3-cell L objects, each occupying three corners of a 2x2 box.

    Rule:
    - Keep the top row and the grey separator unchanged.
    - For each teal L below the separator, find the unique missing black corner in its 2x2 bounding box.
    - Color that missing corner with the color from the top row at the same column, then fill every remaining black cell below the separator with its column's top-row color.
    - Preserve all original teal cells and every newly completed corner.

    Key insight:
    - The missing-corner color is determined by the column lookup into the top-row sequence, and the same column reference also drives the later background fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 2, "grid is too small for the described layout"

    separator = _first_separator_row(g)
    top_row = g[0].copy()

    out = g.copy()
    for obj in _teal_l_objects(g, separator + 1):
        missing_r, missing_c = obj["missing"]
        out[missing_r, missing_c] = int(top_row[missing_c])

    if separator + 1 < h:
        lower = out[separator + 1 :, :]
        column_colors = np.broadcast_to(top_row, lower.shape)
        fill_mask = lower == BLACK
        lower[fill_mask] = column_colors[fill_mask]

    return out


def _sample_top_row(width: int) -> np.ndarray:
    motif = list(random.choice(TOP_ROW_MOTIFS))
    if random.random() < 0.5:
        motif.reverse()
    shift = random.randrange(len(motif))
    motif = motif[shift:] + motif[:shift]
    row = np.array([motif[c % len(motif)] for c in range(width)], dtype=int)
    assert len(set(map(int, row))) >= 2, "top row must expose a non-trivial color mapping"
    return row


def _box_from_anchor(anchor: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = anchor
    return r, c, r + 1, c + 1


def _boxes_compatible(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0, ar1, ac1 = _box_from_anchor(a)
    br0, bc0, br1, bc1 = _box_from_anchor(b)
    rows_overlap = ar0 <= br1 and br0 <= ar1
    cols_overlap = ac0 <= bc1 and bc0 <= ac1
    side_touch = (
        rows_overlap and (ac1 + 1 == bc0 or bc1 + 1 == ac0)
    ) or (
        cols_overlap and (ar1 + 1 == br0 or br1 + 1 == ar0)
    )
    return not ((rows_overlap and cols_overlap) or side_touch)


def _choose_anchors(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(2, height - 1) for c in range(width - 1)]

    def search(
        chosen: list[tuple[int, int]], remaining: list[tuple[int, int]]
    ) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return chosen
        need = count - len(chosen)
        if len(remaining) < need:
            return None

        order = remaining[:]
        random.shuffle(order)
        for anchor in order:
            if not all(_boxes_compatible(anchor, prev) for prev in chosen):
                continue
            filtered = [cand for cand in remaining if cand != anchor and _boxes_compatible(anchor, cand)]
            result = search(chosen + [anchor], filtered)
            if result is not None:
                return result
        return None

    return search([], candidates)


def _missing_column(anchor: tuple[int, int], missing_index: int) -> int:
    return anchor[1] + L_OFFSETS[missing_index][1]


def _assignment_features(
    anchors: list[tuple[int, int]], missing_indices: tuple[int, ...], top_row: np.ndarray
) -> tuple[int, int, int, bool, bool]:
    missing_cols = [_missing_column(anchor, idx) for anchor, idx in zip(anchors, missing_indices)]
    missing_colors = [int(top_row[c]) for c in missing_cols]

    same_orientation_diff_color = any(
        missing_indices[i] == missing_indices[j] and missing_colors[i] != missing_colors[j]
        for i, j in itertools.combinations(range(len(missing_indices)), 2)
    )
    diff_orientation_same_color = any(
        missing_indices[i] != missing_indices[j] and missing_colors[i] == missing_colors[j]
        for i, j in itertools.combinations(range(len(missing_indices)), 2)
    )
    return (
        len(set(missing_indices)),
        len(set(missing_cols)),
        len(set(missing_colors)),
        same_orientation_diff_color,
        diff_orientation_same_color,
    )


def _choose_missing_indices(
    anchors: list[tuple[int, int]], top_row: np.ndarray
) -> list[int] | None:
    target_distinct_orientations = 4 if len(anchors) >= 5 else 3
    valid: list[tuple[tuple[int, int, int], tuple[int, ...]]] = []

    for assignment in itertools.product(range(4), repeat=len(anchors)):
        distinct_orients, distinct_cols, distinct_colors, same_orient_diff_color, diff_orient_same_color = (
            _assignment_features(anchors, assignment, top_row)
        )
        if distinct_orients < target_distinct_orientations:
            continue
        if distinct_cols < 2 or distinct_colors < 2:
            continue
        if not same_orient_diff_color or not diff_orient_same_color:
            continue
        valid.append(((distinct_orients, distinct_cols, distinct_colors), assignment))

    if not valid:
        return None

    best_score = max(score for score, _ in valid)
    best_assignments = [assignment for score, assignment in valid if score == best_score]
    return list(random.choice(best_assignments))


def _render_l(grid: np.ndarray, anchor: tuple[int, int], missing_index: int) -> None:
    base_r, base_c = anchor
    for idx, (dr, dc) in enumerate(L_OFFSETS):
        if idx != missing_index:
            grid[base_r + dr, base_c + dc] = TEAL


def _validate_solver_behavior(grid: np.ndarray) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30, "generated grid must respect ARC size limits"
    assert np.array_equal(grid[1], np.full(grid.shape[1], GREY, dtype=int)), "row 1 must be the grey separator"
    assert set(map(int, np.unique(grid[0]))) <= set(TOP_COLORS), "top row must use the intended palette"
    assert len(set(map(int, grid[0]))) >= 2, "top row must expose multiple column colors"

    separator = _first_separator_row(grid)
    lower_input = grid[separator + 1 :, :]
    assert set(map(int, np.unique(lower_input))) <= {BLACK, TEAL}, "lower area should start as black plus teal L objects only"

    objects = _teal_l_objects(grid, separator + 1)
    assert 4 <= len(objects) <= 5, "generator should create several L objects"

    missing_indices = [
        _missing_index_from_box(obj["box"], obj["missing"])
        for obj in objects
    ]
    distinct_orients, distinct_cols, distinct_colors, same_orient_diff_color, diff_orient_same_color = (
        _assignment_features(
            [(obj["box"][0], obj["box"][1]) for obj in objects],
            tuple(missing_indices),
            grid[0],
        )
    )
    assert distinct_orients >= (4 if len(objects) >= 5 else 3), "orientation coverage is too weak"
    assert distinct_cols >= 2 and distinct_colors >= 2, "missing corners must span multiple column mappings"
    assert same_orient_diff_color, "generator must refute a fixed-color completion shortcut"
    assert diff_orient_same_color, "generator must refute an orientation-only color shortcut"

    out = main(grid)
    assert np.array_equal(out[0], grid[0]), "top row must stay unchanged"
    assert np.array_equal(out[separator], grid[separator]), "separator row must stay unchanged"
    assert np.array_equal(out[grid == TEAL], grid[grid == TEAL]), "teal cells must be preserved"
    assert not np.any(out[separator + 1 :, :] == BLACK), "lower background must be fully filled"

    changed = 0
    for r in range(separator + 1, grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == TEAL:
                continue
            assert out[r, c] == grid[0, c], "every non-teal lower cell must match its top-row column color"
            if out[r, c] != grid[r, c]:
                changed += 1
    assert changed > 0, "the transformation should add visible colored cells"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(9, 12)
        width = random.randint(10, 13)
        count = 5 if random.random() < 0.65 else 4

        top_row = _sample_top_row(width)
        anchors = _choose_anchors(height, width, count)
        if anchors is None:
            continue

        missing_indices = _choose_missing_indices(anchors, top_row)
        if missing_indices is None:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        grid[0, :] = top_row
        grid[1, :] = GREY
        for anchor, missing_index in zip(anchors, missing_indices):
            _render_l(grid, anchor, missing_index)

        try:
            _validate_solver_behavior(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
