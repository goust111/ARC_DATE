from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, paste


BLACK = 0
BLUE = 1
RED = 2
SEP = 5

CELL_SIZE = 5
GRID_SIZE = CELL_SIZE * 3 + 2
MARKER_TOP_LEFTS = ((1, 1), (1, 2), (2, 1), (2, 2))
MAX_GEN_ATTEMPTS = 400

AMBIGUOUS_PAIRS = {
    (1, 1): (((0, 1), (1, 1)), ((1, 0), (1, 1))),
    (1, 2): (((0, 1), (1, 1)), ((1, 1), (1, 2))),
    (2, 1): (((1, 0), (1, 1)), ((1, 1), (2, 1))),
    (2, 2): (((1, 1), (1, 2)), ((1, 1), (2, 1))),
}


def _cell_pattern_key(cell: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(cell.shape), cell.tobytes()


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if bool((grid[r, :] == SEP).all())]
    return [c for c in range(grid.shape[1]) if bool((grid[:, c] == SEP).all())]


def _segment_bounds(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for idx in separators + [size]:
        if idx <= start:
            raise ValueError("invalid separator layout")
        bounds.append((start, idx - 1))
        start = idx + 1
    if len(bounds) != 3:
        raise ValueError("expected a 3x3 separator grid")
    return bounds


def _cell_bboxes(grid: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    row_bounds = _segment_bounds(grid.shape[0], _separator_indices(grid, axis=0))
    col_bounds = _segment_bounds(grid.shape[1], _separator_indices(grid, axis=1))
    bboxes: list[list[tuple[int, int, int, int]]] = []
    for r0, r1 in row_bounds:
        row_boxes = []
        for c0, c1 in col_bounds:
            row_boxes.append((r0, c0, r1, c1))
        bboxes.append(row_boxes)
    return bboxes


def _extract_cells(grid: np.ndarray, bboxes: list[list[tuple[int, int, int, int]]]) -> list[list[np.ndarray]]:
    return [[crop_bbox(grid, bboxes[r][c]) for c in range(3)] for r in range(3)]


def _classify_cells(cells: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    counts = Counter(_cell_pattern_key(cells[r][c]) for r in range(3) for c in range(3))
    if len(counts) != 2:
        raise ValueError("expected exactly two cell patterns")

    majority_key, majority_count = counts.most_common(1)[0]
    minority_keys = [key for key in counts if key != majority_key]
    minority_key = minority_keys[0]
    minority_count = counts[minority_key]
    if majority_count <= minority_count:
        raise ValueError("majority pattern is not unique")
    if minority_count not in (2, 3):
        raise ValueError("expected 2 or 3 visible minority cells")

    majority = None
    minority = None
    minority_positions: list[tuple[int, int]] = []
    for r in range(3):
        for c in range(3):
            key = _cell_pattern_key(cells[r][c])
            if key == majority_key and majority is None:
                majority = cells[r][c]
            if key == minority_key:
                minority_positions.append((r, c))
                if minority is None:
                    minority = cells[r][c]

    if majority is None or minority is None:
        raise ValueError("failed to recover cell templates")
    return majority.copy(), minority.copy(), minority_positions


def _find_marker_top_left(cell: np.ndarray) -> tuple[int, int]:
    h, w = cell.shape
    matches: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if bool((cell[r : r + 2, c : c + 2] == BLACK).all()):
                matches.append((r, c))
    interior_matches = [
        (r, c)
        for r, c in matches
        if r > 0 and c > 0 and r + 1 < h - 1 and c + 1 < w - 1
    ]
    if len(interior_matches) != 1:
        raise ValueError("expected one interior 2x2 black marker block")
    return interior_matches[0]


def _target_positions(marker_top_left: tuple[int, int]) -> list[tuple[int, int]]:
    mr, mc = marker_top_left
    rows = (mr - 1, mr)
    cols = (mc - 1, mc)
    targets = sorted((r, c) for r in rows for c in cols)
    if any(r not in range(3) or c not in range(3) for r, c in targets):
        raise ValueError("marker does not map to a 3x3 macro blueprint")
    return targets


def _cross_mask(shape: tuple[int, int], marker_top_left: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mr, mc = marker_top_left
    mask = np.zeros((h, w), dtype=bool)
    mask[mr : mr + 2, :] = True
    mask[:, mc : mc + 2] = True
    return mask


def _marker_mask(shape: tuple[int, int], marker_top_left: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mr, mc = marker_top_left
    mask = np.zeros((h, w), dtype=bool)
    mask[mr : mr + 2, mc : mc + 2] = True
    return mask


def _erase_cross(cell: np.ndarray, marker_top_left: tuple[int, int]) -> np.ndarray:
    out = cell.copy()
    cross = _cross_mask(out.shape, marker_top_left)
    out[cross & (out != RED)] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a self-similar minority-cell blueprint and erase a protected cross inside every target cell.

    Input structure:
    - A gray separator grid splits the canvas into a 3x3 array of equal square cells.
    - Most cells contain the same blue-and-black checkerboard pattern.
    - The visible minority cells share one different pattern: a 2x2 black marker block plus some red protected pixels.

    Rule:
    - Find the majority checkerboard cell and the repeated minority cell template.
    - Read the minority template's interior 2x2 black block as a macro blueprint: its occupied rows and columns, after removing the outer cell border, indicate which 2x2 block of macro cells should be minority cells.
    - Any macro position demanded by that blueprint but still showing the majority pattern is replaced with the minority template.
    - Then, inside every blueprint-selected minority cell, erase the marker's two rows and two columns to black while keeping red pixels unchanged.
    - Majority cells and gray separator lines stay unchanged.

    Key insight:
    - The minority cell carries two meanings at once: its marker location scales up to choose the macro cells, and that same marker defines the local cross that must be erased after the macro completion.
    """
    g = np.asarray(grid, dtype=int)
    bboxes = _cell_bboxes(g)
    cells = _extract_cells(g, bboxes)
    _majority, minority, _visible_positions = _classify_cells(cells)
    marker = _find_marker_top_left(minority)
    targets = set(_target_positions(marker))
    erased = _erase_cross(minority, marker)

    out = g.copy()
    for r in range(3):
        for c in range(3):
            if (r, c) not in targets:
                continue
            r0, c0, _r1, _c1 = bboxes[r][c]
            paste(out, minority, r0, c0)
            paste(out, erased, r0, c0)
    return out


def _checkerboard_cell(phase: int) -> np.ndarray:
    return np.fromfunction(
        lambda r, c: np.where((r + c + phase) % 2 == 0, BLUE, BLACK),
        (CELL_SIZE, CELL_SIZE),
        dtype=int,
    ).astype(int)


def _sample_red_positions(
    majority: np.ndarray, marker_top_left: tuple[int, int]
) -> list[tuple[int, int]]:
    block = _marker_mask(majority.shape, marker_top_left)
    cross = _cross_mask(majority.shape, marker_top_left)
    protected_blue = [
        (r, c)
        for r in range(CELL_SIZE)
        for c in range(CELL_SIZE)
        if cross[r, c] and not block[r, c] and int(majority[r, c]) == BLUE
    ]
    all_slots = [
        (r, c)
        for r in range(CELL_SIZE)
        for c in range(CELL_SIZE)
        if not block[r, c]
    ]

    for _ in range(80):
        reds = [random.choice(protected_blue)]
        extra_count = random.randint(0, 3)
        extra_pool = [pos for pos in all_slots if pos not in reds]
        if extra_count:
            reds.extend(random.sample(extra_pool, min(extra_count, len(extra_pool))))
        reds = sorted(set(reds))

        still_erasable = any(pos not in reds for pos in protected_blue)
        has_off_cross = any(not cross[r, c] for r, c in reds)
        if still_erasable and (has_off_cross or len(reds) == 1 or random.random() < 0.5):
            return reds
    raise RuntimeError("failed to sample non-degenerate red positions")


def _minority_template(
    phase: int, marker_top_left: tuple[int, int], red_positions: list[tuple[int, int]]
) -> np.ndarray:
    cell = _checkerboard_cell(phase)
    mr, mc = marker_top_left
    cell[mr : mr + 2, mc : mc + 2] = BLACK
    for r, c in red_positions:
        cell[r, c] = RED
    return cell


def _visible_minority_positions(
    marker_top_left: tuple[int, int], red_positions: list[tuple[int, int]], phase: int
) -> list[tuple[int, int]]:
    targets = _target_positions(marker_top_left)
    score = phase + sum((r + 1) * 7 + (c + 1) * 11 for r, c in red_positions)
    if score % 3 == 0:
        missing_index = score % 4
        return [pos for i, pos in enumerate(targets) if i != missing_index]
    pair_options = AMBIGUOUS_PAIRS[marker_top_left]
    return list(pair_options[score % 2])


def _render_macro_grid(
    majority: np.ndarray,
    minority: np.ndarray,
    selected_positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), SEP, dtype=int)
    for r in range(3):
        for c in range(3):
            patch = minority if (r, c) in set(selected_positions) else majority
            r0 = r * (CELL_SIZE + 1)
            c0 = c * (CELL_SIZE + 1)
            paste(grid, patch, r0, c0)
    return grid


def _case_metadata() -> dict[str, object]:
    phase = random.randint(0, 1)
    marker_top_left = random.choice(MARKER_TOP_LEFTS)
    majority = _checkerboard_cell(phase)
    red_positions = _sample_red_positions(majority, marker_top_left)
    minority = _minority_template(phase, marker_top_left, red_positions)
    erased = _erase_cross(minority, marker_top_left)
    target_positions = _target_positions(marker_top_left)
    visible_positions = _visible_minority_positions(marker_top_left, red_positions, phase)
    return {
        "phase": phase,
        "marker_top_left": marker_top_left,
        "red_positions": red_positions,
        "majority": majority,
        "minority": minority,
        "erased": erased,
        "target_positions": target_positions,
        "visible_positions": visible_positions,
    }


def _candidate_targets_from_visible(visible_positions: list[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    candidates = []
    for marker_top_left in MARKER_TOP_LEFTS:
        targets = set(_target_positions(marker_top_left))
        if set(visible_positions).issubset(targets):
            candidates.append(targets)
    return candidates


def _validate_constructed_case(inp: np.ndarray, expected: np.ndarray, meta: dict[str, object]) -> bool:
    cells = _extract_cells(inp, _cell_bboxes(inp))
    majority, minority, minority_positions = _classify_cells(cells)
    marker = _find_marker_top_left(minority)
    targets = _target_positions(marker)
    erased = _erase_cross(minority, marker)

    target_set = set(meta["target_positions"])
    visible_set = set(meta["visible_positions"])
    red_positions = list(meta["red_positions"])
    marker_mask = _marker_mask(minority.shape, marker)
    cross_mask = _cross_mask(minority.shape, marker)

    if not np.array_equal(majority, meta["majority"]):
        return False
    if not np.array_equal(minority, meta["minority"]):
        return False
    if marker != meta["marker_top_left"]:
        return False
    if set(minority_positions) != visible_set:
        return False
    if set(targets) != target_set:
        return False
    if not np.array_equal(erased, meta["erased"]):
        return False
    if np.array_equal(minority, erased):
        return False
    if len(minority_positions) not in (2, 3):
        return False
    if visible_set == target_set:
        return False

    blue_protected = [
        (r, c)
        for r, c in red_positions
        if cross_mask[r, c] and not marker_mask[r, c] and int(majority[r, c]) == BLUE
    ]
    if not blue_protected:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    candidates = _candidate_targets_from_visible(minority_positions)
    if len(minority_positions) == 2 and len(candidates) < 2:
        return False
    if len(minority_positions) == 3 and len(candidates) != 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    meta = _case_metadata()
    inp = _render_macro_grid(meta["majority"], meta["minority"], meta["visible_positions"])
    expected = _render_macro_grid(meta["majority"], meta["erased"], meta["target_positions"])
    if _validate_constructed_case(inp, expected, meta):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
