from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

CELL_SIZE = 5
CELL_GAP = 1
MACRO_ROWS = 3
MACRO_COLS = 4
GRID_SHAPE = (
    MACRO_ROWS * CELL_SIZE + (MACRO_ROWS - 1) * CELL_GAP,
    MACRO_COLS * CELL_SIZE + (MACRO_COLS - 1) * CELL_GAP,
)
HOLE_TO_COLOR = {0: GREY, 1: BLUE, 2: RED, 3: GREEN}
MAX_GENERATION_ATTEMPTS = 1200

_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()
_MASK_LIBRARY: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}


def _parse_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_BASE_MASKS = {
    0: [
        _parse_mask(["10001", "11011", "01110", "11011", "10001"]),
        _parse_mask(["11100", "11110", "01111", "11110", "00111"]),
        _parse_mask(["11011", "11111", "00100", "11111", "11011"]),
    ],
    1: [
        _parse_mask(["11111", "10001", "10111", "10001", "11111"]),
        _parse_mask(["11111", "10001", "11101", "10001", "11111"]),
        _parse_mask(["11111", "10101", "10001", "10111", "11111"]),
    ],
    2: [
        _parse_mask(["11111", "10001", "11111", "10001", "11111"]),
        _parse_mask(["11111", "10101", "10101", "10101", "11111"]),
        _parse_mask(["11111", "10101", "10111", "10101", "11111"]),
    ],
    3: [
        _parse_mask(["11111", "10001", "11111", "10101", "11111"]),
        _parse_mask(["11111", "10101", "11111", "10001", "11111"]),
        _parse_mask(["11111", "10001", "11111", "10101", "11110"]),
    ],
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_holes(mask: np.ndarray) -> int:
    comps = _components(mask)
    assert len(comps) == 1, "each cell must contain exactly one 4-connected teal shape"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    cropped = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    cropped[coords[:, 0] - r0, coords[:, 1] - c0] = True

    padded = np.pad(cropped, 1, constant_values=False)
    holes = 0
    for bg_comp in _components(~padded):
        rr = bg_comp[:, 0]
        cc = bg_comp[:, 1]
        touches_edge = np.any(
            (rr == 0)
            | (rr == padded.shape[0] - 1)
            | (cc == 0)
            | (cc == padded.shape[1] - 1)
        )
        if not touches_edge:
            holes += 1
    return int(holes)


def _macro_spans(indices: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for idx in indices + [limit]:
        if idx - prev > 1:
            spans.append((prev + 1, idx - 1))
        prev = idx
    return spans


def _parse_macro_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    zero_rows = [r for r in range(h) if np.all(grid[r] == BLACK)]
    zero_cols = [c for c in range(w) if np.all(grid[:, c] == BLACK)]
    row_spans = _macro_spans(zero_rows, h)
    col_spans = _macro_spans(zero_cols, w)
    assert len(row_spans) == MACRO_ROWS and len(col_spans) == MACRO_COLS
    assert all(r1 - r0 + 1 == CELL_SIZE for r0, r1 in row_spans)
    assert all(c1 - c0 + 1 == CELL_SIZE for c0, c1 in col_spans)
    assert len(zero_rows) == MACRO_ROWS - 1 and len(zero_cols) == MACRO_COLS - 1
    return row_spans, col_spans


def _cell_mask(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int]) -> np.ndarray:
    r0, r1 = row_span
    c0, c1 = col_span
    cell = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert cell.shape == (CELL_SIZE, CELL_SIZE)
    assert np.all((cell == BLACK) | (cell == TEAL))
    return cell == TEAL


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans, col_spans = _parse_macro_grid(grid)
    hole_counts = np.zeros((MACRO_ROWS, MACRO_COLS), dtype=int)
    areas = np.zeros((MACRO_ROWS, MACRO_COLS), dtype=int)
    for mr, row_span in enumerate(row_spans):
        for mc, col_span in enumerate(col_spans):
            mask = _cell_mask(grid, row_span, col_span)
            areas[mr, mc] = int(mask.sum())
            hole_counts[mr, mc] = _count_holes(mask)
            assert 10 <= areas[mr, mc] <= 20
            assert 0 <= hole_counts[mr, mc] <= 3
    return hole_counts, areas, row_spans, col_spans


def _stable_desc_order(values: list[int] | np.ndarray) -> tuple[int, ...]:
    vals = [int(v) for v in values]
    return tuple(sorted(range(len(vals)), key=lambda idx: (-vals[idx], idx)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count holes in each teal macro-cell, sort macro-columns by column hole total, then recolor each cell by its own hole count.

    Input structure:
    - The grid is a 3x4 macro-layout of 5x5 cell regions separated by single-pixel black gutter rows and columns.
    - Each cell contains exactly one 4-connected teal shape on black background.
    - A cell shape may enclose 0 to 3 black 4-connected holes.

    Rule:
    - Parse the 5x5 cells using the full-black gutter rows and columns.
    - Count each cell's hole count h, then sum hole counts within each macro-column to get H.
    - Reorder the four macro-columns from largest H to smallest H, preserving whole column contents and black gutters.
    - After sorting, recolor every teal cell by its own h using 0->grey, 1->blue, 2->red, 3->green, while all black cells remain black.

    Key insight:
    - The same topological measurement is used twice at different scales: per-cell hole counts determine recoloring, while column-wise aggregates of those same counts determine the spatial reordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    hole_counts, _areas, row_spans, col_spans = _analyze(g)
    col_sums = hole_counts.sum(axis=0)
    order = _stable_desc_order(col_sums)

    out = np.zeros_like(g)
    for new_c, old_c in enumerate(order):
        target_c0, target_c1 = col_spans[new_c]
        source_c0, source_c1 = col_spans[old_c]
        for mr, row_span in enumerate(row_spans):
            r0, r1 = row_span
            cell = g[r0 : r1 + 1, source_c0 : source_c1 + 1]
            recolored = np.where(cell == TEAL, HOLE_TO_COLOR[int(hole_counts[mr, old_c])], BLACK)
            out[r0 : r1 + 1, target_c0 : target_c1 + 1] = recolored
    return out


def _canonical_mask(mask: np.ndarray) -> str:
    return "".join("1" if value else "0" for value in mask.ravel())


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    transforms: list[np.ndarray] = []
    current = mask
    for _ in range(4):
        transforms.append(current)
        transforms.append(np.fliplr(current))
        current = np.rot90(current)
    unique: dict[str, np.ndarray] = {}
    for transformed in transforms:
        unique[_canonical_mask(transformed)] = transformed
    return list(unique.values())


def _valid_shape(mask: np.ndarray, hole_count: int) -> bool:
    if mask.shape != (CELL_SIZE, CELL_SIZE):
        return False
    area = int(mask.sum())
    if area < 10 or area > 20:
        return False
    if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
        return False
    comps = _components(mask)
    if len(comps) != 1:
        return False
    return _count_holes(mask) == hole_count


def _mutate_mask(mask: np.ndarray, hole_count: int) -> np.ndarray:
    candidate = mask.copy()
    for _ in range(random.randint(8, 18)):
        trial = candidate.copy()
        for _ in range(random.randint(1, 3)):
            r = random.randrange(CELL_SIZE)
            c = random.randrange(CELL_SIZE)
            trial[r, c] = ~trial[r, c]
        if _valid_shape(trial, hole_count):
            candidate = trial
    return candidate


def _ensure_mask_library(hole_count: int, target_size: int = 20) -> None:
    library = _MASK_LIBRARY[hole_count]
    if len(library) >= target_size:
        return

    known = {_canonical_mask(mask) for mask in library}
    frontier = list(library) if library else list(_BASE_MASKS[hole_count])
    for base in _BASE_MASKS[hole_count]:
        for transformed in _all_transforms(base):
            if _valid_shape(transformed, hole_count):
                key = _canonical_mask(transformed)
                if key not in known:
                    known.add(key)
                    library.append(transformed.copy())
                    frontier.append(transformed.copy())

    attempts = 0
    while len(library) < target_size and attempts < 4000:
        attempts += 1
        seed = random.choice(frontier if frontier else _BASE_MASKS[hole_count])
        mutated = _mutate_mask(random.choice(_all_transforms(seed)), hole_count)
        key = _canonical_mask(mutated)
        if key in known or not _valid_shape(mutated, hole_count):
            continue
        known.add(key)
        library.append(mutated.copy())
        frontier.append(mutated.copy())

    assert library, f"failed to build a mask library for hole count {hole_count}"


def _sample_shape(hole_count: int) -> np.ndarray:
    _ensure_mask_library(hole_count)
    seed = random.choice(_MASK_LIBRARY[hole_count]).copy()
    return _mutate_mask(random.choice(_all_transforms(seed)), hole_count)


def _sample_hole_counts() -> np.ndarray:
    for _ in range(500):
        counts = np.random.randint(0, 4, size=(MACRO_ROWS, MACRO_COLS), dtype=int)
        present = set(map(int, counts.ravel()))
        require_all = random.random() < 0.6
        if require_all and present != {0, 1, 2, 3}:
            continue
        if not require_all and len(present) < 3:
            continue
        col_sums = counts.sum(axis=0)
        if len(set(map(int, col_sums))) != MACRO_COLS:
            continue
        order = _stable_desc_order(col_sums)
        if order == tuple(range(MACRO_COLS)):
            continue
        nonzero_counts = (counts > 0).sum(axis=0)
        if _stable_desc_order(nonzero_counts) == order:
            continue
        return counts
    raise RuntimeError("failed to sample non-degenerate hole counts")


def _build_grid(hole_counts: np.ndarray, masks: list[list[np.ndarray]]) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    for mr in range(MACRO_ROWS):
        for mc in range(MACRO_COLS):
            r0 = mr * (CELL_SIZE + CELL_GAP)
            c0 = mc * (CELL_SIZE + CELL_GAP)
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = np.where(masks[mr][mc], TEAL, BLACK)
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    hole_counts, areas, _row_spans, _col_spans = _analyze(grid)
    if len(set(map(int, hole_counts.sum(axis=0)))) != MACRO_COLS:
        return False

    order = _stable_desc_order(hole_counts.sum(axis=0))
    if order == tuple(range(MACRO_COLS)):
        return False

    output = main(grid)
    if output.shape != grid.shape or np.array_equal(output, grid):
        return False

    if len(set(map(int, hole_counts.ravel()))) < 3:
        return False

    color_counts = set(int(v) for v in output.ravel() if int(v) != BLACK)
    if len(color_counts) < 3:
        return False

    area_order = _stable_desc_order(areas.sum(axis=0))
    if area_order == order:
        return False

    zero_count_order = _stable_desc_order(((CELL_SIZE * CELL_SIZE) - areas).sum(axis=0))
    if zero_count_order == order:
        return False

    high_hole_order = _stable_desc_order((hole_counts >= 2).sum(axis=0))
    if high_hole_order == order:
        return False

    recolor_grid = np.array(
        [[HOLE_TO_COLOR[int(h)] for h in row] for row in hole_counts],
        dtype=int,
    )
    if len(set(map(int, recolor_grid.ravel()))) < 3:
        return False

    return True


def _candidate_signatures(grid: np.ndarray) -> tuple[str, str]:
    out = main(grid)
    inp_sig = hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
    out_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    return inp_sig, out_sig


def _gen_attempt() -> np.ndarray | None:
    hole_counts = _sample_hole_counts()
    masks: list[list[np.ndarray]] = []
    for mr in range(MACRO_ROWS):
        row_masks: list[np.ndarray] = []
        for mc in range(MACRO_COLS):
            row_masks.append(_sample_shape(int(hole_counts[mr, mc])))
        masks.append(row_masks)

    grid = _build_grid(hole_counts, masks)
    if not _semantic_checks(grid):
        return None

    inp_sig, out_sig = _candidate_signatures(grid)
    if inp_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None

    _SEEN_INPUTS.add(inp_sig)
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-sort recolor input")
