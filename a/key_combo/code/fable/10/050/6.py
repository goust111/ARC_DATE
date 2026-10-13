from __future__ import annotations

import random

import numpy as np


CELL_GRID = 3
CELL_SIZE = 3
STEP = CELL_SIZE + 1
GRID_SIZE = CELL_GRID * STEP + 1

CORNER_ORBIT = ((0, 0), (0, 2), (2, 2), (2, 0))
EDGE_ORBIT = ((0, 1), (1, 2), (2, 1), (1, 0))
CENTER_ORBIT = ((1, 1),)
ALL_ORBITS = (CORNER_ORBIT, EDGE_ORBIT, CENTER_ORBIT)

SYMMETRIC_MASK_PARTS_WITH_BG = (
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, False, True),
    (False, True, True),
    (True, True, False),
)
ASYM_FINAL_MASK_PARTS = (
    (True, False, False),
    (False, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, False),
    (True, True, True),
)

MAX_GENERATION_ATTEMPTS = 400


def _cell_origin(cell_r: int, cell_c: int) -> tuple[int, int]:
    return cell_r * STEP + 1, cell_c * STEP + 1


def _iter_cells() -> list[tuple[int, int]]:
    return [(r, c) for r in range(CELL_GRID) for c in range(CELL_GRID)]


def _infer_layout(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 3x3 separator grid with 3x3 interiors"
    line = int(g[0, 0])
    row_divs = [r for r in range(g.shape[0]) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(g.shape[1]) if np.all(g[:, c] == line)]
    assert row_divs == [0, 4, 8, 12], "unexpected row dividers"
    assert col_divs == [0, 4, 8, 12], "unexpected col dividers"
    return line, row_divs, col_divs


def _extract_cells(grid: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    cells: dict[tuple[int, int], np.ndarray] = {}
    for cell_r, cell_c in _iter_cells():
        r0, c0 = _cell_origin(cell_r, cell_c)
        cells[(cell_r, cell_c)] = grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE].copy()
    return cells


def _infer_inner_background(cells: dict[tuple[int, int], np.ndarray]) -> int:
    common: set[int] | None = None
    for cell in cells.values():
        colors = set(int(v) for v in np.unique(cell))
        common = colors if common is None else (common & colors)
    assert common is not None and len(common) == 1, "expected one interior background color shared by all cells"
    return int(next(iter(common)))


def _is_c4_symmetric(cell: np.ndarray) -> bool:
    return bool(np.array_equal(cell, np.rot90(cell)))


def _mask_from_parts(has_corners: bool, has_edges: bool, has_center: bool) -> np.ndarray:
    mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
    if has_corners:
        for r, c in CORNER_ORBIT:
            mask[r, c] = True
    if has_edges:
        for r, c in EDGE_ORBIT:
            mask[r, c] = True
    if has_center:
        mask[1, 1] = True
    return mask


def _orbit_closure(mask: np.ndarray) -> np.ndarray:
    closure = np.zeros_like(mask, dtype=bool)
    for orbit in ALL_ORBITS:
        if any(bool(mask[r, c]) for r, c in orbit):
            for r, c in orbit:
                closure[r, c] = True
    return closure


def _render_cell(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    cell = np.full((CELL_SIZE, CELL_SIZE), bg, dtype=int)
    cell[mask] = int(fg)
    return cell


def _analyze_asymmetric_cell(cell: np.ndarray, bg: int) -> dict[str, object] | None:
    fg_colors = sorted(set(int(v) for v in np.unique(cell)) - {int(bg)})
    if len(fg_colors) != 1:
        return None

    fg = int(fg_colors[0])
    mask = cell == fg
    closure = _orbit_closure(mask)
    missing_mask = closure & ~mask
    if not bool(missing_mask.any()):
        return None

    completed = _render_cell(closure, fg, bg)
    if not _is_c4_symmetric(completed):
        return None

    missing = [tuple(map(int, coord)) for coord in np.argwhere(missing_mask)]
    return {
        "fg": fg,
        "input_mask": mask,
        "closure_mask": closure,
        "missing": missing,
        "completed_cell": completed,
    }


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    line, _row_divs, _col_divs = _infer_layout(g)
    cells = _extract_cells(g)
    bg = _infer_inner_background(cells)

    asymmetric: list[tuple[int, int, dict[str, object]]] = []
    for (cell_r, cell_c), cell in cells.items():
        if _is_c4_symmetric(cell):
            continue
        info = _analyze_asymmetric_cell(cell, bg)
        assert info is not None, "every non-symmetric cell must be completable without color conflict"
        asymmetric.append((cell_r, cell_c, info))

    assert len(asymmetric) == 1, "expected exactly one completable non-symmetric cell"
    asym_r, asym_c, asym_info = asymmetric[0]
    missing = asym_info["missing"]
    assert isinstance(missing, list) and missing, "the asymmetric cell must require at least one added pixel"
    assert (asym_r, asym_c) not in missing, "missing positions must not point to the asymmetric cell itself"

    return {
        "line": line,
        "bg": bg,
        "cells": cells,
        "asym_pos": (asym_r, asym_c),
        "asym_info": asym_info,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the unique non-symmetric cell, then reinterpret its missing pixels as whole-cell paint commands.

    Input structure:
    - Divider lines split the grid into a 3x3 arrangement of cells, each with a 3x3 interior.
    - Eight cells are already 90-degree rotationally symmetric around their own centers.
    - One cell uses one foreground color on the shared interior background, is not symmetric, and can be uniquely completed by adding pixels of that same foreground color.

    Rule:
    - Find the unique non-symmetric cell and complete it to 4-fold rotational symmetry by adding exactly the missing foreground pixels.
    - Read the added pixel coordinates inside that 3x3 cell as 3x3 cell-grid coordinates.
    - For every referenced cell, repaint its entire 3x3 interior solid with the asymmetric cell's foreground color.
    - Keep the divider lines and every unreferenced cell unchanged.

    Key insight:
    - The completed motif is only the first layer; the real instructions are the coordinates of the pixels that had to be added, because those local gaps become global whole-cell commands.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    asym_r, asym_c = info["asym_pos"]
    asym_info = info["asym_info"]
    assert isinstance(asym_info, dict)
    fg = int(asym_info["fg"])
    completed_cell = np.asarray(asym_info["completed_cell"], dtype=int)
    missing = list(asym_info["missing"])

    out = g.copy()
    r0, c0 = _cell_origin(asym_r, asym_c)
    out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = completed_cell

    solid = np.full((CELL_SIZE, CELL_SIZE), fg, dtype=int)
    for target_r, target_c in missing:
        tr0, tc0 = _cell_origin(target_r, target_c)
        out[tr0 : tr0 + CELL_SIZE, tc0 : tc0 + CELL_SIZE] = solid
    return out


def _build_empty_grid(line: int) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), line, dtype=int)
    for cell_r, cell_c in _iter_cells():
        r0, c0 = _cell_origin(cell_r, cell_c)
        grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = 0
    return grid


def _paint_cells(grid: np.ndarray, cells: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for (cell_r, cell_c), cell in cells.items():
        r0, c0 = _cell_origin(cell_r, cell_c)
        out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = cell
    return out


def _choose_asym_masks(asym_pos: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    for _ in range(200):
        parts = random.choice(ASYM_FINAL_MASK_PARTS)
        final_mask = _mask_from_parts(*parts)

        present_orbits = []
        if parts[0]:
            present_orbits.append(CORNER_ORBIT)
        if parts[1]:
            present_orbits.append(EDGE_ORBIT)

        partial_count = 1 if len(present_orbits) == 1 else random.choice((1, 2))
        chosen_orbits = random.sample(present_orbits, partial_count)

        keep_mask = final_mask.copy()
        for orbit in chosen_orbits:
            keep_count = random.randint(1, 3)
            kept = set(random.sample(list(orbit), keep_count))
            for r, c in orbit:
                keep_mask[r, c] = (r, c) in kept

        missing = [tuple(map(int, coord)) for coord in np.argwhere(final_mask & ~keep_mask)]
        if not missing:
            continue
        if asym_pos in missing:
            continue
        return keep_mask, final_mask, missing
    raise RuntimeError("failed to choose a valid asymmetric completion mask")


def _random_symmetric_mask_with_bg() -> np.ndarray:
    return _mask_from_parts(*random.choice(SYMMETRIC_MASK_PARTS_WITH_BG))


def _cell_signature(cell: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in cell.ravel())


def _expected_output_from_info(inp: np.ndarray, info: dict[str, object]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    asym_r, asym_c = info["asym_pos"]
    asym_info = info["asym_info"]
    assert isinstance(asym_info, dict)
    r0, c0 = _cell_origin(asym_r, asym_c)
    out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = np.asarray(asym_info["completed_cell"], dtype=int)

    fg = int(asym_info["fg"])
    solid = np.full((CELL_SIZE, CELL_SIZE), fg, dtype=int)
    for target_r, target_c in asym_info["missing"]:
        tr0, tc0 = _cell_origin(target_r, target_c)
        out[tr0 : tr0 + CELL_SIZE, tc0 : tc0 + CELL_SIZE] = solid
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        info = _analyze_grid(inp)
    except AssertionError:
        return False

    computed = main(inp)
    if not np.array_equal(computed, expected):
        return False

    asym_r, asym_c = info["asym_pos"]
    asym_info = info["asym_info"]
    assert isinstance(asym_info, dict)
    fg = int(asym_info["fg"])
    missing = list(asym_info["missing"])
    completed_cell = np.asarray(asym_info["completed_cell"], dtype=int)
    solid = np.full((CELL_SIZE, CELL_SIZE), fg, dtype=int)

    completion_only = np.asarray(inp, dtype=int).copy()
    r0, c0 = _cell_origin(asym_r, asym_c)
    completion_only[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = completed_cell
    if np.array_equal(completion_only, expected):
        return False

    paint_only = np.asarray(inp, dtype=int).copy()
    for target_r, target_c in missing:
        tr0, tc0 = _cell_origin(target_r, target_c)
        paint_only[tr0 : tr0 + CELL_SIZE, tc0 : tc0 + CELL_SIZE] = solid
    if np.array_equal(paint_only, expected):
        return False

    # Every target instruction must have a visible effect.
    for target_r, target_c in missing:
        tr0, tc0 = _cell_origin(target_r, target_c)
        before = inp[tr0 : tr0 + CELL_SIZE, tc0 : tc0 + CELL_SIZE]
        if np.array_equal(before, solid):
            return False
        almost = completion_only.copy()
        for other_r, other_c in missing:
            if (other_r, other_c) == (target_r, target_c):
                continue
            or0, oc0 = _cell_origin(other_r, other_c)
            almost[or0 : or0 + CELL_SIZE, oc0 : oc0 + CELL_SIZE] = solid
        if np.array_equal(almost, expected):
            return False

    # The completion clue must be necessary, not replaceable by grouping existing cell appearances.
    cells = info["cells"]
    target_sigs = set()
    other_sigs = set()
    for pos, cell in cells.items():
        if pos == (asym_r, asym_c):
            continue
        sig = _cell_signature(cell)
        if pos in missing:
            target_sigs.add(sig)
        else:
            other_sigs.add(sig)
    if not (target_sigs & other_sigs):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    line, bg, asym_fg = random.sample(range(10), 3)
    other_fg_count = random.randint(1, 3)
    remaining = [c for c in range(10) if c not in (line, bg, asym_fg)]
    extra_fgs = random.sample(remaining, other_fg_count)
    fg_palette = [asym_fg] + extra_fgs

    asym_pos = random.choice(_iter_cells())
    input_mask, _final_mask, missing = _choose_asym_masks(asym_pos)

    cells: dict[tuple[int, int], np.ndarray] = {}
    cells[asym_pos] = _render_cell(input_mask, asym_fg, bg)

    non_asym_positions = [pos for pos in _iter_cells() if pos != asym_pos]
    target_positions = list(missing)
    untarget_positions = [pos for pos in non_asym_positions if pos not in target_positions]
    if not target_positions or not untarget_positions:
        return None

    bridge_target = random.choice(target_positions)
    bridge_other = random.choice(untarget_positions)
    bridge_mask = random.choice(
        [
            _mask_from_parts(*parts)
            for parts in SYMMETRIC_MASK_PARTS_WITH_BG
            if parts != (False, False, True)
        ]
    )
    bridge_cell = _render_cell(bridge_mask, asym_fg, bg)
    cells[bridge_target] = bridge_cell.copy()
    cells[bridge_other] = bridge_cell.copy()

    for pos in non_asym_positions:
        if pos in cells:
            continue
        fg = random.choice(fg_palette)
        mask = _random_symmetric_mask_with_bg()
        cell = _render_cell(mask, fg, bg)
        cells[pos] = cell

    base = _build_empty_grid(line)
    inp = _paint_cells(base, cells)

    try:
        info = _analyze_grid(inp)
    except AssertionError:
        return None

    expected = _expected_output_from_info(inp, info)
    if not _validate_semantics(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
