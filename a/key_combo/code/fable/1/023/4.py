from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    _arc_utils = SourceFileLoader("_arc_utils_fable_1_023_4", str(_ARC_UTILS_PATH)).load_module()
    Color = _arc_utils.Color
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]


PALETTE = tuple(int(c) for c in Color.NOT_BLACK)
MAX_GEN_ATTEMPTS = 400


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _period_model(grid: np.ndarray, pr: int, pc: int) -> tuple[np.ndarray, int, np.ndarray]:
    tile = np.zeros((pr, pc), dtype=int)
    defect_mask = np.zeros((pr, pc), dtype=bool)
    defect_count = 0
    for rr in range(pr):
        for cc in range(pc):
            orbit = grid[rr::pr, cc::pc]
            color = _majority_color(orbit)
            tile[rr, cc] = color
            mismatches = orbit != color
            if np.any(mismatches):
                defect_mask[rr, cc] = True
                defect_count += int(np.count_nonzero(mismatches))
    return tile, defect_count, defect_mask


def _choose_periods(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    h, w = map(int, grid.shape)
    assert h >= 2 and w >= 2, "expected at least two rows and two columns"

    best_score = None
    best_result = None
    for pr in range(1, h // 2 + 1):
        for pc in range(1, w // 2 + 1):
            tile, defect_count, defect_mask = _period_model(grid, pr, pc)
            score = (defect_count + pr * pc, defect_count, pr * pc, pr, pc)
            if best_score is None or score < best_score:
                best_score = score
                best_result = (pr, pc, tile, defect_mask)

    assert best_result is not None
    return best_result


def _transform(grid: np.ndarray) -> np.ndarray:
    pr, pc, tile, defect_mask = _choose_periods(grid)
    out = np.full((pr, pc), int(Color.BLACK), dtype=int)
    out[defect_mask] = tile[defect_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold every defect of a broken 2D periodic tiling into one restored period cell.

    Input structure:
    - The full grid is a rectangular crop of a pattern that repeats by one row period and one column period.
    - A small number of cells have been overwritten to colors that disagree with their true 2D translation orbit.
    - Defects may form short horizontal or vertical runs, so one-direction shortcuts are unreliable.

    Rule:
    - Find the smallest axis-aligned period rectangle whose orbit-majority reconstruction needs only a few edits.
    - For each residue class modulo that row/column period, take the majority color across the full 2D orbit as the restored color.
    - Output exactly one period cell, writing each defective residue at its folded coordinate with that restored majority color and leaving every other cell black.

    Key insight:
    - The period must be inferred jointly with the defect restoration: the visible wrong cells are collected by quotient coordinates, but they are colored by the hidden orbit majority rather than by their current colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    return _transform(g)


def _is_primitive_tile(tile: np.ndarray) -> bool:
    pr, pc = tile.shape
    for shift in range(1, pr):
        if all(np.array_equal(tile[r], tile[(r + shift) % pr]) for r in range(pr)):
            return False
    for shift in range(1, pc):
        if all(np.array_equal(tile[:, c], tile[:, (c + shift) % pc]) for c in range(pc)):
            return False
    return True


def _random_tile(pr: int, pc: int, repeats_r: int, repeats_c: int) -> np.ndarray | None:
    for _ in range(200):
        colors = random.sample(PALETTE, random.randint(3, 5))
        tile = np.array([[random.choice(colors) for _ in range(pc)] for _ in range(pr)], dtype=int)
        if len(set(map(int, tile.reshape(-1)))) < 3:
            continue
        if not _is_primitive_tile(tile):
            continue
        clean = np.tile(tile, (repeats_r, repeats_c))
        got_pr, got_pc, _tile, _mask = _choose_periods(clean)
        if (got_pr, got_pc) == (pr, pc):
            return tile
    return None


def _pick_chain(
    h: int,
    w: int,
    pr: int,
    pc: int,
    orientation: str,
    used_residues: set[tuple[int, int]],
    used_cells: set[tuple[int, int]],
    max_len: int,
) -> list[tuple[int, int]] | None:
    length_limit = min(max_len, pc if orientation == "h" else pr)
    if length_limit < 2:
        return None

    for _ in range(120):
        length = random.randint(2, length_limit)
        if orientation == "h":
            row = random.randrange(h)
            start = random.randrange(w - length + 1)
            cells = [(row, c) for c in range(start, start + length)]
        else:
            col = random.randrange(w)
            start = random.randrange(h - length + 1)
            cells = [(r, col) for r in range(start, start + length)]

        residues = [(r % pr, c % pc) for r, c in cells]
        if len(set(residues)) != len(residues):
            continue
        if any(cell in used_cells for cell in cells):
            continue
        if any(residue in used_residues for residue in residues):
            continue
        return cells
    return None


def _pick_isolated(
    h: int,
    w: int,
    pr: int,
    pc: int,
    used_residues: set[tuple[int, int]],
    used_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for _ in range(200):
        cell = (random.randrange(h), random.randrange(w))
        residue = (cell[0] % pr, cell[1] % pc)
        if cell in used_cells or residue in used_residues:
            continue
        return cell
    return None


def _choose_wrong_color(forbidden: set[int]) -> int:
    options = [color for color in PALETTE if color not in forbidden]
    if options:
        return random.choice(options)
    color = next(iter(forbidden))
    return random.choice([candidate for candidate in PALETTE if candidate != color])


def _apply_group(
    grid: np.ndarray,
    tile: np.ndarray,
    cells: list[tuple[int, int]],
    pr: int,
    pc: int,
    used_residues: set[tuple[int, int]],
    used_cells: set[tuple[int, int]],
) -> None:
    residues = [(r % pr, c % pc) for r, c in cells]
    originals = {int(tile[rr, cc]) for rr, cc in residues}
    wrong = _choose_wrong_color(originals)
    for (r, c), residue in zip(cells, residues):
        grid[r, c] = wrong
        used_cells.add((r, c))
        used_residues.add(residue)


def _generate_case(return_meta: bool = False):
    for _ in range(MAX_GEN_ATTEMPTS):
        pr = random.randint(2, 6)
        pc = random.randint(2, 6)
        repeats_r = random.randint(3, 5)
        repeats_c = random.randint(3, 5)
        h = pr * repeats_r
        w = pc * repeats_c
        if h > 30 or w > 30:
            continue

        tile = _random_tile(pr, pc, repeats_r, repeats_c)
        if tile is None:
            continue

        clean = np.tile(tile, (repeats_r, repeats_c))
        grid = clean.copy()
        used_residues: set[tuple[int, int]] = set()
        used_cells: set[tuple[int, int]] = set()
        max_defects = min(6, pr * pc - 1)
        if max_defects < 2:
            continue

        branch = random.choice(("h", "v", "mixed"))
        if branch in {"h", "mixed"}:
            chain = _pick_chain(h, w, pr, pc, "h", used_residues, used_cells, max_defects - len(used_residues))
            if chain is not None:
                _apply_group(grid, tile, chain, pr, pc, used_residues, used_cells)
        if branch in {"v", "mixed"} and len(used_residues) < max_defects:
            chain = _pick_chain(h, w, pr, pc, "v", used_residues, used_cells, max_defects - len(used_residues))
            if chain is not None:
                _apply_group(grid, tile, chain, pr, pc, used_residues, used_cells)
        if not used_residues:
            fallback = _pick_chain(
                h,
                w,
                pr,
                pc,
                random.choice(("h", "v")),
                used_residues,
                used_cells,
                max_defects - len(used_residues),
            )
            if fallback is None:
                continue
            _apply_group(grid, tile, fallback, pr, pc, used_residues, used_cells)

        target_defects = random.randint(max(2, len(used_residues)), max_defects)
        while len(used_residues) < target_defects:
            cell = _pick_isolated(h, w, pr, pc, used_residues, used_cells)
            if cell is None:
                break
            _apply_group(grid, tile, [cell], pr, pc, used_residues, used_cells)

        if len(used_residues) < 2 or len(used_residues) >= pr * pc:
            continue

        expected = np.full((pr, pc), int(Color.BLACK), dtype=int)
        for rr, cc in used_residues:
            expected[rr, cc] = int(tile[rr, cc])

        got_pr, got_pc, _tile, _mask = _choose_periods(grid)
        if (got_pr, got_pc) != (pr, pc):
            continue
        if not np.array_equal(_transform(grid), expected):
            continue

        if return_meta:
            return grid, expected, {"period": (pr, pc), "branch": branch}
        return grid

    raise RuntimeError("failed to generate a valid broken periodic tiling")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
