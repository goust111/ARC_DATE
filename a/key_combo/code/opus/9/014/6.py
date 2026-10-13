from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


BLACK = Color.BLACK
RED = Color.RED
GREY = Color.GREY
NON_SPECIAL_COLORS = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)

GEN_RETRIES = 400
_GEN_COUNTER = 0


def _assemble_quadrants(grid: np.ndarray) -> np.ndarray:
    n = int(grid.shape[0])
    out = np.empty((2 * n, 2 * n), dtype=int)
    out[:n, :n] = grid
    out[:n, n:] = np.rot90(grid, 1)
    out[n:, :n] = np.rot90(grid, 2)
    out[n:, n:] = np.rot90(grid, 3)
    return out


def _marker_union_mask(marker_coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    rows = np.unique(marker_coords[:, 0])
    cols = np.unique(marker_coords[:, 1])
    mask[rows, :] = True
    mask[:, cols] = True
    return mask


def _replicated_positions(n: int, r: int, c: int) -> list[tuple[int, int]]:
    return [
        (int(r), int(c)),
        (int(n - 1 - c), int(n + r)),
        (int(2 * n - 1 - r), int(n - 1 - c)),
        (int(n + c), int(2 * n - 1 - r)),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate one marked square into four quadrants, then erase every row and column touched by the four gray markers while preserving red pixels.

    Input structure:
    - The input is a square grid with no black background; every cell is a visible color.
    - Exactly one gray marker pixel appears somewhere away from the central row and column.
    - A small number of red pixels are scattered among other non-black colors.

    Rule:
    - Build a 2n x 2n output with the original grid in the top-left, the 90-degree counterclockwise rotation in the top-right,
      the 180-degree rotation in the bottom-left, and the 270-degree counterclockwise rotation in the bottom-right.
    - Locate all gray pixels in this assembled grid and take the union of their full rows and full columns.
    - Inside that union, turn every non-red, non-gray pixel black.
    - Leave red pixels, gray pixels, and every cell outside the union unchanged.

    Key insight:
    - The final erase pattern is not a single cross from the input marker; it is the combined row-and-column imprint of the marker after all four rotational placements.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square input grid"
    assert max(g.shape) <= 15, "input must stay within ARC size limits after doubling"

    tiled = _assemble_quadrants(g)
    marker_coords = np.argwhere(tiled == GREY)
    assert marker_coords.shape[0] >= 1, "expected at least one gray marker"

    erase_mask = _marker_union_mask(marker_coords, tiled.shape)
    out = tiled.copy()
    out[erase_mask & (tiled != RED) & (tiled != GREY)] = BLACK
    return out


def _choose_marker(n: int, branch: str) -> tuple[int, int]:
    mid = n // 2
    candidates: list[tuple[int, int]] = []
    for r in range(n):
        for c in range(n):
            if r == mid or c == mid:
                continue
            if branch == "diag":
                if r == c:
                    candidates.append((r, c))
            elif branch == "anti":
                if r + c == n - 1:
                    candidates.append((r, c))
            else:
                if r != c and r + c != n - 1:
                    candidates.append((r, c))
    assert candidates, f"no marker candidates for branch {branch}"
    return random.choice(candidates)


def _make_base_fill(n: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(80):
        grid = np.random.choice(palette, size=(n, n)).astype(int)
        if len({int(x) for x in np.unique(grid)}) < min(3, len(palette)):
            continue
        if any(np.array_equal(grid, np.rot90(grid, k)) for k in (1, 2, 3)):
            continue
        return grid
    return None


def _inside_counts_for_cells(n: int, marker: tuple[int, int]) -> dict[tuple[int, int], int]:
    marker_positions = np.array(_replicated_positions(n, marker[0], marker[1]), dtype=int)
    erase_mask = _marker_union_mask(marker_positions, (2 * n, 2 * n))
    counts: dict[tuple[int, int], int] = {}
    for r in range(n):
        for c in range(n):
            repl = _replicated_positions(n, r, c)
            counts[(r, c)] = sum(int(erase_mask[rr, cc]) for rr, cc in repl)
    return counts


def _pick_spread(
    candidates: list[tuple[int, int]],
    taken: list[tuple[int, int]],
    marker: tuple[int, int],
) -> tuple[int, int] | None:
    if not candidates:
        return None

    def well_spaced(cell: tuple[int, int]) -> bool:
        if abs(cell[0] - marker[0]) + abs(cell[1] - marker[1]) <= 1:
            return False
        return all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in taken)

    spaced = [cell for cell in candidates if well_spaced(cell)]
    pool = spaced if spaced else candidates
    return random.choice(pool)


def _choose_red_positions(n: int, marker: tuple[int, int], red_count: int) -> list[tuple[int, int]] | None:
    inside_counts = _inside_counts_for_cells(n, marker)
    cells = [cell for cell in inside_counts if cell != marker]
    strong_inside = [cell for cell in cells if inside_counts[cell] >= 2]
    mixed_inside = [cell for cell in cells if 0 < inside_counts[cell] < 4]
    outside_only = [cell for cell in cells if inside_counts[cell] == 0]
    weak_inside = [cell for cell in cells if inside_counts[cell] > 0]

    if not weak_inside or not outside_only:
        return None

    chosen: list[tuple[int, int]] = []
    first = _pick_spread(strong_inside or mixed_inside or weak_inside, chosen, marker)
    if first is None:
        return None
    chosen.append(first)

    second = _pick_spread(outside_only, chosen, marker)
    if second is None:
        return None
    chosen.append(second)

    remaining_pool = [cell for cell in cells if cell not in chosen]
    while len(chosen) < red_count and remaining_pool:
        cell = _pick_spread(remaining_pool, chosen, marker)
        if cell is None:
            break
        chosen.append(cell)
        remaining_pool = [cand for cand in remaining_pool if cand != cell]

    if len(chosen) != red_count:
        return None
    return chosen


def _build_candidate(branch: str, n: int) -> np.ndarray | None:
    palette = random.sample(list(NON_SPECIAL_COLORS), k=random.randint(3, 5))
    grid = _make_base_fill(n, palette)
    if grid is None:
        return None

    marker = _choose_marker(n, branch)
    red_count = random.randint(2, 4)
    red_positions = _choose_red_positions(n, marker, red_count)
    if red_positions is None:
        return None

    grid = grid.copy()
    for r, c in red_positions:
        grid[r, c] = RED
    grid[marker] = GREY
    return grid


def _semantic_checks(grid: np.ndarray, branch: str) -> bool:
    n = int(grid.shape[0])
    if grid.shape != (n, n) or n not in (5, 7):
        return False
    if np.any(grid == BLACK):
        return False

    marker_cells = np.argwhere(grid == GREY)
    if marker_cells.shape[0] != 1:
        return False
    marker = tuple(map(int, marker_cells[0]))
    mid = n // 2
    if marker[0] == mid or marker[1] == mid:
        return False

    red_count = int(np.count_nonzero(grid == RED))
    if red_count < 2 or red_count > 4:
        return False
    if any(np.array_equal(grid, np.rot90(grid, k)) for k in (1, 2, 3)):
        return False

    raw = _assemble_quadrants(grid)
    out = main(grid)
    if out.shape != (2 * n, 2 * n):
        return False

    gray_positions = np.argwhere(raw == GREY)
    if gray_positions.shape[0] != 4:
        return False

    row_count = len(np.unique(gray_positions[:, 0]))
    col_count = len(np.unique(gray_positions[:, 1]))
    if branch == "general" and (row_count != 4 or col_count != 4):
        return False
    if branch != "general" and row_count == 4 and col_count == 4:
        return False

    erase_mask = _marker_union_mask(gray_positions, raw.shape)
    expected = raw.copy()
    expected[erase_mask & (raw != RED) & (raw != GREY)] = BLACK
    if not np.array_equal(out, expected):
        return False

    if np.array_equal(out, raw):
        return False

    erased_non_special = erase_mask & (raw != RED) & (raw != GREY)
    preserved_non_special = (~erase_mask) & (raw != RED) & (raw != GREY)
    if int(np.count_nonzero(erased_non_special)) < max(8, n):
        return False
    if int(np.count_nonzero(preserved_non_special)) < max(8, n):
        return False

    red_inside = int(np.count_nonzero(erase_mask & (raw == RED)))
    red_outside = int(np.count_nonzero((~erase_mask) & (raw == RED)))
    if red_inside < 2 or red_outside < 2:
        return False

    no_exempt = raw.copy()
    no_exempt[erase_mask & (raw != GREY)] = BLACK
    if np.array_equal(out, no_exempt):
        return False

    tl_only = np.zeros(raw.shape, dtype=bool)
    tl_only[marker[0], :] = True
    tl_only[:, marker[1]] = True
    if not np.any((erase_mask & ~tl_only) & (raw != RED) & (raw != GREY)):
        return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    branch_cycle = ("general", "diag", "anti")
    size_cycle = (5, 7)

    for attempt in range(GEN_RETRIES):
        branch = branch_cycle[(_GEN_COUNTER + attempt) % len(branch_cycle)]
        n = size_cycle[(_GEN_COUNTER // len(branch_cycle) + attempt) % len(size_cycle)]
        grid = _build_candidate(branch, n)
        if grid is None:
            continue
        if _semantic_checks(grid, branch):
            _GEN_COUNTER += 1
            return grid

    raise RuntimeError("failed to generate a valid rotational cross-erasure example")
