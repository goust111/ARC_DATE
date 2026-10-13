from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.is_dir():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, colors_present
except Exception:  # pragma: no cover - fallback keeps SourceFileLoader usage robust
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = set(map(int, np.asarray(grid).ravel()))
        if exclude is not None:
            cols -= set(map(int, exclude))
        return sorted(cols)


DOWN_RIGHT = "down_right"
DOWN_LEFT = "down_left"
ORIENTATIONS = (DOWN_RIGHT, DOWN_LEFT)
PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GEN_N_RANGE = (6, 9)
GEN_K_CYCLE = (2, 3, 4)
GEN_MIN_DIAGONAL_LEN = 3
GEN_ATTEMPTS = 300
_GEN_CALL_COUNT = 0


def _diagonal_cells(n: int, orient: str, invariant: int) -> np.ndarray:
    if orient == DOWN_RIGHT:
        r0 = max(0, -invariant)
        r1 = min(n - 1, n - 1 - invariant)
        if r0 > r1:
            return np.zeros((0, 2), dtype=int)
        rows = np.arange(r0, r1 + 1, dtype=int)
        cols = rows + invariant
    else:
        assert orient == DOWN_LEFT
        r0 = max(0, invariant - (n - 1))
        r1 = min(n - 1, invariant)
        if r0 > r1:
            return np.zeros((0, 2), dtype=int)
        rows = np.arange(r0, r1 + 1, dtype=int)
        cols = invariant - rows
    return np.column_stack((rows, cols)).astype(int)


def _coord_invariant(coords: np.ndarray, orient: str) -> np.ndarray:
    if orient == DOWN_RIGHT:
        return coords[:, 1] - coords[:, 0]
    assert orient == DOWN_LEFT
    return coords[:, 1] + coords[:, 0]


def _parse_with_orientation(grid: np.ndarray, orient: str, k: int) -> dict[int, int] | None:
    n = int(grid.shape[0])
    residue_to_color: dict[int, int] = {}

    for color in colors_present(grid, exclude={Color.BLACK}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            return None
        invariants = _coord_invariant(coords, orient)
        if np.unique(invariants).size != 1:
            return None

        invariant = int(invariants[0])
        diag = _diagonal_cells(n, orient, invariant)
        if diag.shape != coords.shape or not np.array_equal(diag, coords):
            return None

        residue = invariant % k
        if residue in residue_to_color:
            return None
        residue_to_color[residue] = int(color)

    if len(residue_to_color) != k:
        return None
    return residue_to_color


def _parse_grid(grid: np.ndarray) -> tuple[int, str, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    colors = colors_present(g, exclude={Color.BLACK})
    k = len(colors)
    assert 2 <= k <= 4, "expected between 2 and 4 non-black diagonal colors"

    candidates: list[tuple[str, dict[int, int]]] = []
    for orient in ORIENTATIONS:
        mapping = _parse_with_orientation(g, orient, k)
        if mapping is not None:
            candidates.append((orient, mapping))

    assert len(candidates) == 1, "expected a unique diagonal orientation and residue mapping"
    orient, residue_to_color = candidates[0]
    return k, orient, residue_to_color


def _fill_pattern(n: int, k: int, orient: str, residue_to_color: dict[int, int]) -> np.ndarray:
    rows, cols = np.indices((n, n))
    invariants = cols - rows if orient == DOWN_RIGHT else cols + rows
    filled = np.zeros((n, n), dtype=int)
    for residue, color in residue_to_color.items():
        filled[invariants % k == residue] = int(color)
    assert np.all(filled != Color.BLACK), "all K residue classes must be covered"
    return filled


def _shear_grid(grid: np.ndarray, step: int) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((h, w + step - 1), dtype=int)
    for r in range(h):
        start = step - 1 - (r % step)
        out[r, start : start + w] = grid[r]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the colored diagonals, use that count as a shared period, then fill and shear by the same K.

    Input structure:
    - The input is an N x N black grid containing K complete non-black diagonals, where K is the number of distinct foreground colors.
    - Each color appears on exactly one maximal diagonal, and all diagonals share the same slope direction.
    - The diagonal offsets may be irregular, but together they cover all K residue classes modulo K for that slope's diagonal invariant.

    Rule:
    - Count the distinct non-black colors to obtain K.
    - Identify each diagonal's residue class: use (c - r) mod K for down-right diagonals and (c + r) mod K for down-left diagonals.
    - Fill every cell in the N x N grid whose diagonal invariant falls in that residue class with the diagonal's color.
    - Create an N x (N + K - 1) black canvas and copy filled row r so it starts at column K - 1 - (r mod K).

    Key insight:
    - The hidden parameter K is encoded only by the number of colored diagonals, and that same K simultaneously controls both the modulo fill pattern and the row-shear offsets.
    """
    g = np.asarray(grid, dtype=int)
    k, orient, residue_to_color = _parse_grid(g)
    filled = _fill_pattern(int(g.shape[0]), k, orient, residue_to_color)
    return _shear_grid(filled, k)


def _valid_invariants(n: int, orient: str, k: int) -> dict[int, list[int]]:
    by_residue = {residue: [] for residue in range(k)}
    if orient == DOWN_RIGHT:
        domain = range(-(n - 1), n)
    else:
        domain = range(0, 2 * n - 1)

    for invariant in domain:
        if _diagonal_cells(n, orient, invariant).shape[0] < GEN_MIN_DIAGONAL_LEN:
            continue
        by_residue[invariant % k].append(int(invariant))

    for residue, values in by_residue.items():
        assert values, f"missing a drawable diagonal for residue {residue}"
    return by_residue


def _canonical_invariants(n: int, orient: str, k: int) -> dict[int, int]:
    candidates = _valid_invariants(n, orient, k)
    center = 0 if orient == DOWN_RIGHT else n - 1
    best_score = None
    best_combo = None

    residue_order = tuple(range(k))
    for combo in itertools.product(*(candidates[residue] for residue in residue_order)):
        ordered = sorted(combo)
        gaps = tuple(b - a for a, b in zip(ordered, ordered[1:]))
        irregular = int(k < 3 or len(set(gaps)) > 1)
        spans_center = int(min(combo) < center < max(combo))
        edge_bias = sum(abs(invariant - center) for invariant in combo)
        not_too_centered = -sum(abs(invariant - center) <= 1 for invariant in combo)
        score = (irregular, spans_center, edge_bias, not_too_centered, tuple(-v for v in ordered))
        if best_score is None or score > best_score:
            best_score = score
            best_combo = combo

    assert best_combo is not None
    return {residue: int(invariant) for residue, invariant in zip(residue_order, best_combo)}


def _render_input(n: int, orient: str, residue_to_color: dict[int, int], invariants: dict[int, int]) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    for residue, color in residue_to_color.items():
        for r, c in _diagonal_cells(n, orient, invariants[residue]):
            grid[r, c] = int(color)
    return grid


def _cycle_choice(counter: int) -> tuple[int, str]:
    k = GEN_K_CYCLE[counter % len(GEN_K_CYCLE)]
    if k == 2:
        return k, DOWN_RIGHT
    orient = ORIENTATIONS[(counter // len(GEN_K_CYCLE)) % len(ORIENTATIONS)]
    return k, orient


def _semantic_checks(
    inp: np.ndarray,
    *,
    n: int,
    k: int,
    orient: str,
    residue_to_color: dict[int, int],
) -> bool:
    expected_fill = _fill_pattern(n, k, orient, residue_to_color)
    expected_out = _shear_grid(expected_fill, k)

    try:
        parsed_k, parsed_orient, parsed_map = _parse_grid(inp)
    except AssertionError:
        return False

    if parsed_k != k or parsed_orient != orient or parsed_map != residue_to_color:
        return False
    if not np.array_equal(main(inp), expected_out):
        return False
    if np.array_equal(inp, expected_fill):
        return False
    if np.array_equal(_shear_grid(inp, k), expected_out):
        return False
    if np.array_equal(_shear_grid(expected_fill, 1), expected_out):
        return False
    if k > 2:
        other = DOWN_LEFT if orient == DOWN_RIGHT else DOWN_RIGHT
        alt = _shear_grid(_fill_pattern(n, k, other, residue_to_color), k)
        if np.array_equal(alt, expected_out):
            return False
    return True


def _gen_attempt(counter: int) -> np.ndarray | None:
    k, orient = _cycle_choice(counter)
    n = random.randint(*GEN_N_RANGE)
    color_list = random.sample(PALETTE, k)
    random.shuffle(color_list)
    residue_to_color = {residue: int(color_list[residue]) for residue in range(k)}
    invariants = _canonical_invariants(n, orient, k)
    inp = _render_input(n, orient, residue_to_color, invariants)

    if not _semantic_checks(inp, n=n, k=k, orient=orient, residue_to_color=residue_to_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    base_counter = _GEN_CALL_COUNT
    for extra in range(GEN_ATTEMPTS):
        inp = _gen_attempt(base_counter + extra)
        if inp is not None:
            _GEN_CALL_COUNT = base_counter + extra + 1
            return inp
    raise RuntimeError("failed to generate a valid diagonal-fill-plus-shear input")
