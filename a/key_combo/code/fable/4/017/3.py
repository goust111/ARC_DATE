from __future__ import annotations

from pathlib import Path
import numpy as np
import sys

try:
    from arc_utils import Color
except Exception:
    _root = Path(__file__).resolve()
    for _parent in _root.parents:
        _solver = _parent / "new" / "solvers" / "arc_utils.py"
        if _solver.exists():
            _solver_dir = str(_solver.parent)
            if _solver_dir not in sys.path:
                sys.path.insert(0, _solver_dir)
            break
    from arc_utils import Color


def _sources_of(grid: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(grid != Color.BLACK)}


def _diagonal_signatures(sources: set[tuple[int, int]]) -> tuple[set[int], set[int]]:
    nw_diag = {r - c for r, c in sources}
    ne_diag = {r + c for r, c in sources}
    return nw_diag, ne_diag


def _find_new_markers(
    grid: np.ndarray,
    sources: set[tuple[int, int]],
    nw_diag: set[int],
    ne_diag: set[int],
) -> set[tuple[int, int]]:
    new_markers: set[tuple[int, int]] = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != Color.BLACK:
                continue
            if (r - c) in nw_diag and (r + c) in ne_diag:
                new_markers.add((r, c))
    new_markers -= sources
    return new_markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit two diagonals from every marker and materialize only cross-point intersections as orange markers until convergence.

    Input structure:
    - The canvas is black.
    - All non-black cells are initial marker pixels (same marker color class in valid inputs).

    Rule:
    - Each marker emits both diagonals (NW-SE and NE-SW) continuously to grid bounds.
    - A black cell becomes a new orange marker if it lies on one NW-SE line from one marker and one NE-SW line from a different marker.
    - New orange markers also emit diagonals, and the process repeats until no further new markers appear.
    - Original markers remain unchanged; only newly generated markers are recolored orange.

    Key insight:
    - The visible output is the fixed-point closure of diagonal-intersection points, not painted lines, so only cells satisfying both diagonal families are filled.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects a 2D grid")

    out = g.copy()
    sources = _sources_of(out)
    if not sources:
        return out

    while True:
        nw_diag, ne_diag = _diagonal_signatures(sources)
        candidates = _find_new_markers(out, sources, nw_diag, ne_diag)

        if not candidates:
            break

        for r, c in candidates:
            out[r, c] = Color.ORANGE
        sources.update(candidates)

    return out


def _candidate_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def _build_grid(seed: np.random.Generator) -> np.ndarray:
    h = int(seed.integers(8, 25))
    w = int(seed.integers(8, 25))

    mode = "focused" if seed.random() < 0.5 else "distributed"
    base_count = 3 if mode == "focused" else 5
    n_markers = int(seed.integers(max(2, base_count - 1), min(base_count + 2, min(h, w, 7)) + 1))

    positions = _candidate_positions(h, w)
    seed.shuffle(positions)

    if mode == "focused":
        r0, c0 = positions.pop()
        parity = (r0 + c0) % 2
        same_parity = [p for p in positions if (p[0] + p[1]) % 2 == parity]
        if not same_parity:
            positions = _candidate_positions(h, w)
            seed.shuffle(positions)
            same_parity = [p for p in positions if (p[0] + p[1]) % 2 == parity]

        r1, c1 = same_parity.pop()
        markers = {(r0, c0), (r1, c1)}

        for p in positions:
            if len(markers) >= n_markers:
                break
            r, c = p
            if abs(r - r0) + abs(c - c0) <= 1:
                continue
            markers.add((r, c))
    else:
        markers = set()
        while len(markers) < n_markers:
            r = int(seed.integers(h))
            c = int(seed.integers(w))
            candidate = (r, c)
            if candidate in markers:
                continue
            # Keep a little distance so intersections stay legible.
            if any(abs(r - rr) + abs(c - cc) < 2 for rr, cc in markers):
                continue
            markers.add(candidate)

    grid = np.full((h, w), Color.BLACK, dtype=int)
    for r, c in markers:
        grid[r, c] = Color.TEAL
    return grid


def _passes_visibility_filter(inp: np.ndarray, out: np.ndarray) -> bool:
    h, w = map(int, inp.shape)
    orange = int(np.count_nonzero(out == Color.ORANGE))
    teal = int(np.count_nonzero(inp == Color.TEAL))
    if orange == 0:
        return False
    if teal == 0:
        return False
    ratio = orange / (h * w)
    return 0.02 <= ratio <= 0.55


def generate_input() -> np.ndarray:
    """
    Randomly build black grids with 2+ visible teal markers so ray intersections generate a meaningful chain of orange condensations.
    """
    seed = np.random.default_rng()
    for _ in range(400):
        inp = _build_grid(seed)
        if np.any(inp != Color.BLACK):
            out = main(inp)
            if _passes_visibility_filter(inp, out):
                # Ensure main is stable for this seed sample.
                try:
                    assert np.array_equal(main(inp), out)
                except Exception:
                    continue
                return inp

    raise RuntimeError("failed to generate valid chain-reaction input for puzzle 3")
