from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.is_dir() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color
except Exception:  # pragma: no cover - fallback keeps SourceFileLoader use simple
    class Color:
        BLACK = 0
        GREY = 5


GRID_SIDE = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _diag_coords(shape: tuple[int, int], k: int) -> list[tuple[int, int]]:
    """Return all cells with r - c == k on a grid of shape (H, W)."""
    H, W = map(int, shape)
    coords: list[tuple[int, int]] = []
    for r in range(H):
        c = r - int(k)
        if 0 <= c < W:
            coords.append((r, c))
    return coords


def _extract_line_map(grid: np.ndarray) -> dict[int, int]:
    """Map each non-black, non-grey color to its complete diagonal offset k = r - c."""
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) not in (Color.BLACK, Color.GREY)]
    line_map: dict[int, int] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        ks = coords[:, 0] - coords[:, 1]
        assert np.all(ks == ks[0]), "each color must occupy exactly one diagonal"
        k = int(ks[0])
        expected = set(_diag_coords(g.shape, k))
        actual = {tuple(map(int, rc)) for rc in coords}
        assert actual == expected, "each input diagonal must be complete"
        line_map[color] = k
    return line_map


def _validate_layout(line_map: dict[int, int]) -> tuple[int, int]:
    """Return (D, gap) after checking equal spacing and distinct modulo classes."""
    D = len(line_map)
    assert D in (2, 3), "expected two or three colored diagonals"
    ks = sorted(int(k) for k in line_map.values())
    gaps = [b - a for a, b in zip(ks, ks[1:])]
    assert gaps and len(set(gaps)) == 1, "diagonals must be equally spaced"
    gap = int(gaps[0])
    assert gap > 0
    residues = {int(k) % D for k in ks}
    assert len(residues) == D, "diagonal classes must be distinct modulo D"
    return D, gap


def _fill_periodic(shape: tuple[int, int], line_map: dict[int, int], period: int) -> np.ndarray | None:
    """Tile the whole grid by diagonal congruence classes; return None on overlap/incomplete cover."""
    H, W = map(int, shape)
    rows = np.arange(H, dtype=int)[:, None]
    cols = np.arange(W, dtype=int)[None, :]

    out = np.full((H, W), Color.BLACK, dtype=int)
    owner = np.full((H, W), -1, dtype=int)
    for color, k in sorted(line_map.items()):
        mask = ((rows - cols - int(k)) % int(period)) == 0
        if np.any(mask & (owner != -1)):
            return None
        out[mask] = int(color)
        owner[mask] = int(color)

    if np.any(owner == -1):
        return None
    return out


def _apply_exclusion(base: np.ndarray, blockers: list[tuple[int, int]], radius: int) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    H, W = out.shape
    for r, c in blockers:
        r0 = max(0, int(r) - int(radius))
        r1 = min(H - 1, int(r) + int(radius))
        c0 = max(0, int(c) - int(radius))
        c1 = min(W - 1, int(c) + int(radius))
        out[r0 : r1 + 1, c0 : c1 + 1] = Color.BLACK
    for r, c in blockers:
        out[int(r), int(c)] = Color.GREY
    return out


def _render_original_diagonals(shape: tuple[int, int], line_map: dict[int, int]) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    for color, k in line_map.items():
        for r, c in _diag_coords(shape, int(k)):
            out[r, c] = int(color)
    return out


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], list[tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIDE, GRID_SIDE)
    line_map = _extract_line_map(g)
    D, gap = _validate_layout(line_map)
    blockers = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.GREY)]
    assert 1 <= len(blockers) <= 2
    for r, c in blockers:
        assert int(g[r, c]) == Color.GREY
    if len(blockers) == 2:
        (r0, c0), (r1, c1) = blockers
        assert max(abs(r0 - r1), abs(c0 - c1)) > 1, "grey blockers should stay isolated"
    for r, c in blockers:
        assert all((r, c) not in _diag_coords(g.shape, k) for k in line_map.values())
    return g, line_map, blockers, D, gap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Periodically extend the input diagonals, then clear blocker-centered Chebyshev exclusion squares.

    Input structure:
    - The input is a 9x9 black grid containing 2 or 3 complete colored diagonals of slope down-right.
    - Each colored diagonal has a distinct non-black, non-grey color, and the diagonals are equally spaced.
    - One or two isolated grey pixels act as blockers and do not lie on the colored diagonals.

    Rule:
    - Count the number of colored diagonals D by the number of non-black, non-grey colors.
    - For each colored diagonal with offset k = r - c, fill every cell whose diagonal offset is congruent to k modulo D with that same color, producing a full-grid interlaced diagonal tiling.
    - For each grey blocker, erase every filled pixel inside its Chebyshev-radius-D square; after the erasure, restore the blocker cell itself to grey.

    Key insight:
    - The same global count D controls both the diagonal tiling period and the blocker exclusion radius, so adding or removing one input diagonal changes two linked mechanisms at once.
    """
    g, line_map, blockers, D, _gap = _parse_input(grid)
    filled = _fill_periodic(g.shape, line_map, D)
    assert filled is not None
    return _apply_exclusion(filled, blockers, D)


def _sample_line_offsets(rng: np.random.Generator, D: int) -> list[int]:
    max_gap = (GRID_SIDE - 1) // max(1, D - 1)
    gaps = [gap for gap in range(1, max_gap + 1) if gap % D != 0]
    gap = int(rng.choice(gaps))
    start_min = -(GRID_SIDE - 1)
    start_max = GRID_SIDE - 1 - (D - 1) * gap
    start = int(rng.integers(start_min, start_max + 1))
    return [start + i * gap for i in range(D)]


def _sample_blockers(
    rng: np.random.Generator, line_offsets: list[int], D: int
) -> list[tuple[int, int]]:
    line_set = set(map(int, line_offsets))
    candidates = [
        (r, c)
        for r in range(GRID_SIDE)
        for c in range(GRID_SIDE)
        if (r - c) not in line_set
    ]
    rng.shuffle(candidates)
    blocker_count = 1 if (D == 3 and rng.random() < 0.65) else int(rng.integers(1, 3))
    picked: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in picked):
            picked.append((r, c))
            if len(picked) == blocker_count:
                return picked
    raise RuntimeError("failed to place blockers")


def _build_input(
    line_offsets: list[int], colors: list[int], blockers: list[tuple[int, int]]
) -> np.ndarray:
    grid = np.full((GRID_SIDE, GRID_SIDE), Color.BLACK, dtype=int)
    for color, k in zip(colors, line_offsets):
        for r, c in _diag_coords(grid.shape, int(k)):
            grid[r, c] = int(color)
    for r, c in blockers:
        grid[int(r), int(c)] = Color.GREY
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        g, line_map, blockers, D, gap = _parse_input(inp)
    except AssertionError:
        return False

    periodic = _fill_periodic(g.shape, line_map, D)
    if periodic is None:
        return False
    out = _apply_exclusion(periodic, blockers, D)

    # Both sub-steps must matter: periodic fill must add pixels, and blockers must erase some.
    original = _render_original_diagonals(g.shape, line_map)
    if np.array_equal(out, original):
        return False
    if np.array_equal(out, periodic):
        return False
    erased = int(np.count_nonzero((periodic != Color.BLACK) & (out == Color.BLACK)))
    survivors = int(np.count_nonzero((out != Color.BLACK) & (out != Color.GREY)))
    if erased < 6 or survivors < 15:
        return False

    out_colors = {int(c) for c in np.unique(out) if int(c) not in (Color.BLACK, Color.GREY)}
    if out_colors != set(line_map):
        return False

    # Reject shortcuts that use the visible spacing gap instead of D, or a wrong radius.
    alt_period = _fill_periodic(g.shape, line_map, gap)
    if alt_period is not None and np.array_equal(_apply_exclusion(alt_period, blockers, D), out):
        return False
    wrong_radius = max(1, len(blockers))
    if wrong_radius != D and np.array_equal(_apply_exclusion(periodic, blockers, wrong_radius), out):
        return False
    if gap != D and np.array_equal(_apply_exclusion(periodic, blockers, gap), out):
        return False

    return np.array_equal(main(inp), out)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        D = int(rng.integers(2, 4))
        line_offsets = _sample_line_offsets(rng, D)
        colors = [int(c) for c in rng.choice(PALETTE, size=D, replace=False)]
        rng.shuffle(colors)
        blockers = _sample_blockers(rng, line_offsets, D)
        inp = _build_input(line_offsets, colors, blockers)
        if _semantic_checks(inp):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
