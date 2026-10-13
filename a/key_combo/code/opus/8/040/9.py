from __future__ import annotations

import random

import numpy as np


BLACK = 0
BAND = 2
TRAIL = 7
MARKER = 9
AUX_COLORS = (1, 3, 4, 6)

HEIGHT_RANGE = (10, 16)
WIDTH_RANGE = (10, 16)
EXTRA_HOLES_RANGE = (1, 5)
GEN_RETRIES = 400


def _marker_side(grid: np.ndarray) -> str:
    coords = np.argwhere(grid == MARKER)
    assert coords.size > 0, "expected at least one marker pixel"
    h, w = grid.shape
    distances = {
        "top": max(int(r) for r, _c in coords),
        "bottom": max(h - 1 - int(r) for r, _c in coords),
        "left": max(int(c) for _r, c in coords),
        "right": max(w - 1 - int(c) for _r, c in coords),
    }
    best = min(distances.values())
    candidates = [side for side, value in distances.items() if value == best]
    assert len(candidates) == 1, "marker pixels must identify exactly one edge"
    return candidates[0]


def _band_coords(shape: tuple[int, int], side: str) -> list[tuple[int, int]]:
    h, w = shape
    if side == "left":
        return [(r, c) for r in range(h) for c in range(BAND)]
    if side == "right":
        return [(r, w - BAND + c) for r in range(h) for c in range(BAND)]
    if side == "top":
        return [(r, c) for r in range(BAND) for c in range(w)]
    if side == "bottom":
        return [(h - BAND + r, c) for r in range(BAND) for c in range(w)]
    raise AssertionError(f"unknown side {side!r}")


def _line_count(shape: tuple[int, int], side: str) -> int:
    return int(shape[0] if side in ("left", "right") else shape[1])


def _get_source_slice(grid: np.ndarray, side: str, line: int) -> np.ndarray:
    h, w = grid.shape
    if side == "left":
        return grid[line, :BAND].copy()
    if side == "right":
        return grid[line, w - BAND : w].copy()
    if side == "top":
        return grid[:BAND, line].copy()
    if side == "bottom":
        return grid[h - BAND : h, line].copy()
    raise AssertionError(f"unknown side {side!r}")


def _set_source_slice(grid: np.ndarray, side: str, line: int, values: np.ndarray) -> None:
    h, w = grid.shape
    if side == "left":
        grid[line, :BAND] = values
    elif side == "right":
        grid[line, w - BAND : w] = values
    elif side == "top":
        grid[:BAND, line] = values
    elif side == "bottom":
        grid[h - BAND : h, line] = values
    else:
        raise AssertionError(f"unknown side {side!r}")


def _set_dest_slice(grid: np.ndarray, side: str, line: int, values: np.ndarray) -> None:
    h, w = grid.shape
    if side == "left":
        grid[line, w - BAND : w] = values
    elif side == "right":
        grid[line, :BAND] = values
    elif side == "top":
        grid[h - BAND : h, line] = values
    elif side == "bottom":
        grid[:BAND, line] = values
    else:
        raise AssertionError(f"unknown side {side!r}")


def _infer_cycle(grid: np.ndarray, side: str) -> tuple[int, dict[int, int]]:
    visible: list[tuple[int, int, int]] = []
    for r, c in _band_coords(grid.shape, side):
        value = int(grid[r, c])
        if value not in (BLACK, MARKER):
            visible.append((r, c, value))
    assert visible, "need visible band colors to infer the cycle"

    candidates: list[tuple[int, dict[int, int]]] = []
    for period in (2, 3):
        residue_to_color: dict[int, int] = {}
        ok = True
        for r, c, color in visible:
            residue = (int(r) + int(c)) % period
            prior = residue_to_color.get(residue)
            if prior is None:
                residue_to_color[residue] = color
            elif prior != color:
                ok = False
                break
        if not ok:
            continue
        if len(residue_to_color) != period:
            continue
        if len(set(residue_to_color.values())) != period:
            continue
        if TRAIL not in residue_to_color.values():
            continue
        candidates.append((period, residue_to_color))

    assert len(candidates) == 1, "cycle length and residue mapping must be uniquely inferable"
    return candidates[0]


def _fill_band(grid: np.ndarray, side: str, period: int, residue_to_color: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for r, c in _band_coords(out.shape, side):
        if int(out[r, c]) == BLACK:
            out[r, c] = residue_to_color[(int(r) + int(c)) % period]
    return out


def _apply_transfer(filled: np.ndarray, side: str) -> np.ndarray:
    out = filled.copy()
    for line in range(_line_count(out.shape, side)):
        src = _get_source_slice(filled, side, line)
        if np.any(src == TRAIL):
            _set_dest_slice(out, side, line, src)
            _set_source_slice(out, side, line, np.where(src == MARKER, MARKER, BLACK))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the diagonal edge cycle, then move only the rows or columns whose filled band reveals the trail color.

    Input structure:
    - A black background contains a single 2-cell-wide edge band, plus 1-2 maroon marker pixels on the active edge.
    - Visible non-black, non-maroon cells in that band follow a diagonal color cycle based on `(row + col) mod K`, with `K` equal to 2 or 3.
    - Some band cells are black holes that hide missing cycle cells; cells outside the band are background.

    Rule:
    - Use the maroon markers to identify the source edge.
    - Infer the unique period `K` and residue-to-color mapping from the visible band cells, then fill every black hole in the source band except marker cells.
    - After filling, every row or column whose source-band slice contains the fixed trail color copies that 2-cell slice to the opposite edge, preserving order.
    - On those moving rows or columns, erase only the non-maroon cells from the original source slice; inactive slices remain filled in place.

    Key insight:
    - The missing cells are part of a global diagonal cycle, and completing that cycle is necessary before you can know which slices truly contain the trail color and therefore move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    side = _marker_side(g)
    period, residue_to_color = _infer_cycle(g, side)
    filled = _fill_band(g, side, period, residue_to_color)
    return _apply_transfer(filled, side)


def _build_full_grid(height: int, width: int, period: int, residue_to_color: dict[int, int], marker_rows: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(BAND):
            grid[r, c] = residue_to_color[(r + c) % period]
    for r in marker_rows:
        grid[r, 0] = MARKER
    return grid


def _visible_residues(grid: np.ndarray, period: int) -> set[int]:
    residues: set[int] = set()
    for r in range(grid.shape[0]):
        for c in range(BAND):
            value = int(grid[r, c])
            if value not in (BLACK, MARKER):
                residues.add((r + c) % period)
    return residues


def _hidden_active_rows(inp: np.ndarray, filled: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(inp.shape[0]):
        before_has_trail = bool(np.any(inp[r, :BAND] == TRAIL))
        after_has_trail = bool(np.any(filled[r, :BAND] == TRAIL))
        if after_has_trail and not before_has_trail:
            rows.append(r)
    return rows


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)

    period = 3 if random.random() < 0.7 else 2
    cycle_colors = [TRAIL, *random.sample(AUX_COLORS, period - 1)]
    random.shuffle(cycle_colors)
    residue_to_color = {residue: color for residue, color in enumerate(cycle_colors)}

    base_band = np.array([[residue_to_color[(r + c) % period] for c in range(BAND)] for r in range(height)], dtype=int)
    trail_on_col0_rows = [r for r in range(height) if int(base_band[r, 0]) == TRAIL]

    marker_count = random.randint(1, 2)
    marker_rows: list[int] = []
    if trail_on_col0_rows and random.random() < 0.75:
        marker_rows.append(random.choice(trail_on_col0_rows))
    remaining_rows = [r for r in range(height) if r not in marker_rows]
    while len(marker_rows) < marker_count:
        marker_rows.append(random.choice(remaining_rows))
        remaining_rows = [r for r in remaining_rows if r != marker_rows[-1]]

    filled = _build_full_grid(height, width, period, residue_to_color, marker_rows)

    trail_positions = [(r, c) for r in range(height) for c in range(BAND) if int(filled[r, c]) == TRAIL]
    if len(trail_positions) < 3:
        return None

    required_trail_holes = random.sample(trail_positions, 2)
    eligible = [(r, c) for r in range(height) for c in range(BAND) if int(filled[r, c]) != MARKER]
    optional = [pos for pos in eligible if pos not in required_trail_holes]
    if not optional:
        return None

    extra_count = min(len(optional), random.randint(*EXTRA_HOLES_RANGE))
    extra_holes = random.sample(optional, extra_count)
    holes = set(required_trail_holes + extra_holes)

    inp = filled.copy()
    for r, c in holes:
        inp[r, c] = BLACK

    if _visible_residues(inp, period) != set(range(period)):
        return None
    if not any(np.any(inp[r, :BAND] == TRAIL) for r in range(height)):
        return None
    hidden_rows = _hidden_active_rows(inp, filled)
    if not hidden_rows:
        return None
    trail_holes = sum(1 for r, c in holes if residue_to_color[(r + c) % period] == TRAIL)
    if trail_holes < 2:
        return None

    try:
        expected = _apply_transfer(filled, "left")
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
