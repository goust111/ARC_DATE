import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
MARKER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
CASE_ORDER = ("main_upper", "anti_upper", "main_lower", "anti_lower")
GENERATION_ATTEMPTS = 300
_GEN_COUNTER = 0


def _neighbors4(r: int, c: int, n: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            out.append((rr, cc))
    return out


def _neighbors8(r: int, c: int, n: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                out.append((rr, cc))
    return out


def _sector_of(r: int, c: int, n: int) -> str:
    anti = n - 1 - c
    if r < c and r < anti:
        return "top"
    if r < c and r > anti:
        return "right"
    if r > c and r < anti:
        return "left"
    if r > c and r > anti:
        return "bottom"
    raise ValueError("cell is on a diagonal and does not belong to a unique sector")


def _case_from_marker(r: int, c: int, n: int) -> str:
    sector = _sector_of(r, c, n)
    mapping = {
        "right": "main_upper",
        "left": "main_lower",
        "top": "anti_upper",
        "bottom": "anti_lower",
    }
    return mapping[sector]


def _case_params(case_name: str) -> tuple[str, str]:
    if case_name == "main_upper":
        return "main", "upper"
    if case_name == "main_lower":
        return "main", "lower"
    if case_name == "anti_upper":
        return "anti", "upper"
    if case_name == "anti_lower":
        return "anti", "lower"
    raise ValueError(f"unknown case: {case_name}")


def _source_sectors(case_name: str) -> tuple[str, str]:
    mapping = {
        "main_upper": ("top", "right"),
        "main_lower": ("left", "bottom"),
        "anti_upper": ("top", "left"),
        "anti_lower": ("right", "bottom"),
    }
    return mapping[case_name]


def _marker_sector(case_name: str) -> str:
    mapping = {
        "main_upper": "right",
        "main_lower": "left",
        "anti_upper": "top",
        "anti_lower": "bottom",
    }
    return mapping[case_name]


def _is_source_cell(r: int, c: int, n: int, axis: str, source_side: str) -> bool:
    if axis == "main":
        return r < c if source_side == "upper" else r > c
    return r < (n - 1 - c) if source_side == "upper" else r > (n - 1 - c)


def _reflect_coord(r: int, c: int, n: int, axis: str) -> tuple[int, int]:
    if axis == "main":
        return c, r
    return n - 1 - c, n - 1 - r


def _solve_with_case(grid: np.ndarray, case_name: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n, m = g.shape
    assert n == m and n % 2 == 1 and 7 <= n <= 11, "expected an odd square grid of size 7..11"

    axis, source_side = _case_params(case_name)
    out = g.copy()
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BACKGROUND, MARKER))
    for color in colors:
        coords = np.argwhere(g == color)
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if not _is_source_cell(rr, cc, n, axis, source_side):
                continue
            tr, tc = _reflect_coord(rr, cc, n, axis)
            if int(out[tr, tc]) == BACKGROUND:
                out[tr, tc] = color

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one gray marker"
    mr, mc = map(int, marker_coords[0])
    out[mr, mc] = BACKGROUND
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a partially mirrored multicolor pattern using the diagonal and source side encoded by one gray marker.

    Input structure:
    - The grid is an odd square canvas with black background.
    - One gray singleton marker lies off both diagonals, inside one of the four sectors cut by the main and anti diagonals.
    - Two or three non-black, non-gray colors form a pattern that is complete on one side of a diagonal and only partially present on the mirrored side.

    Rule:
    - Read the gray marker sector to choose one diagonal and one source side:
      right sector -> main diagonal with the upper side as source;
      left sector -> main diagonal with the lower side as source;
      top sector -> anti diagonal with the upper side as source;
      bottom sector -> anti diagonal with the lower side as source.
    - For each foreground color independently, reflect every source-side pixel across the chosen diagonal.
    - If the reflected cell is black, fill it with that color; otherwise keep the existing non-black cell unchanged.
    - Erase the gray marker back to black and leave the source side untouched.

    Key insight:
    - The marker is the whole instruction: its sector jointly encodes both which diagonal is active and which half of the pattern should be trusted as the complete source.
    """
    g = np.asarray(grid, dtype=int)
    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one gray marker"
    n, m = g.shape
    assert n == m, "expected a square grid"
    mr, mc = map(int, marker_coords[0])
    case_name = _case_from_marker(mr, mc, n)
    return _solve_with_case(g, case_name)


def _sector_cells(n: int, sector: str) -> list[tuple[int, int]]:
    out = []
    for r in range(n):
        for c in range(n):
            try:
                if _sector_of(r, c, n) == sector:
                    out.append((r, c))
            except ValueError:
                pass
    return out


def _dilated(coords: Iterable[tuple[int, int]], n: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in coords:
        out.add((r, c))
        out.update(_neighbors8(r, c, n))
    return out


def _grow_component(
    n: int,
    sector: str,
    forbidden: set[tuple[int, int]],
    size_lo: int,
    size_hi: int,
) -> set[tuple[int, int]] | None:
    sector_cells = [cell for cell in _sector_cells(n, sector) if cell not in forbidden]
    if len(sector_cells) < size_lo:
        return None

    for _ in range(120):
        start = random.choice(sector_cells)
        target_size = random.randint(size_lo, min(size_hi, len(sector_cells)))
        cells = {start}
        while len(cells) < target_size:
            frontier = []
            for r, c in cells:
                for rr, cc in _neighbors4(r, c, n):
                    if (rr, cc) in cells or (rr, cc) in forbidden:
                        continue
                    try:
                        if _sector_of(rr, cc, n) == sector:
                            frontier.append((rr, cc))
                    except ValueError:
                        pass
            if not frontier:
                break
            frontier = list(dict.fromkeys(frontier))
            cells.add(random.choice(frontier))

        if len(cells) < size_lo:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) == 1 or len(cols) == 1:
            continue
        return cells

    return None


def _build_source_components(n: int, case_name: str, colors: list[int]) -> dict[int, set[tuple[int, int]]] | None:
    required_sectors = list(_source_sectors(case_name))
    assigned_sectors = required_sectors[:]
    while len(assigned_sectors) < len(colors):
        assigned_sectors.append(random.choice(required_sectors))
    random.shuffle(assigned_sectors)

    size_hi = 3 if n == 7 else 4 if n == 9 else 5
    forbidden: set[tuple[int, int]] = set()
    components: dict[int, set[tuple[int, int]]] = {}
    for color, sector in zip(colors, assigned_sectors):
        component = _grow_component(n, sector, forbidden, 2, size_hi)
        if component is None:
            return None
        components[color] = component
        forbidden |= _dilated(component, n)

    return components


def _place_marker(grid: np.ndarray, case_name: str) -> tuple[int, int] | None:
    n = grid.shape[0]
    sector = _marker_sector(case_name)
    candidates = []
    for r, c in _sector_cells(n, sector):
        if int(grid[r, c]) != BACKGROUND:
            continue
        if any(int(grid[rr, cc]) != BACKGROUND for rr, cc in _neighbors8(r, c, n)):
            continue
        candidates.append((r, c))
    if not candidates:
        candidates = [(r, c) for r, c in _sector_cells(n, sector) if int(grid[r, c]) == BACKGROUND]
    if not candidates:
        return None
    return random.choice(candidates)


def _validate_generated(inp: np.ndarray, expected_out: np.ndarray, case_name: str) -> bool:
    if not np.array_equal(main(inp), expected_out):
        return False

    alt_outputs = []
    for alt_case in CASE_ORDER:
        if alt_case == case_name:
            continue
        alt_outputs.append(_solve_with_case(inp, alt_case))
    if any(np.array_equal(alt, expected_out) for alt in alt_outputs):
        return False

    if np.array_equal(inp, expected_out):
        return False

    colors = [int(v) for v in np.unique(expected_out) if int(v) not in (BACKGROUND, MARKER)]
    if len(colors) < 2:
        return False
    return True


def _gen_attempt(case_name: str) -> np.ndarray | None:
    n = random.choice((7, 9, 11))
    color_count = random.choice((2, 3))
    colors = random.sample(PALETTE, color_count)
    components = _build_source_components(n, case_name, colors)
    if components is None:
        return None

    axis, _source_side = _case_params(case_name)
    out = np.zeros((n, n), dtype=int)
    missing_total = 0
    for color, cells in components.items():
        for r, c in cells:
            out[r, c] = color
            rr, cc = _reflect_coord(r, c, n, axis)
            out[rr, cc] = color

    inp = out.copy()
    for color, cells in components.items():
        mirrored = [_reflect_coord(r, c, n, axis) for r, c in cells]
        if len(mirrored) < 2:
            return None
        missing_count = random.randint(1, len(mirrored) - 1)
        missing = set(random.sample(mirrored, missing_count))
        for rr, cc in missing:
            inp[rr, cc] = BACKGROUND
        missing_total += len(missing)

    if missing_total < 2:
        return None

    marker = _place_marker(inp, case_name)
    if marker is None:
        return None
    mr, mc = marker
    inp[mr, mc] = MARKER

    if _case_from_marker(mr, mc, n) != case_name:
        return None

    required_sectors = set(_source_sectors(case_name))
    seen_sectors = {
        _sector_of(r, c, n)
        for cells in components.values()
        for r, c in cells
    }
    if not required_sectors.issubset(seen_sectors):
        return None

    if not _validate_generated(inp, out, case_name):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(GENERATION_ATTEMPTS):
        case_name = CASE_ORDER[_GEN_COUNTER % len(CASE_ORDER)]
        _GEN_COUNTER += 1
        grid = _gen_attempt(case_name)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
