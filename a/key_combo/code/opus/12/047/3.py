from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
COLORS = np.arange(1, 10, dtype=int)
MAX_ATTEMPTS = 600


def _compact_lattice(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return g[::2, ::2].copy()


def _axis_window(length: int, axis2: int) -> tuple[int, int]:
    lo = max(0, int(axis2) - (int(length) - 1))
    hi = min(int(length) - 1, int(axis2))
    return int(lo), int(hi)


def _score_vertical(grid: np.ndarray, axis2: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    out_of_bounds = 0
    conflicts = 0
    matches = 0
    supported = 0
    for r, c in np.argwhere(grid != BG):
        c2 = int(axis2) - int(c)
        if not (0 <= c2 < w):
            out_of_bounds += 1
            continue
        supported += 1
        v = int(grid[int(r), int(c2)])
        if v == BG:
            continue
        if v == int(grid[int(r), int(c)]):
            matches += 1
        else:
            conflicts += 1
    return int(out_of_bounds), int(conflicts), -int(matches), -int(supported)


def _score_horizontal(grid: np.ndarray, axis2: int) -> tuple[int, int, int, int]:
    h, _w = grid.shape
    out_of_bounds = 0
    conflicts = 0
    matches = 0
    supported = 0
    for r, c in np.argwhere(grid != BG):
        r2 = int(axis2) - int(r)
        if not (0 <= r2 < h):
            out_of_bounds += 1
            continue
        supported += 1
        v = int(grid[int(r2), int(c)])
        if v == BG:
            continue
        if v == int(grid[int(r), int(c)]):
            matches += 1
        else:
            conflicts += 1
    return int(out_of_bounds), int(conflicts), -int(matches), -int(supported)


def _best_vertical_axis(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[int]]:
    w = int(grid.shape[1])
    scored = [(_score_vertical(grid, axis2), int(axis2)) for axis2 in range(2 * (w - 1) + 1)]
    scored.sort()
    best_score, best_axis2 = scored[0]
    tied = [axis2 for score, axis2 in scored if score == best_score]
    return int(best_axis2), best_score, tied


def _best_horizontal_axis(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[int]]:
    h = int(grid.shape[0])
    scored = [(_score_horizontal(grid, axis2), int(axis2)) for axis2 in range(2 * (h - 1) + 1)]
    scored.sort()
    best_score, best_axis2 = scored[0]
    tied = [axis2 for score, axis2 in scored if score == best_score]
    return int(best_axis2), best_score, tied


def _orbit_positions(shape: tuple[int, int], r: int, c: int, h_axis2: int, v_axis2: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    pts = set()
    for rr in (int(r), int(h_axis2) - int(r)):
        for cc in (int(c), int(v_axis2) - int(c)):
            if 0 <= rr < h and 0 <= cc < w:
                pts.add((int(rr), int(cc)))
    return sorted(pts)


def _fill_by_symmetry(grid: np.ndarray, h_axis2: int, v_axis2: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for _ in range(4):
        changed = False
        for r, c in np.argwhere(out == BG):
            colors = [
                int(out[rr, cc])
                for rr, cc in _orbit_positions(out.shape, int(r), int(c), int(h_axis2), int(v_axis2))
                if int(out[rr, cc]) != BG
            ]
            if not colors:
                continue
            color = int(Counter(colors).most_common(1)[0][0])
            if int(out[int(r), int(c)]) != color:
                out[int(r), int(c)] = color
                changed = True
        if not changed:
            break
    return out


def _quadrant_of(r: int, c: int, h_axis2: int, v_axis2: int) -> str | None:
    dr = -1 if 2 * int(r) < int(h_axis2) else 1 if 2 * int(r) > int(h_axis2) else 0
    dc = -1 if 2 * int(c) < int(v_axis2) else 1 if 2 * int(c) > int(v_axis2) else 0
    if dr == 0 or dc == 0:
        return None
    return {
        (-1, -1): "TL",
        (-1, 1): "TR",
        (1, -1): "BL",
        (1, 1): "BR",
    }[(dr, dc)]


def _collect_valid_orbits(shape: tuple[int, int], h_axis2: int, v_axis2: int) -> list[list[tuple[int, int]]]:
    h, w = map(int, shape)
    r0, r1 = _axis_window(h, h_axis2)
    c0, c1 = _axis_window(w, v_axis2)
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(int(r0), int(r1) + 1):
        for c in range(int(c0), int(c1) + 1):
            orbit = tuple(_orbit_positions(shape, int(r), int(c), int(h_axis2), int(v_axis2)))
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(list(orbit))
    return orbits


def _orbit_sides(
    orbit: list[tuple[int, int]], row_window: tuple[int, int], col_window: tuple[int, int]
) -> set[str]:
    r0, r1 = row_window
    c0, c1 = col_window
    sides = set()
    if any(r == int(r0) for r, _ in orbit):
        sides.add("top")
    if any(r == int(r1) for r, _ in orbit):
        sides.add("bottom")
    if any(c == int(c0) for _, c in orbit):
        sides.add("left")
    if any(c == int(c1) for _, c in orbit):
        sides.add("right")
    return sides


def _is_strictly_symmetric(grid: np.ndarray, h_axis2: int, v_axis2: int) -> bool:
    for r, c in np.argwhere(grid != BG):
        color = int(grid[int(r), int(c)])
        orbit = _orbit_positions(grid.shape, int(r), int(c), int(h_axis2), int(v_axis2))
        if any(int(grid[rr, cc]) != color for rr, cc in orbit):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downsample the separator lattice, infer the hidden double reflection axes, and restore erased colored cells.

    Input structure:
    - The input grid has even height and width, with black background.
    - Every odd-indexed row and every odd-indexed column is an all-black separator.
    - The meaningful pixels lie only on even-row/even-column intersections; these form a compact logical grid.
    - In that logical grid, some colored cells of a doubly symmetric pattern have been erased to black.

    Rule:
    - Remove the separator rows and columns by keeping only even/even lattice intersections.
    - Infer one horizontal and one vertical reflection axis that best preserve the visible non-black cells.
    - For each black cell in the compact grid, inspect its symmetry-equivalent positions under the two axes.
    - If any equivalent positions are colored, fill the black cell with the majority color among those positions.

    Key insight:
    - The separator lattice hides the true geometry; only after downsampling does the off-center bilateral symmetry become visible and usable for completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0, "expected even grid dimensions"
    assert np.all(g[1::2, :] == BG), "odd rows must be separator rows"
    assert np.all(g[:, 1::2] == BG), "odd columns must be separator columns"

    logical = _compact_lattice(g)
    v_axis2, _v_score, _v_tied = _best_vertical_axis(logical)
    h_axis2, _h_score, _h_tied = _best_horizontal_axis(logical)
    return _fill_by_symmetry(logical, int(h_axis2), int(v_axis2))


def _choose_axis2(length: int, rng: np.random.Generator) -> int:
    center = int(length) - 1
    candidates = []
    for axis2 in range(2 * (int(length) - 1) + 1):
        if axis2 == center:
            continue
        lo, hi = _axis_window(int(length), int(axis2))
        if hi - lo + 1 >= 5:
            candidates.append(int(axis2))
    return int(candidates[int(rng.integers(0, len(candidates)))])


def _greedy_boundary_orbits(
    size4_orbits: list[list[tuple[int, int]]],
    row_window: tuple[int, int],
    col_window: tuple[int, int],
    rng: np.random.Generator,
) -> list[list[tuple[int, int]]] | None:
    needed = {"top", "bottom", "left", "right"}
    selected: list[list[tuple[int, int]]] = []
    remaining = list(size4_orbits)
    while needed:
        coverage = [len(_orbit_sides(orbit, row_window, col_window) & needed) for orbit in remaining]
        best = max(coverage, default=0)
        if best == 0:
            return None
        choices = [i for i, count in enumerate(coverage) if count == best]
        pick = int(choices[int(rng.integers(0, len(choices)))])
        orbit = remaining.pop(pick)
        selected.append(orbit)
        needed -= _orbit_sides(orbit, row_window, col_window)
    return selected


def _assign_colors(orbits: list[list[tuple[int, int]]], rng: np.random.Generator) -> list[int]:
    palette_size = int(rng.integers(2, 4))
    palette = [int(c) for c in rng.choice(COLORS, size=palette_size, replace=False)]
    colors = palette[:]
    while len(colors) < len(orbits):
        colors.append(int(palette[int(rng.integers(0, len(palette)))]))
    rng.shuffle(colors)
    return colors


def _embed_lattice(logical: np.ndarray) -> np.ndarray:
    h, w = logical.shape
    out = np.zeros((2 * h, 2 * w), dtype=int)
    out[::2, ::2] = logical
    return out


def _make_complete_logical(
    shape: tuple[int, int], h_axis2: int, v_axis2: int, rng: np.random.Generator
) -> tuple[np.ndarray, list[list[tuple[int, int]]]] | None:
    row_window = _axis_window(int(shape[0]), int(h_axis2))
    col_window = _axis_window(int(shape[1]), int(v_axis2))
    orbits = _collect_valid_orbits(shape, int(h_axis2), int(v_axis2))
    size4_orbits = [orbit for orbit in orbits if len(orbit) == 4]
    size2_orbits = [orbit for orbit in orbits if len(orbit) == 2]
    if len(size4_orbits) < 5:
        return None

    mandatory = _greedy_boundary_orbits(size4_orbits, row_window, col_window, rng)
    if mandatory is None:
        return None

    remaining4 = [orbit for orbit in size4_orbits if orbit not in mandatory]
    rng.shuffle(remaining4)
    target4 = int(rng.integers(5, min(len(size4_orbits), 7) + 1))
    selected4 = mandatory + remaining4[: max(0, target4 - len(mandatory))]

    selected = list(selected4)
    if size2_orbits and rng.random() < 0.6:
        rng.shuffle(size2_orbits)
        selected.extend(size2_orbits[: int(rng.integers(0, min(len(size2_orbits), 2) + 1))])

    colors = _assign_colors(selected, rng)
    full = np.zeros(shape, dtype=int)
    for orbit, color in zip(selected, colors):
        for r, c in orbit:
            full[int(r), int(c)] = int(color)

    if int(np.count_nonzero(full)) < 14:
        return None
    if not _is_strictly_symmetric(full, int(h_axis2), int(v_axis2)):
        return None
    return full, selected4


def _erase_cells(
    full: np.ndarray,
    colored_size4_orbits: list[list[tuple[int, int]]],
    h_axis2: int,
    v_axis2: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    if len(colored_size4_orbits) < 5:
        return None
    n_holes = int(rng.integers(3, 6))
    chosen_indices = rng.choice(len(colored_size4_orbits), size=n_holes, replace=False)
    quadrants = ["TL", "TR", "BL", "BR"]
    rng.shuffle(quadrants)
    if n_holes == 3:
        target_quadrants = quadrants[:3]
    elif n_holes == 4:
        target_quadrants = quadrants
    else:
        target_quadrants = quadrants + [quadrants[int(rng.integers(0, 4))]]

    incomplete = full.copy()
    holes: list[tuple[int, int]] = []
    for idx, quad in zip(chosen_indices, target_quadrants):
        orbit = colored_size4_orbits[int(idx)]
        candidates = [(r, c) for r, c in orbit if _quadrant_of(r, c, int(h_axis2), int(v_axis2)) == quad]
        if len(candidates) != 1:
            return None
        r, c = candidates[0]
        incomplete[int(r), int(c)] = BG
        holes.append((int(r), int(c)))
    return incomplete, holes


def _validate_case(
    inp: np.ndarray,
    full: np.ndarray,
    incomplete: np.ndarray,
    holes: list[tuple[int, int]],
    h_axis2: int,
    v_axis2: int,
) -> bool:
    if not np.array_equal(main(inp), full):
        return False

    got_v_axis2, _v_score, v_tied = _best_vertical_axis(incomplete)
    got_h_axis2, _h_score, h_tied = _best_horizontal_axis(incomplete)
    if got_v_axis2 != int(v_axis2) or got_h_axis2 != int(h_axis2):
        return False
    if len(v_tied) != 1 or len(h_tied) != 1:
        return False
    if len({_quadrant_of(r, c, int(h_axis2), int(v_axis2)) for r, c in holes}) < 3:
        return False

    for r, c in holes:
        if int(full[int(r), int(c)]) == BG or int(incomplete[int(r), int(c)]) != BG:
            return False
        orbit = _orbit_positions(full.shape, int(r), int(c), int(h_axis2), int(v_axis2))
        seen = [int(incomplete[rr, cc]) for rr, cc in orbit if int(incomplete[rr, cc]) != BG]
        if not seen:
            return False
        if any(color != int(full[int(r), int(c)]) for color in seen):
            return False
    return True


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    shape = (int(rng.integers(7, 9)), int(rng.integers(7, 9)))
    h_axis2 = _choose_axis2(int(shape[0]), rng)
    v_axis2 = _choose_axis2(int(shape[1]), rng)

    built = _make_complete_logical(shape, int(h_axis2), int(v_axis2), rng)
    if built is None:
        return None
    full, colored_size4_orbits = built

    erased = _erase_cells(full, colored_size4_orbits, int(h_axis2), int(v_axis2), rng)
    if erased is None:
        return None
    incomplete, holes = erased

    inp = _embed_lattice(incomplete)
    if not _validate_case(inp, full, incomplete, holes, int(h_axis2), int(v_axis2)):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 3")
