from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_corner_bracket(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2 or len(coords) != h + w - 1 or not (3 <= len(coords) <= 5):
        return False
    candidates = [
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
    ]
    return any(pts == cand for cand in candidates)


def _find_bracket_and_marker(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int, int, int], tuple[int, int], int]:
    comps = _find_components(grid == GREEN)
    brackets = [comp for comp in comps if _is_corner_bracket(comp)]
    assert len(brackets) == 1, "expected exactly one green corner bracket"
    bracket = brackets[0]
    bracket_set = set(bracket)
    box = _bbox(bracket)
    r0, c0, r1, c1 = box
    markers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in bracket_set and grid[r, c] != BG
    ]
    assert len(markers) == 1, "expected one colored marker inside the bracket bbox"
    marker = markers[0]
    marker_color = int(grid[marker])
    assert marker_color != BG and marker_color != GREEN
    return bracket_set, box, marker, marker_color


def _best_axis_sum(grid: np.ndarray, *, along_rows: bool) -> int:
    limit = grid.shape[0] if along_rows else grid.shape[1]
    candidates: list[tuple[int, int]] = []
    for axis_sum in range(2 * limit - 1):
        evidence = 0
        conflicts = 0
        for i in range(limit):
            j = axis_sum - i
            if not (0 <= j < limit and i < j):
                continue
            a = grid[i, :] if along_rows else grid[:, i]
            b = grid[j, :] if along_rows else grid[:, j]
            known = (a != BG) & (b != BG)
            evidence += int(np.count_nonzero(known))
            conflicts += int(np.count_nonzero(known & (a != b)))
        if conflicts == 0 and evidence > 0:
            candidates.append((evidence, axis_sum))
    assert candidates, "expected at least one conflict-free reflection axis"
    best_evidence = max(score for score, _axis in candidates)
    best_axes = [axis for score, axis in candidates if score == best_evidence]
    assert len(best_axes) == 1, "expected a unique strongest reflection axis"
    return int(best_axes[0])


def _erase_pollution(
    grid: np.ndarray,
    marker: tuple[int, int],
    pollution_color: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    erased = np.array(grid, dtype=int, copy=True)
    polluted = [
        (int(r), int(c))
        for r, c in np.argwhere(erased == int(pollution_color))
        if (int(r), int(c)) != marker
    ]
    for r, c in polluted:
        erased[r, c] = BG
    erased[marker] = pollution_color
    return erased, polluted


def _masked_pattern_grid(grid: np.ndarray, bracket_box: tuple[int, int, int, int]) -> np.ndarray:
    masked = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = bracket_box
    masked[r0 : r1 + 1, c0 : c1 + 1] = BG
    return masked


def _repair_color(
    ref: np.ndarray,
    pos: tuple[int, int],
    row_axis: int,
    col_axis: int,
) -> int:
    r, c = pos
    probes = [
        (row_axis - r, c),
        (r, col_axis - c),
        (row_axis - r, col_axis - c),
    ]
    for rr, cc in probes:
        if _in_bounds(ref.shape, rr, cc) and int(ref[rr, cc]) != BG:
            return int(ref[rr, cc])
    raise AssertionError(f"polluted cell {pos} has no visible symmetry mate")


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the bracket-indicated contaminant color, then restore the central design from its exposed two-axis reflection symmetry.

    Input structure:
    - Black is the background, and a central blue/red design occupies the middle of the grid.
    - A unique monochromatic green 4-connected corner bracket sits away from the design, with exactly one colored marker pixel inside its bbox.
    - Several orange contaminant pixels overwrite cells of the central design, while the marker pixel itself uses the same contaminant color.

    Rule:
    - Find the unique green corner bracket and read the color of the single non-bracket marker inside its bbox.
    - Replace every other pixel of that color with background, but keep the bracket and the marker unchanged.
    - Ignore the entire bracket bbox, infer the unique horizontal and vertical reflection axes of the remaining non-background design, and use those axes to refill each erased design cell.
    - For each erased position, first try its horizontal reflection mate, then its vertical reflection mate, then its 180-degree counterpart; copy the first non-background color found.

    Key insight:
    - The bracket names which pixels are pollution, and erasing them is what reveals the hidden reflection center needed to reconstruct the damaged cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bracket_set, bracket_box, marker, pollution_color = _find_bracket_and_marker(g)
    erased, polluted = _erase_pollution(g, marker, pollution_color)

    masked = _masked_pattern_grid(erased, bracket_box)
    row_axis = _best_axis_sum(masked, along_rows=True)
    col_axis = _best_axis_sum(masked, along_rows=False)

    ref = masked
    out = erased.copy()
    for cell in polluted:
        out[cell] = _repair_color(ref, cell, row_axis, col_axis)
    for r, c in bracket_set:
        out[r, c] = GREEN
    out[marker] = pollution_color
    return out


def _axis_orbit(r: int, c: int, row_axis: int, col_axis: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (r, c),
        (row_axis - r, c),
        (r, col_axis - c),
        (row_axis - r, col_axis - c),
    }
    return tuple(sorted((int(rr), int(cc)) for rr, cc in pts))


def _component_size(mask: np.ndarray) -> int:
    comps = _find_components(mask)
    return len(comps)


def _grow_quarter_mask(qh: int, qw: int, target: int) -> np.ndarray:
    mask = np.zeros((qh, qw), dtype=bool)
    center = (qh - 1, qw - 1)
    mask[center] = True
    frontier = {center}
    while int(mask.sum()) < target:
        choices = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < qh and 0 <= nc < qw and not mask[nr, nc]:
                    choices.add((nr, nc))
        if not choices:
            return mask
        cell = random.choice(tuple(choices))
        mask[cell] = True
        frontier.add(cell)
    return mask


def _reflect_quarter_mask(quarter: np.ndarray, center: tuple[int, int], radii: tuple[int, int]) -> np.ndarray:
    rad_r, rad_c = radii
    cr, cc = center
    full = np.zeros((2 * rad_r + 1, 2 * rad_c + 1), dtype=bool)
    for qr, qc in np.argwhere(quarter):
        r = cr - rad_r + int(qr)
        c = cc - rad_c + int(qc)
        for rr, cc2 in _axis_orbit(r, c, 2 * cr, 2 * cc):
            full[rr - (cr - rad_r), cc2 - (cc - rad_c)] = True
    return full


def _embed_mask(local_mask: np.ndarray, grid_shape: tuple[int, int], top_left: tuple[int, int]) -> np.ndarray:
    h, w = grid_shape
    out = np.zeros((h, w), dtype=bool)
    r0, c0 = top_left
    hh, ww = local_mask.shape
    out[r0 : r0 + hh, c0 : c0 + ww] = local_mask
    return out


def _orbit_groups(mask: np.ndarray, row_axis: int, col_axis: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    groups: list[tuple[tuple[int, int], ...]] = []
    for r, c in np.argwhere(mask):
        orbit = _axis_orbit(int(r), int(c), row_axis, col_axis)
        if orbit in seen:
            continue
        seen.add(orbit)
        if all(mask[rr, cc] for rr, cc in orbit):
            groups.append(orbit)
    return groups


def _paint_pattern(mask: np.ndarray, orbits: list[tuple[tuple[int, int], ...]]) -> np.ndarray:
    grid = np.zeros(mask.shape, dtype=int)
    colors = []
    for orbit in orbits:
        color = random.choice([BLUE, RED])
        colors.append(color)
        for r, c in orbit:
            grid[r, c] = color
    counts = Counter(colors)
    if counts[BLUE] == 0 or counts[RED] == 0:
        return grid
    return grid


def _branch_type(
    ref: np.ndarray,
    cell: tuple[int, int],
    row_axis: int,
    col_axis: int,
) -> str:
    r, c = cell
    probes = [
        ("horizontal", (row_axis - r, c)),
        ("vertical", (r, col_axis - c)),
        ("rotation", (row_axis - r, col_axis - c)),
    ]
    for name, (rr, cc) in probes:
        if _in_bounds(ref.shape, rr, cc) and int(ref[rr, cc]) != BG:
            return name
    raise AssertionError("unrepairable polluted cell")


def _choose_pollution(orbits4: list[tuple[tuple[int, int], ...]], row_axis: int, col_axis: int) -> list[tuple[int, int]]:
    template_options = [
        ["triple"],
        ["triple", "single"],
        ["triple", "hpair"],
        ["triple", "hpair", "single"],
    ]
    valid = [parts for parts in template_options if len(orbits4) >= len(parts)]
    assert valid
    parts = random.choice(valid)
    chosen_orbits = random.sample(orbits4, len(parts))
    polluted: list[tuple[int, int]] = []
    for orbit, part in zip(chosen_orbits, parts):
        top_left = min(orbit)
        p = top_left
        h = (row_axis - p[0], p[1])
        v = (p[0], col_axis - p[1])
        if part == "triple":
            polluted.extend([p, h, v])
        elif part == "hpair":
            polluted.extend([p, h])
        else:
            polluted.append(p)
    return sorted(set((int(r), int(c)) for r, c in polluted))


def _bracket_candidates(h: int, w: int) -> list[tuple[set[tuple[int, int]], tuple[int, int, int, int], tuple[int, int]]]:
    candidates = []
    dims = [(2, 2), (2, 3), (3, 2)]
    corners = ["tl", "tr", "bl", "br"]
    for bh, bw in dims:
        for corner in corners:
            if corner == "tl":
                r0, c0 = 0, 0
            elif corner == "tr":
                r0, c0 = 0, w - bw
            elif corner == "bl":
                r0, c0 = h - bh, 0
            else:
                r0, c0 = h - bh, w - bw
            r1, c1 = r0 + bh - 1, c0 + bw - 1
            shapes = [
                {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
                {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
                {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
                {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
            ]
            for bracket in shapes:
                interior = [
                    (r, c)
                    for r in range(r0, r1 + 1)
                    for c in range(c0, c1 + 1)
                    if (r, c) not in bracket
                ]
                if not interior:
                    continue
                marker = random.choice(interior)
                candidates.append((bracket, (r0, c0, r1, c1), marker))
    return candidates


def _design_pattern(grid_shape: tuple[int, int]) -> tuple[np.ndarray, int, int, list[tuple[tuple[int, int], ...]]] | None:
    h, w = grid_shape
    cr, cc = h // 2, w // 2
    row_axis = 2 * cr
    col_axis = 2 * cc
    for _ in range(120):
        rad_r = random.randint(3, min(5, cr - 1))
        rad_c = random.randint(3, min(5, cc - 1))
        qh, qw = rad_r + 1, rad_c + 1
        target = random.randint(max(7, (qh * qw) // 2), qh * qw)
        quarter = _grow_quarter_mask(qh, qw, target)
        local_mask = _reflect_quarter_mask(quarter, (rad_r, rad_c), (rad_r, rad_c))
        area = int(np.count_nonzero(local_mask))
        if not (30 <= area <= 50):
            continue
        if _component_size(local_mask) != 1:
            continue
        top_left = (cr - rad_r, cc - rad_c)
        full_mask = _embed_mask(local_mask, grid_shape, top_left)
        orbits = _orbit_groups(full_mask, row_axis, col_axis)
        orbits4 = [orbit for orbit in orbits if len(orbit) == 4]
        if len(orbits4) < 2:
            continue
        clean = _paint_pattern(full_mask, orbits)
        if not np.any(clean == BLUE) or not np.any(clean == RED):
            continue
        try:
            if _best_axis_sum(clean, along_rows=True) != row_axis:
                continue
            if _best_axis_sum(clean, along_rows=False) != col_axis:
                continue
        except AssertionError:
            continue
        return clean, row_axis, col_axis, orbits4
    return None


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    polluted: list[tuple[int, int]],
) -> bool:
    try:
        bracket_set, box, marker, pollution_color = _find_bracket_and_marker(inp)
        if pollution_color != ORANGE:
            return False
        erased, polluted_found = _erase_pollution(inp, marker, pollution_color)
        if sorted(polluted_found) != sorted(polluted):
            return False
        masked = _masked_pattern_grid(erased, box)
        row_axis = _best_axis_sum(masked, along_rows=True)
        col_axis = _best_axis_sum(masked, along_rows=False)
        branch_set = {_branch_type(masked, cell, row_axis, col_axis) for cell in polluted}
        if branch_set != {"horizontal", "vertical", "rotation"}:
            return False
        if not np.array_equal(main(inp), expected):
            return False
        if np.array_equal(inp, expected):
            return False
        if not bracket_set:
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.choice([13, 15, 17])
    w = random.choice([13, 15, 17])
    designed = _design_pattern((h, w))
    if designed is None:
        return None
    clean, row_axis, col_axis, orbits4 = designed

    full_grid = np.zeros((h, w), dtype=int)
    full_grid[clean != BG] = clean[clean != BG]

    bracket_options = _bracket_candidates(h, w)
    random.shuffle(bracket_options)
    bracket = None
    for bracket_cells, box, marker in bracket_options:
        r0, c0, r1, c1 = box
        if np.any(clean[r0 : r1 + 1, c0 : c1 + 1] != BG):
            continue
        bracket = (bracket_cells, box, marker)
        break
    if bracket is None:
        return None
    bracket_cells, box, marker = bracket

    polluted = _choose_pollution(orbits4, row_axis, col_axis)
    if len(polluted) < 3 or len(polluted) > 6:
        return None

    expected = full_grid.copy()
    for r, c in bracket_cells:
        expected[r, c] = GREEN
    expected[marker] = ORANGE
    out_sig = (expected.shape, expected.tobytes())
    if out_sig in SEEN_OUTPUT_SIGNATURES:
        return None

    inp = expected.copy()
    for r, c in polluted:
        inp[r, c] = ORANGE

    if not _semantic_ok(inp, expected, polluted):
        return None
    SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
