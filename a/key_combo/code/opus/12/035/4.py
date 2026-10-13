import random
from collections import deque
from typing import Iterable

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
WIRE = 5
HOLE = 9
MAX_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _largest_pattern_component(grid: np.ndarray) -> np.ndarray:
    comps = _components((grid != BG) & (grid != WIRE))
    if not comps:
        raise AssertionError("expected a non-wire foreground component")
    return max(comps, key=len)


def _pattern_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    return _bbox_coords(_largest_pattern_component(grid))


def _pattern_holes(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts = np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == HOLE)
    return [(int(r0 + r), int(c0 + c)) for r, c in pts]


def _wire_neighbor(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    found = []
    for nr, nc in _neighbors4(r, c):
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == WIRE:
            found.append((nr, nc))
    if not found:
        return None
    assert len(found) == 1, "each wired hole should touch exactly one wire direction"
    return found[0]


def _trace_wire(grid: np.ndarray, hole: tuple[int, int]) -> tuple[int, list[tuple[int, int]], tuple[int, int]] | None:
    r, c = hole
    first = _wire_neighbor(grid, r, c)
    if first is None:
        return None
    dr = first[0] - r
    dc = first[1] - c
    rr, cc = first
    path: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, rr, cc) and grid[rr, cc] == WIRE:
        path.append((rr, cc))
        rr += dr
        cc += dc
    assert _in_bounds(grid.shape, rr, cc), "wire must terminate inside the grid at a source pixel"
    color = int(grid[rr, cc])
    assert color not in {BG, WIRE, HOLE}, "wire must terminate at a colored source"
    return color, path, (rr, cc)


def _fill_by_symmetry(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = bbox
    row_sum = r0 + r1
    col_sum = c0 + c1

    changed = True
    while changed:
        changed = False
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r, c] != HOLE:
                    continue
                colors = set()
                for pr, pc in ((row_sum - r, c), (r, col_sum - c)):
                    if not (r0 <= pr <= r1 and c0 <= pc <= c1):
                        continue
                    value = int(out[pr, pc])
                    if value not in {BG, WIRE, HOLE}:
                        colors.add(value)
                if len(colors) == 1:
                    out[r, c] = colors.pop()
                    changed = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the occluded symmetric pattern by first reading line-anchored colors, then using reflection to fill the remaining holes.

    Input structure:
    - A black background contains one main multicolor pattern plus maroon hole pixels inside that pattern.
    - The main pattern is the largest non-black, non-grey connected component, and its tight bbox is horizontally and vertically reflection-symmetric.
    - Exactly some holes touch straight grey wires that run outward to remote colored source pixels outside the pattern.

    Rule:
    - For every hole adjacent to a grey wire, follow that wire in its straight direction to the terminal colored source and copy that color into the hole.
    - Remove all grey wire cells and their source pixels back to black.
    - Then, inside the pattern bbox, fill every remaining maroon hole from the unique visible color at its horizontal or vertical reflection partner.

    Key insight:
    - The wires are not decorative: they reveal the colors of symmetry-deadlocked holes, and those repaired cells restore enough visible anchors for the remaining reflection-based completion.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bbox = _pattern_bbox(out)

    to_clear: set[tuple[int, int]] = set()
    for hole in _pattern_holes(out, bbox):
        traced = _trace_wire(out, hole)
        if traced is None:
            continue
        color, path, source = traced
        out[hole] = color
        to_clear.update(path)
        to_clear.add(source)

    for r, c in to_clear:
        out[r, c] = BG

    out = _fill_by_symmetry(out, bbox)
    assert not np.any(out[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == HOLE), "all holes must be repairable"
    return out


def _orbit_cells(r: int, c: int, height: int, width: int) -> tuple[tuple[int, int], ...]:
    row_sum = height - 1
    col_sum = width - 1
    pts = {
        (r, c),
        (row_sum - r, c),
        (r, col_sum - c),
        (row_sum - r, col_sum - c),
    }
    return tuple(sorted(pts))


def _make_symmetric_pattern(height: int, width: int, rng: random.Random) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    half_h = height // 2
    half_w = width // 2
    seed = np.zeros((half_h + 1, half_w + 1), dtype=int)
    for r in range(half_h + 1):
        for c in range(half_w + 1):
            options = list(PALETTE)
            if r > 0 and rng.random() < 0.55:
                options.extend([int(seed[r - 1, c])] * 3)
            if c > 0 and rng.random() < 0.55:
                options.extend([int(seed[r, c - 1])] * 3)
            if r > 0 and c > 0 and rng.random() < 0.35:
                options.extend([int(seed[r - 1, c - 1])] * 2)
            seed[r, c] = rng.choice(options)
    for r in range(height):
        for c in range(width):
            sr = min(r, height - 1 - r)
            sc = min(c, width - 1 - c)
            out[r, c] = seed[sr, sc]
    if len(set(map(int, np.unique(out)))) < 3:
        return _make_symmetric_pattern(height, width, rng)
    return out


def _choose_single_orbit(
    height: int,
    width: int,
    used_orbits: set[tuple[tuple[int, int], ...]],
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]] | None:
    half_h = height // 2
    half_w = width // 2
    candidates = []
    for r in range(height):
        for c in range(width):
            orbit = _orbit_cells(r, c, height, width)
            if len(orbit) != 4 or orbit in used_orbits:
                continue
            if r == half_h or c == half_w:
                continue
            candidates.append((r, c, orbit))
    if not candidates:
        return None
    r, c, orbit = rng.choice(candidates)
    return (r, c), orbit


def _choose_pair_orbit(
    height: int,
    width: int,
    used_orbits: set[tuple[tuple[int, int], ...]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], tuple[tuple[int, int], ...]] | None:
    row_sum = height - 1
    col_sum = width - 1
    half_h = height // 2
    half_w = width // 2
    candidates = []
    for r in range(height):
        for c in range(width):
            orbit = _orbit_cells(r, c, height, width)
            if len(orbit) != 4 or orbit in used_orbits:
                continue
            if r == half_h or c == half_w:
                continue
            h_pair = sorted([(r, c), (row_sum - r, c)])
            v_pair = sorted([(r, c), (r, col_sum - c)])
            if h_pair[0] != h_pair[1]:
                candidates.append((h_pair, orbit))
            if v_pair[0] != v_pair[1]:
                candidates.append((v_pair, orbit))
    if not candidates:
        return None
    return rng.choice(candidates)


def _draw_wire(
    grid: np.ndarray,
    hole: tuple[int, int],
    direction: tuple[int, int],
    source_color: int,
) -> None:
    h, w = grid.shape
    dr, dc = direction
    rr, cc = hole[0] + dr, hole[1] + dc
    path: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, rr + dr, cc + dc):
        path.append((rr, cc))
        rr += dr
        cc += dc
    assert path, "wire needs at least one grey segment"
    assert grid[rr, cc] == BG, "source endpoint should start on background"
    for pr, pc in path:
        assert grid[pr, pc] == BG, "wire may not cross existing non-background pixels"
        grid[pr, pc] = WIRE
    grid[rr, cc] = source_color


def _build_axis_pair_case(
    pattern: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    height, width = pattern.shape
    half_h = height // 2
    half_w = width // 2
    holes: list[tuple[int, int]] = []
    wires: list[tuple[tuple[int, int], tuple[int, int]]] = []
    used_orbits: set[tuple[tuple[int, int], ...]] = set()

    if rng.random() < 0.5:
        orbit = _orbit_cells(half_h, 0, height, width)
        used_orbits.add(orbit)
        holes.extend([(half_h, 0), (half_h, width - 1)])
        wires.extend([((half_h, 0), (0, -1)), ((half_h, width - 1), (0, 1))])
    else:
        orbit = _orbit_cells(0, half_w, height, width)
        used_orbits.add(orbit)
        holes.extend([(0, half_w), (height - 1, half_w)])
        wires.extend([((0, half_w), (-1, 0)), ((height - 1, half_w), (1, 0))])

    pair = _choose_pair_orbit(height, width, used_orbits, rng)
    assert pair is not None
    pair_holes, pair_orbit = pair
    used_orbits.add(pair_orbit)
    holes.extend(pair_holes)

    extra_count = rng.choice([0, 1, 2])
    for _ in range(extra_count):
        choice = _choose_single_orbit(height, width, used_orbits, rng)
        assert choice is not None
        cell, orbit = choice
        used_orbits.add(orbit)
        holes.append(cell)

    return pattern, holes, wires


def _build_separate_deadlock_case(
    pattern: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    height, width = pattern.shape
    row_sum = height - 1
    col_sum = width - 1
    half_h = height // 2
    half_w = width // 2

    r_choices = [r for r in range(1, half_h) if r != half_h]
    c_choices = [c for c in range(1, half_w) if c != half_w]
    assert r_choices and c_choices
    r1 = rng.choice(r_choices)
    c2 = rng.choice(c_choices)

    holes = [
        (r1, 0),
        (row_sum - r1, 0),
        (r1, col_sum),
        (0, c2),
        (row_sum, c2),
        (0, col_sum - c2),
    ]
    wires = [((r1, 0), (0, -1)), ((0, c2), (-1, 0))]
    return pattern, holes, wires


def _construct_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    pattern_h = rng.choice([7, 9, 11])
    pattern_w = rng.choice([7, 9, 11])
    pattern = _make_symmetric_pattern(pattern_h, pattern_w, rng)

    if rng.random() < 0.5:
        _, local_holes, local_wires = _build_axis_pair_case(pattern, rng)
    else:
        _, local_holes, local_wires = _build_separate_deadlock_case(pattern, rng)

    top = rng.randint(3, 7)
    bottom = rng.randint(3, 7)
    left = rng.randint(3, 7)
    right = rng.randint(3, 7)
    out = np.zeros((pattern_h + top + bottom, pattern_w + left + right), dtype=int)
    pr0 = top
    pc0 = left
    out[pr0 : pr0 + pattern_h, pc0 : pc0 + pattern_w] = pattern
    inp = out.copy()

    global_holes = [(pr0 + r, pc0 + c) for r, c in local_holes]
    for r, c in global_holes:
        inp[r, c] = HOLE

    for (lr, lc), direction in local_wires:
        gr, gc = pr0 + lr, pc0 + lc
        source_color = int(out[gr, gc])
        _draw_wire(inp, (gr, gc), direction, source_color)

    k = rng.randrange(4)
    if k:
        inp = np.rot90(inp, k=k)
        out = np.rot90(out, k=k)
    return inp, out


def _partner_colors(grid: np.ndarray, hole: tuple[int, int], bbox: tuple[int, int, int, int]) -> set[int]:
    r, c = hole
    r0, c0, r1, c1 = bbox
    row_sum = r0 + r1
    col_sum = c0 + c1
    colors = set()
    for pr, pc in ((row_sum - r, c), (r, col_sum - c)):
        if not (r0 <= pr <= r1 and c0 <= pc <= c1):
            continue
        value = int(grid[pr, pc])
        if value not in {BG, WIRE, HOLE}:
            colors.add(value)
    return colors


def _wire_only(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bbox = _pattern_bbox(out)
    to_clear: set[tuple[int, int]] = set()
    for hole in _pattern_holes(out, bbox):
        traced = _trace_wire(out, hole)
        if traced is None:
            continue
        color, path, source = traced
        out[hole] = color
        to_clear.update(path)
        to_clear.add(source)
    for r, c in to_clear:
        out[r, c] = BG
    return out


def _symmetry_only(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bbox = _pattern_bbox(out)
    return _fill_by_symmetry(out, bbox)


def _semantic_ok(inp: np.ndarray, target: np.ndarray) -> bool:
    try:
        bbox = _pattern_bbox(inp)
        holes = _pattern_holes(inp, bbox)
        if not (4 <= len(holes) <= 6):
            return False

        wired = [hole for hole in holes if _trace_wire(inp, hole) is not None]
        nonwired = [hole for hole in holes if _trace_wire(inp, hole) is None]
        if len(wired) != 2 or len(nonwired) < 2:
            return False

        for hole in wired:
            if _partner_colors(inp, hole, bbox):
                return False
        for hole in nonwired:
            if not _partner_colors(inp, hole, bbox):
                return False

        symmetry_only = _symmetry_only(inp)
        if not np.any(symmetry_only[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == HOLE):
            return False

        wire_only = _wire_only(inp)
        if not np.any(wire_only[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == HOLE):
            return False

        solved = main(inp)
        if not np.array_equal(solved, target):
            return False
        if np.any(solved == WIRE) or np.any(solved == HOLE):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        inp, out = _construct_example(rng)
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        if _semantic_ok(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid line-anchored symmetry puzzle")
