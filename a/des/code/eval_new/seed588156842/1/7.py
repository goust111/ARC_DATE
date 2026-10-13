import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    return int(min(int(v) for v, cnt in zip(vals, counts) if int(cnt) == max_count))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_filled_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _reflect(cell: tuple[int, int], sum_r: int, sum_c: int) -> tuple[int, int]:
    return sum_r - cell[0], sum_c - cell[1]


def _find_marker_and_roles(
    grid: np.ndarray, bg: int
) -> tuple[int, int, set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    assert len(colors) == 2

    comps_by_color = {color: _components_of_color(grid, color) for color in colors}
    comp_by_cell: dict[tuple[int, int], tuple[int, int]] = {}
    rectangle_comps: dict[int, list[list[tuple[int, int]]]] = {}
    for color in colors:
        rectangle_comps[color] = [comp for comp in comps_by_color[color] if len(comp) >= 4 and _is_filled_rectangle(comp)]
        for idx, comp in enumerate(comps_by_color[color]):
            for cell in comp:
                comp_by_cell[cell] = (color, idx)

    candidates = []
    for top_color in colors:
        rects = rectangle_comps[top_color]
        if not rects:
            continue
        for comp in comps_by_color[top_color]:
            if len(comp) != 1:
                continue
            r, c = comp[0]
            if r + 1 >= h:
                continue
            bottom_color = int(grid[r + 1, c])
            if bottom_color == bg or bottom_color == top_color:
                continue
            bottom_key = comp_by_cell.get((r + 1, c))
            if bottom_key is None or bottom_key[0] != bottom_color:
                continue
            if len(comps_by_color[bottom_color][bottom_key[1]]) != 1:
                continue
            best_rect = max(rects, key=len)
            candidates.append(
                (
                    len(best_rect),
                    (int(r), int(c)),
                    (int(r + 1), int(c)),
                    int(top_color),
                    int(bottom_color),
                    set(best_rect),
                )
            )

    assert candidates
    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]), reverse=True)
    _, marker_top, marker_bottom, mask_color, pattern_color, mask_cells = candidates[0]
    return mask_color, pattern_color, mask_cells, marker_top, marker_bottom


def _search_centers(
    visible: set[tuple[int, int]], mask_cells: set[tuple[int, int]], shape: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int, float], int]:
    h, w = shape
    mean_sum_r = 2.0 * sum(r for r, _ in visible) / len(visible)
    mean_sum_c = 2.0 * sum(c for _, c in visible) / len(visible)
    best_center = None
    best_key = None
    ties = 0

    for sum_r in range(2 * h - 1):
        for sum_c in range(2 * w - 1):
            overlap = 0
            hidden = 0
            valid = True
            for cell in visible:
                rr, cc = _reflect(cell, sum_r, sum_c)
                if not (0 <= rr < h and 0 <= cc < w):
                    valid = False
                    break
                partner = (int(rr), int(cc))
                if partner in mask_cells:
                    hidden += 1
                elif partner in visible:
                    overlap += 1
                else:
                    valid = False
                    break
            if not valid:
                continue
            dist = abs(sum_r - mean_sum_r) + abs(sum_c - mean_sum_c)
            key = (int(overlap), int(hidden), -float(dist))
            if best_key is None or key > best_key:
                best_key = key
                best_center = (int(sum_r), int(sum_c))
                ties = 1
            elif key == best_key:
                ties += 1
                assert best_center is not None
                best_center = min(best_center, (int(sum_r), int(sum_c)))

    assert best_center is not None and best_key is not None
    return best_center, best_key, ties


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    mask_color, pattern_color, mask_cells, marker_top, marker_bottom = _find_marker_and_roles(g, bg)

    visible = {tuple(map(int, cell)) for cell in np.argwhere(g == pattern_color)}
    visible.discard(marker_bottom)
    assert visible

    center, score, ties = _search_centers(visible, mask_cells, g.shape)
    sum_r, sum_c = center
    full_pattern = set(visible)
    for cell in tuple(visible):
        full_pattern.add(_reflect(cell, sum_r, sum_c))

    out = np.full(g.shape, bg, dtype=int)
    for r, c in full_pattern:
        out[r, c] = pattern_color

    info = {
        "bg": bg,
        "mask_color": mask_color,
        "pattern_color": pattern_color,
        "mask_cells": mask_cells,
        "marker_top": marker_top,
        "marker_bottom": marker_bottom,
        "center": center,
        "score": score,
        "ties": ties,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove a rectangular mask and restore the hidden part of a 180-degree rotationally symmetric pattern.

    Input structure:
    - The background is the most common color.
    - One non-background color is a pattern color whose visible pixels belong to a shape with 180-degree rotational symmetry.
    - A solid rectangular mask in another color covers part of that shape by replacing covered pattern pixels.
    - Elsewhere there is a separate two-pixel vertical marker: mask color on top, pattern color below.

    Rule:
    - Identify the large rectangular component of the mask color as the occluding mask and the isolated vertical color pair as the hint marker.
    - Ignore the marker's lower pattern-colored pixel, then search for the rotation center that makes the visible pattern pixels overlap maximally under a 180-degree rotation while allowing unmatched partners only inside the mask rectangle.
    - Reflect every visible pattern pixel through that center; any reflected position inside the mask is recovered as pattern color.
    - Remove both the mask rectangle and the hint marker, leaving background everywhere except the reconstructed full pattern.

    Key insight:
    - The small marker explains that mask-colored cells can hide pattern-colored cells, and the missing cells are determined by the pattern's hidden 180-degree rotational symmetry.
    """
    out, _info = _solve(grid)
    return out


# --------------- generator ---------------

ODD_SIZES = (5, 7, 9, 11)
EVEN_SIZES = (4, 6, 8, 10)
MIN_GRID = 14
MAX_GRID = 30


def _build_orbits(
    r0: int, r1: int, c0: int, c1: int
) -> tuple[int, int, list[list[tuple[int, int]]], list[list[int]]]:
    sum_r = r0 + r1
    sum_c = c0 + c1
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[list[tuple[int, int]]] = []
    orbit_by_cell: dict[tuple[int, int], int] = {}

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            partner = _reflect((r, c), sum_r, sum_c)
            key = tuple(sorted(((int(r), int(c)), (int(partner[0]), int(partner[1])))))
            if key in seen:
                continue
            seen.add(key)
            cells = [key[0]]
            if key[1] != key[0]:
                cells.append(key[1])
            idx = len(orbits)
            orbits.append(cells)
            for cell in cells:
                orbit_by_cell[cell] = idx

    neighbors = [set() for _ in orbits]
    for idx, cells in enumerate(orbits):
        for r, c in cells:
            for dr, dc in DIRS4:
                nbr = (r + dr, c + dc)
                other = orbit_by_cell.get(nbr)
                if other is not None and other != idx:
                    neighbors[idx].add(other)

    return sum_r, sum_c, orbits, [sorted(items) for items in neighbors]


def _sample_pattern(orbits: list[list[tuple[int, int]]], neighbors: list[list[int]], sum_r: int, sum_c: int) -> set[tuple[int, int]] | None:
    orbit_count = len(orbits)
    if orbit_count < 8:
        return None

    dists = [min(abs(2 * r - sum_r) + abs(2 * c - sum_c) for r, c in orbit) for orbit in orbits]
    min_dist = min(dists)
    start = random.choice([idx for idx, dist in enumerate(dists) if dist == min_dist])

    low = max(7, orbit_count // 4)
    high = min(orbit_count - 1, max(low, orbit_count // 2 + 2))
    target = random.randint(low, high)

    selected = {start}
    while len(selected) < target:
        frontier = []
        for idx in selected:
            frontier.extend(nbr for nbr in neighbors[idx] if nbr not in selected)
        if frontier:
            chosen = random.choice(frontier)
        else:
            chosen = random.choice([idx for idx in range(orbit_count) if idx not in selected])
        selected.add(chosen)

    pattern = {cell for idx in selected for cell in orbits[idx]}
    rows = {r for r, _ in pattern}
    cols = {c for _, c in pattern}
    if len(pattern) < 14 or len(pattern) > 48:
        return None
    if len(rows) < 4 or len(cols) < 4:
        return None
    if len(pattern) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1):
        return None
    return pattern


def _sample_mask(
    pattern: set[tuple[int, int]], r0: int, r1: int, c0: int, c1: int, sum_r: int, sum_c: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    max_h = min(6, r1 - r0 + 1)
    max_w = min(6, c1 - c0 + 1)
    for _ in range(200):
        mask_h = random.randint(2, max_h)
        mask_w = random.randint(2, max_w)
        mr0 = random.randint(r0, r1 - mask_h + 1)
        mc0 = random.randint(c0, c1 - mask_w + 1)
        mask_cells = {(r, c) for r in range(mr0, mr0 + mask_h) for c in range(mc0, mc0 + mask_w)}
        hidden = pattern & mask_cells
        if len(hidden) < 2 or len(hidden) > len(pattern) // 2:
            continue
        if any(_reflect(cell, sum_r, sum_c) in mask_cells for cell in hidden):
            continue
        visible = pattern - mask_cells
        if len(visible) < 8:
            continue
        overlap = sum(1 for cell in visible if _reflect(cell, sum_r, sum_c) in visible)
        if overlap < 4:
            continue
        if len(hidden) == len(mask_cells) and random.random() < 0.5:
            continue
        return mask_cells, hidden
    return None


def _marker_candidates(
    shape: tuple[int, int],
    pattern: set[tuple[int, int]],
    mask_cells: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    occupied = pattern | mask_cells
    candidates = []

    for r in range(h - 1):
        for c in range(w):
            if (r, c) in occupied or (r + 1, c) in occupied:
                continue
            if not (r + 1 < r0 - 1 or r > r1 + 1 or c < c0 - 1 or c > c1 + 1):
                continue
            clear = True
            for rr in range(max(0, r - 1), min(h, r + 3)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if (rr, cc) in occupied:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                candidates.append((int(r), int(c)))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    bg, pattern_color, mask_color = random.sample(PALETTE, 3)

    pattern_h = random.choice(ODD_SIZES if random.random() < 0.5 else EVEN_SIZES)
    pattern_w = random.choice(ODD_SIZES if random.random() < 0.5 else EVEN_SIZES)

    h_min = max(MIN_GRID, pattern_h + 6)
    w_min = max(MIN_GRID, pattern_w + 6)
    if h_min > MAX_GRID or w_min > MAX_GRID:
        return None
    h = random.randint(h_min, MAX_GRID)
    w = random.randint(w_min, MAX_GRID)

    r0 = random.randint(2, h - pattern_h - 3)
    c0 = random.randint(2, w - pattern_w - 3)
    r1 = r0 + pattern_h - 1
    c1 = c0 + pattern_w - 1
    sum_r, sum_c, orbits, neighbors = _build_orbits(r0, r1, c0, c1)

    pattern = _sample_pattern(orbits, neighbors, sum_r, sum_c)
    if pattern is None:
        return None

    mask_result = _sample_mask(pattern, r0, r1, c0, c1, sum_r, sum_c)
    if mask_result is None:
        return None
    mask_cells, _hidden = mask_result

    pattern_bbox = _bbox(pattern)
    marker_options = _marker_candidates((h, w), pattern, mask_cells, pattern_bbox)
    if not marker_options:
        return None
    marker_top = random.choice(marker_options)
    marker_bottom = (marker_top[0] + 1, marker_top[1])

    out = np.full((h, w), bg, dtype=int)
    for r, c in pattern:
        out[r, c] = pattern_color

    inp = out.copy()
    for r, c in mask_cells:
        inp[r, c] = mask_color
    inp[marker_top] = mask_color
    inp[marker_bottom] = pattern_color

    counts = {int(v): int(np.count_nonzero(inp == v)) for v in np.unique(inp)}
    if counts[bg] <= max(cnt for color, cnt in counts.items() if color != bg):
        return None

    try:
        actual, info = _solve(inp)
    except Exception:
        return None
    if info["ties"] != 1:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
