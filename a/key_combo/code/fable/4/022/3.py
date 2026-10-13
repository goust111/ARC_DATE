from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _flood_region(blocked: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = blocked.shape
    sr, sc = map(int, start)
    if blocked[sr, sc]:
        raise AssertionError("start point cannot be inside the wall")

    region = np.zeros((h, w), dtype=bool)
    queue = deque([(sr, sc)])
    region[sr, sc] = True

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not region[nr, nc]:
                region[nr, nc] = True
                queue.append((nr, nc))
    return region


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0

    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if seen[rr, cc]:
            continue
        count += 1
        queue = deque([(rr, cc)])
        seen[rr, cc] = True
        while queue:
            cr, cc = queue.popleft()
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
    return count


def _infer_axis(grid: np.ndarray, *, vertical: bool) -> int:
    h, w = grid.shape
    size = w if vertical else h
    center = size - 1
    coords = np.argwhere(grid != BLACK)
    best: tuple[tuple[int, int, int, int], int] | None = None

    for axis_sum in range(2 * size - 1):
        contradictions = 0
        support = 0
        missing = 0
        for r, c in coords:
            rr, cc = int(r), int(c)
            reflected = axis_sum - (cc if vertical else rr)
            if not 0 <= reflected < size:
                contradictions += 1
                continue
            value = int(grid[rr, cc])
            other = int(grid[rr, reflected] if vertical else grid[reflected, cc])
            if other == BLACK:
                missing += 1
            elif other == value:
                support += 1
            else:
                contradictions += 1

        score = (contradictions, -support, missing, abs(axis_sum - center))
        if best is None or score < best[0]:
            best = (score, axis_sum)

    assert best is not None
    return int(best[1])


def _symmetry_orbit(
    r: int,
    c: int,
    horizontal_sum: int,
    vertical_sum: int,
    h: int,
    w: int,
) -> tuple[tuple[int, int], ...]:
    orbit = {
        (int(r), int(c)),
        (int(horizontal_sum - r), int(c)),
        (int(r), int(vertical_sum - c)),
        (int(horizontal_sum - r), int(vertical_sum - c)),
    }
    return tuple(sorted((rr, cc) for rr, cc in orbit if 0 <= rr < h and 0 <= cc < w))


def _repair_wall_orbits(
    grid: np.ndarray,
    wall_color: int,
    horizontal_sum: int,
    vertical_sum: int,
) -> tuple[np.ndarray, int, int, int]:
    h, w = grid.shape
    repaired = grid.copy()
    seen: set[tuple[tuple[int, int], ...]] = set()
    added = 0
    mixed_orbits = 0
    touched_orbits = 0

    for r in range(h):
        for c in range(w):
            orbit = _symmetry_orbit(r, c, horizontal_sum, vertical_sum, h, w)
            if orbit in seen:
                continue
            seen.add(orbit)

            present = {int(grid[rr, cc]) for rr, cc in orbit if int(grid[rr, cc]) != BLACK}
            if wall_color not in present:
                continue

            touched_orbits += 1
            if any(color not in (BLACK, wall_color) for color in present):
                mixed_orbits += 1
                continue

            for rr, cc in orbit:
                if int(repaired[rr, cc]) == BLACK:
                    repaired[rr, cc] = wall_color
                    added += 1

    return repaired, int(added), int(mixed_orbits), int(touched_orbits)


def _adjacent_wall_mask(region: np.ndarray, wall_mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = region.shape
    adjacent = np.zeros((h, w), dtype=bool)
    contacts = 0

    for r, c in np.argwhere(region):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and wall_mask[nr, nc]:
                adjacent[nr, nc] = True
                contacts += 1

    return adjacent, int(contacts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the broken symmetric wall, then flood-fill the indicator's enclosed chamber.

    Input structure:
    - The full canvas is a colored pattern with both horizontal and vertical mirror symmetry,
      but some cells have been erased to black.
    - One non-black color forms a one-cell-thick wall loop; some wall cells are missing because
      their positions were erased to black.
    - Exactly one non-black color appears only once in the input: that singleton is the
      indicator pixel, located inside the wall.

    Rule:
    - Infer the two mirror axes from the surviving non-black pixels.
    - For each candidate wall color, use its symmetry orbits to restore black cells that must
      also be wall-colored, and keep the unique color whose repaired wall truly encloses the
      indicator.
    - Flood-fill from the indicator with 4-connectivity, treating the repaired wall as blocked,
      and recolor every reached non-wall cell to the indicator color.

    Key insight:
    - The visible wall is incomplete, so the enclosure only becomes detectable after restoring
      its missing symmetric counterparts; the repaired wall then defines the region to fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    non_black = g[g != BLACK]
    colors, counts = np.unique(non_black, return_counts=True)
    unique_colors = [int(color) for color, count in zip(colors, counts) if int(count) == 1]
    assert len(unique_colors) == 1, "expected a unique indicator color"
    indicator_color = int(unique_colors[0])
    indicator_pos = tuple(map(int, np.argwhere(g == indicator_color)[0]))

    horizontal_sum = _infer_axis(g, vertical=False)
    vertical_sum = _infer_axis(g, vertical=True)

    candidates = []
    for wall_color in sorted(set(map(int, non_black)) - {indicator_color}):
        repaired, added, mixed_orbits, touched_orbits = _repair_wall_orbits(
            g, wall_color, horizontal_sum, vertical_sum
        )
        if mixed_orbits:
            continue

        wall_mask = repaired == wall_color
        if wall_mask[indicator_pos]:
            continue

        region = _flood_region(wall_mask, indicator_pos)
        if _touches_border(region):
            continue

        adjacent_wall, contacts = _adjacent_wall_mask(region, wall_mask)
        if contacts == 0:
            continue

        score = (
            0 if added > 0 else 1,
            touched_orbits,
            int(wall_mask.sum()),
            -contacts,
            wall_color,
        )
        candidates.append((score, wall_color, repaired, region))

    assert candidates, "failed to identify the repaired wall"
    candidates.sort(key=lambda item: item[0])
    _score, _wall_color, repaired, inside_region = candidates[0]

    out = repaired.copy()
    out[inside_region] = indicator_color
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)

    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                boundary[rr, cc] = True
                break
    return boundary


def _sample_blob(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_r = h // 2
    center_c = w // 2
    max_radius = min(center_r - 2, 7)
    max_half_width = min(center_c - 2, 8)

    for _ in range(200):
        radius = rng.randint(3, max_radius)
        center_half_width = rng.randint(3, max_half_width)
        profile = [center_half_width]
        current = center_half_width

        for _dist in range(1, radius + 1):
            current = rng.randint(max(1, current - 2), current)
            profile.append(current)

        filled = np.zeros((h, w), dtype=bool)
        for dist, half_width in enumerate(profile):
            row_slice = slice(center_c - half_width, center_c + half_width + 1)
            filled[center_r - dist, row_slice] = True
            filled[center_r + dist, row_slice] = True

        wall = _boundary(filled)
        inside = filled & ~wall
        if int(wall.sum()) >= 16 and int(inside.sum()) >= 8:
            return filled, wall, inside

    raise RuntimeError("failed to sample a valid enclosed region")


def _all_orbits(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    horizontal_sum = h - 1
    vertical_sum = w - 1
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []

    for r in range(h):
        for c in range(w):
            orbit = _symmetry_orbit(r, c, horizontal_sum, vertical_sum, h, w)
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def _erase_subset(
    grid: np.ndarray,
    orbit: tuple[tuple[int, int], ...],
    rng: random.Random,
    *,
    keep_at_least_one: bool,
) -> None:
    max_erase = len(orbit) - 1 if keep_at_least_one else len(orbit)
    if max_erase <= 0:
        return
    erase_count = rng.randint(1, max_erase)
    for r, c in rng.sample(list(orbit), erase_count):
        grid[r, c] = BLACK


def _unique_color_count(grid: np.ndarray) -> int:
    non_black = grid[grid != BLACK]
    if non_black.size == 0:
        return 0
    _colors, counts = np.unique(non_black, return_counts=True)
    return int(np.count_nonzero(counts == 1))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randrange(13, 30, 2)
    w = rng.randrange(13, 30, 2)
    _filled, wall, inside = _sample_blob(h, w, rng)

    wall_color, indicator_color = rng.sample(PALETTE, 2)
    filler_palette = rng.sample(
        [color for color in PALETTE if color not in (wall_color, indicator_color)],
        rng.randint(2, 4),
    )

    full = np.zeros((h, w), dtype=int)
    orbits = _all_orbits(h, w)
    wall_cells = {tuple(cell) for cell in np.argwhere(wall)}
    inside_cells = {tuple(cell) for cell in np.argwhere(inside)}

    inside_orbits: list[tuple[tuple[int, int], ...]] = []
    for orbit in orbits:
        if all(cell in wall_cells for cell in orbit):
            for r, c in orbit:
                full[r, c] = wall_color
        elif all(cell in inside_cells for cell in orbit):
            inside_orbits.append(orbit)
        else:
            fill_color = rng.choice(filler_palette)
            for r, c in orbit:
                full[r, c] = fill_color

    indicator_choices = [orbit for orbit in inside_orbits if len(orbit) > 1]
    indicator_orbit = rng.choice(indicator_choices or inside_orbits)
    for r, c in indicator_orbit:
        full[r, c] = indicator_color

    for orbit in inside_orbits:
        if orbit == indicator_orbit:
            continue
        fill_color = rng.choice(filler_palette)
        for r, c in orbit:
            full[r, c] = fill_color

    inp = full.copy()

    wall_orbits = [orbit for orbit in orbits if any(cell in wall_cells for cell in orbit)]
    erasable_wall_orbits = [orbit for orbit in wall_orbits if len(orbit) >= 2]
    if not erasable_wall_orbits:
        return None

    gap_count = rng.randint(1, min(3, len(erasable_wall_orbits)))
    for orbit in rng.sample(erasable_wall_orbits, gap_count):
        _erase_subset(inp, orbit, rng, keep_at_least_one=True)

    survivor = rng.choice(list(indicator_orbit))
    for r, c in indicator_orbit:
        if (r, c) != survivor:
            inp[r, c] = BLACK

    interior_orbits = []
    exterior_orbits = []
    for orbit in orbits:
        if orbit == indicator_orbit or any(cell in wall_cells for cell in orbit):
            continue
        if all(cell in inside_cells for cell in orbit):
            interior_orbits.append(orbit)
        elif not any(cell in inside_cells for cell in orbit):
            exterior_orbits.append(orbit)

    if interior_orbits:
        for orbit in rng.sample(interior_orbits, rng.randint(1, min(4, len(interior_orbits)))):
            _erase_subset(inp, orbit, rng, keep_at_least_one=False)

    if exterior_orbits:
        for orbit in rng.sample(exterior_orbits, rng.randint(0, min(4, len(exterior_orbits)))):
            _erase_subset(inp, orbit, rng, keep_at_least_one=False)

    if _unique_color_count(inp) != 1 or int(np.count_nonzero(inp == indicator_color)) != 1:
        return None

    leak_region = _flood_region(inp == wall_color, survivor)
    if not _touches_border(leak_region):
        return None

    expected = inp.copy()
    expected[wall] = wall_color
    expected[inside] = indicator_color

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
