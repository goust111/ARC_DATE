from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

GRID_SIZE = 13
CENTER = GRID_SIZE // 2

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rot90_ccw(coord: Coord, center: int) -> Coord:
    r, c = coord
    dr = r - center
    dc = c - center
    return center - dc, center + dr


def _orbit(coord: Coord, center: int) -> list[Coord]:
    cells = [coord]
    for _ in range(3):
        cells.append(_rot90_ccw(cells[-1], center))
    return cells


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _ray_direction(r: int, c: int, center: int) -> Coord:
    dr0 = _sign(r - center)
    dc0 = _sign(c - center)
    if dr0 != 0 and dc0 != 0:
        return dr0, dc0
    if dr0 == 0 and dc0 != 0:
        # Center-row tie-break: follow the adjacent CCW orbit quadrant.
        return -dc0, dc0
    if dc0 == 0 and dr0 != 0:
        # Center-column tie-break: follow the adjacent CCW orbit quadrant.
        return dr0, -dr0
    raise AssertionError("the center cell cannot be a ray seed")


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _mirror_map(grid: np.ndarray) -> tuple[dict[Coord, str], list[tuple[str, list[Coord]]]]:
    red_comps = _components(grid == RED)
    cell_to_orientation: dict[Coord, str] = {}
    segments: list[tuple[str, list[Coord]]] = []
    for comp in red_comps:
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) == 1:
            row = next(iter(rows))
            ordered = sorted(comp, key=lambda rc: rc[1])
            assert [c for _, c in ordered] == list(range(ordered[0][1], ordered[-1][1] + 1))
            orientation = "horizontal"
        elif len(cols) == 1:
            col = next(iter(cols))
            ordered = sorted(comp, key=lambda rc: rc[0])
            assert [r for r, _ in ordered] == list(range(ordered[0][0], ordered[-1][0] + 1))
            orientation = "vertical"
        else:
            raise AssertionError("red mirrors must be straight horizontal or vertical segments")
        for cell in ordered:
            cell_to_orientation[cell] = orientation
        segments.append((orientation, ordered))
    return cell_to_orientation, segments


def _repair_pattern(grid: np.ndarray) -> tuple[np.ndarray, list[Coord]]:
    repaired = np.asarray(grid, dtype=int).copy()
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(repaired == GREY)]
    for r, c in seeds:
        orb = _orbit((r, c), CENTER)
        visible = {int(repaired[rr, cc]) for rr, cc in orb if int(repaired[rr, cc]) != GREY}
        assert len(visible) == 1, "each masked orbit must expose a unique visible color"
        fill = visible.pop()
        assert fill in (BLUE, GREEN), "masked symmetry orbits must belong to the blue/green pattern"
        repaired[r, c] = fill
    return repaired, sorted(seeds)


def _trace_ray(
    canvas: np.ndarray,
    mirrors: dict[Coord, str],
    seed: Coord,
    direction: Coord,
    *,
    paint: bool,
) -> dict:
    out = canvas
    r, c = map(int, seed)
    dr, dc = map(int, direction)
    path: list[Coord] = []
    reflections = 0
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray path must not loop"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(out.shape, nr, nc):
            return {
                "path": path,
                "reflections": reflections,
                "reason": "boundary",
                "seed": seed,
                "direction": direction,
            }

        cell = int(out[nr, nc])
        if cell == RED:
            orientation = mirrors.get((nr, nc))
            assert orientation is not None
            if orientation == "horizontal":
                dr = -dr
            else:
                dc = -dc
            reflections += 1
            continue

        if cell != BLACK:
            return {
                "path": path,
                "reflections": reflections,
                "reason": "blocked",
                "seed": seed,
                "direction": direction,
            }

        if paint:
            out[nr, nc] = ORANGE
        path.append((nr, nc))
        r, c = nr, nc


def _solve(grid: np.ndarray, *, paint_rays: bool) -> tuple[np.ndarray, np.ndarray, list[dict], list[Coord]]:
    repaired, seeds = _repair_pattern(grid)
    mirrors, _segments = _mirror_map(repaired)
    out = repaired.copy()
    stats: list[dict] = []
    for seed in seeds:
        direction = _ray_direction(seed[0], seed[1], CENTER)
        stats.append(_trace_ray(out, mirrors, seed, direction, paint=paint_rays))
    return out, repaired, stats, seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the masked 4-fold rotational pattern, then emit quadrant-directed diagonal rays from the repaired pixels.

    Input structure:
    - The grid is a 13x13 black canvas whose blue and green pixels form a 90-degree rotationally symmetric pattern around the center.
    - Several pattern pixels are replaced by gray mask cells, and separate red straight segments act as mirrors.
    - The red segments never belong to the symmetric pattern and stay fixed in place.

    Rule:
    - For each gray cell, inspect its 4-cell rotational orbit around the center and restore the gray cell with the orbit's unique visible non-gray color.
    - Every originally gray cell then becomes a ray seed: its location relative to the center determines the diagonal ray direction, with center-line seeds using the adjacent orbit quadrant as a tie-break.
    - Starting from the next cell after the seed, extend the ray diagonally through black cells, reflect on red horizontal or vertical segments without painting the red cell, and stop when the next cell is out of bounds or is any non-black non-red cell.

    Key insight:
    - The gray damage is not just missing content: once each masked orbit member is restored, that repaired position also determines a unique outward diagonal ray whose route is altered only by the red mirror segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 13x13 grid"
    out, _repaired, _stats, _seeds = _solve(g, paint_rays=True)
    return out


FUNDAMENTAL_OFFSETS = [
    (dr, dc)
    for dr in range(-5, 0)
    for dc in range(0, 6)
    if not (dr == 0 and dc == 0)
    and (dc > 0 or dr < 0)
    and abs(dr) + abs(dc) <= 6
    and max(abs(dr), abs(dc)) >= 2
]
GEN_ATTEMPTS = 500


def _offset_to_cells(offset: Coord) -> list[Coord]:
    dr, dc = offset
    return _orbit((CENTER + dr, CENTER + dc), CENTER)


def _grow_offsets(count: int) -> list[Coord] | None:
    candidates = list(FUNDAMENTAL_OFFSETS)
    start = random.choice(candidates)
    chosen = [start]
    while len(chosen) < count:
        frontier = [
            cand
            for cand in candidates
            if cand not in chosen
            and any(max(abs(cand[0] - rr), abs(cand[1] - cc)) == 1 for rr, cc in chosen)
        ]
        if not frontier:
            return None
        chosen.append(random.choice(frontier))
    return chosen


def _build_pattern() -> tuple[np.ndarray, dict[Coord, int], list[Coord]] | None:
    orbit_count = random.randint(4, 6)
    offsets = _grow_offsets(orbit_count)
    if offsets is None:
        return None

    grid = np.full((GRID_SIZE, GRID_SIZE), BLACK, dtype=int)
    orbit_by_cell: dict[Coord, int] = {}
    colors = [random.choice([BLUE, GREEN]) for _ in offsets]
    if len(set(colors)) == 1:
        colors[random.randrange(len(colors))] = BLUE if colors[0] == GREEN else GREEN

    for orbit_id, (offset, color) in enumerate(zip(offsets, colors, strict=False)):
        for cell in _offset_to_cells(offset):
            if cell in orbit_by_cell:
                return None
            orbit_by_cell[cell] = orbit_id
            grid[cell] = color
    return grid, orbit_by_cell, offsets


def _mask_candidates(pattern: np.ndarray, orbit_by_cell: dict[Coord, int]) -> list[dict]:
    candidates: list[dict] = []
    for (r, c), orbit_id in orbit_by_cell.items():
        dr, dc = _ray_direction(r, c, CENTER)
        nr, nc = r + dr, c + dc
        if not _in_bounds(pattern.shape, nr, nc):
            continue
        if int(pattern[nr, nc]) != BLACK:
            continue
        candidates.append(
            {
                "coord": (r, c),
                "orbit_id": orbit_id,
                "axis": (r == CENTER or c == CENTER),
                "quadrant": (_sign(r - CENTER), _sign(c - CENTER)),
            }
        )
    return candidates


def _choose_masks(candidates: list[dict]) -> list[Coord] | None:
    if len(candidates) < 5:
        return None

    by_quadrant: dict[tuple[int, int], list[dict]] = {}
    axis_candidates = []
    for item in candidates:
        if item["axis"]:
            axis_candidates.append(item)
        else:
            by_quadrant.setdefault(item["quadrant"], []).append(item)

    wanted = random.randint(3, 5)
    masks: list[dict] = []
    used_cells: set[Coord] = set()
    used_orbits: dict[int, int] = {}

    non_axis_quadrants = list(by_quadrant.keys())
    random.shuffle(non_axis_quadrants)
    for quadrant in non_axis_quadrants[: min(4, wanted)]:
        options = [item for item in by_quadrant[quadrant] if item["coord"] not in used_cells]
        if not options:
            continue
        choice = random.choice(options)
        masks.append(choice)
        used_cells.add(choice["coord"])
        used_orbits[choice["orbit_id"]] = used_orbits.get(choice["orbit_id"], 0) + 1
        if len(masks) == wanted:
            break

    if len(masks) < wanted and axis_candidates and random.random() < 0.4:
        random.shuffle(axis_candidates)
        for item in axis_candidates:
            if item["coord"] in used_cells:
                continue
            if used_orbits.get(item["orbit_id"], 0) >= 2:
                continue
            masks.append(item)
            used_cells.add(item["coord"])
            used_orbits[item["orbit_id"]] = used_orbits.get(item["orbit_id"], 0) + 1
            break

    remaining = candidates[:]
    random.shuffle(remaining)
    for item in remaining:
        if len(masks) == wanted:
            break
        if item["coord"] in used_cells:
            continue
        if used_orbits.get(item["orbit_id"], 0) >= 2:
            continue
        masks.append(item)
        used_cells.add(item["coord"])
        used_orbits[item["orbit_id"]] = used_orbits.get(item["orbit_id"], 0) + 1

    if len(masks) != wanted:
        return None
    covered_quadrants = {item["quadrant"] for item in masks if not item["axis"]}
    if len(covered_quadrants) < 3:
        return None
    return sorted(item["coord"] for item in masks)


def _segment_cells(start: Coord, orientation: str, length: int) -> list[Coord]:
    r0, c0 = start
    if orientation == "horizontal":
        return [(r0, c0 + i) for i in range(length)]
    return [(r0 + i, c0) for i in range(length)]


def _segment_valid(
    cells: list[Coord],
    occupied_pattern: set[Coord],
    red_cells: set[Coord],
    shape: tuple[int, int],
) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in occupied_pattern or (r, c) in red_cells:
            return False
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in red_cells and (nr, nc) not in cell_set:
                return False
    return True


def _segments_through(
    cell: Coord,
    orientation: str,
    occupied_pattern: set[Coord],
    red_cells: set[Coord],
    shape: tuple[int, int],
) -> list[list[Coord]]:
    r, c = cell
    options: list[list[Coord]] = []
    for length in (3, 4, 5):
        for offset in range(length):
            if orientation == "horizontal":
                start = (r, c - offset)
            else:
                start = (r - offset, c)
            cells = _segment_cells(start, orientation, length)
            if _segment_valid(cells, occupied_pattern, red_cells, shape):
                options.append(cells)
    return options


def _independent_ray_stats(repaired_with_red: np.ndarray, seeds: list[Coord]) -> list[dict]:
    mirrors, _segments = _mirror_map(repaired_with_red)
    stats = []
    for seed in seeds:
        direction = _ray_direction(seed[0], seed[1], CENTER)
        stats.append(_trace_ray(repaired_with_red.copy(), mirrors, seed, direction, paint=False))
    return stats


def _plan_reflection_segment(
    repaired_with_red: np.ndarray,
    occupied_pattern: set[Coord],
    seed: Coord,
) -> list[Coord] | None:
    mirrors, _segments = _mirror_map(repaired_with_red)
    direction = _ray_direction(seed[0], seed[1], CENTER)
    stat = _trace_ray(repaired_with_red.copy(), mirrors, seed, direction, paint=False)
    current_red = {tuple(map(int, rc)) for rc in np.argwhere(repaired_with_red == RED)}

    candidate_steps = stat["path"][:-1] if len(stat["path"]) >= 2 else []
    if not candidate_steps:
        return None
    random.shuffle(candidate_steps)

    for prev in candidate_steps:
        dr, dc = direction
        temp_grid = repaired_with_red.copy()
        # Recover the direction at this step by replaying up to the cell just before it.
        rr, cc = seed
        cur_dr, cur_dc = direction
        seen: set[tuple[int, int, int, int]] = set()
        while (rr, cc) != prev:
            state = (rr, cc, cur_dr, cur_dc)
            if state in seen:
                return None
            seen.add(state)
            nr, nc = rr + cur_dr, cc + cur_dc
            if int(temp_grid[nr, nc]) == RED:
                orientation = mirrors[(nr, nc)]
                if orientation == "horizontal":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc
                continue
            rr, cc = nr, nc
        hit = (rr + cur_dr, cc + cur_dc)
        if not _in_bounds(temp_grid.shape, hit[0], hit[1]) or int(temp_grid[hit]) != BLACK:
            continue

        orientations = ["horizontal", "vertical"]
        random.shuffle(orientations)
        for orientation in orientations:
            if orientation == "horizontal":
                new_dir = (-cur_dr, cur_dc)
            else:
                new_dir = (cur_dr, -cur_dc)
            step_after = (rr + new_dir[0], cc + new_dir[1])
            if not _in_bounds(temp_grid.shape, step_after[0], step_after[1]):
                continue
            if int(temp_grid[step_after]) != BLACK:
                continue
            segment_options = _segments_through(hit, orientation, occupied_pattern, current_red, temp_grid.shape)
            random.shuffle(segment_options)
            for segment in segment_options:
                trial = temp_grid.copy()
                for sr, sc in segment:
                    trial[sr, sc] = RED
                trial_stat = _trace_ray(
                    trial.copy(),
                    _mirror_map(trial)[0],
                    seed,
                    direction,
                    paint=False,
                )
                if trial_stat["reflections"] >= stat["reflections"] + 1 and len(trial_stat["path"]) >= len(stat["path"]) + 1:
                    return segment
    return None


def _all_valid_segments(
    occupied_pattern: set[Coord],
    red_cells: set[Coord],
    shape: tuple[int, int],
) -> list[list[Coord]]:
    options: list[list[Coord]] = []
    for orientation in ("horizontal", "vertical"):
        for length in (3, 4, 5):
            if orientation == "horizontal":
                for r in range(shape[0]):
                    for c in range(shape[1] - length + 1):
                        cells = _segment_cells((r, c), orientation, length)
                        if _segment_valid(cells, occupied_pattern, red_cells, shape):
                            options.append(cells)
            else:
                for r in range(shape[0] - length + 1):
                    for c in range(shape[1]):
                        cells = _segment_cells((r, c), orientation, length)
                        if _segment_valid(cells, occupied_pattern, red_cells, shape):
                            options.append(cells)
    return options


def _add_segment(grid: np.ndarray, segment: list[Coord]) -> None:
    for r, c in segment:
        grid[r, c] = RED


def _validate_case(inp: np.ndarray) -> bool:
    try:
        out, repaired, ray_stats, seeds = _solve(inp, paint_rays=True)
    except AssertionError:
        return False

    if not (3 <= len(seeds) <= 5):
        return False

    mirrors, segments = _mirror_map(repaired)
    if len(segments) not in (2, 3):
        return False
    if not all(3 <= len(seg) <= 5 for _orient, seg in segments):
        return False

    independent = _independent_ray_stats(repaired, seeds)
    orange_sets = [set(stat["path"]) for stat in independent]
    for i in range(len(orange_sets)):
        for j in range(i + 1, len(orange_sets)):
            if orange_sets[i] & orange_sets[j]:
                return False

    total_orange = sum(len(stat["path"]) for stat in independent)
    total_reflections = sum(stat["reflections"] for stat in independent)
    if total_orange < 7 or total_reflections < 1:
        return False
    if not any(stat["reflections"] == 0 for stat in independent):
        return False
    if not any(stat["reflections"] > 0 for stat in independent):
        return False

    repaired_only = repaired.copy()
    no_mask_out = main(repaired_only)
    if np.array_equal(no_mask_out, out):
        return False

    no_red = inp.copy()
    no_red[no_red == RED] = BLACK
    if np.array_equal(main(no_red), out):
        return False

    if int(np.sum(out == ORANGE)) != total_orange:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    built = _build_pattern()
    if built is None:
        return None
    pattern, orbit_by_cell, _offsets = built

    candidates = _mask_candidates(pattern, orbit_by_cell)
    masks = _choose_masks(candidates)
    if masks is None:
        return None

    occupied_pattern = set(orbit_by_cell.keys())
    repaired_with_red = pattern.copy()

    target_reflections = 1 if random.random() < 0.45 else 2
    targeted_seeds = masks[:]
    random.shuffle(targeted_seeds)
    actual_reflections = 0
    for seed in targeted_seeds:
        if actual_reflections >= target_reflections:
            break
        segment = _plan_reflection_segment(repaired_with_red, occupied_pattern, seed)
        if segment is None:
            continue
        _add_segment(repaired_with_red, segment)
        actual_reflections += 1
    if actual_reflections < 1:
        return None

    desired_segments = random.randint(2, 3)
    while len(_mirror_map(repaired_with_red)[1]) < desired_segments:
        red_cells = {tuple(map(int, rc)) for rc in np.argwhere(repaired_with_red == RED)}
        options = _all_valid_segments(occupied_pattern, red_cells, repaired_with_red.shape)
        if not options:
            return None
        random.shuffle(options)
        _add_segment(repaired_with_red, options[0])

    inp = repaired_with_red.copy()
    for r, c in masks:
        inp[r, c] = GREY
    return inp if _validate_case(inp) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-ray ARC input")
