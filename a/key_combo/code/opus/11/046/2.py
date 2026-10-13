import random
from itertools import combinations

import numpy as np


BG = 0
MASK = 5
WALL = 9
PALETTE = (1, 2, 3, 4, 7)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _d4_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = n - 1
    return tuple(
        sorted(
            {
                (r, c),
                (c, r),
                (r, last - c),
                (last - r, c),
                (last - r, last - c),
                (last - c, last - r),
                (c, last - r),
                (last - c, r),
            }
        )
    )


_ORBIT_CACHE: dict[int, list[tuple[tuple[int, int], ...]]] = {}


def _all_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    if n not in _ORBIT_CACHE:
        seen = set()
        orbits = []
        for r in range(n):
            for c in range(n):
                orbit = _d4_orbit(n, r, c)
                if orbit not in seen:
                    seen.add(orbit)
                    orbits.append(orbit)
        _ORBIT_CACHE[n] = orbits
    return _ORBIT_CACHE[n]


def _axial_subset(orbit: tuple[tuple[int, int], ...], n: int) -> tuple[tuple[int, int], ...] | None:
    last = n - 1
    orbit_set = set(orbit)
    candidates = []
    for r, c in orbit:
        quad = {
            (r, c),
            (r, last - c),
            (last - r, c),
            (last - r, last - c),
        }
        if len(quad) == 4 and quad.issubset(orbit_set):
            candidates.append(tuple(sorted(quad)))
    return min(candidates) if candidates else None


def _global_orbit(grid: np.ndarray, r: int, c: int) -> tuple[tuple[int, int], ...]:
    n, m = grid.shape
    assert n == m
    return _d4_orbit(n, r, c)


def main(grid: np.ndarray) -> np.ndarray:
    """Restore grey masks by completing each wall-bounded square region's local eight-way symmetry.

    Input structure:
    - The grid is square, with black background and one-pixel-wide maroon wall pixels.
    - Wall pixels split the non-wall space into independent 4-connected regions; the relevant rooms are complete odd-sized squares.
    - Inside each square room, a multi-color pattern is D4-symmetric about the room center, but some cells are replaced by grey masks.

    Rule:
    - Treat maroon pixels as barriers and find connected non-wall regions with 4-connectivity.
    - For each region that occupies an entire square bounding box, use that local square's center as the symmetry center.
    - Replace each grey mask with the unique visible color from its local D4 orbit; leave visible cells, walls, and non-square regions unchanged.

    Key insight:
    - The symmetry is not global: walls first define the local square domain, and only then do the eight rotations/reflections determine each masked color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for coords in _components(g != WALL):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        region = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        full_square = h == w and coords.shape[0] == h * w
        if not full_square:
            assert not np.any(region == MASK)
            continue

        changed = False
        for orbit in _all_orbits(h):
            values = [int(region[r, c]) for r, c in orbit]
            if MASK not in values:
                continue
            visible = {v for v in values if v != MASK}
            assert len(visible) == 1
            fill = visible.pop()
            for r, c in orbit:
                if region[r, c] == MASK:
                    region[r, c] = fill
                    changed = True
        if changed:
            out[r0 : r1 + 1, c0 : c1 + 1] = region
    return out


def _draw_layout(kind: str, side: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    grid = np.zeros((2 * side + 1, 2 * side + 1), dtype=int)
    m = side
    if kind == "cross":
        grid[m, :] = WALL
        grid[:, m] = WALL
        rooms = [
            (0, 0, m - 1, m - 1),
            (0, m + 1, m - 1, 2 * m),
            (m + 1, 0, 2 * m, m - 1),
            (m + 1, m + 1, 2 * m, 2 * m),
        ]
    elif kind == "t_top":
        grid[m, :] = WALL
        grid[: m + 1, m] = WALL
        rooms = [(0, 0, m - 1, m - 1), (0, m + 1, m - 1, 2 * m)]
    elif kind == "t_bottom":
        grid[m, :] = WALL
        grid[m:, m] = WALL
        rooms = [(m + 1, 0, 2 * m, m - 1), (m + 1, m + 1, 2 * m, 2 * m)]
    elif kind == "t_left":
        grid[:, m] = WALL
        grid[m, : m + 1] = WALL
        rooms = [(0, 0, m - 1, m - 1), (m + 1, 0, 2 * m, m - 1)]
    elif kind == "t_right":
        grid[:, m] = WALL
        grid[m, m:] = WALL
        rooms = [(0, m + 1, m - 1, 2 * m), (m + 1, m + 1, 2 * m, 2 * m)]
    else:
        raise ValueError(kind)
    return grid, rooms


def _paint_orbit(region: np.ndarray, orbit: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in orbit:
        region[r, c] = color


def _make_region_case(
    side: int,
    colors: tuple[int, ...],
    *,
    masked: bool,
    require_hard: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    area = side * side
    orbits = _all_orbits(side)
    general_orbits = [orbit for orbit in orbits if len(orbit) == 8]

    for _ in range(200):
        output = np.zeros((side, side), dtype=int)
        special = random.choice(general_orbits)
        min_orbits = max(len(colors) + 1, 4)
        max_orbits = min(len(orbits), len(colors) + 4)
        count = random.randint(min_orbits, max_orbits)

        chosen = [special]
        rest = [orbit for orbit in orbits if orbit != special]
        random.shuffle(rest)
        chosen.extend(rest[: count - 1])
        random.shuffle(chosen)

        assign_queue = list(colors)
        random.shuffle(assign_queue)
        orbit_colors: dict[tuple[tuple[int, int], ...], int] = {}
        for orbit in chosen:
            if assign_queue:
                color = assign_queue.pop()
            else:
                color = random.choice(colors)
            orbit_colors[orbit] = color
            _paint_orbit(output, orbit, color)

        colored = int(np.count_nonzero(output))
        if colored < max(side + 2, area // 4) or colored > (3 * area) // 4:
            continue
        if len({int(v) for v in output.flat if v != 0}) != len(colors):
            continue

        input_region = output.copy()
        masked_positions: set[tuple[int, int]] = set()
        hard_cells: set[tuple[int, int]] = set()
        budget = (2 * area) // 5

        if masked:
            if require_hard:
                quad = _axial_subset(special, side)
                if quad is None:
                    continue
                masked_positions.update(quad)
                hard_cells.update(quad)

            extra_orbits = [orbit for orbit in chosen if len(orbit) > 1 and orbit != special]
            random.shuffle(extra_orbits)
            target_masks = random.randint(
                max(1, len(masked_positions)),
                max(len(masked_positions), min(budget, side + random.randint(1, side))),
            )
            for orbit in extra_orbits:
                remaining = budget - len(masked_positions)
                if remaining <= 0 or len(masked_positions) >= target_masks:
                    break
                max_take = min(len(orbit) - 1, remaining, target_masks - len(masked_positions))
                if max_take <= 0:
                    continue
                take = random.randint(1, max_take)
                picks = random.sample(list(orbit), take)
                masked_positions.update(picks)

            if not masked_positions:
                picks = random.sample(list(special), 1)
                masked_positions.update(picks)

            if len(masked_positions) > budget:
                continue
            if any(output[r, c] == 0 for r, c in masked_positions):
                continue

        for r, c in masked_positions:
            input_region[r, c] = MASK

        info = {
            "masked_positions": tuple(sorted(masked_positions)),
            "hard_cells": tuple(sorted(hard_cells)),
            "special_orbit": special,
        }
        return input_region, output, info

    raise AssertionError("failed to build region")


def _all_color_sets() -> list[tuple[int, ...]]:
    out = []
    for k in (2, 3):
        out.extend(combinations(PALETTE, k))
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    color_sets = _all_color_sets()

    for _ in range(400):
        side = random.choice((5, 7))
        layout_kind = random.choice(
            ["cross", "cross", "cross", "t_top", "t_bottom", "t_left", "t_right"]
        )
        grid, rooms = _draw_layout(layout_kind, side)
        expected = grid.copy()

        room_count = len(rooms)
        if layout_kind == "cross":
            masked_room_count = random.choice((3, 4))
        else:
            masked_room_count = 2
        masked_room_ids = set(random.sample(range(room_count), masked_room_count))
        hard_room = random.choice(sorted(masked_room_ids))
        chosen_sets = random.sample(color_sets, room_count)

        room_meta = []
        for idx, box in enumerate(rooms):
            colors = chosen_sets[idx]
            inp_patch, out_patch, info = _make_region_case(
                side,
                colors,
                masked=idx in masked_room_ids,
                require_hard=idx == hard_room,
            )
            r0, c0, r1, c1 = box
            grid[r0 : r1 + 1, c0 : c1 + 1] = inp_patch
            expected[r0 : r1 + 1, c0 : c1 + 1] = out_patch
            room_meta.append({"box": box, "colors": colors, **info})

        got = main(grid)
        if not np.array_equal(got, expected):
            continue
        if not _verify_case(grid, expected, room_meta):
            continue
        return grid, expected, {"layout": layout_kind, "rooms": room_meta}

    raise AssertionError("failed to generate case")


def _verify_case(inp: np.ndarray, out: np.ndarray, rooms: list[dict]) -> bool:
    if inp.shape[0] != inp.shape[1] or max(inp.shape) > 30:
        return False
    if not np.any(inp == MASK):
        return False

    hard_ok = False
    local_needed = False

    for room in rooms:
        r0, c0, r1, c1 = room["box"]
        patch_in = inp[r0 : r1 + 1, c0 : c1 + 1]
        patch_out = out[r0 : r1 + 1, c0 : c1 + 1]
        side = patch_in.shape[0]
        masked_positions = set(room["masked_positions"])
        if masked_positions:
            if len(masked_positions) > (2 * side * side) // 5:
                return False
        seen_orbits = set()
        for r, c in masked_positions:
            orbit = _d4_orbit(side, r, c)
            if orbit in seen_orbits:
                continue
            seen_orbits.add(orbit)
            visible = {int(patch_in[rr, cc]) for rr, cc in orbit if int(patch_in[rr, cc]) != MASK}
            if len(visible) != 1:
                return False
            fill = next(iter(visible))
            if any(int(patch_out[rr, cc]) != fill for rr, cc in orbit):
                return False

            global_visible = {
                int(inp[rr, cc])
                for rr, cc in _global_orbit(inp, r0 + r, c0 + c)
                if int(inp[rr, cc]) not in (MASK, WALL)
            }
            if global_visible != {fill}:
                local_needed = True

        for r, c in room["hard_cells"]:
            orbit = _d4_orbit(side, r, c)
            quad = set(_axial_subset(orbit, side) or ())
            axial_visible = {int(patch_in[rr, cc]) for rr, cc in quad if int(patch_in[rr, cc]) != MASK}
            full_visible = {int(patch_in[rr, cc]) for rr, cc in orbit if int(patch_in[rr, cc]) != MASK}
            if not axial_visible and len(full_visible) == 1:
                hard_ok = True

    return hard_ok and local_needed


def generate_input() -> np.ndarray:
    return _generate_case()[0]
