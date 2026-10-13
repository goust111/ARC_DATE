import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
INPUT_COLORS = (1, 4, 5, 6, 7, 8, 9)
BASE_TILES = (
    np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 1, 2],
            [1, 1, 1, 2],
            [0, 2, 2, 2],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 0],
            [1, 0, 1, 2],
            [1, 1, 1, 2],
            [3, 3, 3, 2],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 1, 2],
            [3, 3, 3, 0],
        ],
        dtype=int,
    ),
)


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _find_components(grid: np.ndarray, target_color: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            if target_color is not None and color != int(target_color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return out


def _black_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != 0 or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append({"coords": coords, "touches_border": touches_border})
    return out


def _hole_owner_map(grid: np.ndarray) -> dict[int, bool]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comp_id = np.full((h, w), -1, dtype=int)
    components = _find_components(g)
    for idx, comp in enumerate(components):
        for r, c in comp["coords"]:
            comp_id[r, c] = idx

    hole_owner = {idx: False for idx in range(len(components))}
    for hole in _black_components(g):
        if hole["touches_border"]:
            continue
        owners = set()
        for r, c in hole["coords"]:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and comp_id[nr, nc] != -1:
                    owners.add(int(comp_id[nr, nc]))
        if len(owners) == 1:
            hole_owner[next(iter(owners))] = True
    return hole_owner


def _recolor_by_topology(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    components = _find_components(g)
    hole_owner = _hole_owner_map(g)
    out = np.zeros_like(g)
    for idx, comp in enumerate(components):
        color = 3 if hole_owner.get(idx, False) else 2
        for r, c in comp["coords"]:
            out[r, c] = color
    return out


def _period_is_consistent(grid: np.ndarray, pr: int, pc: int) -> tuple[bool, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    residue_colors: dict[tuple[int, int], set[int]] = {}
    residue_counts: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value == 0:
                continue
            key = (r % pr, c % pc)
            residue_colors.setdefault(key, set()).add(value)
            residue_counts[key] = residue_counts.get(key, 0) + 1
            if len(residue_colors[key]) > 1:
                return False, 0
    support = sum(max(0, count - 1) for count in residue_counts.values())
    return True, support


def _detect_period(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    for pr in range(1, h):
        for pc in range(1, w):
            ok, support = _period_is_consistent(g, pr, pc)
            if not ok or support <= 0:
                continue
            key = (pr * pc, pr, pc)
            if best_key is None or key < best_key:
                best_key = key
                best = (pr, pc)
    if best is None:
        raise ValueError("no translational period detected")
    return best


def _repair_grid(grid: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pr, pc = map(int, period)
    h, w = g.shape
    residue_colors: dict[tuple[int, int], set[int]] = {}
    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value == 0:
                continue
            residue_colors.setdefault((r % pr, c % pc), set()).add(value)

    repaired = g.copy()
    for r in range(h):
        for c in range(w):
            if int(repaired[r, c]) != 0:
                continue
            colors = residue_colors.get((r % pr, c % pc), set())
            if len(colors) == 1:
                repaired[r, c] = next(iter(colors))
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a damaged periodic pattern and recolor repaired components by hollow-vs-solid topology.

    Input structure:
    - The input is a black-background translational tiling whose visible foreground repeats with a small rectangular period.
    - Foreground cells use a few non-black colors and form monochromatic 4-connected components, some hollow and some solid in the undamaged pattern.
    - Some foreground cells were replaced by black, so black can mean either true background or periodic damage.

    Rule:
    - Detect the smallest row and column periods whose translational residue classes keep all visible non-black colors consistent.
    - For each black cell, inspect its translational orbit under that period; if all visible non-black copies in the orbit agree on one color, repair the black cell with that color.
    - On the fully repaired grid, find every monochromatic 4-connected foreground component and determine whether it encloses at least one 4-connected black hole region.
    - Recolor hollow components green (3), recolor solid components red (2), and leave black background cells unchanged.

    Key insight:
    - The damage removes exactly the boundary evidence needed for hole detection, so topology can be classified correctly only after periodic inpainting restores the missing pixels.
    """
    g = np.asarray(grid, dtype=int)
    period = _detect_period(g)
    repaired = _repair_grid(g, period)
    return _recolor_by_topology(repaired)


def _coord_transform(
    shape: tuple[int, int], coord: tuple[int, int], rotations: int, flip_lr: bool
) -> tuple[int, int]:
    h, w = map(int, shape)
    r, c = map(int, coord)
    for _ in range(rotations % 4):
        r, c = c, h - 1 - r
        h, w = w, h
    if flip_lr:
        c = w - 1 - c
    return int(r), int(c)


def _instantiate_tile() -> np.ndarray:
    base = np.array(random.choice(BASE_TILES), dtype=int, copy=True)
    rotations = random.randrange(4)
    flip_lr = random.random() < 0.5
    tile = np.rot90(base, rotations)
    if flip_lr:
        tile = np.fliplr(tile)

    roles = sorted(set(int(v) for v in np.unique(tile) if int(v) != 0))
    mapped_colors = random.sample(INPUT_COLORS, k=len(roles))
    mapping = {role: color for role, color in zip(roles, mapped_colors)}
    out = np.zeros_like(tile)
    for role, color in mapping.items():
        out[tile == role] = color
    return out


def _articulation_cells(tile: np.ndarray, comp: dict) -> list[tuple[int, int]]:
    if len(comp["coords"]) < 3:
        return []
    out: list[tuple[int, int]] = []
    for removed in comp["coords"]:
        remaining = comp["coord_set"] - {removed}
        if not remaining:
            continue
        start = next(iter(remaining))
        seen = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        if len(seen) != len(remaining):
            out.append(removed)
    return out


def _hole_break_cells(tile: np.ndarray, comp: dict) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    original_color = int(comp["color"])
    for removed in comp["coords"]:
        broken = np.array(tile, dtype=int, copy=True)
        broken[removed] = 0
        broken_components = _find_components(broken)
        broken_holes = _hole_owner_map(broken)
        still_hollow = any(
            int(candidate["color"]) == original_color and broken_holes.get(idx, False)
            for idx, candidate in enumerate(broken_components)
        )
        if not still_hollow:
            out.append(removed)
    return out


def _choose_local_damage(tile: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    hole_owner = _hole_owner_map(tile)
    hollow_candidates: list[tuple[int, int]] = []
    solid_candidates: list[tuple[int, int]] = []
    for idx, comp in enumerate(_find_components(tile)):
        if hole_owner.get(idx, False):
            hollow_candidates.extend(_hole_break_cells(tile, comp))
        else:
            solid_candidates.extend(_articulation_cells(tile, comp))
    if not hollow_candidates or not solid_candidates:
        return None
    return random.choice(hollow_candidates), random.choice(solid_candidates)


def _tile_grid(tile: np.ndarray, repeats_r: int, repeats_c: int) -> np.ndarray:
    return np.tile(tile, (int(repeats_r), int(repeats_c)))


def _all_foreground_groups(grid: np.ndarray, period: tuple[int, int]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    pr, pc = map(int, period)
    h, w = g.shape
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == 0:
                continue
            groups.setdefault((r % pr, c % pc), []).append((r, c))
    return groups


def _damage_grid(
    full: np.ndarray,
    period: tuple[int, int],
    mandatory: list[tuple[int, int]],
    damage_target: int,
) -> np.ndarray:
    damaged = np.array(full, dtype=int, copy=True)
    groups = _all_foreground_groups(full, period)
    damaged_count = {key: 0 for key in groups}
    used = set()

    for cell in mandatory:
        if cell in used:
            continue
        key = (cell[0] % period[0], cell[1] % period[1])
        if len(groups[key]) <= 1:
            raise ValueError("cannot damage a singleton orbit")
        damaged[cell] = 0
        damaged_count[key] += 1
        used.add(cell)

    candidates = [
        cell
        for group in groups.values()
        for cell in group
        if cell not in used
    ]
    random.shuffle(candidates)

    for cell in candidates:
        if len(used) >= int(damage_target):
            break
        key = (cell[0] % period[0], cell[1] % period[1])
        if damaged_count[key] >= len(groups[key]) - 1:
            continue
        damaged[cell] = 0
        damaged_count[key] += 1
        used.add(cell)

    return damaged


def _tile_patch(
    grid: np.ndarray, period: tuple[int, int], tile_index: tuple[int, int]
) -> np.ndarray:
    pr, pc = map(int, period)
    tr, tc = map(int, tile_index)
    r0 = tr * pr
    c0 = tc * pc
    return np.asarray(grid[r0 : r0 + pr, c0 : c0 + pc], dtype=int)


def _count_hollow_components(grid: np.ndarray) -> int:
    hole_owner = _hole_owner_map(grid)
    return sum(1 for value in hole_owner.values() if value)


def _semantic_checks(
    inp: np.ndarray,
    complete: np.ndarray,
    period: tuple[int, int],
    hole_copy: tuple[int, int],
    solid_copy: tuple[int, int],
    expected_hole_color: int,
    expected_solid_color: int,
) -> bool:
    if _detect_period(inp) != tuple(map(int, period)):
        return False

    repaired = _repair_grid(inp, period)
    if not np.array_equal(repaired, complete):
        return False

    correct_output = _recolor_by_topology(complete)
    if not np.array_equal(main(inp), correct_output):
        return False

    shortcut_output = _recolor_by_topology(inp)
    if np.array_equal(shortcut_output, correct_output):
        return False

    if not np.any(correct_output == 2) or not np.any(correct_output == 3):
        return False

    if np.count_nonzero(inp == 0) - np.count_nonzero(complete == 0) > int(0.3 * inp.size):
        return False

    for comp in _find_components(complete):
        if len(comp["coords"]) < 3:
            return False

    hole_patch_damaged = _tile_patch(inp, period, hole_copy)
    hole_patch_full = _tile_patch(complete, period, hole_copy)
    if _count_hollow_components(hole_patch_full) == 0:
        return False
    if _count_hollow_components(hole_patch_damaged) != 0:
        return False
    if np.count_nonzero(hole_patch_damaged == expected_hole_color) >= np.count_nonzero(hole_patch_full == expected_hole_color):
        return False

    solid_patch_damaged = _tile_patch(inp, period, solid_copy)
    solid_patch_full = _tile_patch(complete, period, solid_copy)
    full_solid_count = len(_find_components(solid_patch_full, target_color=expected_solid_color))
    damaged_solid_count = len(_find_components(solid_patch_damaged, target_color=expected_solid_color))
    if damaged_solid_count <= full_solid_count:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(800):
        tile = _instantiate_tile()
        period = tuple(map(int, tile.shape))
        local_damage = _choose_local_damage(tile)
        if local_damage is None:
            continue
        hole_local, solid_local = local_damage
        expected_hole_color = int(tile[hole_local])
        expected_solid_color = int(tile[solid_local])

        repeats_r = random.randint(3, 6)
        repeats_c = random.randint(3, 6)
        complete = _tile_grid(tile, repeats_r, repeats_c)

        max_extra_black = int(0.3 * complete.size)
        if max_extra_black < 2:
            continue

        hole_copy = (random.randrange(repeats_r), random.randrange(repeats_c))
        solid_copy = (random.randrange(repeats_r), random.randrange(repeats_c))
        hole_cell = (
            hole_copy[0] * period[0] + hole_local[0],
            hole_copy[1] * period[1] + hole_local[1],
        )
        solid_cell = (
            solid_copy[0] * period[0] + solid_local[0],
            solid_copy[1] * period[1] + solid_local[1],
        )
        if hole_cell == solid_cell:
            continue

        groups = _all_foreground_groups(complete, period)
        if len(groups[(hole_local[0], hole_local[1])]) <= 1 or len(groups[(solid_local[0], solid_local[1])]) <= 1:
            continue

        damage_target = random.randint(2, min(max_extra_black, max(5, int(np.count_nonzero(complete) * 0.22))))
        inp = _damage_grid(complete, period, [hole_cell, solid_cell], damage_target)
        if _semantic_checks(
            inp,
            complete,
            period,
            hole_copy,
            solid_copy,
            expected_hole_color,
            expected_solid_color,
        ):
            return inp

    raise RuntimeError("failed to generate a valid periodic-topology instance")
