from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
VISIBLE_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
BASE_SIGNAL_COLORS = [1, 2, 3, 6]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(h: int, w: int, r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if _in_bounds(h, w, nr, nc):
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(h, w, cr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_legend(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(grid != BG):
        vals = [int(grid[r, c]) for r, c in coords]
        if YELLOW in vals:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if int(coords.shape[0]) != hh * ww:
            continue
        if not (ww == 3 or hh == 3):
            continue
        if int(coords.shape[0]) < 6:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        candidates.append(((r0, c0, r1, c1), coords))

    assert len(candidates) == 1, "expected one dense edge legend"
    (r0, c0, r1, c1), coords = candidates[0]

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[coords[:, 0], coords[:, 1]] = True

    rows: list[tuple[int, int, int]] = []
    if c1 - c0 + 1 == 3:
        for r in range(r0, r1 + 1):
            row = tuple(int(x) for x in grid[r, c0 : c0 + 3])
            assert BG not in row and YELLOW not in row
            rows.append(row)
    else:
        for c in range(c0, c1 + 1):
            row = (int(grid[r0, c]), int(grid[r0 + 1, c]), int(grid[r0 + 2, c]))
            assert BG not in row and YELLOW not in row
            rows.append(row)

    return legend_mask, rows


def _extract_scene(
    grid: np.ndarray,
) -> tuple[list[dict], dict[tuple[int, int], int], np.ndarray, dict[tuple[int, int], int]]:
    legend_mask, legend_rows = _find_legend(grid)
    mix_lookup: dict[tuple[int, int], int] = {}
    for a, b, c in legend_rows:
        key = tuple(sorted((a, b)))
        if key in mix_lookup:
            assert mix_lookup[key] == c, "legend rows must agree for repeated pairs"
        mix_lookup[key] = c

    h, w = grid.shape
    towers: list[dict] = []
    tower_positions: set[tuple[int, int]] = set()
    indicator_positions: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val in (BG, YELLOW) or legend_mask[r, c]:
                continue
            yellow_neighbors = [(nr, nc) for nr, nc in _neighbors4(h, w, r, c) if int(grid[nr, nc]) == YELLOW]
            if len(yellow_neighbors) == 1:
                ir, ic = yellow_neighbors[0]
                direction = (ir - r, ic - c)
                towers.append({"pos": (r, c), "color": val, "dir": direction, "indicator": (ir, ic)})
                tower_positions.add((r, c))
                indicator_positions.add((ir, ic))
            else:
                assert len(yellow_neighbors) == 0, "non-tower colored cells cannot touch yellow"

    all_yellow = set(map(tuple, np.argwhere((grid == YELLOW) & (~legend_mask))))
    assert towers and all_yellow == indicator_positions, "yellow pixels must be one-to-one with towers"

    mirrors: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val in (BG, YELLOW) or legend_mask[r, c] or (r, c) in tower_positions:
                continue
            assert all((nr, nc) not in tower_positions for nr, nc in _neighbors4(h, w, r, c)), "mirrors cannot touch towers"
            mirrors[(r, c)] = val

    towers.sort(key=lambda item: item["pos"])
    return towers, mirrors, legend_mask, mix_lookup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace tower rays, turning them clockwise at colored mirrors and recoloring each reflected segment from the legend mix table.

    Input structure:
    - The background is black.
    - There are 1-3 single-pixel signal towers in non-black colors, and each tower has exactly one yellow 4-neighbor that marks its launch direction.
    - Additional isolated non-black, non-yellow single pixels are mirrors, and one dense edge-attached Nx3 legend lists color-mix rules as row triples A,B,C.

    Rule:
    - Copy the input, then clear every yellow direction marker to black.
    - Each tower emits one ray in the direction indicated by the yellow marker, but the marker cell itself is only a pointer, so the first drawable step is the cell beyond it.
    - As the ray travels, paint traversed black cells with the current ray color; when it hits a mirror, rotate the direction 90 degrees clockwise and replace the ray color by the legend result for current_color + mirror_color.
    - Rays stop when the next step would leave the grid or enter any original non-black obstacle that is not a mirror; towers, mirrors, and the legend keep their original colors.

    Key insight:
    - The task combines two coupled state updates: spatial ray tracing from the yellow pointer side and color evolution from repeated legend lookups at mirror collisions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    towers, mirrors, legend_mask, mix_lookup = _extract_scene(g)

    out = g.copy()
    out[g == YELLOW] = BG

    h, w = g.shape
    dir_to_idx = {d: i for i, d in enumerate(DIRS)}
    for tower in towers:
        tr, tc = tower["pos"]
        ir, ic = tower["indicator"]
        color = int(tower["color"])
        d = dir_to_idx[tower["dir"]]
        pos = (ir, ic)
        seen: set[tuple[int, int, int, int]] = set()

        while True:
            state = (pos[0], pos[1], d, color)
            assert state not in seen, "ray path must not cycle"
            seen.add(state)

            dr, dc = DIRS[d]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not _in_bounds(h, w, nr, nc):
                break

            if (nr, nc) in mirrors:
                key = tuple(sorted((color, mirrors[(nr, nc)])))
                assert key in mix_lookup, "missing legend entry for ray color and mirror color"
                color = int(mix_lookup[key])
                pos = (nr, nc)
                d = (d + 1) % 4
                continue

            if int(g[nr, nc]) != BG:
                break

            out[nr, nc] = color
            pos = (nr, nc)

        out[tr, tc] = int(g[tr, tc])

    out[legend_mask] = g[legend_mask]
    for (r, c), color in mirrors.items():
        out[r, c] = color
    return out


GEN_ATTEMPTS = 400
TOWER_PLACEMENT_ATTEMPTS = 250
UNUSED_MIRROR_ATTEMPTS = 120
LEGEND_PLACEMENT_ATTEMPTS = 120


def _free_run(
    h: int,
    w: int,
    pos: tuple[int, int],
    d: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = DIRS[d]
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(h, w, r, c):
            break
        if (r, c) in blocked:
            break
        cells.append((r, c))
    return cells


def _can_add_object_cells(
    h: int,
    w: int,
    new_cells: Iterable[tuple[int, int]],
    object_cells: set[tuple[int, int]],
    allowed_pairs: set[frozenset[tuple[int, int]]] | None = None,
) -> bool:
    allowed_pairs = allowed_pairs or set()
    new_list = list(new_cells)
    new_set = set(new_list)
    if len(new_set) != len(new_list):
        return False
    for cell in new_set:
        if not _in_bounds(h, w, cell[0], cell[1]) or cell in object_cells:
            return False
    for cell in new_set:
        for nb in _neighbors4(h, w, cell[0], cell[1]):
            pair = frozenset((cell, nb))
            if nb in object_cells and pair not in allowed_pairs:
                return False
            if nb in new_set and nb != cell and pair not in allowed_pairs:
                return False
    return True


def _sample_mix_color(pair: tuple[int, int], rng: random.Random) -> int:
    options = [color for color in VISIBLE_COLORS if color not in pair]
    if not options:
        options = list(VISIBLE_COLORS)
    return int(rng.choice(options))


def _plan_tower(
    h: int,
    w: int,
    turns: int,
    object_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    for _ in range(TOWER_PLACEMENT_ATTEMPTS):
        d0 = int(rng.randrange(4))
        dr0, dc0 = DIRS[d0]
        tower = (rng.randrange(1, h - 1), rng.randrange(1, w - 1))
        indicator = (tower[0] + dr0, tower[1] + dc0)
        beyond = (indicator[0] + dr0, indicator[1] + dc0)
        if not _in_bounds(h, w, beyond[0], beyond[1]):
            continue

        allowed = {frozenset((tower, indicator))}
        if not _can_add_object_cells(h, w, [tower, indicator], object_cells, allowed_pairs=allowed):
            continue
        if tower in path_cells or indicator in path_cells:
            continue

        local_objects = [tower, indicator]
        local_mirrors: list[tuple[int, int]] = []
        local_path: set[tuple[int, int]] = set()
        pos = indicator
        direction = d0
        drawn = 0
        ok = True

        for turn_idx in range(turns):
            blocked = object_cells | path_cells | set(local_objects) | local_path
            cells = _free_run(h, w, pos, direction, blocked)
            if len(cells) < 2:
                ok = False
                break

            candidates: list[tuple[int, tuple[int, int]]] = []
            for j in range(1, len(cells)):
                mirror = cells[j]
                if not _can_add_object_cells(h, w, [mirror], object_cells | set(local_objects), allowed_pairs=set()):
                    continue
                next_dir = (direction + 1) % 4
                next_blocked = blocked | {mirror} | set(cells[:j])
                next_cells = _free_run(h, w, mirror, next_dir, next_blocked)
                min_needed = 2 if turn_idx + 1 < turns else 1
                if len(next_cells) < min_needed:
                    continue
                candidates.append((j, mirror))

            if not candidates:
                ok = False
                break

            j, mirror = rng.choice(candidates)
            for cell in cells[:j]:
                local_path.add(cell)
            drawn += j
            local_objects.append(mirror)
            local_mirrors.append(mirror)
            pos = mirror
            direction = (direction + 1) % 4

        if not ok:
            continue

        blocked = object_cells | path_cells | set(local_objects) | local_path
        final_cells = _free_run(h, w, pos, direction, blocked)
        if drawn + len(final_cells) < 2:
            continue
        local_path.update(final_cells)

        return {
            "tower": tower,
            "indicator": indicator,
            "dir_idx": d0,
            "mirrors": local_mirrors,
            "path_cells": local_path,
        }

    return None


def _place_unused_mirrors(
    h: int,
    w: int,
    count: int,
    object_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    mirrors: list[tuple[int, int]] = []
    current_objects = set(object_cells)
    for _ in range(count):
        placed = False
        for _ in range(UNUSED_MIRROR_ATTEMPTS):
            cell = (rng.randrange(h), rng.randrange(w))
            if cell in path_cells:
                continue
            if not _can_add_object_cells(h, w, [cell], current_objects, allowed_pairs=set()):
                continue
            mirrors.append(cell)
            current_objects.add(cell)
            placed = True
            break
        if not placed:
            break
    return mirrors


def _legend_candidates(h: int, w: int, rows: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for c0 in range(0, w - 2):
        candidates.append((0, c0))
        candidates.append((h - rows, c0))
    for r0 in range(0, h - rows + 1):
        candidates.append((r0, 0))
        candidates.append((r0, w - 3))
    dedup: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup


def _place_legend(
    h: int,
    w: int,
    rows: list[tuple[int, int, int]],
    object_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    candidates = _legend_candidates(h, w, len(rows))
    rng.shuffle(candidates)
    blocked = object_cells | path_cells
    for r0, c0 in candidates[:LEGEND_PLACEMENT_ATTEMPTS]:
        r1 = r0 + len(rows) - 1
        c1 = c0 + 2
        ok = True
        for r in range(max(0, r0 - 1), min(h, r1 + 2)):
            for c in range(max(0, c0 - 1), min(w, c1 + 2)):
                if (r, c) in blocked:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return r0, c0
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tower_count = int(rng.randint(1, 3))
    turns_per_tower = [int(rng.choice((0, 1, 2))) for _ in range(tower_count)]
    complexity = tower_count + sum(turns_per_tower)
    min_side = max(15, 10 + 2 * complexity)
    max_side = min(30, max(min_side, 24))
    h = int(rng.randint(min_side, max_side))
    w = int(rng.randint(min_side, max_side))

    object_cells: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()
    tower_specs: list[dict] = []

    order = list(range(tower_count))
    rng.shuffle(order)
    turns_ordered = [turns_per_tower[i] for i in order]
    for turns in turns_ordered:
        spec = _plan_tower(h, w, turns, object_cells, path_cells, rng)
        if spec is None:
            return None
        tower_specs.append(spec)
        object_cells.add(spec["tower"])
        object_cells.add(spec["indicator"])
        object_cells.update(spec["mirrors"])
        path_cells.update(spec["path_cells"])

    unused_mirrors = _place_unused_mirrors(
        h,
        w,
        count=int(rng.randint(0, 2)),
        object_cells=object_cells,
        path_cells=path_cells,
        rng=rng,
    )
    object_cells.update(unused_mirrors)

    tower_colors = [int(rng.choice(BASE_SIGNAL_COLORS)) for _ in tower_specs]
    mirror_colors: dict[tuple[int, int], int] = {}
    for spec in tower_specs:
        for mirror in spec["mirrors"]:
            mirror_colors[mirror] = int(rng.choice(VISIBLE_COLORS))
    for mirror in unused_mirrors:
        mirror_colors[mirror] = int(rng.choice(VISIBLE_COLORS))

    mix_lookup: dict[tuple[int, int], int] = {}
    for tower_color, spec in zip(tower_colors, tower_specs):
        current = int(tower_color)
        for mirror in spec["mirrors"]:
            pair = tuple(sorted((current, mirror_colors[mirror])))
            if pair not in mix_lookup:
                mix_lookup[pair] = _sample_mix_color(pair, rng)
            current = int(mix_lookup[pair])

    legend_rows = [(a, b, c) for (a, b), c in mix_lookup.items()]
    all_pairs = [(a, b) for i, a in enumerate(VISIBLE_COLORS) for b in VISIBLE_COLORS[i:]]
    rng.shuffle(all_pairs)
    target_rows = max(3, len(legend_rows) + int(rng.randint(0, 2)))
    for a, b in all_pairs:
        if len(legend_rows) >= target_rows:
            break
        pair = tuple(sorted((a, b)))
        if pair in mix_lookup:
            continue
        legend_rows.append((pair[0], pair[1], _sample_mix_color(pair, rng)))
    rng.shuffle(legend_rows)

    legend_pos = _place_legend(h, w, legend_rows, object_cells, path_cells, rng)
    if legend_pos is None:
        return None
    legend_r0, legend_c0 = legend_pos

    grid = np.full((h, w), BG, dtype=int)
    for spec, tower_color in zip(tower_specs, tower_colors):
        tr, tc = spec["tower"]
        ir, ic = spec["indicator"]
        grid[tr, tc] = int(tower_color)
        grid[ir, ic] = YELLOW
    for mirror, color in mirror_colors.items():
        grid[mirror] = int(color)
    for idx, row in enumerate(legend_rows):
        grid[legend_r0 + idx, legend_c0 : legend_c0 + 3] = np.array(row, dtype=int)

    if np.count_nonzero(grid == YELLOW) != tower_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for seed in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(10**9) + seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-mirror ARC input")
