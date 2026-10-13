from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
OBSTACLE = 5
SEED = 7
COMPONENT_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (14, 18)
COMPONENT_COUNT_RANGE = (4, 6)
OBSTACLE_COUNT_OPTIONS = (2, 3)
MAX_GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(coords: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return out


def _spiral_records(shape: tuple[int, int], seed: tuple[int, int]) -> list[tuple[int, int, bool]]:
    """
    Return conceptual spiral steps as (r, c, in_bounds) records.

    The leg sequence is 2, 2, 4, 4, 6, 6, ... and directions cycle
    up, right, down, left. Out-of-bounds steps remain in the record because
    they still consume the conceptual path length.
    """
    h, w = shape
    r, c = seed
    max_delta = max(r, h - 1 - r, c, w - 1 - c)
    max_leg_length = 2 * (max_delta + 1)

    records: list[tuple[int, int, bool]] = []
    leg_length = 2
    direction_index = 0
    while leg_length <= max_leg_length:
        dr, dc = DIR4[direction_index]
        for _ in range(leg_length):
            r += dr
            c += dc
            records.append((r, c, _in_bounds(shape, r, c)))
        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg_length += 2
    return records


def _foreground_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, OBSTACLE, SEED) or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _select_component(components: list[dict]) -> dict:
    return min(
        components,
        key=lambda comp: (
            -int(comp["area"]),
            int(comp["bbox"][1]),
            int(comp["bbox"][0]),
        ),
    )


def _analyze_grid(grid: np.ndarray, *, stop_on_obstacle: bool) -> dict:
    g = np.asarray(grid, dtype=int)
    seed_coords = np.argwhere(g == SEED)
    assert seed_coords.shape == (1, 2), "expected exactly one seed pixel"
    seed = (int(seed_coords[0, 0]), int(seed_coords[0, 1]))

    records = _spiral_records(g.shape, seed)
    visited = [seed]
    visited_order = {seed: 0}
    inbounds_path: list[tuple[int, int]] = []
    inbounds_to_record: list[int] = []
    stop_cell = None

    for record_index, (r, c, ok) in enumerate(records):
        if not ok:
            continue
        inbounds_path.append((r, c))
        inbounds_to_record.append(record_index)
        if stop_on_obstacle and int(g[r, c]) == OBSTACLE:
            stop_cell = (r, c)
            break
        if (r, c) not in visited_order:
            visited_order[(r, c)] = len(visited)
        visited.append((r, c))

    comps = _foreground_components(g)
    visited_set = set(visited)
    touched = [comp for comp in comps if any((int(r), int(c)) in visited_set for r, c in comp["coords"])]

    touch_order: dict[int, int] = {}
    for comp in touched:
        touch_order[int(comp["color"])] = min(
            visited_order[(int(r), int(c))]
            for r, c in comp["coords"]
            if (int(r), int(c)) in visited_order
        )

    return {
        "seed": seed,
        "records": records,
        "visited": visited,
        "visited_order": visited_order,
        "inbounds_path": inbounds_path,
        "inbounds_to_record": inbounds_to_record,
        "stop_cell": stop_cell,
        "components": comps,
        "touched": touched,
        "touch_order": touch_order,
        "winner": _select_component(touched) if touched else None,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use an invisible clipped square spiral to select the largest touched component and return its crop.

    Input structure:
    - The background is black.
    - Exactly one orange seed pixel marks the spiral origin and 2-3 gray pixels are obstacles.
    - All other non-black pixels form 4-connected monochrome foreground components of distinct colors.

    Rule:
    - Starting at the seed, conceptually follow a spiral with directions up, right, down, left and leg lengths
      2, 2, 4, 4, 6, 6, ... ; out-of-bounds steps are skipped but still consume length.
    - Stop immediately before the first in-bounds gray obstacle that would be entered, and collect every
      in-bounds position visited before that stop, including the seed.
    - Among foreground components that occupy at least one collected position, choose the one with maximum area;
      break ties by the leftmost bounding-box column, then by the topmost bounding-box row, and return its tight crop.

    Key insight:
    - The spiral is not drawn; it only defines which components are eligible, so obstacle placement changes the
      candidate set before the area comparison is applied.
    """
    analysis = _analyze_grid(grid, stop_on_obstacle=True)
    winner = analysis["winner"]
    assert winner is not None, "expected at least one touched component"
    return _crop_component(winner["coords"], winner["color"])


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if _in_bounds(shape, rr, cc):
                out.add((rr, cc))
        out.add((r, c))
    return out


def _edge_distance(shape: tuple[int, int], cell: tuple[int, int]) -> int:
    h, w = shape
    r, c = cell
    return min(r, c, h - 1 - r, w - 1 - c)


def _pick_separated_positions(
    candidates: list[tuple[int, int]],
    count: int,
    *,
    min_chebyshev: int,
    banned: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    banned = set() if banned is None else set(banned)
    ordered = candidates[:]
    random.shuffle(ordered)
    picked: list[tuple[int, int]] = []
    for cell in ordered:
        if cell in banned:
            continue
        if any(_chebyshev(cell, prev) < min_chebyshev for prev in picked):
            continue
        picked.append(cell)
        if len(picked) == count:
            return picked
    return None


def _grow_component(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    area: int,
    blocked: set[tuple[int, int]],
    banned_path_cells: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    if anchor in blocked:
        return None

    for radius in (3, 4, 5):
        for _ in range(60):
            cells = {anchor}
            while len(cells) < area:
                frontier = set()
                for r, c in cells:
                    for dr, dc in DIR4:
                        nr, nc = r + dr, c + dc
                        if not _in_bounds(shape, nr, nc):
                            continue
                        if (nr, nc) in cells or (nr, nc) in blocked:
                            continue
                        if (nr, nc) in banned_path_cells:
                            continue
                        if max(abs(nr - anchor[0]), abs(nc - anchor[1])) > radius:
                            continue
                        frontier.add((nr, nc))
                if not frontier:
                    break
                cells.add(random.choice(tuple(frontier)))
            if len(cells) != area:
                continue
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            if area >= 4 and (max(rows) == min(rows) or max(cols) == min(cols)):
                continue
            return cells
    return None


def _build_component_specs(
    prefix: list[tuple[int, int]],
    suffix: list[tuple[int, int]],
    shape: tuple[int, int],
    reach_count: int,
    unreach_count: int,
    tie_mode: bool,
) -> list[dict] | None:
    usable_prefix = [p for p in prefix if _edge_distance(shape, p) >= 2]
    usable_suffix = [p for p in suffix if _edge_distance(shape, p) >= 2]
    if len(usable_prefix) < reach_count or len(usable_suffix) < unreach_count:
        return None

    prefix_index = {cell: i for i, cell in enumerate(prefix)}

    if tie_mode:
        pair_candidates = [
            (a, b)
            for a in usable_prefix
            for b in usable_prefix
            if prefix_index[a] < prefix_index[b]
            and _chebyshev(a, b) >= 4
            and b[1] <= a[1] - 3
        ]
        if not pair_candidates:
            return None
        early_anchor, winning_anchor = random.choice(pair_candidates)
        chosen_prefix = [early_anchor, winning_anchor]
        remaining_prefix = [p for p in usable_prefix if p not in chosen_prefix]
        if reach_count > 2:
            extra = _pick_separated_positions(
                remaining_prefix,
                reach_count - 2,
                min_chebyshev=4,
                banned=set(chosen_prefix),
            )
            if extra is None:
                return None
            chosen_prefix.extend(extra)
        winner_area = random.randint(4, 6)
        reachable_specs = [
            {"anchor": early_anchor, "area": winner_area, "role": "reachable_tie"},
            {"anchor": winning_anchor, "area": winner_area, "role": "reachable_tie"},
        ]
        for anchor in chosen_prefix[2:]:
            reachable_specs.append(
                {
                    "anchor": anchor,
                    "area": random.randint(3, max(3, winner_area - 1)),
                    "role": "reachable_small",
                }
            )
    else:
        ordered_prefix = sorted(usable_prefix, key=lambda p: prefix_index[p])
        pair_candidates = [
            (a, b)
            for a in ordered_prefix
            for b in ordered_prefix
            if prefix_index[a] + 1 < prefix_index[b]
            and _chebyshev(a, b) >= 4
        ]
        if not pair_candidates:
            return None
        first_anchor, winning_anchor = random.choice(pair_candidates)
        chosen_prefix = [first_anchor, winning_anchor]
        remaining_prefix = [p for p in usable_prefix if p not in chosen_prefix]
        if reach_count > 2:
            extra = _pick_separated_positions(
                remaining_prefix,
                reach_count - 2,
                min_chebyshev=4,
                banned=set(chosen_prefix),
            )
            if extra is None:
                return None
            chosen_prefix.extend(extra)
        winner_area = random.randint(4, 7)
        reachable_specs = [
            {
                "anchor": first_anchor,
                "area": random.randint(3, max(3, winner_area - 1)),
                "role": "reachable_first",
            },
            {"anchor": winning_anchor, "area": winner_area, "role": "reachable_winner"},
        ]
        for anchor in chosen_prefix[2:]:
            reachable_specs.append(
                {
                    "anchor": anchor,
                    "area": random.randint(3, max(3, winner_area - 1)),
                    "role": "reachable_small",
                }
            )

    suffix_picks = _pick_separated_positions(
        usable_suffix,
        unreach_count,
        min_chebyshev=4,
    )
    if suffix_picks is None:
        return None

    big_area = random.randint(max(winner_area + 1, 5), 8)
    unreachable_specs = [{"anchor": suffix_picks[0], "area": big_area, "role": "unreachable_big"}]
    for anchor in suffix_picks[1:]:
        unreachable_specs.append(
            {
                "anchor": anchor,
                "area": random.randint(3, min(8, max(big_area - 1, 3))),
                "role": "unreachable_small",
            }
        )

    specs = reachable_specs + unreachable_specs
    random.shuffle(specs)
    return specs


def _place_components(
    grid: np.ndarray,
    path_cells: list[tuple[int, int]],
    specs: list[dict],
    reserved: set[tuple[int, int]],
) -> list[dict] | None:
    colors = random.sample(COMPONENT_COLORS, len(specs))
    banned_path_all = set(path_cells)
    placed: list[dict] = []
    blocked = set(reserved)

    for spec, color in sorted(zip(specs, colors), key=lambda x: (-x[0]["area"], x[0]["role"])):
        anchor = spec["anchor"]
        banned_path_cells = banned_path_all - {anchor}
        cells = _grow_component(grid.shape, anchor, spec["area"], blocked, banned_path_cells)
        if cells is None:
            return None
        for r, c in cells:
            grid[r, c] = int(color)
        blocked |= _halo(cells, grid.shape)
        placed.append(
            {
                "color": int(color),
                "anchor": anchor,
                "area": int(spec["area"]),
                "role": spec["role"],
                "cells": cells,
            }
        )
    return placed


def _first_oob_record(records: list[tuple[int, int, bool]]) -> int | None:
    for i, (_r, _c, ok) in enumerate(records):
        if not ok:
            return i
    return None


def _grid_signature(grid: np.ndarray) -> tuple:
    analysis = _analyze_grid(grid, stop_on_obstacle=True)
    winner = analysis["winner"]
    full = _analyze_grid(grid, stop_on_obstacle=False)
    touched_by_order = sorted(
        analysis["touched"],
        key=lambda comp: analysis["touch_order"][int(comp["color"])],
    )
    return (
        tuple(grid.shape),
        int(winner["color"]) if winner is not None else -1,
        int(full["winner"]["color"]) if full["winner"] is not None else -1,
        tuple(sorted(int(comp["area"]) for comp in analysis["components"])),
        tuple(sorted(int(comp["bbox"][1]) for comp in analysis["components"])),
        tuple(int(comp["color"]) for comp in touched_by_order),
    )


def _validate_generated(grid: np.ndarray, *, expect_tie_mode: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if np.count_nonzero(g == SEED) != 1:
        return False
    obstacle_count = int(np.count_nonzero(g == OBSTACLE))
    if obstacle_count not in OBSTACLE_COUNT_OPTIONS:
        return False

    blocked = _analyze_grid(g, stop_on_obstacle=True)
    full = _analyze_grid(g, stop_on_obstacle=False)
    if blocked["stop_cell"] is None:
        return False
    if blocked["winner"] is None or full["winner"] is None:
        return False

    components = blocked["components"]
    if not (COMPONENT_COUNT_RANGE[0] <= len(components) <= COMPONENT_COUNT_RANGE[1]):
        return False
    if len({int(comp["color"]) for comp in components}) != len(components):
        return False
    if not all(3 <= int(comp["area"]) <= 8 for comp in components):
        return False
    if len(blocked["touched"]) < 2:
        return False

    blocked_winner = blocked["winner"]
    full_winner = full["winner"]
    if int(blocked_winner["color"]) == int(full_winner["color"]):
        return False
    if int(blocked_winner["area"]) >= max(int(comp["area"]) for comp in components):
        return False

    touched_sorted = sorted(
        blocked["touched"],
        key=lambda comp: blocked["touch_order"][int(comp["color"])],
    )
    if int(touched_sorted[0]["color"]) == int(blocked_winner["color"]):
        return False

    max_touched_area = max(int(comp["area"]) for comp in blocked["touched"])
    num_max_touched = sum(int(comp["area"]) == max_touched_area for comp in blocked["touched"])
    if expect_tie_mode:
        if num_max_touched < 2:
            return False
    else:
        if num_max_touched != 1:
            return False

    rendered = _crop_component(blocked_winner["coords"], blocked_winner["color"])
    if not np.array_equal(main(g), rendered):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    center_r = h // 2
    center_c = w // 2
    seed_candidates = [
        (r, c)
        for r in range(max(2, center_r - 2), min(h - 2, center_r + 3))
        for c in range(max(2, center_c - 2), min(w - 2, center_c + 3))
        if (r, c) != (center_r, center_c)
    ]
    random.shuffle(seed_candidates)

    for seed in seed_candidates:
        records = _spiral_records(shape, seed)
        inbounds_path = [(r, c) for r, c, ok in records if ok]
        inbounds_to_record = [i for i, (_r, _c, ok) in enumerate(records) if ok]
        if len(inbounds_path) < 16:
            continue

        first_oob = _first_oob_record(records)
        require_oob = random.random() < 0.45
        stop_candidates = []
        for i in range(8, len(inbounds_path) - 6):
            if require_oob and (first_oob is None or inbounds_to_record[i] <= first_oob):
                continue
            stop_candidates.append(i)
        if not stop_candidates:
            continue

        random.shuffle(stop_candidates)
        for stop_index in stop_candidates[:20]:
            stop_cell = inbounds_path[stop_index]
            prefix = inbounds_path[:stop_index]
            suffix = inbounds_path[stop_index + 1 :]

            total_components = random.randint(*COMPONENT_COUNT_RANGE)
            reach_count = random.choice((2, 3))
            if reach_count >= total_components:
                reach_count = total_components - 1
            unreach_count = total_components - reach_count
            tie_mode = random.random() < 0.35

            specs = _build_component_specs(
                prefix,
                suffix,
                shape,
                reach_count,
                unreach_count,
                tie_mode,
            )
            if specs is None:
                continue

            grid = np.zeros(shape, dtype=int)
            grid[seed] = SEED
            grid[stop_cell] = OBSTACLE

            reserved = _halo({seed, stop_cell}, shape)
            placed = _place_components(grid, inbounds_path, specs, reserved)
            if placed is None:
                continue

            free_cells = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if int(grid[r, c]) == BG and (r, c) not in prefix
            ]
            extra_obstacle_count = random.choice(OBSTACLE_COUNT_OPTIONS) - 1
            later_path_candidates = [cell for cell in suffix if int(grid[cell]) == BG]
            extra_obstacles: list[tuple[int, int]] = []
            if extra_obstacle_count >= 1 and later_path_candidates:
                extra_obstacles.append(random.choice(later_path_candidates))
            if extra_obstacle_count == 2:
                pool = [cell for cell in free_cells if cell not in extra_obstacles]
                if not pool:
                    continue
                extra_obstacles.append(random.choice(pool))
            elif extra_obstacle_count == 1 and not extra_obstacles:
                pool = free_cells
                if not pool:
                    continue
                extra_obstacles.append(random.choice(pool))
            if len(extra_obstacles) != extra_obstacle_count:
                continue
            for cell in extra_obstacles:
                grid[cell] = OBSTACLE

            if _validate_generated(grid, expect_tie_mode=tie_mode):
                return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-component ARC input")
