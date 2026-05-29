# Combo variant of 28a6681f (index 46)
# combo_spec: D1/1+D7/1
# Name: BFS泛洪扩散+连通器原理

from __future__ import annotations

import random
from typing import Any

import numpy as np


# Design analysis
# a) Concrete realization and behavior trace
#    - Keep the original 10x10 "blue water + colored bottom cups" visual language, but
#      replace local gravity motion with two explicit reasoning stages.
#    - Stage 1 is visible BFS reachability: starting from the supported source-water block,
#      spread through 4-neighbor background cells and mark every cavity the source can
#      visibly enter.
#    - Stage 2 is hidden communicating-vessel equalization: once any cup in one bottom
#      vessel-group is reached, that whole visible group acts like it has hidden base pipes,
#      so its basins end at one common waterline.
#    - Example: the bottom contains an open two-basin group and, elsewhere, a roofed single
#      cup. In the original task, water would fall and slide locally, and a sealed cup would
#      stay dry simply because there is no local path. In this fusion, the open group is
#      selected because BFS can enter it, then its two basins equalize to the same surface;
#      the roofed cup stays dry because BFS never reaches its cavity. The difference is:
#      reachability is decided globally by flood-fill, and final height is decided globally
#      by pooled equalization, not by per-pixel falling order.
# b) Complexity calibration
#    - Use exactly one reachable vessel-group with two visible basins plus one unreachable
#      decoy cup. That makes both parts of the combo observable without overcomplicating a
#      10x10 grid.
#    - Basin widths stay in {1,2}; floor depths stay in {8,9}; wall tops stay in {5,6}.
#      These small discrete families are easy to parse and create repeated evidence for the
#      equal-waterline rule.
#    - Water counts are sampled from valid shared levels so the output is stable and the
#      two active basins both visibly participate.
# c) Implicit assumption audit against the original solver
#    - Original `settle()` assumed local downward / diagonal / horizontal motion determines
#      the answer. That assumption fails here because hidden base links can move water into
#      a neighboring basin without any visible local trajectory.
#    - Original iteration order over individual water pixels also no longer matters; the BFS
#      stage only measures visible access, and the equalization stage only depends on total
#      water count plus candidate basin cells.
#    - Original guide-specific connectivity checks are dropped. The new solver only assumes
#      supported source water and a controlled cup geometry that can be parsed from
#      bottom-anchored stacks.
#    - Original stable-output checks still make sense, but now stability is produced by
#      filling parsed basin cells bottom-up to a common level rather than by simulating
#      movement until convergence.
# d) Placement legality
#    - The generator draws bottom-anchored wall/floor stacks first, then adds a non-bottom
#      source shelf over the active side and a non-bottom roof over the decoy cup.
#    - Active and decoy groups occupy disjoint column spans with at least one empty gap, so
#      there is no overlap. Every object is placed with fixed in-bounds templates.
#    - The roof sits above the decoy rim, so it blocks BFS access without altering the
#      parser's bottom-stack interpretation of that cup.
# e) Visual observability
#    - The input always shows an obvious source block, one open multi-basin structure, and
#      one visibly cup-like but roofed decoy. Across examples, the dry decoy reveals that
#      reachability matters, while the matched water surface across the two active basins
#      reveals the communicating-vessel rule.
#    - This avoids the D7-only degeneration where all visible basins always fill, making BFS
#      irrelevant.
# f) Visual cleanliness
#    - Only three semantic layers are used: background, fixed solids, and blue water.
#    - The decoy roof is a single short bar, so it reads as a closure cue rather than noisy
#      clutter. Source shelf and bottom vessels remain vertically separated.
# g) Color-change inferability
#    - No color remapping occurs. Blue always means water; the other colors are static solid
#      material. Different solid colors only help segment structures visually.
# h) Variant metric used after implementation
#    - Measure `(reachable_groups, total_groups, active_surface_rows, dry_decoy_columns)`.
#      Target behavior is exactly one reachable group, at least one unreachable group, equal
#      top rows across the active basins, and a fully dry decoy cup.
# i) New solver assumptions to re-check
#    - `_extract_groups()` assumes every parsed basin is an interval between two tall
#      bottom-anchored walls and that every inner column has a bottom-anchored floor stack.
#    - `_active_group_ids()` assumes the source can reach candidate cells of exactly one
#      vessel-group through visible background connectivity.
#    - `generate_input()` guarantees those assumptions by construction and rejects any sample
#      whose metric does not show one wet equalized group plus one dry unreachable decoy.


H = 10
W = 10
BG = 0
WATER = 1


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _bottom_stack_top(structure: np.ndarray, c: int) -> int | None:
    """Top row of the bottom-connected solid stack in a column, else None."""
    if int(structure[H - 1, c]) == BG:
        return None
    r = H - 1
    while r - 1 >= 0 and int(structure[r - 1, c]) != BG:
        r -= 1
    return int(r)


def _extract_groups(structure: np.ndarray) -> list[list[dict[str, Any]]]:
    """
    Parse the controlled cup family.

    Tall bottom stacks (top <= 6) are cup walls.
    Inner bottom stacks (top >= 8) are floors.
    Consecutive basins that share a wall belong to one communicating vessel-group.
    """
    stack_top = [_bottom_stack_top(structure, c) for c in range(W)]
    wall_cols = [c for c, top in enumerate(stack_top) if top is not None and int(top) <= 6]
    basins: list[dict[str, Any]] = []
    for left, right in zip(wall_cols, wall_cols[1:]):
        if right - left <= 1:
            continue
        inner = list(range(left + 1, right))
        floor_tops = [stack_top[c] for c in inner]
        if any(top is None or int(top) <= 6 for top in floor_tops):
            continue
        rim = max(int(stack_top[left]), int(stack_top[right]))
        supports = {int(c): int(stack_top[c]) - 1 for c in inner if int(stack_top[c]) - 1 >= rim}
        if not supports:
            continue
        basins.append(
            {
                "left": int(left),
                "right": int(right),
                "rim": int(rim),
                "supports": supports,
            }
        )

    groups: list[list[dict[str, Any]]] = []
    for basin in basins:
        if groups and int(groups[-1][-1]["right"]) == int(basin["left"]):
            groups[-1].append(basin)
        else:
            groups.append([basin])

    for group_id, group in enumerate(groups):
        for basin in group:
            basin["group_id"] = int(group_id)
    return groups


def _group_candidate_cells(group: list[dict[str, Any]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for basin in group:
        rim = int(basin["rim"])
        for c, support in basin["supports"].items():
            for r in range(int(support), rim - 1, -1):
                cells.append((int(r), int(c)))
    cells.sort(key=lambda rc: (-int(rc[0]), int(rc[1])))
    return cells


def _reachable_mask(structure: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    seen = np.zeros((H, W), dtype=bool)
    passable = structure == BG
    stack = [(int(r), int(c)) for r, c in starts if 0 <= int(r) < H and 0 <= int(c) < W and passable[int(r), int(c)]]
    for r, c in stack:
        seen[int(r), int(c)] = True
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(int(r), int(c)):
            if 0 <= rr < H and 0 <= cc < W and passable[int(rr), int(cc)] and not seen[int(rr), int(cc)]:
                seen[int(rr), int(cc)] = True
                stack.append((int(rr), int(cc)))
    return seen


def _active_group_ids(structure: np.ndarray, starts: list[tuple[int, int]]) -> set[int]:
    reachable = _reachable_mask(structure, starts)
    active: set[int] = set()
    for group in _extract_groups(structure):
        for r, c in _group_candidate_cells(group):
            if reachable[int(r), int(c)]:
                active.add(int(group[0]["group_id"]))
                break
    return active


def _fill_equalized_group(structure: np.ndarray, group: list[dict[str, Any]], water_count: int) -> np.ndarray:
    cells = _group_candidate_cells(group)
    if int(water_count) > len(cells):
        raise AssertionError("too much water for reachable communicating-vessel capacity")
    out = np.array(structure, dtype=int).copy()
    for r, c in cells[: int(water_count)]:
        out[int(r), int(c)] = WATER
    return out


def _surface_rows(grid: np.ndarray, group: list[dict[str, Any]]) -> list[int]:
    tops: list[int] = []
    for basin in group:
        basin_tops = []
        for c in basin["supports"]:
            rows = np.argwhere(grid[:, int(c)] == WATER)
            if rows.size:
                basin_tops.append(int(rows[0, 0]))
        if basin_tops:
            tops.append(min(basin_tops))
    return tops


def _valid_levels(group: list[dict[str, Any]]) -> list[tuple[int, int]]:
    max_rim = max(int(basin["rim"]) for basin in group)
    max_support = max(int(support) for basin in group for support in basin["supports"].values())
    levels: list[tuple[int, int]] = []
    for level in range(max_rim, max_support + 1):
        water_count = 0
        active_basins = 0
        for basin in group:
            basin_has_water = False
            for support in basin["supports"].values():
                if int(support) >= level:
                    basin_has_water = True
                    water_count += int(support) - level + 1
            if basin_has_water:
                active_basins += 1
        if active_basins == len(group) and 5 <= water_count <= 10:
            levels.append((int(level), int(water_count)))
    return levels


def _fusion_metric(grid: np.ndarray) -> dict[str, Any]:
    structure = np.where(grid == WATER, BG, grid)
    groups = _extract_groups(structure)
    starts = [(int(r), int(c)) for r, c in np.argwhere(grid == WATER)]
    active_ids = _active_group_ids(structure, starts)
    solved = main(grid, check_input=False, check_output=False)
    dry_columns = 0
    for group in groups:
        if int(group[0]["group_id"]) in active_ids:
            continue
        for _, c in _group_candidate_cells(group):
            if not np.any(solved[:, int(c)] == WATER):
                dry_columns += 1
    active_surface_rows: list[int] = []
    for group in groups:
        if int(group[0]["group_id"]) in active_ids:
            active_surface_rows.extend(_surface_rows(solved, group))
    return {
        "reachable_groups": int(len(active_ids)),
        "total_groups": int(len(groups)),
        "active_surface_rows": active_surface_rows,
        "dry_decoy_columns": int(dry_columns),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Route blue water by visible BFS reachability, then equalize it across the reached vessel-group.

    Input structure:
    - bg = 0 and water = 1. Other colors are fixed solids.
    - The lower half contains one open two-basin cup group plus one separate roofed decoy cup.
    - Water starts as a supported source block on an upper shelf, with no water already spread through the basins.

    Rule:
    - Remove the water temporarily and run a 4-neighbor BFS from the source positions through background cells.
    - Any vessel-group whose basin cells are touched by that BFS is reachable; in the generated family exactly one group is reachable and the roofed decoy is not.
    - Pool all water into the reachable group and refill its basin cells from the lowest positions upward so every reached basin ends at the same waterline.

    Key insight: BFS decides which visible container system the source can access, and the hidden communicating-vessel rule decides the final shared height inside that system.
    """
    g = np.array(grid, dtype=int).copy()
    if g.shape != (H, W):
        raise AssertionError("grid must be 10x10")

    starts = [(int(r), int(c)) for r, c in np.argwhere(g == WATER)]
    if not starts:
        return g

    if check_input:
        for r, c in starts:
            if int(r) < H - 1:
                assert int(g[int(r) + 1, int(c)]) != BG, "input water must be supported"

    water_count = int(len(starts))
    structure = np.where(g == WATER, BG, g)
    groups = _extract_groups(structure)
    if len(groups) < 2:
        raise AssertionError("expected at least one active vessel-group and one decoy group")

    active_ids = _active_group_ids(structure, starts)
    if len(active_ids) != 1:
        raise AssertionError("expected exactly one BFS-reachable vessel-group")

    active_id = next(iter(active_ids))
    active_group = next(group for group in groups if int(group[0]["group_id"]) == int(active_id))
    out = _fill_equalized_group(structure, active_group, water_count)

    if check_output:
        water_cells = {(int(r), int(c)) for r, c in np.argwhere(out == WATER)}
        allowed = set(_group_candidate_cells(active_group))
        assert water_cells <= allowed, "water escaped the reachable vessel-group"
        surface_rows = _surface_rows(out, active_group)
        assert len(surface_rows) == len(active_group), "every active basin must receive water"
        assert len(set(surface_rows)) == 1, "reachable basins must share one waterline"
        for r, c in water_cells:
            assert r == H - 1 or int(out[int(r) + 1, int(c)]) != BG
            assert c == 0 or int(out[int(r), int(c) - 1]) != BG
            assert c == W - 1 or int(out[int(r), int(c) + 1]) != BG
    return out


def _draw_bottom_stack(grid: np.ndarray, c: int, top: int, color: int) -> None:
    grid[int(top) :, int(c)] = int(color)


def _build_structure(rng: random.Random) -> tuple[np.ndarray, int]:
    grid = np.zeros((H, W), dtype=int)
    palette = rng.sample([2, 3, 4, 5, 6, 7, 8, 9], k=4)

    active_start = rng.choice([0, 4])
    decoy_start = 7 if active_start == 0 else 0
    widths = list(rng.choice([(1, 2), (2, 1)]))
    wall_cols = [int(active_start)]
    cur = int(active_start)
    for width in widths:
        cur += int(width) + 1
        wall_cols.append(int(cur))

    wall_tops = [rng.choice([5, 6]) for _ in wall_cols]
    support_tops_flat = [8, 8, 9]
    rng.shuffle(support_tops_flat)

    basin_inner_cols = [
        list(range(wall_cols[0] + 1, wall_cols[1])),
        list(range(wall_cols[1] + 1, wall_cols[2])),
    ]
    split = len(basin_inner_cols[0])
    support_profiles = [support_tops_flat[:split], support_tops_flat[split:]]

    for idx, c in enumerate(wall_cols):
        _draw_bottom_stack(grid, int(c), int(wall_tops[idx]), int(palette[idx % 2]))
    for basin_idx, cols in enumerate(basin_inner_cols):
        for offset, c in enumerate(cols):
            _draw_bottom_stack(grid, int(c), int(support_profiles[basin_idx][offset]), int(palette[2 + basin_idx]))

    decoy_wall_cols = [int(decoy_start), int(decoy_start + 2)]
    decoy_wall_tops = [rng.choice([5, 6]), rng.choice([5, 6])]
    decoy_floor_top = rng.choice([8, 9])
    for idx, c in enumerate(decoy_wall_cols):
        _draw_bottom_stack(grid, int(c), int(decoy_wall_tops[idx]), int(palette[1 + idx]))
    _draw_bottom_stack(grid, int(decoy_start + 1), int(decoy_floor_top), int(palette[3]))
    grid[4, decoy_start : decoy_start + 3] = int(palette[0])

    groups = _extract_groups(grid)
    active_group_id = None
    for group in groups:
        if len(group) == 2:
            active_group_id = int(group[0]["group_id"])
            break
    if active_group_id is None or len(groups) != 2:
        raise AssertionError("generated structure did not match the expected 2-group family")
    return grid, int(active_group_id)


def _add_source_shelf(grid: np.ndarray, active_start: int, color: int) -> str:
    if int(active_start) == 0:
        grid[4, 0:3] = int(color)
        return "left"
    grid[4, 7:10] = int(color)
    return "right"


def _place_source_water(grid: np.ndarray, water_count: int, side: str) -> None:
    cols = [0, 1, 2] if side == "left" else [7, 8, 9]
    col_order = cols if side == "left" else list(reversed(cols))
    order: list[tuple[int, int]] = []
    for r in [3, 2, 1, 0]:
        for c in col_order:
            order.append((int(r), int(c)))
    if int(water_count) > len(order):
        raise AssertionError("source shelf cannot support this much water")
    for r, c in order[: int(water_count)]:
        if int(grid[int(r), int(c)]) != BG:
            raise AssertionError("source water overlapped a solid")
        grid[int(r), int(c)] = WATER


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random(random.randrange(1 << 30))
        structure, active_group_id = _build_structure(rng)
        groups = _extract_groups(structure)
        active_group = next(group for group in groups if int(group[0]["group_id"]) == int(active_group_id))
        levels = _valid_levels(active_group)
        if not levels:
            continue

        base = np.array(structure, dtype=int).copy()
        active_start = min(int(basin["left"]) for basin in active_group)
        shelf_color = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
        source_side = _add_source_shelf(base, active_start, shelf_color)

        _, water_count = rng.choice(levels)
        inp = np.array(base, dtype=int).copy()
        _place_source_water(inp, int(water_count), source_side)

        try:
            out = main(inp)
        except Exception:
            continue

        metric = _fusion_metric(inp)
        if metric["reachable_groups"] != 1:
            continue
        if metric["total_groups"] != 2:
            continue
        if len(metric["active_surface_rows"]) != 2 or len(set(metric["active_surface_rows"])) != 1:
            continue
        if metric["dry_decoy_columns"] < 1:
            continue

        active_cells = {(int(r), int(c)) for r, c in np.argwhere(out == WATER)}
        if not active_cells:
            continue
        return inp
    raise RuntimeError("failed to generate a valid BFS+communicating-vessel sample")
