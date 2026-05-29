# Combo variant of 2b83f449 (index 24)
# combo_spec: D3/2+D7/0+D9/1
# Name: 惯性+多出口+逐个占位
#
# Design analysis
# - Concrete behavior on the base template (W=17):
#   wall rows use centers [[8], [8], [8,12], [6,13], [6,12]].
#   After opening, a token keeps trying to go downward. If the current column is
#   already an available exit, it continues straight. Otherwise it slides inside
#   its current pipe segment to a chosen exit. When a segment offers two exits,
#   the token chooses by approach side: "from left" -> leftmost exit,
#   "from right" -> rightmost exit. Example:
#   row0-left goes 0 -> 7 -> 7 -> 7 -> 7 and settles at row10,col7.
#   row2-left arrives later and follows the same route until row8. Static row8
#   has exits {7,11}, but row10,col7 is already occupied, so only 11 remains and
#   the later token is forced to reroute to row10,col11. This is the intended
#   D3+D7+D9 coupling.
# - Complexity calibration:
#   the template repeats only a small number of fork motifs, so the rule is
#   visible from a few examples. At the same time there are multiple repeated
#   2-exit segments and multiple blocking events, so the combo is not visually
#   degenerate to the base task.
# - Broken base assumptions:
#   the original solver assumed <=1 exit per segment and used a final global
#   packing phase. Both assumptions are invalid here. The new solver removes all
#   single-exit assertions, resolves exits locally by approach direction, and
#   settles tokens immediately one-by-one so settled tokens become new blockers.
# - Placement legality:
#   the generator uses a fixed relative gate pattern, then only translates it
#   horizontally within width 17-19 and optionally mirrors it. This preserves
#   non-overlap, keeps every 1x3 bar inside the grid, and preserves the segment
#   geometry needed by the solver.
# - Visual observability / cleanliness:
#   the opened network has two repeated "fork then fork again" zones. Earlier
#   tokens occupy one branch, later same-side tokens either switch to the other
#   branch or stop one level earlier. No hidden ordering by color or numeric
#   value is used: all cues are geometric and visible.
# - Generator guarantees for new assumptions:
#   tokens always start on edge cells, every opened fork comes from actual gate
#   spacing in the pixel grid, and the sampled instance is accepted only if it
#   shows nontrivial coupled behavior under quantitative checks.

from __future__ import annotations

from typing import Literal

import numpy as np

WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8

Approach = Literal["left", "right"]


def _parse_gate_centers(grid: np.ndarray) -> list[list[int]]:
    h, w = grid.shape
    centers: list[list[int]] = []
    for wr in range(1, h, 2):
        row_centers: list[int] = []
        c = 1
        while c < w - 1:
            if int(grid[wr, c - 1]) == GATE and int(grid[wr, c]) == GATE and int(grid[wr, c + 1]) == GATE:
                row_centers.append(c)
                c += 3
            else:
                c += 1
        centers.append(row_centers)
    return centers


def _collect_sources(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    sources: list[tuple[int, int]] = []
    for r in range(0, h, 2):
        for c in range(w):
            if int(grid[r, c]) == TOKEN:
                sources.append((r, c))
    return sorted(sources)


def _open_input(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[list[int]]]:
    g = np.array(grid, dtype=int)
    centers = _parse_gate_centers(g)
    sources = _collect_sources(g)
    opened = g.copy()
    for r, c in sources:
        opened[r, c] = BG
    for wall_idx, row_centers in enumerate(centers):
        wr = 2 * wall_idx + 1
        for c in row_centers:
            opened[wr, c - 1 : c + 2] = BG
            opened[wr - 1, c] = OPEN
            opened[wr, c] = OPEN
            opened[wr + 1, c] = OPEN
    return opened, sources, centers


def _find_segment(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return None
    if int(grid[r, c]) != BG:
        return None
    a = c
    while a - 1 >= 0 and int(grid[r, a - 1]) == BG:
        a -= 1
    b = c
    while b + 1 < grid.shape[1] and int(grid[r, b + 1]) == BG:
        b += 1
    return (a, b)


def _reachable_exits(grid: np.ndarray, r: int, seg: tuple[int, int] | None) -> list[int]:
    if seg is None or r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    exits: list[int] = []
    for c in range(a, b + 1):
        if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG:
            exits.append(c)
    return exits


def _initial_approach(c: int, width: int) -> Approach:
    return "left" if c <= (width - 1) // 2 else "right"


def _choose_exit(c: int, exits: list[int], approach: Approach) -> int:
    if c in exits:
        return c
    if approach == "left":
        return min(exits)
    return max(exits)


def _simulate_tokens(
    opened: np.ndarray,
    sources: list[tuple[int, int]],
    sequential: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    static_grid = np.array(opened, dtype=int)
    dynamic_grid = np.array(opened, dtype=int)
    out = np.array(opened, dtype=int)
    records: list[dict[str, object]] = []
    width = opened.shape[1]

    for source_r, source_c in sources:
        r, c = source_r, source_c
        approach: Approach = _initial_approach(c, width)
        step_records: list[dict[str, object]] = []

        while True:
            cur_grid = dynamic_grid if sequential else static_grid
            dyn_seg = _find_segment(cur_grid, r, c)
            static_seg = _find_segment(static_grid, r, c)
            dyn_exits = _reachable_exits(cur_grid, r, dyn_seg)
            static_exits = _reachable_exits(static_grid, r, static_seg)
            if not dyn_exits:
                step_records.append(
                    {
                        "row": r,
                        "col": c,
                        "static_exits": tuple(static_exits),
                        "dynamic_exits": tuple(dyn_exits),
                        "target": None,
                    }
                )
                break

            target = _choose_exit(c, dyn_exits, approach)
            step_records.append(
                {
                    "row": r,
                    "col": c,
                    "static_exits": tuple(static_exits),
                    "dynamic_exits": tuple(dyn_exits),
                    "target": target,
                }
            )
            if target > c:
                approach = "left"
            elif target < c:
                approach = "right"
            r += 2
            c = target

        if sequential and int(dynamic_grid[r, c]) == BG:
            dynamic_grid[r, c] = TOKEN
        if int(out[r, c]) == BG:
            out[r, c] = TOKEN
        records.append({"source": (source_r, source_c), "final": (r, c), "steps": tuple(step_records)})

    return out, records


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    opened, sources, _ = _open_input(grid)
    _, seq_records = _simulate_tokens(opened, sources, sequential=True)
    _, free_records = _simulate_tokens(opened, sources, sequential=False)

    final_differences = 0
    coupled_steps = 0
    multi_exit_steps = 0
    horizontal_moves = 0
    max_depth = 0

    for seq_rec, free_rec in zip(seq_records, free_records):
        if seq_rec["final"] != free_rec["final"]:
            final_differences += 1
        max_depth = max(max_depth, len(seq_rec["steps"]) - 1)
        for step in seq_rec["steps"]:
            static_exits = step["static_exits"]
            dynamic_exits = step["dynamic_exits"]
            target = step["target"]
            col = step["col"]
            if len(static_exits) > 1:
                multi_exit_steps += 1
            if target is not None and target != col:
                horizontal_moves += 1
            if len(static_exits) > len(dynamic_exits):
                coupled_steps += 1

    return {
        "final_differences": final_differences,
        "coupled_steps": coupled_steps,
        "multi_exit_steps": multi_exit_steps,
        "horizontal_moves": horizontal_moves,
        "max_depth": max_depth,
    }


def main(grid: np.ndarray) -> np.ndarray:
    opened, sources, _ = _open_input(np.array(grid, dtype=int))
    out, _ = _simulate_tokens(opened, sources, sequential=True)
    return out


def _transform_pattern(base: list[list[int]], width: int, offset: int, mirror: bool) -> list[list[int]]:
    transformed: list[list[int]] = []
    for row in base:
        cols = [c + offset for c in row]
        if mirror:
            cols = [width - 1 - c for c in cols]
        transformed.append(sorted(cols))
    return transformed


def _build_input_from_centers(centers: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(centers)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        grid[2 * pipe_idx, :] = BG
    for wall_idx, row_centers in enumerate(centers):
        wr = 2 * wall_idx + 1
        for c in row_centers:
            grid[wr, c - 1 : c + 2] = GATE
    for pipe_idx in range(n_walls):
        grid[2 * pipe_idx, 0] = TOKEN
        grid[2 * pipe_idx, width - 1] = TOKEN
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    base_centers = [[8], [8], [8, 12], [6, 13], [6, 12]]

    for _ in range(100):
        width = int(rng.integers(17, 20))
        offset = int(rng.integers(0, width - 16))
        mirror = bool(rng.integers(2))
        centers = _transform_pattern(base_centers, width, offset, mirror)
        grid = _build_input_from_centers(centers, width)
        metrics = _behavior_metrics(grid)
        if metrics["coupled_steps"] < 2:
            continue
        if metrics["final_differences"] < 3:
            continue
        if metrics["multi_exit_steps"] < 4:
            continue
        if metrics["horizontal_moves"] < 7:
            continue
        if metrics["max_depth"] < 4:
            continue
        opened, sources, _ = _open_input(grid)
        if not np.array_equal(main(grid), _simulate_tokens(opened, sources, sequential=True)[0]):
            continue
        return grid

    raise RuntimeError("failed to generate combo input for 2b83f449/24")
