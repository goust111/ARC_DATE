from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WALL = 5
ARROW_PALETTE = (1, 2, 3, 4, 7)
DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
BOUNCE_TYPES = ("front", "row", "col")
GEN_ATTEMPTS = 300
ARROW_ATTEMPTS = 240


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield int(r + dr), int(c + dc)


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        steps = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in steps:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _arrow_cells_from_bbox(r0: int, c0: int, start: tuple[int, int]) -> set[tuple[int, int]]:
    bbox_cells = {
        (int(r0), int(c0)),
        (int(r0), int(c0 + 1)),
        (int(r0 + 1), int(c0)),
        (int(r0 + 1), int(c0 + 1)),
    }
    bbox_cells.remove((int(start[0]), int(start[1])))
    return bbox_cells


def _find_arrows(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    arrows: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {BG, WALL}):
        comps = _components_mask(g == int(color), connectivity=8)
        assert len(comps) == 1, f"expected one component for arrow color {color}"
        coords = comps[0]
        assert int(coords.shape[0]) == 3, f"arrow color {color} must have area 3"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert (r1 - r0, c1 - c0) == (1, 1), f"arrow color {color} must fit a 2x2 box"
        cells = {(int(r), int(c)) for r, c in coords}
        bbox_cells = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        missing = list(bbox_cells - cells)
        assert len(missing) == 1, f"arrow color {color} must be a 2x2 missing-corner L"
        start = tuple(map(int, missing[0]))
        dr = -1 if start[0] == r0 else 1
        dc = -1 if start[1] == c0 else 1
        arrows.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "cells": cells,
                "start": start,
                "dir": (int(dr), int(dc)),
            }
        )
    arrows.sort(key=lambda a: (a["bbox"][0], a["bbox"][1], a["color"]))
    return arrows


def _is_wall(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    r, c = map(int, pos)
    return _in_bounds(grid.shape, r, c) and int(grid[r, c]) == WALL


def _simulate_arrow(grid: np.ndarray, arrow: dict, all_arrow_cells: set[tuple[int, int]]) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    r, c = map(int, arrow["start"])
    dr, dc = map(int, arrow["dir"])
    self_cells = set(arrow["cells"])
    bounces = 0
    trace: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    step_limit = int(g.shape[0] * g.shape[1] * 16)

    for _ in range(step_limit):
        state = (int(r), int(c), int(dr), int(dc))
        assert state not in seen, "invalid input: ray entered a loop"
        seen.add(state)

        diag = (int(r + dr), int(c + dc))
        row_side = (int(r + dr), int(c))
        col_side = (int(r), int(c + dc))
        event = "none"
        walls: list[tuple[int, int]] = []

        if not _in_bounds(g.shape, diag[0], diag[1]):
            return int(bounces), trace
        if diag in all_arrow_cells - self_cells:
            return int(bounces), trace
        assert diag not in self_cells, "invalid input: ray entered its own arrow body"

        if _is_wall(g, diag):
            dr, dc = -int(dr), -int(dc)
            bounces += 1
            trace.append(
                {
                    "from": (int(r), int(c)),
                    "to": (int(r), int(c)),
                    "dir_after": (int(dr), int(dc)),
                    "event": "front",
                    "walls": (diag,),
                }
            )
            continue

        nr, nc = map(int, diag)
        row_hit = _is_wall(g, row_side)
        col_hit = _is_wall(g, col_side)
        if row_hit or col_hit:
            if row_hit:
                dr = -int(dr)
                walls.append(row_side)
            if col_hit:
                dc = -int(dc)
                walls.append(col_side)
            bounces += 1
            event = "row+col" if row_hit and col_hit else ("row" if row_hit else "col")

        trace.append(
            {
                "from": (int(r), int(c)),
                "to": (nr, nc),
                "dir_after": (int(dr), int(dc)),
                "event": event,
                "walls": tuple(walls),
            }
        )
        r, c = nr, nc

    raise AssertionError("invalid input: step budget exceeded")


def _simulate_all(grid: np.ndarray) -> tuple[list[dict], list[int], list[list[dict]]]:
    arrows = _find_arrows(grid)
    all_arrow_cells = set().union(*(arrow["cells"] for arrow in arrows))
    counts: list[int] = []
    traces: list[list[dict]] = []
    for arrow in arrows:
        count, trace = _simulate_arrow(grid, arrow, all_arrow_cells)
        counts.append(int(count))
        traces.append(trace)
    return arrows, counts, traces


def _histogram(arrows: list[dict], counts: list[int]) -> np.ndarray:
    height = max(1, max(counts))
    out = np.zeros((height, len(arrows)), dtype=int)
    order = sorted(
        range(len(arrows)),
        key=lambda i: (int(counts[i]), int(arrows[i]["bbox"][0]), int(arrows[i]["bbox"][1])),
    )
    for col, idx in enumerate(order):
        count = int(counts[idx])
        if count:
            out[-count:, col] = int(arrows[idx]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert diagonal bounce counts from missing-corner arrows into a sorted bottom-aligned histogram.

    Input structure:
    - Black background with 3-5 uniquely colored 3-pixel arrows, each a 2x2 L whose missing corner gives one diagonal launch direction.
    - Grey single-pixel walls are scattered away from the arrows.
    - Arrow colors are identifiers only; geometry comes from the arrow bbox and missing corner.

    Rule:
    - For each arrow, start at its missing corner and cast a diagonal ray in that corner's direction.
    - If the diagonal next cell is a wall, reverse both direction components in place; otherwise step into that diagonal cell and, if one axial side cell is a wall, reverse only that axis for subsequent motion.
    - Stop before stepping out of bounds or into another arrow body, and count how many bounces occurred.
    - Sort arrows by bounce count ascending, breaking ties by arrow bbox top-left position, then output one bottom-aligned color column per arrow with height equal to its count; if every count is zero, use height 1.

    Key insight:
    - The visible ray paths are only an intermediate geometric computation; the final answer keeps just one integer per arrow and discards all spatial detail in favor of a sorted histogram.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    arrows, counts, _ = _simulate_all(g)
    return _histogram(arrows, counts)


def _touches_arrow(existing_arrow_cells: set[tuple[int, int]], cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in existing_arrow_cells:
            return True
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in existing_arrow_cells:
                return True
    return False


def _straight_ok(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    arrow_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> bool:
    r, c = map(int, pos)
    dr, dc = map(int, direction)
    diag = (int(r + dr), int(c + dc))
    row_side = (int(r + dr), int(c))
    col_side = (int(r), int(c + dc))
    if not _in_bounds(shape, diag[0], diag[1]):
        return False
    if diag in arrow_cells or diag in wall_cells:
        return False
    if row_side in wall_cells or col_side in wall_cells:
        return False
    return True


def _step_guard(pos: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = map(int, pos)
    dr, dc = map(int, direction)
    return {
        (int(r), int(c)),
        (int(r + dr), int(c + dc)),
        (int(r + dr), int(c)),
        (int(r), int(c + dc)),
    }


def _apply_bounce(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    bounce_type: str,
    arrow_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    global_bg_guard: set[tuple[int, int]],
    local_guard: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    r, c = map(int, pos)
    dr, dc = map(int, direction)
    diag = (int(r + dr), int(c + dc))
    row_side = (int(r + dr), int(c))
    col_side = (int(r), int(c + dc))
    new_walls: set[tuple[int, int]] = set()
    needed_clear = {(int(r), int(c))}

    if bounce_type == "front":
        wall_pos = diag
        needed_clear |= {row_side, col_side}
        new_dir = (-int(dr), -int(dc))
        new_pos = (int(r), int(c))
    elif bounce_type == "row":
        wall_pos = row_side
        needed_clear |= {diag, col_side}
        new_dir = (-int(dr), int(dc))
        new_pos = diag
    else:
        wall_pos = col_side
        needed_clear |= {diag, row_side}
        new_dir = (int(dr), -int(dc))
        new_pos = diag

    if not _in_bounds(shape, wall_pos[0], wall_pos[1]):
        return None
    if wall_pos in arrow_cells or wall_pos in wall_cells or wall_pos in global_bg_guard or wall_pos in local_guard:
        return None
    for cell in needed_clear:
        rr, cc = cell
        if _in_bounds(shape, rr, cc) and (cell in arrow_cells or cell in wall_cells):
            return None
    if not _in_bounds(shape, new_pos[0], new_pos[1]):
        return None
    if new_pos in wall_cells or new_pos in arrow_cells:
        return None

    new_walls.add(wall_pos)
    new_guard = {cell for cell in needed_clear if _in_bounds(shape, cell[0], cell[1])}
    new_guard.add(new_pos)
    return new_pos, new_dir, new_walls, new_guard


def _plan_arrow(
    rng: random.Random,
    shape: tuple[int, int],
    desired_bounces: int,
    all_arrow_cells: set[tuple[int, int]],
    all_wall_cells: set[tuple[int, int]],
    global_bg_guard: set[tuple[int, int]],
    forced_duplicate_ok: bool,
    required_prefix: tuple[str, ...] = (),
) -> dict | None:
    h, w = map(int, shape)
    for _ in range(ARROW_ATTEMPTS):
        r0 = rng.randint(1, h - 3)
        c0 = rng.randint(1, w - 3)
        start = (int(r0 + rng.choice((0, 1))), int(c0 + rng.choice((0, 1))))
        direction = (-1 if start[0] == r0 else 1, -1 if start[1] == c0 else 1)
        cells = _arrow_cells_from_bbox(r0, c0, start)
        if _touches_arrow(all_arrow_cells, cells):
            continue
        if cells & (all_wall_cells | global_bg_guard):
            continue

        path_state = (int(start[0]), int(start[1]), int(direction[0]), int(direction[1]))
        seen_states = {path_state}
        local_guard = {start}
        local_walls: set[tuple[int, int]] = set()
        pos = start
        cur_dir = direction
        bounce_types_used: list[str] = []

        ok = True
        for bounce_index in range(desired_bounces):
            straight_len = rng.randint(0, 3)
            for _ in range(straight_len):
                combined_walls = all_wall_cells | local_walls
                combined_arrows = all_arrow_cells | cells
                if not _straight_ok(shape, pos, cur_dir, combined_arrows, combined_walls):
                    ok = False
                    break
                local_guard |= {cell for cell in _step_guard(pos, cur_dir) if _in_bounds(shape, cell[0], cell[1])}
                pos = (int(pos[0] + cur_dir[0]), int(pos[1] + cur_dir[1]))
                state = (int(pos[0]), int(pos[1]), int(cur_dir[0]), int(cur_dir[1]))
                if state in seen_states:
                    ok = False
                    break
                seen_states.add(state)
            if not ok:
                break

            if bounce_index < len(required_prefix):
                bounce_choices = [str(required_prefix[bounce_index])]
            else:
                bounce_choices = ["row", "col", "front"]
                rng.shuffle(bounce_choices)
            applied = None
            for bounce_type in bounce_choices:
                result = _apply_bounce(
                    shape,
                    pos,
                    cur_dir,
                    bounce_type,
                    all_arrow_cells | cells,
                    all_wall_cells | local_walls,
                    global_bg_guard,
                    local_guard,
                )
                if result is None:
                    continue
                next_pos, next_dir, new_walls, new_guard = result
                state = (int(next_pos[0]), int(next_pos[1]), int(next_dir[0]), int(next_dir[1]))
                if state in seen_states:
                    continue
                applied = (bounce_type, next_pos, next_dir, new_walls, new_guard, state)
                break
            if applied is None:
                ok = False
                break

            bounce_type, pos, cur_dir, new_walls, new_guard, state = applied
            local_walls |= new_walls
            local_guard |= new_guard
            seen_states.add(state)
            bounce_types_used.append(bounce_type)

        if not ok:
            continue

        combined_walls = all_wall_cells | local_walls
        combined_arrows = all_arrow_cells | cells
        exit_steps = 0
        while True:
            nr = int(pos[0] + cur_dir[0])
            nc = int(pos[1] + cur_dir[1])
            if not _in_bounds(shape, nr, nc):
                break
            if not _straight_ok(shape, pos, cur_dir, combined_arrows, combined_walls):
                ok = False
                break
            local_guard |= {cell for cell in _step_guard(pos, cur_dir) if _in_bounds(shape, cell[0], cell[1])}
            pos = (nr, nc)
            exit_steps += 1
            if exit_steps > h * w:
                ok = False
                break
            state = (int(pos[0]), int(pos[1]), int(cur_dir[0]), int(cur_dir[1]))
            if state in seen_states:
                ok = False
                break
            seen_states.add(state)
        if not ok:
            continue

        if desired_bounces > 0 and not bounce_types_used:
            continue
        if desired_bounces == 0 and forced_duplicate_ok and exit_steps == 0:
            continue

        plan = {
            "color": None,
            "bbox": (int(r0), int(c0), int(r0 + 1), int(c0 + 1)),
            "cells": cells,
            "start": start,
            "dir": direction,
            "walls": local_walls,
            "guard": local_guard,
            "bounce_types": bounce_types_used,
        }
        return plan
    return None


def _plan_front_macro_arrow(
    rng: random.Random,
    shape: tuple[int, int],
    all_arrow_cells: set[tuple[int, int]],
    all_wall_cells: set[tuple[int, int]],
    global_bg_guard: set[tuple[int, int]],
) -> dict | None:
    h, w = map(int, shape)
    for _ in range(ARROW_ATTEMPTS):
        r0 = rng.randint(1, h - 3)
        c0 = rng.randint(1, w - 3)
        start = (int(r0 + rng.choice((0, 1))), int(c0 + rng.choice((0, 1))))
        direction = (-1 if start[0] == r0 else 1, -1 if start[1] == c0 else 1)
        dr, dc = map(int, direction)
        first_step = (int(start[0] + dr), int(start[1] + dc))
        front_wall = (int(start[0]), int(start[1] + 2 * dc))
        post_col_step = (int(start[0] + 2 * dr), int(start[1]))
        if not _in_bounds(shape, first_step[0], first_step[1]):
            continue
        if not _in_bounds(shape, front_wall[0], front_wall[1]):
            continue
        if not _in_bounds(shape, post_col_step[0], post_col_step[1]):
            continue

        cells = _arrow_cells_from_bbox(r0, c0, start)
        if _touches_arrow(all_arrow_cells, cells):
            continue
        if cells & (all_wall_cells | global_bg_guard):
            continue

        local_guard = {start}
        local_walls: set[tuple[int, int]] = set()
        seen_states = {(int(start[0]), int(start[1]), int(dr), int(dc))}

        first = _apply_bounce(
            shape,
            start,
            direction,
            "row",
            all_arrow_cells | cells,
            all_wall_cells | local_walls,
            global_bg_guard,
            local_guard,
        )
        if first is None:
            continue
        pos, cur_dir, new_walls, new_guard = first
        local_walls |= new_walls
        local_guard |= new_guard
        seen_states.add((int(pos[0]), int(pos[1]), int(cur_dir[0]), int(cur_dir[1])))

        second = _apply_bounce(
            shape,
            pos,
            cur_dir,
            "front",
            all_arrow_cells | cells,
            all_wall_cells | local_walls,
            global_bg_guard,
            local_guard,
        )
        if second is None:
            continue
        pos, cur_dir, new_walls, new_guard = second
        local_walls |= new_walls
        local_guard |= new_guard
        state = (int(pos[0]), int(pos[1]), int(cur_dir[0]), int(cur_dir[1]))
        if state in seen_states:
            continue
        seen_states.add(state)

        combined_arrows = all_arrow_cells | cells
        combined_walls = all_wall_cells | local_walls
        if not _in_bounds(shape, int(pos[0] + cur_dir[0]), int(pos[1] + cur_dir[1])):
            continue
        if (int(pos[0] + cur_dir[0]), int(pos[1] + cur_dir[1])) in combined_arrows:
            continue
        row_side = (int(pos[0] + cur_dir[0]), int(pos[1]))
        col_side = (int(pos[0]), int(pos[1] + cur_dir[1]))
        if row_side in combined_walls or col_side not in combined_walls:
            continue
        local_guard |= {cell for cell in _step_guard(pos, cur_dir) if _in_bounds(shape, cell[0], cell[1])}
        pos = (int(pos[0] + cur_dir[0]), int(pos[1] + cur_dir[1]))
        cur_dir = (int(cur_dir[0]), -int(cur_dir[1]))
        state = (int(pos[0]), int(pos[1]), int(cur_dir[0]), int(cur_dir[1]))
        if state in seen_states:
            continue
        seen_states.add(state)

        ok = True
        exit_steps = 0
        while True:
            nr = int(pos[0] + cur_dir[0])
            nc = int(pos[1] + cur_dir[1])
            if not _in_bounds(shape, nr, nc):
                break
            if not _straight_ok(shape, pos, cur_dir, all_arrow_cells | cells, all_wall_cells | local_walls):
                ok = False
                break
            local_guard |= {cell for cell in _step_guard(pos, cur_dir) if _in_bounds(shape, cell[0], cell[1])}
            pos = (nr, nc)
            exit_steps += 1
            if exit_steps > h * w:
                ok = False
                break
            state = (int(pos[0]), int(pos[1]), int(cur_dir[0]), int(cur_dir[1]))
            if state in seen_states:
                ok = False
                break
            seen_states.add(state)
        if not ok:
            continue

        return {
            "color": None,
            "bbox": (int(r0), int(c0), int(r0 + 1), int(c0 + 1)),
            "cells": cells,
            "start": start,
            "dir": direction,
            "walls": local_walls,
            "guard": local_guard,
            "bounce_types": ["row", "front", "col"],
        }
    return None


def _sample_counts(rng: random.Random, n_arrows: int) -> tuple[list[int], bool] | None:
    for _ in range(120):
        want_duplicate = rng.random() < 0.6
        counts = [rng.randint(0, 5) for _ in range(n_arrows)]
        if len(set(counts)) == 1:
            continue
        if max(counts) == 0:
            continue
        if sum(counts) > 16:
            continue
        if want_duplicate and len(set(counts)) == len(counts):
            continue
        if not want_duplicate and len(set(counts)) < len(counts) and rng.random() < 0.5:
            continue
        return counts, want_duplicate
    return None


def _add_distractor_walls(
    rng: random.Random,
    shape: tuple[int, int],
    grid: np.ndarray,
    arrow_cells: set[tuple[int, int]],
    bg_guard: set[tuple[int, int]],
    wall_target: int,
) -> None:
    h, w = map(int, shape)
    wall_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == WALL)}
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (int(r), int(c))
            if cell in arrow_cells or cell in bg_guard or cell in wall_cells:
                continue
            if any((nr, nc) in arrow_cells for nr, nc in _neighbors8(r, c)):
                continue
            candidates.append(cell)
    rng.shuffle(candidates)
    while len(wall_cells) < wall_target and candidates:
        cell = candidates.pop()
        grid[cell] = WALL
        wall_cells.add(cell)


def _validate_generated(grid: np.ndarray) -> tuple[list[dict], list[int], list[list[dict]]]:
    g = np.asarray(grid, dtype=int)
    arrows, counts, traces = _simulate_all(g)
    arrow_cells = set().union(*(arrow["cells"] for arrow in arrows))

    assert 3 <= len(arrows) <= 5
    assert len({arrow["color"] for arrow in arrows}) == len(arrows)
    assert 8 <= int(np.count_nonzero(g == WALL)) <= 20
    assert all(0 <= int(count) <= 5 for count in counts)
    assert len(set(counts)) > 1

    for i, arrow in enumerate(arrows):
        for cell in arrow["cells"]:
            for nr, nc in _neighbors8(*cell):
                if (nr, nc) in arrow_cells and (nr, nc) not in arrow["cells"]:
                    raise AssertionError("arrows must be non-adjacent")
        for step in traces[i]:
            event = step["event"]
            if event == "front":
                assert len(step["walls"]) == 1
            if event in {"row", "col", "row+col"}:
                assert len(step["walls"]) >= 1

    total_bounces = sum(counts)
    event_kinds = {
        step["event"]
        for trace in traces
        for step in trace
        if step["event"] != "none"
    }
    if total_bounces >= 2:
        assert "front" in event_kinds
        assert any(kind in event_kinds for kind in ("row", "col", "row+col"))

    if len(set(counts)) < len(counts):
        out = main(g)
        dup_counts = {count for count in counts if counts.count(count) > 1}
        for dup in dup_counts:
            dup_arrows = [arrow for arrow, count in zip(arrows, counts) if count == dup]
            dup_arrows.sort(key=lambda a: (a["bbox"][0], a["bbox"][1]))
            expected_colors = [arrow["color"] for arrow in dup_arrows]
            sorted_indices = sorted(
                [i for i, count in enumerate(counts) if count == dup],
                key=lambda i: (counts[i], arrows[i]["bbox"][0], arrows[i]["bbox"][1]),
            )
            actual_colors = [arrows[i]["color"] for i in sorted_indices]
            assert actual_colors == expected_colors
            cols = [int(col) for col in range(out.shape[1]) if int(np.max(out[:, col])) in expected_colors or int(np.min(out[:, col])) in expected_colors]
            assert cols

    checked = 0
    for trace in traces:
        for step in trace:
            if step["event"] == "none":
                continue
            mod = g.copy()
            for wr, wc in step["walls"]:
                mod[wr, wc] = BG
            _, mod_counts, _ = _simulate_all(mod)
            if mod_counts != counts:
                checked += 1
                break
        if checked >= 2:
            break
    assert checked >= 1, "removing a bounce wall must change at least one count"

    return arrows, counts, traces


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 24)
    w = rng.randint(18, 24)
    n_arrows = rng.randint(3, 5)
    sampled = _sample_counts(rng, n_arrows)
    if sampled is None:
        return None
    desired_counts, want_duplicate = sampled
    front_idx = rng.randrange(n_arrows)
    desired_counts[front_idx] = 3
    if sum(desired_counts) > 16 or len(set(desired_counts)) == 1:
        return None

    all_arrow_cells: set[tuple[int, int]] = set()
    all_wall_cells: set[tuple[int, int]] = set()
    global_bg_guard: set[tuple[int, int]] = set()
    plans: list[dict] = []

    front_plan = _plan_front_macro_arrow(rng, (h, w), all_arrow_cells, all_wall_cells, global_bg_guard)
    if front_plan is None:
        return None
    plans.append(front_plan)
    all_arrow_cells |= front_plan["cells"]
    all_wall_cells |= front_plan["walls"]
    global_bg_guard |= front_plan["guard"]

    order = [i for i in range(n_arrows) if i != front_idx]
    rng.shuffle(order)
    for idx in order:
        plan = _plan_arrow(
            rng,
            (h, w),
            desired_counts[idx],
            all_arrow_cells,
            all_wall_cells,
            global_bg_guard,
            want_duplicate,
        )
        if plan is None:
            return None
        plans.append(plan)
        all_arrow_cells |= plan["cells"]
        all_wall_cells |= plan["walls"]
        global_bg_guard |= plan["guard"]

    used_types = {bounce_type for plan in plans for bounce_type in plan["bounce_types"]}
    if "front" not in used_types or not ({"row", "col"} & used_types):
        return None
    if len(all_wall_cells) > 20:
        return None

    colors = list(ARROW_PALETTE)
    rng.shuffle(colors)
    for plan, color in zip(plans, colors[:n_arrows]):
        plan["color"] = int(color)

    grid = np.zeros((h, w), dtype=int)
    for plan in plans:
        for r, c in plan["cells"]:
            grid[r, c] = int(plan["color"])
    for r, c in all_wall_cells:
        grid[r, c] = WALL

    wall_target = rng.randint(max(8, len(all_wall_cells)), 20)
    _add_distractor_walls(rng, (h, w), grid, all_arrow_cells, global_bg_guard, wall_target)

    try:
        arrows, counts, _ = _validate_generated(grid)
    except AssertionError:
        return None

    if counts != desired_counts and sorted(counts) != sorted(desired_counts):
        return None
    if want_duplicate and len(set(counts)) == len(counts):
        return None
    if not want_duplicate and len(set(counts)) < len(counts) and rng.random() < 0.6:
        return None

    out = main(grid)
    if out.shape[1] != len(arrows):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
