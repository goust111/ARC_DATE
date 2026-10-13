from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
CUP_COLORS = (1, 2, 3, 6, 7, 8, 9)
OPENINGS = ("up", "right", "down", "left")
DIR_VECTORS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0a, c0a, r1a, c1a = a
    r0b, c0b, r1b, c1b = b
    dr = max(0, r0b - r1a - 1, r0a - r1b - 1)
    dc = max(0, c0b - c1a - 1, c0a - c1b - 1)
    return dr + dc


def _components_by_color(grid: np.ndarray) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[dict] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _opening_info(local_mask: np.ndarray) -> tuple[str, list[tuple[int, int]]]:
    h, w = local_mask.shape
    gaps = {
        "up": [(0, c) for c in range(w) if not local_mask[0, c]],
        "right": [(r, w - 1) for r in range(h) if not local_mask[r, w - 1]],
        "down": [(h - 1, c) for c in range(w) if not local_mask[h - 1, c]],
        "left": [(r, 0) for r in range(h) if not local_mask[r, 0]],
    }
    present = [(side, coords) for side, coords in gaps.items() if coords]
    assert len(present) == 1, "expected exactly one cup opening side"
    side, coords = present[0]
    if side in ("up", "down"):
        coords = sorted(coords, key=lambda p: p[1])
    else:
        coords = sorted(coords, key=lambda p: p[0])
    return side, coords


def _fill_from_opposite_side(grid: np.ndarray, bbox: tuple[int, int, int, int], opening: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    seeds: list[tuple[int, int]] = []
    if opening == "up":
        if r1 - r0 < 2:
            return set()
        seeds = [(r1 - 1, c) for c in range(c0 + 1, c1)]
    elif opening == "down":
        if r1 - r0 < 2:
            return set()
        seeds = [(r0 + 1, c) for c in range(c0 + 1, c1)]
    elif opening == "left":
        if c1 - c0 < 2:
            return set()
        seeds = [(r, c1 - 1) for r in range(r0 + 1, r1)]
    else:
        if c1 - c0 < 2:
            return set()
        seeds = [(r, c0 + 1) for r in range(r0 + 1, r1)]

    q = deque()
    seen: set[tuple[int, int]] = set()
    for cell in seeds:
        rr, cc = cell
        if int(grid[rr, cc]) == BLACK and cell not in seen:
            seen.add(cell)
            q.append(cell)

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if nr <= r0 or nr >= r1 or nc <= c0 or nc >= c1:
                continue
            if int(grid[nr, nc]) != BLACK or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def _analyze_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for comp in _components_by_color(grid):
        if comp["color"] == GREY and comp["area"] == 1:
            continue
        assert comp["area"] > 1, "non-cup foreground should only be isolated grey blockers"
        r0, c0, r1, c1 = comp["bbox"]
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local = comp["coords"] - np.array([r0, c0])
        local_mask[local[:, 0], local[:, 1]] = True
        opening, gap_local = _opening_info(local_mask)
        gap_abs = [(r0 + rr, c0 + cc) for rr, cc in gap_local]
        fill = _fill_from_opposite_side(grid, comp["bbox"], opening)
        mid = gap_abs[(len(gap_abs) - 1) // 2]
        dr, dc = DIR_VECTORS[opening]
        start = (mid[0] + dr, mid[1] + dc)
        cups.append(
            {
                "color": int(comp["color"]),
                "coords": {(int(r), int(c)) for r, c in comp["coords"]},
                "bbox": comp["bbox"],
                "area": int(comp["area"]),
                "opening": opening,
                "gap": gap_abs,
                "start": start,
                "fill": fill,
            }
        )
    cups.sort(key=lambda cup: (cup["area"], cup["bbox"][0], cup["bbox"][1]))
    for idx, cup in enumerate(cups):
        cup["id"] = idx
    return cups


def _spiral_directions(opening: str) -> list[tuple[int, int]]:
    order = list(OPENINGS)
    start = order.index(opening)
    return [DIR_VECTORS[order[(start + i) % 4]] for i in range(4)]


def _classify_blocker(
    blocker: tuple[int, int],
    cup: dict,
    cups: list[dict],
    spiral_owner: dict[tuple[int, int], int],
) -> tuple[str, int | None]:
    if blocker in spiral_owner:
        owner = spiral_owner[blocker]
        return ("self_spiral" if owner == cup["id"] else "prior_spiral"), owner
    for other in cups:
        if blocker in other["coords"] or blocker in other["fill"]:
            return ("self_cup" if other["id"] == cup["id"] else "other_cup"), other["id"]
    return ("grey", None)


def _solve(grid: np.ndarray, *, trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    cups = _analyze_cups(g)
    assert 2 <= len(cups) <= 3, "expected 2-3 cups"

    out = g.copy()
    for cup in cups:
        for r, c in cup["fill"]:
            out[r, c] = YELLOW

    spiral_owner: dict[tuple[int, int], int] = {}
    traces: list[dict] = []
    for cup in cups:
        directions = _spiral_directions(cup["opening"])
        r, c = cup["start"]
        segment_index = 0
        empty_segments = 0
        painted: list[tuple[int, int]] = []
        stop_reason = "out_of_bounds"
        stop_cell: tuple[int, int] | None = None
        blocker_owner: int | None = None

        while empty_segments < 4:
            dr, dc = directions[segment_index % 4]
            leg_length = 1 + segment_index // 2
            had_in_bounds = False
            for _ in range(leg_length):
                r += dr
                c += dc
                if not _in_bounds(out.shape, r, c):
                    continue
                had_in_bounds = True
                if int(out[r, c]) != BLACK:
                    stop_cell = (r, c)
                    stop_reason, blocker_owner = _classify_blocker(stop_cell, cup, cups, spiral_owner)
                    empty_segments = 4
                    break
                out[r, c] = cup["color"]
                spiral_owner[(r, c)] = cup["id"]
                painted.append((r, c))
            if stop_cell is not None:
                break
            empty_segments = 0 if had_in_bounds else empty_segments + 1
            segment_index += 1

        traces.append(
            {
                "cup_id": cup["id"],
                "color": cup["color"],
                "area": cup["area"],
                "bbox": cup["bbox"],
                "opening": cup["opening"],
                "painted": painted,
                "painted_count": len(painted),
                "stop_reason": stop_reason,
                "stop_cell": stop_cell,
                "blocker_owner": blocker_owner,
            }
        )

    if trace:
        return out, cups, traces
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each cup and then draw an opening-directed spiral whose ordering determines later blockage.

    Input structure:
    - The grid is a black background containing 2-3 monochrome one-cell-thick cup outlines in distinct non-grey colors plus isolated grey blocker pixels.
    - Each cup has exactly one opening on one side of its bounding box, and the cups have different wall areas and different opening directions.
    - Cup interiors and openings are black in the input, cups are separated from each other, and grey blockers are isolated background pixels.

    Rule:
    - For each cup, detect the unique bbox side containing the opening, then fill every black cell inside the cup that is 4-connected to the inner line on the side opposite that opening with yellow.
    - Sort the cups by wall-pixel count ascending, breaking ties by the bbox top-left corner.
    - From the midpoint of each opening, trace a square spiral whose first segment points along the opening direction, then rotates clockwise with segment lengths 1, 1, 2, 2, 3, 3, ...; paint only black cells with the cup color, skip out-of-bounds steps while still consuming them, and stop immediately before the first non-black cell.

    Key insight:
    - The opening is reused as the spiral's orientation seed, while the area-based processing order makes earlier spirals become new blockers for later cups.
    """
    return _solve(grid, trace=False)


def _cup_patch(height: int, width: int, opening: str, gap_start: int, gap_len: int, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if opening == "up":
        patch[0, gap_start : gap_start + gap_len] = BLACK
    elif opening == "down":
        patch[-1, gap_start : gap_start + gap_len] = BLACK
    elif opening == "left":
        patch[gap_start : gap_start + gap_len, 0] = BLACK
    else:
        patch[gap_start : gap_start + gap_len, -1] = BLACK
    return patch


def _local_start(cup: dict) -> tuple[int, int]:
    if cup["opening"] == "up":
        return -1, cup["gap_start"] + (cup["gap_len"] - 1) // 2
    if cup["opening"] == "down":
        return cup["height"], cup["gap_start"] + (cup["gap_len"] - 1) // 2
    if cup["opening"] == "left":
        return cup["gap_start"] + (cup["gap_len"] - 1) // 2, -1
    return cup["gap_start"] + (cup["gap_len"] - 1) // 2, cup["width"]


def _local_blocking_cells(cup: dict) -> list[tuple[int, int]]:
    patch = _cup_patch(cup["height"], cup["width"], cup["opening"], cup["gap_start"], cup["gap_len"], cup["color"])
    wall = [(int(r), int(c)) for r, c in np.argwhere(patch == cup["color"])]
    fill = sorted(_fill_from_opposite_side(patch, (0, 0, cup["height"] - 1, cup["width"] - 1), cup["opening"]))
    return wall + [cell for cell in fill if cell not in wall]


def _gap_choices(height: int, width: int, opening: str, gap_len: int) -> list[int]:
    side_len = width if opening in ("up", "down") else height
    return list(range(1, side_len - gap_len))


def _cup_area(height: int, width: int, gap_len: int) -> int:
    return 2 * height + 2 * width - 4 - gap_len


def _cup_specs_for_opening(opening: str) -> list[dict]:
    specs: list[dict] = []
    for height in range(4, 9):
        for width in range(4, 10):
            side_len = width if opening in ("up", "down") else height
            gap_len = side_len - 2
            area = _cup_area(height, width, gap_len)
            if 8 <= area <= 16 and _gap_choices(height, width, opening, gap_len):
                specs.append(
                    {
                        "height": height,
                        "width": width,
                        "gap_len": gap_len,
                        "area": area,
                    }
                )
    return specs


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _adjacent_to_nonblack(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) != BLACK:
                return True
    return False


def _possible_positions(
    shape: tuple[int, int],
    size: tuple[int, int],
    placed_boxes: list[tuple[int, int, int, int]],
    *,
    min_gap: int,
) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = size
    out: list[tuple[int, int]] = []
    for top in range(1, h - ph):
        for left in range(1, w - pw):
            box = (top, left, top + ph - 1, left + pw - 1)
            if all(_bbox_distance(box, old) >= min_gap for old in placed_boxes):
                out.append((top, left))
    return out


def _candidate_cups(count: int) -> list[dict] | None:
    openings = random.sample(OPENINGS, count)
    records: list[dict] = []
    used_areas: set[int] = set()
    palette = random.sample(list(CUP_COLORS), count)
    for i, opening in enumerate(openings):
        specs = [spec for spec in _cup_specs_for_opening(opening) if spec["area"] not in used_areas]
        if not specs:
            return None
        chosen = random.choice(specs)
        used_areas.add(chosen["area"])
        gap_start = random.choice(_gap_choices(chosen["height"], chosen["width"], opening, chosen["gap_len"]))
        records.append(
            {
                **chosen,
                "opening": opening,
                "gap_start": gap_start,
                "color": palette[i],
            }
        )
    records.sort(key=lambda rec: rec["area"])
    if len({rec["opening"] for rec in records}) != count:
        return None
    if len({rec["area"] for rec in records}) != count:
        return None
    return records


def _build_grid(shape: tuple[int, int], cups: list[dict], placements: list[tuple[int, int]], obstacles: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for cup, (top, left) in zip(cups, placements):
        patch = _cup_patch(cup["height"], cup["width"], cup["opening"], cup["gap_start"], cup["gap_len"], cup["color"])
        _place_patch(grid, patch, top, left)
    for r, c in obstacles:
        grid[r, c] = GREY
    return grid


def _interaction_ok(traces: list[dict]) -> bool:
    return any(trace["stop_reason"] in {"other_cup", "prior_spiral"} for trace in traces)


def _solve_ok(grid: np.ndarray) -> bool:
    try:
        out, cups, traces = _solve(grid, trace=True)
    except Exception:
        return False
    if out.shape != grid.shape or len(cups) < 2:
        return False
    if len({cup["opening"] for cup in cups}) != len(cups):
        return False
    if len({cup["area"] for cup in cups}) != len(cups):
        return False
    if not _interaction_ok(traces):
        return False
    if not all(trace["painted_count"] >= 1 for trace in traces):
        return False
    if not any(trace["painted_count"] >= 4 for trace in traces):
        return False
    return True


def _pair_with_required_interaction(shape: tuple[int, int], cups: list[dict]) -> list[tuple[int, int]] | None:
    first, second = cups[0], cups[1]
    second_positions = _possible_positions(shape, (second["height"], second["width"]), [], min_gap=0)
    if not second_positions:
        return None
    center_r = shape[0] / 2
    center_c = shape[1] / 2
    second_positions.sort(
        key=lambda rc: abs((rc[0] + second["height"] / 2) - center_r) + abs((rc[1] + second["width"] / 2) - center_c)
    )
    second_positions = second_positions[:60]
    first_blockers = _local_blocking_cells(first)
    for pos1 in second_positions:
        start_local = _local_start(second)
        start_abs = (pos1[0] + start_local[0], pos1[1] + start_local[1])
        path = _spiral_probe(start_abs, second["opening"], shape)
        if len(path) < 8:
            continue
        box1 = (pos1[0], pos1[1], pos1[0] + second["height"] - 1, pos1[1] + second["width"] - 1)
        target_cells = [
            cell
            for cell in path[8:70]
            if max(abs(cell[0] - start_abs[0]), abs(cell[1] - start_abs[1])) >= 5
        ]
        for tr, tc in target_cells:
            for ar, ac in first_blockers:
                pos0 = (tr - ar, tc - ac)
                top, left = pos0
                if not (1 <= top <= shape[0] - first["height"] - 1 and 1 <= left <= shape[1] - first["width"] - 1):
                    continue
                box0 = (top, left, top + first["height"] - 1, left + first["width"] - 1)
                if _bbox_distance(box0, box1) < 4:
                    continue
                grid = _build_grid(shape, cups[:2], [pos0, pos1], set())
                try:
                    _, _, traces = _solve(grid, trace=True)
                except Exception:
                    continue
                if len(traces) >= 2 and traces[1]["stop_reason"] in {"other_cup", "prior_spiral"}:
                    return [pos0, pos1]
    return None


def _spiral_probe(start: tuple[int, int], opening: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    directions = _spiral_directions(opening)
    segment_index = 0
    empty_segments = 0
    visited: list[tuple[int, int]] = []
    while empty_segments < 4 and len(visited) < 80:
        dr, dc = directions[segment_index % 4]
        leg_length = 1 + segment_index // 2
        had_in_bounds = False
        for _ in range(leg_length):
            r += dr
            c += dc
            if _in_bounds(shape, r, c):
                visited.append((r, c))
                had_in_bounds = True
        empty_segments = 0 if had_in_bounds else empty_segments + 1
        segment_index += 1
    return visited


def _add_optional_cups(shape: tuple[int, int], cups: list[dict], placements: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    placed_boxes = [
        (top, left, top + cup["height"] - 1, left + cup["width"] - 1)
        for cup, (top, left) in zip(cups[: len(placements)], placements)
    ]
    for cup in cups[len(placements) :]:
        positions = _possible_positions(shape, (cup["height"], cup["width"]), placed_boxes, min_gap=4)
        random.shuffle(positions)
        if not positions:
            return None
        chosen = None
        for pos in positions[:150]:
            trial = placements + [pos]
            grid = _build_grid(shape, cups[: len(trial)], trial, set())
            if _solve_ok(grid):
                chosen = pos
                break
        if chosen is None:
            return None
        placements.append(chosen)
        placed_boxes.append((chosen[0], chosen[1], chosen[0] + cup["height"] - 1, chosen[1] + cup["width"] - 1))
    return placements


def _obstacle_candidates(grid: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK:
                continue
            if _adjacent_to_nonblack(grid, r, c):
                continue
            candidates.append((r, c))
    return candidates


def _try_add_targeted_grey(grid: np.ndarray) -> set[tuple[int, int]]:
    obstacles: set[tuple[int, int]] = set()
    if random.random() >= 0.45:
        return obstacles
    _, cups, traces = _solve(grid, trace=True)
    blocked_ids = {trace["cup_id"] for trace in traces if trace["stop_reason"] in {"other_cup", "prior_spiral"}}
    for trace in reversed(traces):
        if trace["cup_id"] in blocked_ids or trace["painted_count"] < 6:
            continue
        candidate = trace["painted"][random.randint(3, min(5, trace["painted_count"] - 1))]
        if int(grid[candidate]) == BLACK and not _adjacent_to_nonblack(grid, *candidate):
            obstacles.add(candidate)
            break
    return obstacles


def _add_random_obstacles(grid: np.ndarray, existing: set[tuple[int, int]]) -> set[tuple[int, int]]:
    obstacles = set(existing)
    candidates = _obstacle_candidates(grid)
    candidates = [cell for cell in candidates if cell not in obstacles]
    random.shuffle(candidates)
    target_total = random.randint(2, 6)
    for cell in list(obstacles):
        if cell in candidates:
            candidates.remove(cell)
    for r, c in candidates:
        if len(obstacles) >= target_total:
            break
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in obstacles):
            continue
        obstacles.add((r, c))
    return obstacles


def _random_cup_placements(shape: tuple[int, int], cups: list[dict]) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    for cup in cups:
        positions = _possible_positions(shape, (cup["height"], cup["width"]), placed_boxes, min_gap=4)
        random.shuffle(positions)
        if not positions:
            return None
        top, left = positions[0]
        placements.append((top, left))
        placed_boxes.append((top, left, top + cup["height"] - 1, left + cup["width"] - 1))
    return placements


def _valid_input_semantics(grid: np.ndarray) -> bool:
    try:
        out, cups, traces = _solve(grid, trace=True)
    except Exception:
        return False
    if not np.any(out != grid):
        return False
    if len(cups) not in (2, 3):
        return False
    if len({cup["opening"] for cup in cups}) != len(cups):
        return False
    if len({cup["area"] for cup in cups}) != len(cups):
        return False
    if not _interaction_ok(traces):
        return False
    if not all(trace["painted_count"] >= 1 for trace in traces):
        return False
    if not any(trace["stop_reason"] == "other_cup" for trace in traces) and not any(
        trace["stop_reason"] == "prior_spiral" for trace in traces
    ):
        return False

    comps = _components_by_color(grid)
    for comp in comps:
        if comp["color"] == GREY:
            if comp["area"] != 1:
                return False
            r, c = map(int, comp["coords"][0])
            if _adjacent_to_nonblack(np.where(grid == GREY, BLACK, grid), r, c):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(20, 26), random.randint(20, 26))
    cup_count = 2 if random.random() < 0.85 else 3
    cups = _candidate_cups(cup_count)
    if cups is None:
        return None
    placements = _pair_with_required_interaction(shape, cups[:2])
    if placements is None:
        placements = _random_cup_placements(shape, cups)
        if placements is None:
            return None
    if cup_count == 3:
        if len(placements) == cup_count:
            pass
        else:
            placements = _add_optional_cups(shape, cups, placements)
            if placements is None:
                return None

    cup_grid = _build_grid(shape, cups, placements, set())
    try:
        cup_out = main(cup_grid)
    except Exception:
        return None
    if not np.any(cup_out != cup_grid):
        return None

    if not _solve_ok(cup_grid):
        return cup_grid

    obstacles = _try_add_targeted_grey(cup_grid)
    full_grid = _build_grid(shape, cups, placements, obstacles)
    obstacles = _add_random_obstacles(full_grid, obstacles)
    full_grid = _build_grid(shape, cups, placements, obstacles)

    if _valid_input_semantics(full_grid) and np.array_equal(main(full_grid), _solve(full_grid, trace=False)):
        return full_grid
    return cup_grid


def generate_input() -> np.ndarray:
    for _ in range(2000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
