from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]

CORE_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
DIRECTIONS = ("up", "right", "down", "left")
DIR_TO_VEC = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
# This is the 90-degree rotation orbit of one canonical nub placement.
DIR_TO_NUB = {
    "up": (-1, 0),
    "right": (0, 2),
    "down": (2, 1),
    "left": (1, -1),
}

GEN_ROW_RANGE = (14, 23)
GEN_COL_RANGE = (14, 23)
GEN_CHAIN_LENGTHS = (3,)
GEN_LAST_QUOTA = {
    "quota": (1, 2),
    "edge": (2, 3),
}
GEN_HANDOFF_MOVE_RANGE = (1, 2)
GEN_EXTRA_COUNT = (0, 1)
GEN_MAX_ATTEMPTS = 500


class LayoutError(ValueError):
    pass


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    colors: list[int] | tuple[int, ...] | None = None,
) -> list[dict]:
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)
    height, width = map(int, grid.shape)
    if colors is None:
        colors_iter = sorted(set(map(int, grid.ravel())) - {bg})
    else:
        colors_iter = [int(color) for color in colors if int(color) != bg]

    out = []
    for color in colors_iter:
        mask = grid == int(color)
        seen = np.zeros_like(mask, dtype=bool)
        for start_r, start_c in np.argwhere(mask):
            start_r = int(start_r)
            start_c = int(start_c)
            if seen[start_r, start_c]:
                continue
            q = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            coords = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _shape_cells(anchor: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    ar, ac = map(int, anchor)
    cells = {(ar + dr, ac + dc) for dr, dc in CORE_CELLS}
    nr, nc = DIR_TO_NUB[direction]
    cells.add((ar + nr, ac + nc))
    return cells


def _cells_in_bounds(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    return all(0 <= r < height and 0 <= c < width for r, c in cells)


def _all_anchors(direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    rel = list(_shape_cells((0, 0), direction))
    min_r = min(r for r, _c in rel)
    max_r = max(r for r, _c in rel)
    min_c = min(c for _r, c in rel)
    max_c = max(c for _r, c in rel)
    height, width = map(int, shape)
    rows = range(-min_r, height - max_r)
    cols = range(-min_c, width - max_c)
    return [(int(r), int(c)) for r in rows for c in cols]


def _object_cells(obj: dict, anchor: tuple[int, int] | None = None) -> set[tuple[int, int]]:
    return _shape_cells(obj["anchor"] if anchor is None else anchor, obj["direction"])


def _distance_to_edge(obj: dict, shape: tuple[int, int]) -> int:
    dr, dc = DIR_TO_VEC[obj["direction"]]
    cells = _object_cells(obj)
    if dr == -1:
        return min(r for r, _c in cells)
    if dr == 1:
        return min(shape[0] - 1 - r for r, _c in cells)
    if dc == -1:
        return min(c for _r, c in cells)
    return min(shape[1] - 1 - c for _r, c in cells)


def _touches_or_overlaps(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _render(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for obj in objects:
        for r, c in _object_cells(obj):
            assert out[r, c] == BLACK, "objects must not overlap"
            out[r, c] = int(obj["color"])
    return out


def _parse_object(component: dict) -> dict:
    coords = {tuple(map(int, rc)) for rc in component["coords"]}
    assert len(coords) == 5, "every foreground object must have five pixels"
    candidates = []
    for r, c in coords:
        core = {(r + dr, c + dc) for dr, dc in CORE_CELLS}
        if core <= coords:
            candidates.append((int(r), int(c)))
    assert len(candidates) == 1, "every object must contain exactly one 2x2 core"
    anchor = candidates[0]
    extra = list(coords - {(anchor[0] + dr, anchor[1] + dc) for dr, dc in CORE_CELLS})
    assert len(extra) == 1, "expected exactly one nub pixel"
    nub_offset = (extra[0][0] - anchor[0], extra[0][1] - anchor[1])
    direction_matches = [direction for direction, offset in DIR_TO_NUB.items() if offset == nub_offset]
    assert len(direction_matches) == 1, "nub must indicate a unique direction"
    return {
        "color": int(component["color"]),
        "anchor": anchor,
        "direction": direction_matches[0],
    }


def _parse_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == BLACK, "expected black background"

    components = find_components(g, bg=bg)
    assert len(components) >= 2, "expected at least two objects"
    colors = [int(comp["color"]) for comp in components]
    assert len(colors) == len(set(colors)), "object colors must be unique"
    assert colors.count(RED) == 1, "expected exactly one red trigger object"

    objects = [_parse_object(comp) for comp in components]
    for obj in objects:
        assert _cells_in_bounds(_object_cells(obj), g.shape)
    return objects


def _advance_one(
    objects: list[dict],
    idx: int,
    move_direction: str,
    step_cap: int | None,
    shape: tuple[int, int],
    *,
    record_steps: bool = False,
) -> tuple[int, str, int | None, list[tuple[int, int]]]:
    dr, dc = DIR_TO_VEC[move_direction]
    moved = 0
    anchor = tuple(map(int, objects[idx]["anchor"]))
    steps = []

    while step_cap is None or moved < int(step_cap):
        next_anchor = (anchor[0] + dr, anchor[1] + dc)
        next_cells = _object_cells(objects[idx], next_anchor)
        if not _cells_in_bounds(next_cells, shape):
            return moved, "edge", None, steps

        colliders = []
        for other_idx, other in enumerate(objects):
            if other_idx == idx:
                continue
            if next_cells & _object_cells(other):
                colliders.append(other_idx)
        if len(colliders) > 1:
            raise LayoutError("a move would collide with more than one blocker")
        if colliders:
            return moved, "block", int(colliders[0]), steps

        anchor = next_anchor
        objects[idx]["anchor"] = anchor
        moved += 1
        if record_steps:
            steps.append(anchor)

    return moved, "quota", None, steps


def _simulate(
    objects: list[dict],
    shape: tuple[int, int],
    *,
    direction_mode: str = "self",
    movement_mode: str = "quota",
    transfer_mode: str = "remainder",
    record_visited: bool = False,
) -> tuple[list[dict], list[dict], set[tuple[int, int]]]:
    states = [{**obj, "anchor": tuple(map(int, obj["anchor"]))} for obj in objects]
    red_candidates = [idx for idx, obj in enumerate(states) if int(obj["color"]) == RED]
    assert len(red_candidates) == 1

    current = red_candidates[0]
    incoming_direction = None
    active_quota = None
    trace = []
    visited = set()
    if record_visited:
        for obj in states:
            visited |= _object_cells(obj)

    while True:
        obj = states[current]
        if current == red_candidates[0]:
            move_direction = obj["direction"]
            step_cap = None
        else:
            move_direction = incoming_direction if direction_mode == "incoming" else obj["direction"]
            step_cap = None if movement_mode == "unbounded" else int(active_quota)

        moved, event, blocker, anchors = _advance_one(
            states,
            current,
            move_direction,
            step_cap,
            shape,
            record_steps=record_visited,
        )
        if record_visited:
            for anchor in anchors:
                visited |= _object_cells(states[current], anchor)

        next_quota = None
        if event == "block":
            if current == red_candidates[0] or movement_mode == "unbounded":
                next_quota = int(moved)
            elif transfer_mode == "full":
                next_quota = int(active_quota)
            else:
                next_quota = int(active_quota) - int(moved)

        trace.append(
            {
                "color": int(obj["color"]),
                "object_direction": obj["direction"],
                "used_direction": move_direction,
                "quota_in": None if step_cap is None else int(step_cap),
                "moved": int(moved),
                "event": event,
                "blocker": blocker,
                "quota_out": None if next_quota is None else int(next_quota),
            }
        )

        if event != "block" or next_quota is None or next_quota <= 0:
            break
        current = int(blocker)
        incoming_direction = move_direction
        active_quota = int(next_quota)

    return states, trace, visited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the red nub-block first, then relay its traveled-cell quota through the collision chain.

    Input structure:
    - A black background contains several monochromatic 5-pixel objects.
    - Each object is a solid 2x2 block plus one same-color nub pixel attached orthogonally to one side; that nub side is the object's facing direction.
    - Exactly one object is red; every other object has a different color, and the layout keeps every prospective collision unique.

    Rule:
    - Move the red object one cell at a time along its own facing direction until the next step would leave the grid or overlap another object.
    - If it slides d cells before hitting another object, that blocker receives quota d and moves along its own facing direction for at most d cells.
    - If a quota-driven object spends only k<d cells before the next step would hit another object, it stops there and passes the remaining quota d-k to that blocker.
    - The relay ends when quota is exhausted, or when the currently moving object reaches the grid edge; untouched objects stay fixed and all shapes/colors are preserved.

    Key insight:
    - The moving direction always comes from the struck object's own nub, while the travel budget comes from the previous object's actual motion, so both direction and leftover quota must be tracked together.
    """
    g = np.asarray(grid, dtype=int)
    objects = _parse_grid(g)
    final_objects, _trace, _visited = _simulate(objects, g.shape)
    return _render(final_objects, g.shape)


def _sample_chain_spec() -> dict:
    terminal_mode = random.choice(("quota", "edge"))
    chain_len = random.choice(GEN_CHAIN_LENGTHS)
    q_lo, q_hi = GEN_LAST_QUOTA[terminal_mode]
    q_last = random.randint(q_lo, q_hi)
    handoff_moves = [random.randint(*GEN_HANDOFF_MOVE_RANGE) for _ in range(chain_len - 2)]
    red_distance = q_last + sum(handoff_moves)
    return {
        "terminal_mode": terminal_mode,
        "chain_len": chain_len,
        "q_last": q_last,
        "handoff_moves": handoff_moves,
        "red_distance": red_distance,
        "collision_moves": [red_distance] + handoff_moves,
    }


def _sample_directions(chain_len: int) -> list[str]:
    for _ in range(80):
        dirs = [random.choice(DIRECTIONS) for _ in range(chain_len)]
        if any(dirs[i] != dirs[i - 1] for i in range(1, chain_len)):
            return dirs
    raise LayoutError("failed to sample directional diversity")


def _pick_last_anchor(direction: str, shape: tuple[int, int], terminal_mode: str, q_last: int) -> tuple[int, int] | None:
    anchors = _all_anchors(direction, shape)
    random.shuffle(anchors)
    for anchor in anchors:
        obj = {"color": 1, "anchor": anchor, "direction": direction}
        edge_distance = _distance_to_edge(obj, shape)
        if terminal_mode == "quota" and edge_distance > q_last:
            return anchor
        if terminal_mode == "edge" and 1 <= edge_distance < q_last:
            return anchor
    return None


def _place_previous_object(
    suffix: list[dict],
    direction: str,
    move_before_hit: int,
    color: int,
    shape: tuple[int, int],
    terminal_mode: str,
) -> dict | None:
    anchors = _all_anchors(direction, shape)
    random.shuffle(anchors)
    suffix_cells = [_object_cells(obj) for obj in suffix]
    for anchor in anchors:
        candidate = {"color": int(color), "anchor": anchor, "direction": direction}
        candidate_cells = _object_cells(candidate)
        if any(_touches_or_overlaps(candidate_cells, cells) for cells in suffix_cells):
            continue
        trial = [{**candidate, "anchor": tuple(map(int, candidate["anchor"]))}] + [
            {**obj, "anchor": tuple(map(int, obj["anchor"]))} for obj in suffix
        ]
        try:
            moved, event, blocker, _steps = _advance_one(trial, 0, direction, None, shape)
        except LayoutError:
            continue
        if moved != move_before_hit or event != "block" or blocker != 1:
            continue
        if len(suffix) >= 2:
            try:
                relay_quota = int(moved)
                second_dir = trial[1]["direction"]
                second_moved, second_event, second_blocker, _steps = _advance_one(
                    trial,
                    1,
                    second_dir,
                    relay_quota,
                    shape,
                )
            except Exception:
                continue
            if second_event != "block" or second_blocker != 2:
                continue
            last_quota = relay_quota - int(second_moved)
            if last_quota <= 0:
                continue
            try:
                last_dir = trial[2]["direction"]
                _last_moved, last_event, _last_blocker, _steps = _advance_one(
                    trial,
                    2,
                    last_dir,
                    last_quota,
                    shape,
                )
            except Exception:
                continue
            if last_event != ("quota" if terminal_mode == "quota" else "edge"):
                continue
        return candidate
    return None


def _build_chain(shape: tuple[int, int], spec: dict, colors: list[int]) -> list[dict] | None:
    directions = _sample_directions(spec["chain_len"])
    last = {
        "color": colors[-1],
        "direction": directions[-1],
        "anchor": _pick_last_anchor(directions[-1], shape, spec["terminal_mode"], spec["q_last"]),
    }
    if last["anchor"] is None:
        return None

    suffix = [last]
    for idx in range(spec["chain_len"] - 2, -1, -1):
        color = RED if idx == 0 else colors[idx - 1]
        candidate = _place_previous_object(
            suffix,
            directions[idx],
            spec["collision_moves"][idx],
            color,
            shape,
            spec["terminal_mode"],
        )
        if candidate is None:
            return None
        suffix = [candidate] + suffix
    return suffix


def _trace_color_sequence(trace: list[dict]) -> list[int]:
    return [int(step["color"]) for step in trace]


def _variant_differs(
    parsed_objects: list[dict],
    shape: tuple[int, int],
    correct_output: np.ndarray,
    *,
    direction_mode: str = "self",
    movement_mode: str = "quota",
    transfer_mode: str = "remainder",
) -> bool:
    try:
        states, _trace, _visited = _simulate(
            parsed_objects,
            shape,
            direction_mode=direction_mode,
            movement_mode=movement_mode,
            transfer_mode=transfer_mode,
        )
        return not np.array_equal(_render(states, shape), correct_output)
    except Exception:
        return True


def _chain_trace_ok(chain_objects: list[dict], shape: tuple[int, int], spec: dict) -> bool:
    try:
        _final_states, trace, _visited = _simulate(chain_objects, shape)
    except Exception:
        return False
    expected_colors = [int(obj["color"]) for obj in chain_objects]
    if _trace_color_sequence(trace) != expected_colors:
        return False
    if len(trace) != spec["chain_len"]:
        return False
    if trace[0]["moved"] != spec["red_distance"] or trace[0]["event"] != "block":
        return False
    return trace[-1]["event"] == ("quota" if spec["terminal_mode"] == "quota" else "edge")


def _semantic_ok(grid: np.ndarray, chain_objects: list[dict], spec: dict) -> bool:
    try:
        parsed = _parse_grid(grid)
        final_states, trace, visited = _simulate(parsed, grid.shape, record_visited=True)
    except Exception:
        return False

    expected_colors = [int(obj["color"]) for obj in chain_objects]
    if _trace_color_sequence(trace) != expected_colors:
        return False
    if len(trace) != spec["chain_len"]:
        return False
    if trace[-1]["event"] != ("quota" if spec["terminal_mode"] == "quota" else "edge"):
        return False
    if trace[0]["moved"] != spec["red_distance"] or trace[0]["event"] != "block":
        return False
    if spec["terminal_mode"] == "quota":
        last_obj = next(obj for obj in final_states if int(obj["color"]) == expected_colors[-1])
        if _distance_to_edge(last_obj, grid.shape) <= 0:
            return False
    if not any(trace[i]["used_direction"] != trace[i - 1]["used_direction"] for i in range(1, len(trace))):
        return False

    correct_output = _render(final_states, grid.shape)
    same_direction_diff = _variant_differs(
        parsed,
        grid.shape,
        correct_output,
        direction_mode="incoming",
        movement_mode="quota",
        transfer_mode="remainder",
    )
    if not same_direction_diff:
        return False
    if spec["terminal_mode"] == "quota":
        quota_diff = _variant_differs(
            parsed,
            grid.shape,
            correct_output,
            direction_mode="self",
            movement_mode="unbounded",
            transfer_mode="remainder",
        ) or _variant_differs(
            parsed,
            grid.shape,
            correct_output,
            direction_mode="self",
            movement_mode="quota",
            transfer_mode="full",
        )
        if not quota_diff:
            return False

    direct_output = main(grid)
    if not np.array_equal(direct_output, correct_output):
        return False

    initial_chain_colors = {int(obj["color"]) for obj in chain_objects}
    distractor_count = sum(1 for obj in parsed if int(obj["color"]) not in initial_chain_colors)
    if distractor_count > 0 and len(parsed) == len(chain_objects):
        return False
    return len(visited) > 0


def _place_distractors(
    chain_objects: list[dict],
    shape: tuple[int, int],
    remaining_colors: list[int],
) -> list[dict]:
    states, trace, visited = _simulate(chain_objects, shape, record_visited=True)
    assert len(trace) >= 1

    extras = [{**obj, "anchor": tuple(map(int, obj["anchor"]))} for obj in chain_objects]
    extra_count = random.randint(*GEN_EXTRA_COUNT)
    color_pool = remaining_colors[:]
    random.shuffle(color_pool)
    initial_cells = [_object_cells(obj) for obj in extras]

    for color in color_pool:
        if extra_count <= 0:
            break
        directions = list(DIRECTIONS)
        random.shuffle(directions)
        placed = False
        for direction in directions:
            anchors = _all_anchors(direction, shape)
            random.shuffle(anchors)
            for anchor in anchors:
                candidate = {"color": int(color), "anchor": anchor, "direction": direction}
                cells = _object_cells(candidate)
                if cells & visited:
                    continue
                if any(_touches_or_overlaps(cells, prev) for prev in initial_cells):
                    continue
                extras.append(candidate)
                initial_cells.append(cells)
                extra_count -= 1
                placed = True
                break
            if placed:
                break

    return extras


def _attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GEN_ROW_RANGE),
        random.randint(*GEN_COL_RANGE),
    )
    spec = _sample_chain_spec()
    total_objects = spec["chain_len"] + GEN_EXTRA_COUNT[1]
    color_sample = random.sample(PALETTE, total_objects)
    chain_colors = color_sample[: spec["chain_len"] - 1]
    distractor_colors = color_sample[spec["chain_len"] - 1 :]
    chain = _build_chain(shape, spec, chain_colors)
    if chain is None:
        return None
    if not _chain_trace_ok(chain, shape, spec):
        return None

    objects = _place_distractors(chain, shape, distractor_colors)
    grid = _render(objects, shape)
    return grid if _semantic_ok(grid, chain, spec) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid momentum-relay input")
