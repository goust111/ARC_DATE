from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, in_bounds


BORDER_COLORS = (Color.BLUE, Color.GREEN, Color.ORANGE)
DIR_NAMES = ("U", "R", "D", "L")
DIR_TO_VEC = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
STATIC_BLOCKERS = {Color.GRAY, Color.BLUE, Color.GREEN, Color.ORANGE}

GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (14, 20)
NUM_CUPS_CHOICES = (2, 3)
CONTENT_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 150


def _cw(direction: str) -> str:
    idx = DIR_NAMES.index(direction)
    return DIR_NAMES[(idx + 1) % 4]


def _neighbors8(cell: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = cell
    return {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    }


def _expand8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for cell in cells:
        out |= _neighbors8(cell)
    return out


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    seen = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR_TO_VEC.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in seen or (nr, nc) in border_set:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return seen


def _detect_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for border_color in BORDER_COLORS:
        for coords in components_mask(grid == int(border_color), connectivity=8):
            if int(coords.shape[0]) < 5:
                continue
            bbox = tuple(map(int, bbox_coords(coords)))
            r0, c0, r1, c1 = bbox
            border_set = {(int(r), int(c)) for r, c in coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            edges = {
                "U": [p for p in border_set if p[0] == r0],
                "D": [p for p in border_set if p[0] == r1],
                "L": [p for p in border_set if p[1] == c0],
                "R": [p for p in border_set if p[1] == c1],
            }
            candidates = [
                (side, edge_cells[0])
                for side, edge_cells in edges.items()
                if len(edge_cells) == 1 and edge_cells[0] not in corners
            ]

            passing = []
            for side, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[side])
                visible = {int(grid[r, c]) for r, c in mouth if int(grid[r, c]) not in {Color.BLACK, int(border_color)}}
                if visible != {Color.RED}:
                    continue
                content_cells = sorted((int(r), int(c)) for r, c in mouth if int(grid[r, c]) == Color.RED)
                if not content_cells:
                    continue
                passing.append(
                    {
                        "border_color": int(border_color),
                        "tail_dir": side,
                        "spike": (int(spike[0]), int(spike[1])),
                        "content_cells": content_cells,
                        "component_cells": sorted((int(r), int(c)) for r, c in border_set),
                        "bbox": bbox,
                    }
                )
            if len(passing) == 1:
                cups.append(passing[0])
    cups.sort(key=lambda item: item["bbox"])
    return cups


def _simulate_trail(grid: np.ndarray, spike: tuple[int, int], tail_dir: str, steps: int) -> list[tuple[int, int]]:
    current = (int(spike[0]), int(spike[1]))
    direction = str(tail_dir)
    trail: list[tuple[int, int]] = []
    for _ in range(int(steps)):
        dr, dc = DIR_TO_VEC[direction]
        nr, nc = current[0] + dr, current[1] + dc
        blocked = not in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) in STATIC_BLOCKERS
        if blocked:
            direction = _cw(direction)
            dr, dc = DIR_TO_VEC[direction]
            nr, nc = current[0] + dr, current[1] + dc
            blocked = not in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) in STATIC_BLOCKERS
            if blocked:
                break
        current = (int(nr), int(nc))
        trail.append(current)
    return trail


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject red cup contents outward from each tail, turning clockwise when gray blockers obstruct the trail.

    Input structure:
    - The background is black.
    - There are 2-3 hollow cup-shaped monochrome frames in blue, green, or orange, each with a single sharp tail
      spike and an open mouth on the opposite side.
    - Red content pixels sit inside each cup mouth, and gray single-pixel blockers lie outside the cups.

    Rule:
    - Detect each cup's unique tail spike and its outward cardinal direction.
    - Erase the red content pixels inside the cup and count how many were removed.
    - Starting from the spike, emit that many red trail steps outward: advance one cell in the current direction,
      but if that cell is a gray blocker or outside the grid, turn clockwise and try that side instead.
    - If both the forward cell and the clockwise cell are blocked, stop that cup's trail early; keep cup frames and
      gray blockers unchanged.

    Key insight:
    - The content count sets the maximum trail length, but the actual shape is determined by local obstacle-driven
      clockwise detours from the tail spike rather than by a straight ejection line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    cups = _detect_cups(g)

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = Color.BLACK

    for cup in cups:
        trail = _simulate_trail(g, cup["spike"], cup["tail_dir"], len(cup["content_cells"]))
        for r, c in trail:
            out[r, c] = Color.RED

    return out


def _build_canonical_cup(left_arm: int, right_arm: int, left_wall: int, right_wall: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    width = left_arm + right_arm + 1
    height = max(left_arm + left_wall, right_arm + right_wall) + 1
    patch = np.zeros((height, width), dtype=int)
    patch[0, left_arm] = int(color)
    for step in range(1, left_arm + 1):
        patch[step, left_arm - step] = int(color)
    for step in range(1, right_arm + 1):
        patch[step, left_arm + step] = int(color)
    if left_wall > 0:
        patch[left_arm + 1 : left_arm + left_wall + 1, 0] = int(color)
    if right_wall > 0:
        patch[right_arm + 1 : right_arm + right_wall + 1, width - 1] = int(color)
    return patch, (0, left_arm)


def _canonical_mouth_cells(patch: np.ndarray, color: int) -> list[tuple[int, int]]:
    border_set = {(int(r), int(c)) for r, c in np.argwhere(patch == int(color))}
    bbox = (0, 0, int(patch.shape[0] - 1), int(patch.shape[1] - 1))
    mouth = _flood_from_bbox_side(border_set, bbox, "D")
    return sorted((int(r), int(c)) for r, c in mouth)


def _rotate_shape(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth_cells: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (int(sr), int(sc)), list(mouth_cells)
    if direction == "R":
        return (
            np.rot90(patch, k=-1).copy(),
            (int(sc), int(h - 1 - sr)),
            [(int(c), int(h - 1 - r)) for r, c in mouth_cells],
        )
    if direction == "D":
        return (
            np.flipud(patch).copy(),
            (int(h - 1 - sr), int(sc)),
            [(int(h - 1 - r), int(c)) for r, c in mouth_cells],
        )
    return (
        np.rot90(patch, k=1).copy(),
        (int(w - 1 - sc), int(sr)),
        [(int(w - 1 - c), int(r)) for r, c in mouth_cells],
    )


def _sample_cup_patch(rng: random.Random, color: int, direction: str) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    for _ in range(200):
        left_arm = rng.randint(1, 3)
        right_arm = rng.randint(1, 3)
        left_wall = rng.randint(1, 4)
        right_wall = rng.randint(1, 4)
        patch, spike = _build_canonical_cup(left_arm, right_arm, left_wall, right_wall, color)
        mouth = _canonical_mouth_cells(patch, color)
        if len(mouth) < 4:
            continue
        rot_patch, rot_spike, rot_mouth = _rotate_shape(patch, spike, mouth, direction)
        if rot_patch.shape[0] <= 8 and rot_patch.shape[1] <= 8:
            return rot_patch, rot_spike, rot_mouth
    raise RuntimeError("failed to sample a valid cup patch")


def _straight_trail(spike: tuple[int, int], direction: str, steps: int) -> list[tuple[int, int]]:
    dr, dc = DIR_TO_VEC[direction]
    return [(int(spike[0] + dr * k), int(spike[1] + dc * k)) for k in range(1, int(steps) + 1)]


def _simulate_actions(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: str,
    actions: tuple[str, ...],
    occupied: set[tuple[int, int]],
    obstacle_forbidden: set[tuple[int, int]],
) -> tuple[tuple[int, int], str, list[tuple[int, int]], set[tuple[int, int]]] | None:
    current = (int(start[0]), int(start[1]))
    facing = str(direction)
    used = set(occupied)
    trail: list[tuple[int, int]] = []
    obstacles: set[tuple[int, int]] = set()

    for action in actions:
        dr, dc = DIR_TO_VEC[facing]
        front = (current[0] + dr, current[1] + dc)
        if action == "F":
            if not in_bounds(shape, front[0], front[1]) or front in used or front in obstacles:
                return None
            current = front
            trail.append(current)
            used.add(current)
            continue

        if not in_bounds(shape, front[0], front[1]) or front in used or front in obstacles or front in obstacle_forbidden:
            return None
        obstacles.add(front)
        facing = _cw(facing)
        dr, dc = DIR_TO_VEC[facing]
        step = (current[0] + dr, current[1] + dc)
        if not in_bounds(shape, step[0], step[1]) or step in used or step in obstacles:
            return None
        current = step
        trail.append(current)
        used.add(current)

    return current, facing, trail, obstacles


def _force_stop(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: str,
    occupied: set[tuple[int, int]],
    obstacle_forbidden: set[tuple[int, int]],
    existing_obstacles: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    blockers: set[tuple[int, int]] = set()

    def ensure_blocked(target: tuple[int, int]) -> bool:
        if not in_bounds(shape, target[0], target[1]):
            return True
        if target in existing_obstacles or target in blockers:
            return True
        if target in occupied or target in obstacle_forbidden:
            return False
        blockers.add(target)
        return True

    dr, dc = DIR_TO_VEC[direction]
    front = (current[0] + dr, current[1] + dc)
    cw_dir = _cw(direction)
    dr, dc = DIR_TO_VEC[cw_dir]
    clockwise = (current[0] + dr, current[1] + dc)
    if ensure_blocked(front) and ensure_blocked(clockwise):
        return blockers
    return None


def _plan_trail(
    rng: random.Random,
    shape: tuple[int, int],
    spike: tuple[int, int],
    direction: str,
    content_count: int,
    occupied: set[tuple[int, int]],
    obstacle_forbidden: set[tuple[int, int]],
    pattern: str,
) -> dict | None:
    seq_candidates: list[tuple[tuple[str, ...], bool]] = []
    if pattern == "turn_once":
        for seq in itertools.product("FT", repeat=content_count):
            if seq[0] != "F" or seq.count("T") != 1:
                continue
            seq_candidates.append((tuple(seq), False))
    elif pattern == "turn_twice":
        for seq in itertools.product("FT", repeat=content_count):
            if seq[0] != "F" or seq.count("T") < 2:
                continue
            seq_candidates.append((tuple(seq), False))
    else:
        for path_len in range(2, content_count):
            for seq in itertools.product("FT", repeat=path_len):
                if seq[0] != "F" or seq.count("T") < 1:
                    continue
                seq_candidates.append((tuple(seq), True))

    rng.shuffle(seq_candidates)
    for seq, needs_stop in seq_candidates:
        result = _simulate_actions(shape, spike, direction, seq, occupied, obstacle_forbidden)
        if result is None:
            continue
        current, facing, trail, obstacles = result
        stop_obstacles: set[tuple[int, int]] = set()
        if needs_stop:
            stop = _force_stop(shape, current, facing, occupied | set(trail), obstacle_forbidden, obstacles)
            if stop is None:
                continue
            stop_obstacles = stop
        return {
            "trail": trail,
            "obstacles": obstacles | stop_obstacles,
            "turns": int(seq.count("T")),
            "stopped_early": bool(needs_stop),
        }
    return None


def _oracle_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        for r, c in spec["content_abs"]:
            out[r, c] = Color.BLACK
    for spec in specs:
        for r, c in spec["trail"]:
            out[r, c] = Color.RED
    return out


def _validate_generated_case(grid: np.ndarray, specs: list[dict]) -> None:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("generated input must be a 2D ndarray")
    if max(grid.shape) > 30:
        raise ValueError(f"grid exceeds ARC size limit: {grid.shape}")

    expected = _oracle_output(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise ValueError("solver output does not match oracle output")

    parsed = _detect_cups(grid)
    if len(parsed) != len(specs):
        raise ValueError("cup parser did not recover the generated number of cups")

    parsed_by_color = {int(item["border_color"]): item for item in parsed}
    all_paths: set[tuple[int, int]] = set()
    all_obstacles: set[tuple[int, int]] = set()
    for spec in specs:
        color = int(spec["border_color"])
        parsed_cup = parsed_by_color.get(color)
        if parsed_cup is None:
            raise ValueError("generated cup is not uniquely decodable")
        if len(parsed_cup["content_cells"]) != int(spec["content_count"]):
            raise ValueError("parsed content count differs from generator metadata")
        if tuple(parsed_cup["spike"]) != tuple(spec["spike_abs"]) or parsed_cup["tail_dir"] != spec["direction"]:
            raise ValueError("tail detection differs from generator metadata")
        if spec["turns"] < 1:
            raise ValueError("every generated cup must be forced to detour at least once")
        if not spec["obstacles_abs"]:
            raise ValueError("every generated cup must have at least one blocker")
        if _straight_trail(spec["spike_abs"], spec["direction"], spec["content_count"]) == spec["trail"]:
            raise ValueError("detour cup degenerates to a straight ejection")
        if any(cell in all_paths for cell in spec["trail"]):
            raise ValueError("trails must not intersect each other")
        if any(cell in all_obstacles for cell in spec["obstacles_abs"]):
            raise ValueError("obstacle cells should stay distinct across cups")
        all_paths.update(spec["trail"])
        all_obstacles.update(spec["obstacles_abs"])
        cup_cells = set(spec["border_abs"])
        forbidden = _expand8(cup_cells)
        if any(cell in forbidden for cell in spec["obstacles_abs"]):
            raise ValueError("obstacle touches a cup boundary")
        if any(grid[r, c] != Color.GRAY for r, c in spec["obstacles_abs"]):
            raise ValueError("generator metadata and grid obstacle cells disagree")
        if any(expected[r, c] == Color.GRAY for r, c in spec["trail"]):
            raise ValueError("trail overwrote an obstacle")
        if spec["stopped_early"] and len(spec["trail"]) >= spec["content_count"]:
            raise ValueError("early-stop metadata is inconsistent")


def _sample_directions(rng: random.Random, num_cups: int) -> list[str]:
    directions = list(DIR_NAMES)
    rng.shuffle(directions)
    if num_cups == 2:
        return directions[:2]
    return directions[:3]


def _sample_patterns(rng: random.Random, num_cups: int, counts: list[int]) -> list[str]:
    patterns = []
    complex_used = False
    for content_count in counts:
        options = [("turn_once", 4)]
        if content_count >= 3:
            options.append(("turn_twice", 3))
            options.append(("stop", 3))
        total = sum(weight for _name, weight in options)
        pick = rng.randint(1, total)
        acc = 0
        choice = "turn_once"
        for name, weight in options:
            acc += weight
            if pick <= acc:
                choice = name
                break
        patterns.append(choice)
        complex_used |= choice != "turn_once"

    if num_cups >= 2 and not complex_used:
        upgrade = rng.randrange(num_cups)
        if counts[upgrade] >= 3:
            patterns[upgrade] = rng.choice(["turn_twice", "stop"])
    return patterns


def _place_case(rng: random.Random) -> tuple[np.ndarray, list[dict]] | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    num_cups = rng.choice(NUM_CUPS_CHOICES)
    colors = list(BORDER_COLORS)
    rng.shuffle(colors)
    colors = colors[:num_cups]
    directions = _sample_directions(rng, num_cups)
    content_counts = [rng.randint(*CONTENT_COUNT_RANGE) for _ in range(num_cups)]
    patterns = _sample_patterns(rng, num_cups, content_counts)

    grid = np.zeros((height, width), dtype=int)
    occupied_cups: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    reserved_obstacles: set[tuple[int, int]] = set()
    occupied_with_gap: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for border_color, direction, content_count, pattern in zip(colors, directions, content_counts, patterns):
        patch, spike_local, mouth_local = _sample_cup_patch(rng, border_color, direction)
        content_local = rng.sample(mouth_local, k=content_count)
        h, w = patch.shape
        candidates = [(r0, c0) for r0 in range(height - h + 1) for c0 in range(width - w + 1)]
        rng.shuffle(candidates)

        placed = False
        for r0, c0 in candidates[:PLACEMENT_ATTEMPTS]:
            border_abs = {
                (int(r0 + r), int(c0 + c))
                for r, c in np.argwhere(patch == int(border_color))
            }
            content_abs = [(int(r0 + r), int(c0 + c)) for r, c in content_local]
            all_cells = border_abs | set(content_abs)
            if all_cells & occupied_with_gap:
                continue

            spike_abs = (int(r0 + spike_local[0]), int(c0 + spike_local[1]))
            occupied_now = occupied_cups | reserved_paths | reserved_obstacles | border_abs
            obstacle_forbidden = _expand8(occupied_cups | border_abs)
            plan = _plan_trail(
                rng,
                (height, width),
                spike_abs,
                direction,
                content_count,
                occupied_now,
                obstacle_forbidden,
                pattern,
            )
            if plan is None:
                continue

            trail = plan["trail"]
            obstacles_abs = set(plan["obstacles"])
            if set(trail) & occupied_with_gap:
                continue
            if obstacles_abs & occupied_with_gap:
                continue

            for r, c in border_abs:
                grid[r, c] = int(border_color)
            for r, c in content_abs:
                grid[r, c] = Color.RED
            for r, c in obstacles_abs:
                grid[r, c] = Color.GRAY

            occupied_cups |= border_abs
            reserved_paths |= set(trail)
            reserved_obstacles |= obstacles_abs
            occupied_with_gap |= _expand8(border_abs | obstacles_abs | set(trail))
            specs.append(
                {
                    "border_color": int(border_color),
                    "direction": direction,
                    "content_count": int(content_count),
                    "spike_abs": spike_abs,
                    "trail": trail,
                    "obstacles_abs": sorted(obstacles_abs),
                    "border_abs": sorted(border_abs),
                    "content_abs": content_abs,
                    "turns": int(plan["turns"]),
                    "stopped_early": bool(plan["stopped_early"]),
                }
            )
            placed = True
            break

        if not placed:
            return None

    return grid, specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        built = _place_case(rng)
        if built is None:
            continue
        grid, specs = built
        try:
            _validate_generated_case(grid, specs)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid ARC input")
