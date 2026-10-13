from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import find_components, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


    def _components_bool(mask: np.ndarray) -> list[np.ndarray]:
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        parts: list[np.ndarray] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            if seen[rr, cc]:
                continue
            stack = [(rr, cc)]
            seen[rr, cc] = True
            coords = []
            while stack:
                cr, cc2 = stack.pop()
                coords.append((cr, cc2))
                for nr, nc in ((cr - 1, cc2), (cr + 1, cc2), (cr, cc2 - 1), (cr, cc2 + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
        return parts


    def find_components(
        grid: np.ndarray,
        *,
        bg: int = 0,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | tuple[int, ...] | None = None,
        min_area: int = 1,
    ) -> list[dict]:
        assert connectivity == 4
        assert monochromatic
        if colors is None:
            colors = sorted(set(map(int, np.unique(grid))) - {int(bg)})
        out = []
        for color in colors:
            if int(color) == int(bg):
                continue
            for coords in _components_bool(grid == int(color)):
                if int(coords.shape[0]) < int(min_area):
                    continue
                r0, c0 = coords.min(axis=0)
                r1, c1 = coords.max(axis=0)
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(coords.shape[0]),
                    }
                )
        return out


BG = 0
SPARK = 2
RING_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
GAP_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
SPARK_SIDES = (
    ("top", (-1, 1)),
    ("bottom", (3, 1)),
    ("left", (1, -1)),
    ("right", (1, 3)),
)

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
CHAIN_TRAVEL_RANGE = (1, 7)
FINAL_WALL_STEPS_RANGE = (2, 8)
PLACEMENT_RETRIES = 240
GENERATION_RETRIES = 900

SCENARIOS = (
    {
        "chain_len": 3,
        "distractors": 1,
        "need_diagonal": True,
        "need_cardinal": True,
        "need_corner_gap": True,
    },
    {
        "chain_len": 2,
        "distractors": 1,
        "need_diagonal": True,
        "need_cardinal": False,
        "need_corner_gap": True,
    },
    {
        "chain_len": 3,
        "distractors": 0,
        "need_diagonal": True,
        "need_cardinal": True,
        "need_corner_gap": True,
    },
)

_GEN_CALL_COUNT = 0


def _build_local_ring_coords() -> dict[tuple[int, int], np.ndarray]:
    out = {}
    for gap in GAP_OFFSETS:
        coords = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) == (0, 0) or (dr, dc) == gap:
                    continue
                coords.append((1 + dr, 1 + dc))
        out[gap] = np.array(coords, dtype=int)
    return out


LOCAL_RING_COORDS = _build_local_ring_coords()


def _move_from_gap(gap: tuple[int, int]) -> tuple[int, int]:
    return -int(gap[0]), -int(gap[1])


def _is_corner_gap(gap: tuple[int, int]) -> bool:
    return int(gap[0]) != 0 and int(gap[1]) != 0


def _ring_coords(top_left: tuple[int, int], gap: tuple[int, int]) -> np.ndarray:
    return LOCAL_RING_COORDS[gap] + np.array(top_left, dtype=int)


def _ring_cell_set(top_left: tuple[int, int], gap: tuple[int, int]) -> set[tuple[int, int]]:
    coords = _ring_coords(top_left, gap)
    return {tuple(map(int, rc)) for rc in coords}


def _cells_in_bounds(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches_spark(cells: set[tuple[int, int]], spark: tuple[int, int]) -> bool:
    sr, sc = map(int, spark)
    return any(abs(r - sr) + abs(c - sc) == 1 for r, c in cells)


def _parse_ring(component: dict, grid: np.ndarray, bg: int) -> dict | None:
    if int(component["area"]) != 7:
        return None
    r0, c0, r1, c1 = map(int, component["bbox"])
    if r1 != r0 + 2 or c1 != c0 + 2:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(component["color"])
    if int(patch[1, 1]) != int(bg):
        return None

    gaps = []
    for rr in range(3):
        for cc in range(3):
            if (rr, cc) == (1, 1):
                continue
            value = int(patch[rr, cc])
            if value == int(bg):
                gaps.append((rr - 1, cc - 1))
            elif value != color:
                return None

    if len(gaps) != 1:
        return None

    gap = gaps[0]
    return {
        "color": color,
        "gap": gap,
        "move": _move_from_gap(gap),
        "top_left": (r0, c0),
        "bbox": (r0, c0, r1, c1),
    }


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    spark = None
    rings = []

    for comp in components:
        color = int(comp["color"])
        if color == SPARK and int(comp["area"]) == 1:
            assert spark is None
            spark = tuple(map(int, comp["coords"][0]))
            continue

        ring = _parse_ring(comp, g, bg)
        assert ring is not None
        assert ring["color"] != SPARK
        rings.append(ring)

    assert spark is not None
    assert rings

    rings.sort(key=lambda ring: (ring["top_left"][0], ring["top_left"][1], ring["color"]))
    for idx, ring in enumerate(rings):
        ring["id"] = idx

    start_ids = []
    for ring in rings:
        if _touches_spark(_ring_cell_set(ring["top_left"], ring["gap"]), spark):
            start_ids.append(int(ring["id"]))
    assert len(start_ids) == 1

    return {
        "shape": tuple(map(int, g.shape)),
        "bg": bg,
        "rings": rings,
        "spark": spark,
        "start_id": int(start_ids[0]),
    }


def _advance_active(scene: dict, positions: dict[int, tuple[int, int]], active_id: int) -> dict:
    rings = scene["rings"]
    shape = scene["shape"]
    ring = rings[active_id]
    move_r, move_c = ring["move"]
    current = positions[active_id]
    stationary = {
        rid: _ring_cell_set(positions[rid], other["gap"])
        for rid, other in enumerate(rings)
        if rid != active_id
    }

    steps = 0
    while True:
        next_pos = (current[0] + move_r, current[1] + move_c)
        next_cells = _ring_cell_set(next_pos, ring["gap"])
        if not _cells_in_bounds(next_cells, shape):
            return {
                "kind": "wall",
                "active": int(active_id),
                "stop_top_left": current,
                "steps": int(steps),
            }

        hits = [rid for rid, cells in stationary.items() if not next_cells.isdisjoint(cells)]
        assert len(hits) <= 1
        if hits:
            return {
                "kind": "ring",
                "active": int(active_id),
                "stop_top_left": current,
                "steps": int(steps),
                "target": int(hits[0]),
            }

        current = next_pos
        steps += 1


def _simulate_chain(scene: dict) -> tuple[dict[int, tuple[int, int]], list[int], list[dict]]:
    positions = {int(ring["id"]): tuple(map(int, ring["top_left"])) for ring in scene["rings"]}
    active = int(scene["start_id"])
    order = [active]
    seen = {active}
    segments = []

    while True:
        segment = _advance_active(scene, positions, active)
        positions[active] = tuple(map(int, segment["stop_top_left"]))
        segments.append(segment)
        if segment["kind"] == "wall":
            break

        target = int(segment["target"])
        assert target not in seen
        seen.add(target)
        order.append(target)
        active = target

    return positions, order, segments


def _render_scene(
    shape: tuple[int, int],
    bg: int,
    rings: list[dict],
    positions: dict[int, tuple[int, int]],
    *,
    spark: tuple[int, int] | None = None,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for ring in rings:
        coords = _ring_coords(positions[int(ring["id"])], ring["gap"])
        out[coords[:, 0], coords[:, 1]] = int(ring["color"])
    if spark is not None:
        out[int(spark[0]), int(spark[1])] = SPARK
    return out


def _initial_positions(scene: dict) -> dict[int, tuple[int, int]]:
    return {int(ring["id"]): tuple(map(int, ring["top_left"])) for ring in scene["rings"]}


def _start_only_positions(scene: dict) -> dict[int, tuple[int, int]]:
    positions = _initial_positions(scene)
    active = int(scene["start_id"])
    segment = _advance_active(scene, positions, active)
    positions[active] = tuple(map(int, segment["stop_top_left"]))
    return positions


def _independent_positions(scene: dict) -> dict[int, tuple[int, int]]:
    initial = _initial_positions(scene)
    fixed = {
        rid: _ring_cell_set(initial[rid], ring["gap"])
        for rid, ring in enumerate(scene["rings"])
    }
    out = {}
    for rid, ring in enumerate(scene["rings"]):
        move_r, move_c = ring["move"]
        current = initial[rid]
        while True:
            next_pos = (current[0] + move_r, current[1] + move_c)
            next_cells = _ring_cell_set(next_pos, ring["gap"])
            if not _cells_in_bounds(next_cells, scene["shape"]):
                break
            hits = [
                other_id
                for other_id, other_cells in fixed.items()
                if other_id != rid and not next_cells.isdisjoint(other_cells)
            ]
            if hits:
                break
            current = next_pos
        out[rid] = current
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Relay a spark-triggered chain of broken 3x3 rings until the last moving ring reaches a wall.

    Input structure:
    - A black background contains several monochromatic 3x3 ring objects whose centers are black and whose outer border is missing exactly one cell.
    - Exactly one red spark pixel is 4-neighbor adjacent to the outside of one ring.

    Rule:
    - The spark marks the first active ring and disappears in the output.
    - An active ring moves one cell at a time in the direction opposite its missing-cell offset from the ring center.
    - If the next step would leave the grid, that ring stays in its last in-bounds position and the relay ends.
    - If the next step would overlap exactly one stationary ring, the moving ring stops one step earlier and the struck ring becomes the next active ring.
    - Only one ring moves at a time; every non-active ring stays fixed until it is ignited by a collision.

    Key insight:
    - The red spark only identifies the first baton-holder; every later motion is encoded locally by each ring's own gap, so the final layout is produced by a sequential collision relay rather than by one global direction.
    """
    scene = _parse_scene(grid)
    final_positions, _order, _segments = _simulate_chain(scene)
    return _render_scene(scene["shape"], scene["bg"], scene["rings"], final_positions)


def _rel_stop_candidates(
    moving_gap: tuple[int, int], target_gap: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    move_r, move_c = _move_from_gap(moving_gap)
    target_cells = _ring_cell_set((0, 0), target_gap)
    candidates = []
    for rr in range(-4, 5):
        for cc in range(-4, 5):
            current_cells = _ring_cell_set((rr, cc), moving_gap)
            next_cells = _ring_cell_set((rr + move_r, cc + move_c), moving_gap)
            if current_cells.isdisjoint(target_cells) and not next_cells.isdisjoint(target_cells):
                candidates.append((rr, cc))
    return tuple(candidates)


REL_STOP_CANDIDATES = {
    (moving_gap, target_gap): _rel_stop_candidates(moving_gap, target_gap)
    for moving_gap in GAP_OFFSETS
    for target_gap in GAP_OFFSETS
}


def _max_wall_steps(top_left: tuple[int, int], gap: tuple[int, int], shape: tuple[int, int]) -> int:
    move_r, move_c = _move_from_gap(gap)
    current = top_left
    steps = 0
    while True:
        next_pos = (current[0] + move_r, current[1] + move_c)
        if not _cells_in_bounds(_ring_cell_set(next_pos, gap), shape):
            return int(steps)
        current = next_pos
        steps += 1


def _sample_chain_gaps(scenario: dict) -> list[tuple[int, int]] | None:
    chain_len = int(scenario["chain_len"])
    for _ in range(200):
        gaps = [random.choice(GAP_OFFSETS) for _ in range(chain_len)]
        moves = [_move_from_gap(gap) for gap in gaps]
        if len(set(moves)) < 2:
            continue
        if any(moves[i] == moves[i + 1] for i in range(chain_len - 1)):
            continue
        if scenario["need_diagonal"] and not any(move_r != 0 and move_c != 0 for move_r, move_c in moves):
            continue
        if scenario["need_cardinal"] and not any(move_r == 0 or move_c == 0 for move_r, move_c in moves):
            continue
        if scenario["need_corner_gap"] and not any(_is_corner_gap(gap) for gap in gaps):
            continue
        return gaps
    return None


def _path_clear_to_stop(
    start: tuple[int, int],
    steps: int,
    moving_gap: tuple[int, int],
    move: tuple[int, int],
    stationary: dict[int, set[tuple[int, int]]],
    shape: tuple[int, int],
) -> bool:
    current = start
    for step in range(int(steps) + 1):
        cells = _ring_cell_set(current, moving_gap)
        if not _cells_in_bounds(cells, shape):
            return False
        for other_cells in stationary.values():
            if not cells.isdisjoint(other_cells):
                return False
        if step < int(steps):
            current = (current[0] + move[0], current[1] + move[1])
    return True


def _place_chain_rings(
    shape: tuple[int, int],
    chain_gaps: list[tuple[int, int]],
    colors: list[int],
) -> list[dict] | None:
    h, w = shape
    chain_len = len(chain_gaps)
    rings = [
        {
            "id": idx,
            "color": int(colors[idx]),
            "gap": chain_gaps[idx],
            "move": _move_from_gap(chain_gaps[idx]),
            "role": "chain",
        }
        for idx in range(chain_len)
    ]

    last = chain_len - 1
    last_positions = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    random.shuffle(last_positions)
    min_wall, max_wall = FINAL_WALL_STEPS_RANGE
    for pos in last_positions:
        wall_steps = _max_wall_steps(pos, rings[last]["gap"], shape)
        if min_wall <= wall_steps <= max_wall:
            rings[last]["top_left"] = pos
            break
    else:
        return None

    min_travel, max_travel = CHAIN_TRAVEL_RANGE
    for idx in range(chain_len - 2, -1, -1):
        target = rings[idx + 1]
        rel_candidates = list(REL_STOP_CANDIDATES[(rings[idx]["gap"], target["gap"])])
        random.shuffle(rel_candidates)
        later_rings = rings[idx + 1 :]
        later_cells = {
            int(other["id"]): _ring_cell_set(other["top_left"], other["gap"])
            for other in later_rings
        }
        found = False

        for rel_r, rel_c in rel_candidates:
            stop = (target["top_left"][0] + rel_r, target["top_left"][1] + rel_c)
            travel_choices = list(range(min_travel, max_travel + 1))
            random.shuffle(travel_choices)
            for travel in travel_choices:
                start = (
                    stop[0] - travel * rings[idx]["move"][0],
                    stop[1] - travel * rings[idx]["move"][1],
                )
                if not _path_clear_to_stop(
                    start,
                    travel,
                    rings[idx]["gap"],
                    rings[idx]["move"],
                    later_cells,
                    shape,
                ):
                    continue
                next_pos = (
                    stop[0] + rings[idx]["move"][0],
                    stop[1] + rings[idx]["move"][1],
                )
                next_cells = _ring_cell_set(next_pos, rings[idx]["gap"])
                hits = [
                    other_id
                    for other_id, cells in later_cells.items()
                    if not next_cells.isdisjoint(cells)
                ]
                if hits != [idx + 1]:
                    continue

                rings[idx]["top_left"] = start
                found = True
                break
            if found:
                break

        if not found:
            return None

    return rings


def _add_distractors(shape: tuple[int, int], rings: list[dict], count: int, colors: list[int], chain_len: int) -> bool:
    h, w = shape
    for color in colors[chain_len : chain_len + count]:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            gap = random.choice(GAP_OFFSETS)
            top_left = (random.randrange(h - 2), random.randrange(w - 2))
            cells = _ring_cell_set(top_left, gap)
            if not _cells_in_bounds(cells, shape):
                continue
            if any(not cells.isdisjoint(_ring_cell_set(ring["top_left"], ring["gap"])) for ring in rings):
                continue

            rings.append(
                {
                    "id": len(rings),
                    "color": int(color),
                    "gap": gap,
                    "move": _move_from_gap(gap),
                    "top_left": top_left,
                    "role": "distractor",
                }
            )
            test_scene = {
                "shape": shape,
                "bg": BG,
                "rings": rings,
                "start_id": 0,
            }
            try:
                _positions, order, segments = _simulate_chain(test_scene)
            except AssertionError:
                rings.pop()
                continue
            if order == list(range(chain_len)) and segments[-1]["kind"] == "wall":
                placed = True
                break
            rings.pop()
        if not placed:
            return False
    return True


def _front_spark_offsets(move: tuple[int, int]) -> set[tuple[int, int]]:
    move_r, move_c = move
    blocked = set()
    if move_r < 0:
        blocked.add((-1, 1))
    if move_r > 0:
        blocked.add((3, 1))
    if move_c < 0:
        blocked.add((1, -1))
    if move_c > 0:
        blocked.add((1, 3))
    return blocked


def _choose_spark(scene: dict) -> tuple[int, int] | None:
    start_ring = scene["rings"][scene["start_id"]]
    occupied = {
        tuple(cell)
        for ring in scene["rings"]
        for cell in _ring_cell_set(ring["top_left"], ring["gap"])
    }
    blocked = _front_spark_offsets(start_ring["move"])
    choices = list(SPARK_SIDES)
    random.shuffle(choices)

    for _name, (off_r, off_c) in choices:
        if (off_r, off_c) in blocked:
            continue
        spark = (start_ring["top_left"][0] + off_r, start_ring["top_left"][1] + off_c)
        if not (0 <= spark[0] < scene["shape"][0] and 0 <= spark[1] < scene["shape"][1]):
            continue
        if spark in occupied:
            continue
        touching = [
            ring["id"]
            for ring in scene["rings"]
            if _touches_spark(_ring_cell_set(ring["top_left"], ring["gap"]), spark)
        ]
        if touching == [scene["start_id"]]:
            return spark
    return None


def _build_scene(scenario: dict) -> dict | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    chain_gaps = _sample_chain_gaps(scenario)
    if chain_gaps is None:
        return None

    total_rings = int(scenario["chain_len"]) + int(scenario["distractors"])
    colors = random.sample(RING_COLORS, total_rings)
    rings = _place_chain_rings(shape, chain_gaps, colors)
    if rings is None:
        return None

    if not _add_distractors(shape, rings, int(scenario["distractors"]), colors, int(scenario["chain_len"])):
        return None

    scene = {
        "shape": shape,
        "bg": BG,
        "rings": rings,
        "start_id": 0,
        "chain_len": int(scenario["chain_len"]),
    }
    spark = _choose_spark(scene)
    if spark is None:
        return None
    scene["spark"] = spark

    try:
        _positions, order, segments = _simulate_chain(scene)
    except AssertionError:
        return None
    if order != list(range(scene["chain_len"])) or segments[-1]["kind"] != "wall":
        return None
    if any(int(segment["steps"]) <= 0 for segment in segments):
        return None

    return scene


def _scene_grid(scene: dict) -> np.ndarray:
    return _render_scene(
        scene["shape"],
        scene["bg"],
        scene["rings"],
        _initial_positions(scene),
        spark=scene["spark"],
    )


def _semantic_checks(grid: np.ndarray, expected_chain_colors: list[int], expected_ring_count: int) -> bool:
    try:
        scene = _parse_scene(grid)
        final_positions, order, segments = _simulate_chain(scene)
    except Exception:
        return False

    if len(scene["rings"]) != int(expected_ring_count):
        return False

    actual_chain_colors = [int(scene["rings"][ring_id]["color"]) for ring_id in order]
    if actual_chain_colors != [int(color) for color in expected_chain_colors]:
        return False

    if len(order) < 2 or segments[-1]["kind"] != "wall":
        return False
    if any(int(segment["steps"]) <= 0 for segment in segments):
        return False

    moves = [tuple(scene["rings"][ring_id]["move"]) for ring_id in order]
    if len(set(moves)) < 2 or not any(moves[i] != moves[i + 1] for i in range(len(moves) - 1)):
        return False
    if not any(_is_corner_gap(scene["rings"][ring_id]["gap"]) for ring_id in order):
        return False

    expected = _render_scene(scene["shape"], scene["bg"], scene["rings"], final_positions)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual == SPARK):
        return False

    initial_render = _render_scene(scene["shape"], scene["bg"], scene["rings"], _initial_positions(scene))
    if np.array_equal(expected, initial_render):
        return False

    start_only = _render_scene(scene["shape"], scene["bg"], scene["rings"], _start_only_positions(scene))
    if np.array_equal(expected, start_only):
        return False

    independent = _render_scene(scene["shape"], scene["bg"], scene["rings"], _independent_positions(scene))
    if np.array_equal(expected, independent):
        return False

    return True


def _gen_attempt(scenario: dict) -> np.ndarray | None:
    scene = _build_scene(scenario)
    if scene is None:
        return None
    grid = _scene_grid(scene)
    expected_chain_colors = [int(scene["rings"][idx]["color"]) for idx in range(int(scene["chain_len"]))]
    if not _semantic_checks(grid, expected_chain_colors, len(scene["rings"])):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred = SCENARIOS[_GEN_CALL_COUNT % len(SCENARIOS)]
    _GEN_CALL_COUNT += 1

    for attempt in range(GENERATION_RETRIES):
        scenario = preferred if attempt < GENERATION_RETRIES // 2 else random.choice(SCENARIOS)
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-ring input")
