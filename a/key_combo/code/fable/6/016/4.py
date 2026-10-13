from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, find_components, get_bg, neighbors4


BG_COLOR = Color.BLACK
SPARK_COLOR = Color.YELLOW
MAX_GRID_SIZE = 30
GEN_ATTEMPTS = 600
CHAIN_LEN_CHOICES = (2, 3, 4)
DISTRACTOR_CHOICES = (1, 2, 3)
MOVE_STEPS = (1, 2, 3, 4, 5, 6)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

ARROW_SHAPES = {
    "up": np.array([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)], dtype=int),
    "down": np.array([(0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)], dtype=int),
    "left": np.array([(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (1, 3)], dtype=int),
    "right": np.array([(1, 0), (1, 1), (0, 2), (1, 2), (2, 2), (1, 3)], dtype=int),
}
ARROW_BOUNDS = {
    direction: (int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1)
    for direction, coords in ARROW_SHAPES.items()
}
SHAPE_TO_DIRECTION = {
    frozenset(map(tuple, coords.tolist())): direction
    for direction, coords in ARROW_SHAPES.items()
}


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.asarray(coords, dtype=int)}


def _coords_from_anchor(direction: str, anchor: tuple[int, int]) -> np.ndarray:
    return ARROW_SHAPES[direction] + np.array(anchor, dtype=int)


def _shift(coords: np.ndarray, direction: str, steps: int = 1) -> np.ndarray:
    dr, dc = DIRS[direction]
    return np.asarray(coords, dtype=int) + np.array((dr * int(steps), dc * int(steps)), dtype=int)


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == height - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == width - 1)
    )


def _max_steps(coords: np.ndarray, direction: str, shape: tuple[int, int]) -> int:
    height, width = map(int, shape)
    coords = np.asarray(coords, dtype=int)
    if direction == "up":
        return int(coords[:, 0].min())
    if direction == "down":
        return int(height - 1 - coords[:, 0].max())
    if direction == "left":
        return int(coords[:, 1].min())
    if direction == "right":
        return int(width - 1 - coords[:, 1].max())
    raise ValueError(f"unknown direction: {direction}")


def _overlaps(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    return bool(_coord_set(coords_a) & _coord_set(coords_b))


def _is_adjacent(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    bset = _coord_set(coords_b)
    for r, c in np.asarray(coords_a, dtype=int):
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) in bset:
                return True
    return False


def _adjacent_ids(coords: np.ndarray, others: list[np.ndarray]) -> list[int]:
    return [idx for idx, other in enumerate(others) if _is_adjacent(coords, other)]


def _render(shape: tuple[int, int], arrows: list[dict], spark: tuple[int, int] | None, bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for arrow in arrows:
        coords = np.asarray(arrow["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(arrow["color"])
    if spark is not None:
        out[int(spark[0]), int(spark[1])] = SPARK_COLOR
    return out


def _direction_from_component(coords: np.ndarray) -> str:
    coords = np.asarray(coords, dtype=int)
    norm = coords - coords.min(axis=0)
    key = frozenset(map(tuple, norm.tolist()))
    direction = SHAPE_TO_DIRECTION.get(key)
    if direction is None:
        raise ValueError("found a foreground component that is not a supported arrow sprite")
    return direction


def _decode_scene(grid: np.ndarray) -> tuple[int, tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    spark_cells = [
        tuple(map(int, comp["coords"][0]))
        for comp in comps
        if int(comp["color"]) == SPARK_COLOR and int(comp["area"]) == 1
    ]
    if len(spark_cells) != 1:
        raise ValueError("expected exactly one yellow spark pixel")
    spark = spark_cells[0]

    arrows: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        if color == SPARK_COLOR and area == 1:
            continue
        if area != 6:
            raise ValueError("expected every non-spark foreground component to be a 6-cell arrow")
        coords = np.asarray(comp["coords"], dtype=int)
        arrows.append(
            {
                "color": color,
                "direction": _direction_from_component(coords),
                "coords": coords.copy(),
            }
        )

    arrows.sort(key=lambda item: tuple(map(int, item["coords"].min(axis=0))))
    if len(arrows) < 2:
        raise ValueError("expected at least two arrows")
    return bg, spark, arrows


def _spark_target(spark: tuple[int, int], arrows: list[dict]) -> int:
    sr, sc = map(int, spark)
    hits = [idx for idx, arrow in enumerate(arrows) if _is_adjacent(np.array([(sr, sc)], dtype=int), arrow["coords"])]
    if len(hits) != 1:
        raise ValueError("spark must be adjacent to exactly one arrow")
    return int(hits[0])


def _simulate_chain(shape: tuple[int, int], arrows: list[dict], spark: tuple[int, int]) -> tuple[list[dict], list[int]]:
    state = [
        {
            "color": int(arrow["color"]),
            "direction": str(arrow["direction"]),
            "coords": np.asarray(arrow["coords"], dtype=int).copy(),
        }
        for arrow in arrows
    ]
    current = _spark_target(spark, state)
    visited: list[int] = []

    while True:
        if current in visited:
            raise ValueError("movement chain must not loop back to a previously moved arrow")
        visited.append(int(current))

        mover = state[current]
        max_steps = _max_steps(mover["coords"], mover["direction"], shape)
        if max_steps <= 0:
            raise ValueError("triggered arrow must have room to move")

        stop_coords = None
        stop_target = None
        for step in range(1, max_steps + 1):
            trial = _shift(mover["coords"], mover["direction"], step)
            other_coords = [state[idx]["coords"] for idx in range(len(state)) if idx != current]
            if any(_overlaps(trial, other) for other in other_coords):
                raise ValueError("arrows may not overlap while sliding")
            hit_ids = [idx for idx, arrow in enumerate(state) if idx != current and _is_adjacent(trial, arrow["coords"])]
            if hit_ids:
                if len(hit_ids) != 1:
                    raise ValueError("a moving arrow must touch exactly one successor")
                if step == max_steps and _touches_edge(trial, shape):
                    raise ValueError("chain stop is ambiguous between edge and arrow contact")
                stop_coords = trial
                stop_target = int(hit_ids[0])
                break

        if stop_coords is None:
            stop_coords = _shift(mover["coords"], mover["direction"], max_steps)
            if not _touches_edge(stop_coords, shape):
                raise ValueError("non-contact movement must terminate at the grid edge")
            state[current]["coords"] = stop_coords
            return state, visited

        state[current]["coords"] = stop_coords
        current = int(stop_target)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Remove the spark, then propagate a single movement chain through the arrows.
    Each triggered arrow slides along its own pointing direction until it either
    becomes edge-stopped or comes into orthogonal contact with the next arrow.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg, spark, arrows = _decode_scene(grid)
    final_arrows, _visited = _simulate_chain(grid.shape, arrows, spark)
    return _render(grid.shape, final_arrows, None, bg)


def _all_anchors(direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    box_h, box_w = ARROW_BOUNDS[direction]
    return [(r0, c0) for r0 in range(height - box_h + 1) for c0 in range(width - box_w + 1)]


def _edge_anchors(direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    box_h, box_w = ARROW_BOUNDS[direction]
    if direction == "up":
        return [(0, c0) for c0 in range(width - box_w + 1)]
    if direction == "down":
        return [(height - box_h, c0) for c0 in range(width - box_w + 1)]
    if direction == "left":
        return [(r0, 0) for r0 in range(height - box_h + 1)]
    if direction == "right":
        return [(r0, width - box_w) for r0 in range(height - box_h + 1)]
    raise ValueError(f"unknown direction: {direction}")


def _path_states(initial: np.ndarray, direction: str, steps: int) -> list[np.ndarray]:
    return [_shift(initial, direction, step) for step in range(steps + 1)]


def _translation_steps(initial: np.ndarray, final: np.ndarray, direction: str) -> int:
    delta = np.asarray(final, dtype=int) - np.asarray(initial, dtype=int)
    if direction == "up":
        return int(-delta[0, 0])
    if direction == "down":
        return int(delta[0, 0])
    if direction == "left":
        return int(-delta[0, 1])
    if direction == "right":
        return int(delta[0, 1])
    raise ValueError(f"unknown direction: {direction}")


def _pick_chain_specs(
    rng: np.random.Generator,
    shape: tuple[int, int],
    colors: list[int],
) -> list[dict] | None:
    chain_len = len(colors)
    specs: list[dict | None] = [None] * chain_len

    for idx in range(chain_len - 1, -1, -1):
        later_initials = [np.asarray(spec["initial"], dtype=int) for spec in specs[idx + 1 :] if spec is not None]
        candidates: list[dict] = []

        for direction in rng.permutation(np.array(list(DIRS), dtype=object)).tolist():
            anchors = _edge_anchors(direction, shape) if idx == chain_len - 1 else _all_anchors(direction, shape)
            if not anchors:
                continue
            order = rng.permutation(len(anchors)).tolist()
            for anchor_idx in order:
                final_coords = _coords_from_anchor(direction, anchors[anchor_idx])
                if idx != chain_len - 1 and _touches_edge(final_coords, shape):
                    continue
                if any(_overlaps(final_coords, other) for other in later_initials):
                    continue

                if idx < chain_len - 1:
                    hit_ids = _adjacent_ids(final_coords, later_initials)
                    if hit_ids != [0]:
                        continue

                max_back = min(_max_steps(final_coords, OPPOSITE[direction], shape), max(MOVE_STEPS))
                step_choices = [step for step in MOVE_STEPS if step <= max_back]
                if not step_choices:
                    continue
                step_order = rng.permutation(np.array(step_choices, dtype=int)).tolist()
                for steps in step_order:
                    initial_coords = _shift(final_coords, OPPOSITE[direction], int(steps))
                    if _touches_edge(initial_coords, shape):
                        continue
                    if any(_overlaps(initial_coords, other) or _is_adjacent(initial_coords, other) for other in later_initials):
                        continue

                    clear_path = True
                    for mid_step in range(1, int(steps)):
                        trial = _shift(initial_coords, direction, mid_step)
                        if any(_overlaps(trial, other) or _is_adjacent(trial, other) for other in later_initials):
                            clear_path = False
                            break
                    if not clear_path:
                        continue

                    candidates.append(
                        {
                            "color": int(colors[idx]),
                            "direction": direction,
                            "initial": initial_coords,
                            "final": final_coords,
                            "steps": int(steps),
                        }
                    )

        if not candidates:
            return None
        specs[idx] = candidates[int(rng.integers(len(candidates)))]

    return [spec for spec in specs if spec is not None]


def _spark_candidates(first_arrow: np.ndarray, other_arrows: list[np.ndarray], shape: tuple[int, int]) -> list[tuple[int, int]]:
    occupied = set()
    for coords in [first_arrow, *other_arrows]:
        occupied |= _coord_set(coords)

    candidates: list[tuple[int, int]] = []
    for r, c in _coord_set(first_arrow):
        for nr, nc in neighbors4(r, c):
            cell = (int(nr), int(nc))
            if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
                continue
            if cell in occupied:
                continue
            spark_arr = np.array([cell], dtype=int)
            if any(_is_adjacent(spark_arr, other) for other in other_arrows):
                continue
            candidates.append(cell)
    return sorted(set(candidates))


def _place_distractors(
    rng: np.random.Generator,
    shape: tuple[int, int],
    colors: list[int],
    spark: tuple[int, int],
    chain_paths: list[np.ndarray],
    existing_initials: list[np.ndarray],
) -> list[dict] | None:
    distractors: list[dict] = []
    spark_arr = np.array([spark], dtype=int)

    for color in colors:
        candidates: list[dict] = []
        blockers = [*existing_initials, *(item["coords"] for item in distractors)]
        for direction in rng.permutation(np.array(list(DIRS), dtype=object)).tolist():
            anchors = _all_anchors(direction, shape)
            order = rng.permutation(len(anchors)).tolist()
            for anchor_idx in order:
                coords = _coords_from_anchor(direction, anchors[anchor_idx])
                if _touches_edge(coords, shape):
                    continue
                if _is_adjacent(coords, spark_arr):
                    continue
                if any(_overlaps(coords, other) or _is_adjacent(coords, other) for other in blockers):
                    continue
                if any(_overlaps(coords, path) or _is_adjacent(coords, path) for path in chain_paths):
                    continue
                candidates.append({"color": int(color), "direction": direction, "coords": coords})

        if not candidates:
            return None
        distractors.append(candidates[int(rng.integers(len(candidates)))])

    return distractors


def _build_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    shape = (
        int(rng.integers(12, 21)),
        int(rng.integers(12, 21)),
    )
    if shape[0] > MAX_GRID_SIZE or shape[1] > MAX_GRID_SIZE:
        return None

    chain_len = int(rng.choice(np.array(CHAIN_LEN_CHOICES, dtype=int)))
    distractor_count = int(rng.choice(np.array(DISTRACTOR_CHOICES, dtype=int)))
    palette = [color for color in Color.ALL_COLORS if color not in (BG_COLOR, SPARK_COLOR)]
    if chain_len + distractor_count > len(palette):
        return None
    sampled = rng.choice(np.array(palette, dtype=int), size=chain_len + distractor_count, replace=False).tolist()
    chain_colors = [int(color) for color in sampled[:chain_len]]
    distractor_colors = [int(color) for color in sampled[chain_len:]]

    chain_specs = _pick_chain_specs(rng, shape, chain_colors)
    if chain_specs is None:
        return None

    spark_options = _spark_candidates(
        np.asarray(chain_specs[0]["initial"], dtype=int),
        [np.asarray(spec["initial"], dtype=int) for spec in chain_specs[1:]],
        shape,
    )
    if not spark_options:
        return None
    spark = spark_options[int(rng.integers(len(spark_options)))]

    chain_paths = []
    for spec in chain_specs:
        chain_paths.extend(_path_states(np.asarray(spec["initial"], dtype=int), str(spec["direction"]), int(spec["steps"])))

    distractors = _place_distractors(
        rng,
        shape,
        distractor_colors,
        spark,
        chain_paths,
        [np.asarray(spec["initial"], dtype=int) for spec in chain_specs],
    )
    if distractors is None:
        return None

    initial_arrows = [
        {"color": int(spec["color"]), "direction": str(spec["direction"]), "coords": np.asarray(spec["initial"], dtype=int)}
        for spec in chain_specs
    ]
    initial_arrows.extend(distractors)
    expected_arrows = [
        {"color": int(spec["color"]), "direction": str(spec["direction"]), "coords": np.asarray(spec["final"], dtype=int)}
        for spec in chain_specs
    ]
    expected_arrows.extend(
        {
            "color": int(item["color"]),
            "direction": str(item["direction"]),
            "coords": np.asarray(item["coords"], dtype=int),
        }
        for item in distractors
    )

    if len({_coord for arrow in expected_arrows for _coord in _coord_set(arrow["coords"])}) != 6 * len(expected_arrows):
        return None

    input_grid = _render(shape, initial_arrows, spark, BG_COLOR)
    expected_grid = _render(shape, expected_arrows, None, BG_COLOR)

    try:
        solved = main(input_grid)
        bg, decoded_spark, decoded_arrows = _decode_scene(input_grid)
        simulated, visited = _simulate_chain(shape, decoded_arrows, decoded_spark)
    except ValueError:
        return None

    if not np.array_equal(solved, expected_grid):
        return None
    if bg != BG_COLOR:
        return None
    if len(visited) != chain_len or len(decoded_arrows) != chain_len + distractor_count:
        return None
    if not np.array_equal(_render(shape, simulated, None, bg), expected_grid):
        return None

    return input_grid, expected_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance(rng)
        if built is None:
            continue
        input_grid, _expected_grid = built
        return input_grid.astype(int)
    raise RuntimeError("failed to generate a valid chain-reaction arrow instance")
