import numpy as np

from collections import Counter, deque


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4


DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "down", "left", "right")
PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GEN_RETRY_LIMIT = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    seen = np.zeros_like(arr, dtype=bool)
    out: list[np.ndarray] = []
    if connectivity != 4:
        raise ValueError("only 4-connectivity is supported")
    for r in range(h):
        for c in range(w):
            if not arr[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray):
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_direction(mask3: np.ndarray) -> str:
    mask3 = np.asarray(mask3, dtype=bool)
    assert mask3.shape == (3, 3), f"expected 3x3 bbox, got {mask3.shape}"
    tests = {
        "up": bool(mask3[0, 1] and not mask3[2, 1]),
        "down": bool(mask3[2, 1] and not mask3[0, 1]),
        "left": bool(mask3[1, 0] and not mask3[1, 2]),
        "right": bool(mask3[1, 2] and not mask3[1, 0]),
    }
    valid = [name for name, ok in tests.items() if ok]
    assert len(valid) == 1, f"direction code must be unique, got {tests}"
    return valid[0]


def _build_shape_library() -> dict[str, list[np.ndarray]]:
    library = {name: [] for name in DIR_ORDER}
    for bits in range(1, 1 << 9):
        flat = np.array([(bits >> k) & 1 for k in range(9)], dtype=bool)
        mask3 = flat.reshape(3, 3)
        area = int(mask3.sum())
        if not 4 <= area <= 8:
            continue
        if bbox(mask3) != (0, 0, 2, 2):
            continue
        if len(components_mask(mask3, connectivity=4)) != 1:
            continue
        try:
            direction = _decode_direction(mask3)
        except AssertionError:
            continue
        library[direction].append(mask3.copy())
    for direction, shapes in library.items():
        assert shapes, f"no shapes found for {direction}"
    return library


SHAPE_LIBRARY = _build_shape_library()


def _coords_from_mask(mask3: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    return np.argwhere(mask3).astype(int) + np.array([[int(top_left[0]), int(top_left[1])]], dtype=int)


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.asarray(coords, dtype=int)}


def _render(objects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        color = int(obj["color"])
        assert np.all(out[coords[:, 0], coords[:, 1]] == bg), "objects must not overlap"
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    assert comps, "expected at least one foreground sprite"
    objects = []
    for coords in comps:
        values = g[coords[:, 0], coords[:, 1]]
        colors = np.unique(values)
        assert len(colors) == 1, "each sprite must be monochromatic"
        box = _bbox_from_coords(coords)
        patch = np.zeros((box[2] - box[0] + 1, box[3] - box[1] + 1), dtype=bool)
        rel = coords - np.array([[box[0], box[1]]], dtype=int)
        patch[rel[:, 0], rel[:, 1]] = True
        direction = _decode_direction(patch)
        objects.append(
            {
                "color": int(colors[0]),
                "coords": coords.astype(int).copy(),
                "direction": direction,
                "steps": 0,
                "stop_reason": None,
            }
        )
    return bg, objects


def _simulate(objects: list[dict], shape: tuple[int, int], bg: int) -> tuple[np.ndarray, list[dict]]:
    h, w = map(int, shape)
    sim_objects = []
    for obj in objects:
        sim_objects.append(
            {
                "color": int(obj["color"]),
                "coords": np.asarray(obj["coords"], dtype=int).copy(),
                "direction": str(obj["direction"]),
                "steps": int(obj.get("steps", 0)),
                "stop_reason": obj.get("stop_reason"),
                "stopped": bool(obj.get("stopped", False)),
            }
        )

    max_steps = h + w + 10
    for _ in range(max_steps):
        if all(obj["stopped"] for obj in sim_objects):
            break

        stopped_occ = set()
        for obj in sim_objects:
            if obj["stopped"]:
                stopped_occ |= _coords_set(obj["coords"])

        candidates: dict[int, np.ndarray] = {}
        for idx, obj in enumerate(sim_objects):
            if obj["stopped"]:
                continue
            dr, dc = DIR_VECTORS[obj["direction"]]
            candidate = obj["coords"] + np.array([[dr, dc]], dtype=int)
            if (
                np.any(candidate[:, 0] < 0)
                or np.any(candidate[:, 0] >= h)
                or np.any(candidate[:, 1] < 0)
                or np.any(candidate[:, 1] >= w)
            ):
                obj["stopped"] = True
                obj["stop_reason"] = "boundary"
                continue
            if any((int(r), int(c)) in stopped_occ for r, c in candidate):
                obj["stopped"] = True
                obj["stop_reason"] = "collision"
                continue
            candidates[idx] = candidate

        if not candidates:
            assert all(obj["stopped"] for obj in sim_objects), "unresolved moving object"
            break

        seen: dict[tuple[int, int], int] = {}
        for idx, candidate in candidates.items():
            for r, c in candidate:
                key = (int(r), int(c))
                if key in stopped_occ:
                    raise AssertionError("moving object overlapped a stopped object")
                if key in seen:
                    raise AssertionError("two moving objects overlapped in the same step")
                seen[key] = idx

        for idx, candidate in candidates.items():
            sim_objects[idx]["coords"] = candidate
            sim_objects[idx]["steps"] += 1
    else:
        raise AssertionError("simulation did not settle")

    assert all(obj["stopped"] for obj in sim_objects), "all sprites must stop"
    return _render(sim_objects, shape, bg), sim_objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each 3x3 contour sprite as a motion direction, then slide all sprites until border or settled-sprite blockage.

    Input structure:
    - The grid has black background and 3-4 separated monochromatic foreground sprites.
    - Each sprite is a 4-connected shape with a tight 3x3 bounding box and area 4-8.
    - A single occupied middle-edge cell of that 3x3 contour uniquely encodes one direction: up, down, left, or right.

    Rule:
    - Read every sprite's 3x3 contour to determine its motion direction.
    - Move all not-yet-stopped sprites simultaneously by one cell per step in their own directions.
    - A sprite stops when its next one-step move would leave the grid or would overlap a sprite that has already stopped; preserve shape and color.

    Key insight:
    - The sprite geometry is not decorative: the contour itself is the direction code that controls a gravity-like settling process with blocker dependencies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg, objects = _parse_objects(g)
    out, _ = _simulate(objects, g.shape, bg)
    return out


def _max_reverse_shift(direction: str, pos: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    r, c = map(int, pos)
    if direction == "right":
        return c
    if direction == "left":
        return w - 3 - c
    if direction == "down":
        return r
    if direction == "up":
        return h - 3 - r
    raise ValueError(direction)


def _shift_top_left(pos: tuple[int, int], direction: str, steps: int) -> tuple[int, int]:
    dr, dc = DIR_VECTORS[direction]
    r, c = map(int, pos)
    return int(r - dr * steps), int(c - dc * steps)


def _boundary_slots(direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    if direction == "right":
        return [(r, w - 3) for r in range(h - 2)]
    if direction == "left":
        return [(r, 0) for r in range(h - 2)]
    if direction == "down":
        return [(h - 3, c) for c in range(w - 2)]
    if direction == "up":
        return [(0, c) for c in range(w - 2)]
    raise ValueError(direction)


def _search_blocked_position(
    rng: np.random.Generator,
    direction: str,
    prev_coords: np.ndarray,
    prev_top_left: tuple[int, int],
    curr_mask: np.ndarray,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = map(int, shape)
    prev_set = _coords_set(prev_coords)
    candidates = []
    for r in range(h - 2):
        for c in range(w - 2):
            if direction == "right" and c > prev_top_left[1]:
                continue
            if direction == "left" and c < prev_top_left[1]:
                continue
            if direction == "down" and r > prev_top_left[0]:
                continue
            if direction == "up" and r < prev_top_left[0]:
                continue
            curr_coords = _coords_from_mask(curr_mask, (r, c))
            curr_set = _coords_set(curr_coords)
            if curr_set & occupied:
                continue
            dr, dc = DIR_VECTORS[direction]
            shifted = curr_coords + np.array([[dr, dc]], dtype=int)
            if (
                np.any(shifted[:, 0] < 0)
                or np.any(shifted[:, 0] >= h)
                or np.any(shifted[:, 1] < 0)
                or np.any(shifted[:, 1] >= w)
            ):
                continue
            shifted_set = _coords_set(shifted)
            if not (shifted_set & prev_set):
                continue
            if shifted_set & (occupied - prev_set):
                continue
            if direction in ("left", "right"):
                orth_dist = abs(r - prev_top_left[0])
                axis_dist = abs(c - prev_top_left[1])
            else:
                orth_dist = abs(c - prev_top_left[1])
                axis_dist = abs(r - prev_top_left[0])
            candidates.append((orth_dist, axis_dist, (r, c)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    top = candidates[: min(len(candidates), 8)]
    return top[int(rng.integers(len(top)))][2]


def _no_adjacent_pairs(objects: list[dict]) -> bool:
    cell_to_owner = {}
    for idx, obj in enumerate(objects):
        for cell in _coords_set(obj["coords"]):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (cell[0] + dr, cell[1] + dc)
                    other = cell_to_owner.get(neighbor)
                    if other is not None and other != idx:
                        return False
            cell_to_owner[cell] = idx
    return True


def _independent_boundary_result(objects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray | None:
    moved = []
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int).copy()
        dr, dc = DIR_VECTORS[obj["direction"]]
        while True:
            candidate = coords + np.array([[dr, dc]], dtype=int)
            if (
                np.any(candidate[:, 0] < 0)
                or np.any(candidate[:, 0] >= shape[0])
                or np.any(candidate[:, 1] < 0)
                or np.any(candidate[:, 1] >= shape[1])
            ):
                break
            coords = candidate
        moved.append({"color": obj["color"], "coords": coords})
    try:
        return _render(moved, shape, bg)
    except AssertionError:
        return None


def _global_direction_result(objects: list[dict], shape: tuple[int, int], bg: int, direction: str) -> np.ndarray:
    alt = []
    for obj in objects:
        alt.append(
            {
                "color": obj["color"],
                "coords": np.asarray(obj["coords"], dtype=int).copy(),
                "direction": direction,
                "steps": 0,
                "stop_reason": None,
            }
        )
    out, _ = _simulate(alt, shape, bg)
    return out


def _expected_objects_from_specs(specs: list[dict]) -> list[dict]:
    return [
        {
            "color": spec["color"],
            "coords": _coords_from_mask(spec["mask"], spec["final_top_left"]),
            "direction": spec["direction"],
        }
        for spec in specs
    ]


def _choose_chain_shifts(
    rng: np.random.Generator,
    specs: list[dict],
    shape: tuple[int, int],
) -> list[int] | None:
    shifts = []
    previous = 0
    for idx, spec in enumerate(specs):
        max_shift = _max_reverse_shift(spec["direction"], spec["final_top_left"], shape)
        remaining = len(specs) - idx - 1
        low = previous + 1
        high = max_shift - remaining
        if high < low:
            return None
        shift = int(rng.integers(low, high + 1))
        shifts.append(shift)
        previous = shift
    return shifts


def _specs_to_input_specs(specs: list[dict], shifts: list[int]) -> list[dict]:
    out = []
    for spec, shift in zip(specs, shifts):
        inp = dict(spec)
        inp["input_top_left"] = _shift_top_left(spec["final_top_left"], spec["direction"], shift)
        inp["shift"] = int(shift)
        out.append(inp)
    return out


def _render_specs(specs: list[dict], shape: tuple[int, int], bg: int, key: str) -> np.ndarray:
    objects = []
    for spec in specs:
        objects.append(
            {
                "color": spec["color"],
                "coords": _coords_from_mask(spec["mask"], spec[key]),
            }
        )
    return _render(objects, shape, bg)


def _validate_generation(input_grid: np.ndarray, expected_output: np.ndarray) -> bool:
    bg, parsed = _parse_objects(input_grid)
    if len(parsed) not in (3, 4):
        return False
    if len({obj["color"] for obj in parsed}) != len(parsed):
        return False
    if len({obj["direction"] for obj in parsed}) < 2:
        return False
    if not _no_adjacent_pairs(parsed):
        return False

    try:
        actual_output, settled = _simulate(parsed, input_grid.shape, bg)
    except AssertionError:
        return False
    if not np.array_equal(actual_output, expected_output):
        return False
    if np.array_equal(actual_output, input_grid):
        return False
    if not all(obj["steps"] >= 1 for obj in settled):
        return False
    reasons = {obj["stop_reason"] for obj in settled}
    if reasons != {"boundary", "collision"} and not ({"boundary", "collision"} <= reasons):
        return False

    independent = _independent_boundary_result(parsed, input_grid.shape, bg)
    if independent is not None and np.array_equal(independent, actual_output):
        return False

    for direction in DIR_ORDER:
        try:
            alt = _global_direction_result(parsed, input_grid.shape, bg, direction)
        except AssertionError:
            continue
        if np.array_equal(alt, actual_output):
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(13, 17))
    w = int(rng.integers(13, 17))
    shape = (h, w)
    bg = Color.BLACK

    n_objects = 4 if rng.random() < 0.7 else 3
    chain_len = 2
    if n_objects == 4 and rng.random() < 0.35:
        chain_len = 3

    chain_direction = DIR_ORDER[int(rng.integers(len(DIR_ORDER)))]
    other_directions = [d for d in DIR_ORDER if d != chain_direction]
    rng.shuffle(other_directions)
    directions = [chain_direction] * chain_len + other_directions[: n_objects - chain_len]
    colors = list(rng.choice(PALETTE, size=n_objects, replace=False))

    specs = []
    for color, direction in zip(colors, directions):
        shapes = SHAPE_LIBRARY[direction]
        mask = shapes[int(rng.integers(len(shapes)))].copy()
        specs.append({"color": int(color), "direction": direction, "mask": mask})

    occupied_final: set[tuple[int, int]] = set()
    leader = specs[0]
    boundary_positions = _boundary_slots(chain_direction, shape)
    rng.shuffle(boundary_positions)
    leader_pos = None
    for pos in boundary_positions:
        if _max_reverse_shift(chain_direction, pos, shape) < chain_len:
            continue
        coords = _coords_from_mask(leader["mask"], pos)
        coord_set = _coords_set(coords)
        if coord_set & occupied_final:
            continue
        leader_pos = pos
        leader["final_top_left"] = pos
        occupied_final |= coord_set
        prev_coords = coords
        prev_pos = pos
        break
    if leader_pos is None:
        return None

    for idx in range(1, chain_len):
        spec = specs[idx]
        pos = _search_blocked_position(
            rng,
            chain_direction,
            prev_coords,
            prev_pos,
            spec["mask"],
            shape,
            occupied_final,
        )
        if pos is None:
            return None
        spec["final_top_left"] = pos
        prev_coords = _coords_from_mask(spec["mask"], pos)
        prev_pos = pos
        occupied_final |= _coords_set(prev_coords)

    for spec in specs[chain_len:]:
        positions = _boundary_slots(spec["direction"], shape)
        rng.shuffle(positions)
        placed = False
        for pos in positions:
            if _max_reverse_shift(spec["direction"], pos, shape) < 1:
                continue
            coords = _coords_from_mask(spec["mask"], pos)
            coord_set = _coords_set(coords)
            if coord_set & occupied_final:
                continue
            spec["final_top_left"] = pos
            occupied_final |= coord_set
            placed = True
            break
        if not placed:
            return None

    chain_shifts = _choose_chain_shifts(rng, specs[:chain_len], shape)
    if chain_shifts is None:
        return None

    shifts = list(chain_shifts)
    for spec in specs[chain_len:]:
        max_shift = _max_reverse_shift(spec["direction"], spec["final_top_left"], shape)
        if max_shift < 1:
            return None
        shifts.append(int(rng.integers(1, min(max_shift, 6) + 1)))

    input_specs = _specs_to_input_specs(specs, shifts)
    input_objects = [
        {
            "color": spec["color"],
            "coords": _coords_from_mask(spec["mask"], spec["input_top_left"]),
            "direction": spec["direction"],
        }
        for spec in input_specs
    ]
    if not _no_adjacent_pairs(input_objects):
        return None

    try:
        input_grid = _render(input_objects, shape, bg)
    except AssertionError:
        return None

    expected_output = _render_specs(input_specs, shape, bg, "final_top_left")
    if not _validate_generation(input_grid, expected_output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid contour-gravity input")
