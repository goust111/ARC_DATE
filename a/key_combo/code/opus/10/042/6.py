from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 5
CENTER_COLORS = (1, 3, 4, 9)
ARM_COLORS = (2, 6, 7, 8)
CROSS_MASK = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)
ARM_ORDER = ((0, 1), (1, 2), (2, 1), (1, 0))  # N, E, S, W
MOVE_TO_SIDE = {"right": "left", "left": "right", "down": "top", "up": "bottom"}
SIDE_TO_MOVE = {side: move for move, side in MOVE_TO_SIDE.items()}
MOVE_DELTA = {"right": (0, 1), "left": (0, -1), "down": (1, 0), "up": (-1, 0)}
FINAL_PAIR_PATTERNS = (
    ("bottom_right", ("right", "down")),
    ("bottom_left", ("left", "down")),
    ("top_right", ("right", "up")),
    ("top_left", ("left", "up")),
)
GEN_ATTEMPTS = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_offset(side: str, marker_coords: np.ndarray, cross_r0: int, cross_c0: int) -> int:
    if side in {"left", "right"}:
        return int(marker_coords[:, 0].min() - cross_r0)
    return int(marker_coords[:, 1].min() - cross_c0)


def _cross_top_left_from_component(cross_coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cross_coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "cross must occupy a 3x3 box"
    normalized = np.zeros((3, 3), dtype=bool)
    normalized[cross_coords[:, 0] - r0, cross_coords[:, 1] - c0] = True
    assert np.array_equal(normalized, CROSS_MASK), "cross pixels must form a plus shape"
    return r0, c0


def _parse_object(grid: np.ndarray, coords: np.ndarray) -> dict:
    values = grid[coords[:, 0], coords[:, 1]]
    marker_coords = coords[values == MARKER]
    cross_coords = coords[values != MARKER]
    assert len(marker_coords) == 2 and len(cross_coords) == 5, "expected a 5-pixel cross plus 2-pixel marker"

    cross_r0, cross_c0 = _cross_top_left_from_component(cross_coords)
    cross_block = grid[cross_r0 : cross_r0 + 3, cross_c0 : cross_c0 + 3].copy()
    assert np.array_equal(cross_block != BG, CROSS_MASK), "cross bbox must contain only the cross"

    mr0, mc0, mr1, mc1 = _bbox(marker_coords)
    marker_h = mr1 - mr0 + 1
    marker_w = mc1 - mc0 + 1
    assert (marker_h, marker_w) in {(2, 1), (1, 2)}, "marker must be a domino"

    if np.all(marker_coords[:, 1] == cross_c0 - 1):
        side = "left"
    elif np.all(marker_coords[:, 1] == cross_c0 + 3):
        side = "right"
    elif np.all(marker_coords[:, 0] == cross_r0 - 1):
        side = "top"
    elif np.all(marker_coords[:, 0] == cross_r0 + 3):
        side = "bottom"
    else:
        raise AssertionError("marker must sit flush against exactly one side of the cross")

    offset = _marker_offset(side, marker_coords, cross_r0, cross_c0)
    assert offset in {0, 1}, "marker attachment must be offset by 0 or 1 cells along its side"

    obj_r0, obj_c0, obj_r1, obj_c1 = _bbox(coords)
    move = SIDE_TO_MOVE[side]
    h, w = grid.shape
    if move == "right":
        distance = w - 1 - obj_c1
    elif move == "left":
        distance = obj_c0
    elif move == "down":
        distance = h - 1 - obj_r1
    else:
        distance = obj_r0
    assert distance > 0, "objects must start away from their destination edge"

    center_color = int(cross_block[1, 1])
    return {
        "coords": coords.copy(),
        "cross_top_left": (cross_r0, cross_c0),
        "cross_block": cross_block,
        "side": side,
        "move": move,
        "distance": int(distance),
        "marker_offset": int(offset),
        "center_color": center_color,
    }


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    assert 3 <= len(comps) <= 4, "expected 3 or 4 composite objects"
    objects = [_parse_object(g, comp) for comp in comps]
    centers = [obj["center_color"] for obj in objects]
    assert len(set(centers)) == len(centers), "center colors must distinguish the objects"
    distances = [obj["distance"] for obj in objects]
    assert len(set(distances)) == len(distances), "travel distances must be unique"
    return objects


def _rotated_cross(block: np.ndarray, distance: int) -> np.ndarray:
    if distance <= 4:
        return np.rot90(block, -1)
    return np.rot90(block, 2)


def _marker_cells(cross_r: int, cross_c: int, side: str, offset: int) -> list[tuple[int, int]]:
    if side == "left":
        return [(cross_r + offset, cross_c - 1), (cross_r + offset + 1, cross_c - 1)]
    if side == "right":
        return [(cross_r + offset, cross_c + 3), (cross_r + offset + 1, cross_c + 3)]
    if side == "top":
        return [(cross_r - 1, cross_c + offset), (cross_r - 1, cross_c + offset + 1)]
    return [(cross_r + 3, cross_c + offset), (cross_r + 3, cross_c + offset + 1)]


def _final_cross_top_left(cross_r: int, cross_c: int, move: str, distance: int) -> tuple[int, int]:
    dr, dc = MOVE_DELTA[move]
    return cross_r + dr * distance, cross_c + dc * distance


def _object_cells(obj: dict, *, transformed: bool) -> list[tuple[int, int, int]]:
    cross_r, cross_c = obj["cross_top_left"]
    if transformed:
        cross_r, cross_c = _final_cross_top_left(cross_r, cross_c, obj["move"], obj["distance"])
        cross_block = _rotated_cross(obj["cross_block"], obj["distance"])
    else:
        cross_block = obj["cross_block"]

    cells: list[tuple[int, int, int]] = []
    for rr, cc in np.argwhere(cross_block != BG):
        cells.append((cross_r + int(rr), cross_c + int(cc), int(cross_block[rr, cc])))
    for mr, mc in _marker_cells(cross_r, cross_c, obj["side"], obj["marker_offset"]):
        cells.append((int(mr), int(mc), MARKER))
    return cells


def _render_output_from_specs(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for obj in sorted(objects, key=lambda item: item["distance"]):
        for r, c, color in _object_cells(obj, transformed=True):
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked colored cross to the opposite edge, rotate the cross by an amount encoded by the travel distance, and resolve overlaps by longer-travel top layers.

    Input structure:
    - The background is black.
    - There are 3 or 4 separated composite objects.
    - Each object contains a 3x3 five-pixel colored cross and a grey two-pixel domino marker attached flush to exactly one side of that cross.

    Rule:
    - The grey marker indicates the tail side, so the whole object slides in the opposite direction until its leading edge reaches the grid border.
    - Let d be the number of cells moved. Rotate only the 3x3 cross block: if d <= 4, rotate it 90 degrees clockwise; if d > 4, rotate it 180 degrees.
    - Keep the grey marker attached on the same tail side after the move.
    - When multiple final objects cover the same output cell, keep the color from the object with the larger travel distance.

    Key insight:
    - The same travel distance d must first be inferred from the slide to the border, then reused twice: once to choose the cross rotation and once to choose which overlapping object is visible on top.
    """
    g = np.asarray(grid, dtype=int)
    objects = _parse_objects(g)
    return _render_output_from_specs(g.shape, objects)


def _random_cross_block(center_color: int) -> np.ndarray:
    block = np.zeros((3, 3), dtype=int)
    block[1, 1] = center_color
    arms = random.sample(list(ARM_COLORS), 4)
    for color, (r, c) in zip(arms, ARM_ORDER):
        block[r, c] = color
    return block


def _occupancy_cross_block() -> np.ndarray:
    return _random_cross_block(CENTER_COLORS[0])


def _make_input_spec(final_obj: dict, distance: int) -> dict:
    move = final_obj["move"]
    dr, dc = MOVE_DELTA[move]
    final_cross = final_obj["cross_top_left"]
    start_cross = (final_cross[0] - dr * distance, final_cross[1] - dc * distance)
    return {
        "move": move,
        "side": final_obj["side"],
        "cross_top_left": start_cross,
        "distance": int(distance),
        "marker_offset": final_obj["marker_offset"],
        "center_color": final_obj["center_color"],
        "cross_block": final_obj["cross_block"].copy(),
    }


def _final_cross_for_move(shape: tuple[int, int], move: str, ortho: int) -> tuple[int, int]:
    h, w = shape
    if move == "right":
        return int(ortho), w - 3
    if move == "left":
        return int(ortho), 0
    if move == "down":
        return h - 3, int(ortho)
    return 0, int(ortho)


def _pair_orthogonals(shape: tuple[int, int], pattern: str) -> tuple[int, int]:
    h, w = shape
    if pattern == "bottom_right":
        return h - 4, w - 4
    elif pattern == "bottom_left":
        return h - 4, 1
    elif pattern == "top_right":
        return 1, w - 4
    return 1, 1


def _sample_unique_distances(count: int, shape: tuple[int, int]) -> list[int]:
    max_distance = min(shape) - 4
    assert max_distance >= 7
    short = random.choice((2, 3, 4))
    longs = random.sample(list(range(5, max_distance + 1)), count - 1)
    distances = [short, *longs]
    random.shuffle(distances)
    return distances


def _in_bounds_cells(shape: tuple[int, int], cells: list[tuple[int, int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c, _ in cells)


def _can_add_final(existing: list[dict], candidate: dict, shape: tuple[int, int]) -> bool:
    cell_to_objects: dict[tuple[int, int], set[int]] = {}
    for idx, obj in enumerate(existing + [candidate]):
        for r, c, _ in _object_cells(obj, transformed=False):
            cell_to_objects.setdefault((r, c), set()).add(idx)
    overlap_cells = [owners for owners in cell_to_objects.values() if len(owners) >= 2]
    if any(len(owners) > 2 for owners in overlap_cells):
        return False
    return len(overlap_cells) <= 4


def _sample_extra_final(shape: tuple[int, int], candidate: dict, current: list[dict]) -> tuple[int, int] | None:
    h, w = shape
    move = candidate["move"]
    if move in {"right", "left"}:
        choices = list(range(0, h - 2))
    else:
        choices = list(range(0, w - 2))
    random.shuffle(choices)
    for ortho in choices:
        candidate["cross_top_left"] = _final_cross_for_move(shape, move, ortho)
        if _can_add_final(current, candidate, shape):
            return candidate["cross_top_left"]
    return None


def _input_components_well_separated(shape: tuple[int, int], objects: list[dict]) -> bool:
    canvas = np.zeros(shape, dtype=int)
    for idx, obj in enumerate(objects, start=1):
        for r, c, _ in _object_cells(obj, transformed=False):
            if canvas[r, c] != 0:
                return False
            canvas[r, c] = idx
    for r in range(shape[0]):
        for c in range(shape[1]):
            if canvas[r, c] == 0:
                continue
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    if canvas[nr, nc] not in {0, canvas[r, c]}:
                        return False
    return True


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    for obj in objects:
        for r, c, color in _object_cells(obj, transformed=False):
            inp[r, c] = color
    return inp


def _overlap_stats(objects: list[dict], *, already_final: bool = False) -> tuple[int, int]:
    counts: dict[tuple[int, int], int] = {}
    for obj in objects:
        for r, c, _ in _object_cells(obj, transformed=not already_final):
            counts[(r, c)] = counts.get((r, c), 0) + 1
    overlap_cells = sum(1 for value in counts.values() if value >= 2)
    max_stack = max(counts.values())
    return overlap_cells, max_stack


def _intended_signature(objects: list[dict]) -> dict[int, tuple]:
    sig: dict[int, tuple] = {}
    for obj in objects:
        sig[obj["center_color"]] = (
            obj["side"],
            obj["move"],
            obj["distance"],
            obj["marker_offset"],
            obj["cross_top_left"],
            tuple(map(int, obj["cross_block"].flat)),
        )
    return sig


def _parsed_signature(objects: list[dict]) -> dict[int, tuple]:
    sig: dict[int, tuple] = {}
    for obj in objects:
        sig[obj["center_color"]] = (
            obj["side"],
            obj["move"],
            obj["distance"],
            obj["marker_offset"],
            obj["cross_top_left"],
            tuple(map(int, obj["cross_block"].flat)),
        )
    return sig


def _validate_generation(inp: np.ndarray, objects: list[dict]) -> bool:
    try:
        parsed = _parse_objects(inp)
    except AssertionError:
        return False
    if _parsed_signature(parsed) != _intended_signature(objects):
        return False

    distances = [obj["distance"] for obj in objects]
    if not any(d <= 4 for d in distances):
        return False
    if not any(d > 4 for d in distances):
        return False
    if len(set(distances)) != len(distances):
        return False

    if not _input_components_well_separated(inp.shape, objects):
        return False
    overlap_cells, max_stack = _overlap_stats(objects)
    if overlap_cells < 1 or overlap_cells > 4 or max_stack > 2:
        return False

    expected = _render_output_from_specs(inp.shape, objects)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    shape = (h, w)
    count = random.choice((3, 4))
    pattern, pair_moves = random.choice(FINAL_PAIR_PATTERNS)
    extra_moves = [move for move in ("up", "down", "left", "right") if move not in pair_moves]
    if count == 3:
        extra_moves = random.sample(extra_moves, 1)

    directions = [*pair_moves, *extra_moves]
    distances = _sample_unique_distances(len(directions), shape)
    center_colors = random.sample(list(CENTER_COLORS), len(directions))

    pair_ortho_a, pair_ortho_b = _pair_orthogonals(shape, pattern)
    final_objects: list[dict] = [
        {
            "move": pair_moves[0],
            "side": MOVE_TO_SIDE[pair_moves[0]],
            "cross_top_left": _final_cross_for_move(shape, pair_moves[0], pair_ortho_a),
            "distance": 5,
            "marker_offset": random.randint(0, 1),
            "center_color": center_colors[0],
            "cross_block": _random_cross_block(center_colors[0]),
        },
        {
            "move": pair_moves[1],
            "side": MOVE_TO_SIDE[pair_moves[1]],
            "cross_top_left": _final_cross_for_move(shape, pair_moves[1], pair_ortho_b),
            "distance": 5,
            "marker_offset": random.randint(0, 1),
            "center_color": center_colors[1],
            "cross_block": _random_cross_block(center_colors[1]),
        },
    ]

    for idx, move in enumerate(extra_moves, start=2):
        candidate = {
            "move": move,
            "side": MOVE_TO_SIDE[move],
            "cross_top_left": (0, 0),
            "distance": 5,
            "marker_offset": random.randint(0, 1),
            "center_color": center_colors[idx],
            "cross_block": _random_cross_block(center_colors[idx]),
        }
        cross = _sample_extra_final(shape, candidate, final_objects)
        if cross is None:
            return None
        candidate["cross_top_left"] = cross
        final_objects.append(candidate)

    overlap_cells, max_stack = _overlap_stats(final_objects, already_final=True)
    if overlap_cells < 1 or overlap_cells > 5 or max_stack > 2:
        return None

    objects = [
        _make_input_spec(final_obj, distance)
        for final_obj, distance in zip(final_objects, distances)
    ]

    if not all(_in_bounds_cells(shape, _object_cells(obj, transformed=False)) for obj in objects):
        return None
    if not all(_in_bounds_cells(shape, _object_cells(obj, transformed=True)) for obj in objects):
        return None
    if not _input_components_well_separated(shape, objects):
        return None

    inp = _render_input(shape, objects)
    if not _validate_generation(inp, objects):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _attempt_generate()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
