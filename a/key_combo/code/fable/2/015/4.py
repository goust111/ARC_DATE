from __future__ import annotations

import random
import itertools
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC = SourceFileLoader(
    "_arc_utils_shared",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC.Color
get_bg = _ARC.get_bg
find_components = _ARC.find_components


BG = Color.BLACK
MARKER = Color.MAROON
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [c for c in Color.NOT_BLACK if c != MARKER]

GRID_H_RANGE = (9, 16)
LEFT_MARGIN_RANGE = (1, 3)
MARKER_W_RANGE = (1, 2)
RECT_H_RANGE = (1, 3)
RECT_W_RANGE = (1, 3)
GAP_RANGE = (1, 4)
DISTRACTOR_COUNT_WEIGHTS = [0, 0, 1]
DISTRACTOR_SIZE_RANGE = (1, 2)
GENERATE_RETRIES = 300
PREFER_GROUP_PROB = 0.7

GROUP_LAYOUTS = [
    layout
    for n_groups in range(1, 4)
    for layout in itertools.product((1, 2, 3), repeat=n_groups)
    if 2 <= sum(layout) <= 5
]
PASSTHROUGH_LAYOUTS = [layout for layout in GROUP_LAYOUTS if any(size > 1 for size in layout)]
PLAIN_LAYOUTS = [layout for layout in GROUP_LAYOUTS if all(size == 1 for size in layout)]


def _coord_tuples(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in coords.tolist()}


def _touches_marker(coords: np.ndarray, marker_mask: np.ndarray) -> bool:
    for r, c in coords.tolist():
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < marker_mask.shape[0] and 0 <= nc < marker_mask.shape[1] and marker_mask[nr, nc]:
                return True
    return False


def _launch_direction(marker_mask: np.ndarray, coords: np.ndarray) -> tuple[int, int]:
    directions = set()
    for r, c in coords.tolist():
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < marker_mask.shape[0] and 0 <= nc < marker_mask.shape[1] and marker_mask[nr, nc]:
                directions.add((-dr, -dc))
    assert len(directions) == 1, "launcher contact must imply one outward direction"
    return next(iter(directions))


def _objects_touch(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    bset = _coord_tuples(coords_b)
    for r, c in coords_a.tolist():
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS:
            if (rr + dr, cc + dc) in bset:
                return True
    return False


def _front_value(coords: np.ndarray, direction: tuple[int, int]) -> int:
    dr, dc = direction
    if dr < 0:
        return int(coords[:, 0].min())
    if dr > 0:
        return int(coords[:, 0].max())
    if dc < 0:
        return int(coords[:, 1].min())
    return int(coords[:, 1].max())


def _shift(coords: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    dr, dc = direction
    return coords + np.array([dr, dc], dtype=int)


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "expected a black background"

    marker_mask = g == MARKER
    assert bool(marker_mask.any()), "expected a maroon launcher"

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    objects: dict[int, np.ndarray] = {}
    for comp in components:
        color = int(comp["color"])
        if color == MARKER:
            continue
        assert color not in objects, "each moving object must have a unique color"
        objects[color] = comp["coords"].astype(int, copy=False)

    assert objects, "expected at least one non-launcher object"

    touching_marker = [color for color, coords in objects.items() if _touches_marker(coords, marker_mask)]
    assert len(touching_marker) == 1, "exactly one object must touch the launcher"
    start_color = touching_marker[0]
    start_coords = objects[start_color]
    direction = _launch_direction(marker_mask, start_coords)

    for color, coords in objects.items():
        if color == start_color:
            continue
        assert not _objects_touch(start_coords, coords), "the launched object must start isolated from other objects"

    return {
        "grid": g,
        "bg": bg,
        "marker_mask": marker_mask,
        "objects": objects,
        "start_color": start_color,
        "direction": direction,
    }


def _slide_until_blocked(
    coords: np.ndarray,
    occupied: set[tuple[int, int]],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    pos = coords.copy()
    h, w = map(int, shape)
    while True:
        nxt = _shift(pos, direction)
        if not np.all((0 <= nxt[:, 0]) & (nxt[:, 0] < h) & (0 <= nxt[:, 1]) & (nxt[:, 1] < w)):
            return pos, True
        if any((int(r), int(c)) in occupied for r, c in nxt.tolist()):
            return pos, False
        pos = nxt


def _touching_colors(coords: np.ndarray, objects: dict[int, np.ndarray]) -> list[int]:
    return [color for color, other in objects.items() if _objects_touch(coords, other)]


def _contact_group(seed_colors: list[int], objects: dict[int, np.ndarray]) -> set[int]:
    seen = set(seed_colors)
    stack = list(seed_colors)
    while stack:
        color = stack.pop()
        for other, other_coords in objects.items():
            if other in seen:
                continue
            if _objects_touch(objects[color], other_coords):
                seen.add(other)
                stack.append(other)
    return seen


def _farthest_color(group: set[int], objects: dict[int, np.ndarray], direction: tuple[int, int]) -> int:
    values = {color: _front_value(objects[color], direction) for color in group}
    if direction in {(1, 0), (0, 1)}:
        best = max(values.values())
    else:
        best = min(values.values())
    farthest = [color for color, value in values.items() if value == best]
    assert len(farthest) == 1, "the contact group must have a unique far end"
    return farthest[0]


def _render(shape: tuple[int, int], marker_mask: np.ndarray, objects: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[marker_mask] = MARKER
    for color, coords in objects.items():
        assert np.all(out[coords[:, 0], coords[:, 1]] == BG), "objects must remain disjoint"
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _simulate(grid: np.ndarray) -> tuple[dict, dict[int, np.ndarray], list[dict]]:
    scene = _extract_scene(grid)
    stationary = {
        color: coords.copy()
        for color, coords in scene["objects"].items()
        if color != scene["start_color"]
    }
    moving_color = int(scene["start_color"])
    moving_coords = scene["objects"][moving_color].copy()
    trace: list[dict] = []

    for _ in range(len(scene["objects"]) + 1):
        occupied = {
            (int(r), int(c))
            for coords in stationary.values()
            for r, c in coords.tolist()
        }
        final_coords, hit_boundary = _slide_until_blocked(
            moving_coords,
            occupied,
            scene["direction"],
            scene["grid"].shape,
        )

        event = {
            "moving_color": moving_color,
            "start_coords": moving_coords.copy(),
            "final_coords": final_coords.copy(),
            "boundary": hit_boundary,
        }

        if hit_boundary:
            stationary[moving_color] = final_coords
            trace.append(event)
            break

        touched = _touching_colors(final_coords, stationary)
        assert touched, "a blocked mover must stop against another object"
        group = _contact_group(touched, stationary)
        assert set(touched) <= group
        ejected = _farthest_color(group, stationary, scene["direction"])

        stationary[moving_color] = final_coords
        next_coords = stationary.pop(ejected)
        event["touched_colors"] = sorted(touched)
        event["group_colors"] = sorted(group)
        event["ejected_color"] = int(ejected)
        trace.append(event)

        moving_color = int(ejected)
        moving_coords = next_coords.copy()
    else:
        raise AssertionError("momentum relay did not terminate")

    assert trace and trace[-1]["boundary"], "the final momentum recipient must reach the boundary"
    return scene, stationary, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay outward momentum through downstream contact groups until the last object reaches the boundary.

    Input structure:
    - The background is black and maroon cells form one fixed launcher.
    - Exactly one colored 4-connected object touches the launcher on one cardinal side and is pushed outward.
    - Additional uniquely colored 4-connected objects lie farther along that ray; some of them may already touch as a group.

    Rule:
    - Infer the unique outward direction from the launcher-contacting object.
    - Move only one object at a time along that direction until its next step would leave the grid or overlap another object.
    - If it stops against another object, keep the stopped object there, pass momentum through the whole touched contact group, and eject only the direction-farthest object from that group; the final momentum holder slides to the boundary.

    Key insight:
    - A touching group is a momentum conduit rather than a block pile: the near members stay fixed and only the far-end member is launched onward.
    """
    scene, stationary, _trace = _simulate(grid)
    return _render(scene["grid"].shape, scene["marker_mask"], stationary)


def _sample_layout() -> tuple[tuple[int, ...], bool]:
    require_passthrough = random.random() < PREFER_GROUP_PROB
    if require_passthrough:
        return random.choice(PASSTHROUGH_LAYOUTS), True
    return random.choice(PLAIN_LAYOUTS), False


def _sample_band_top(lane: int, obj_h: int, height: int) -> int | None:
    lo = max(1, lane - obj_h + 1)
    hi = min(lane, height - obj_h - 1)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> np.ndarray:
    rows = np.arange(r0, r0 + h, dtype=int)
    cols = np.arange(c0, c0 + w, dtype=int)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    coords = np.stack([rr.ravel(), cc.ravel()], axis=1)
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)
    return coords


def _can_place_with_padding(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    lane: int,
) -> bool:
    if r0 <= lane <= r0 + h - 1:
        return False
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0], r0 + h + 1)
    cc1 = min(grid.shape[1], c0 + w + 1)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BG))


def _rotate_random(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, random.randint(0, 3)).copy()


def _validate_generated_grid(
    grid: np.ndarray,
    *,
    expected_groups: int,
    require_passthrough: bool,
) -> None:
    out = main(grid)
    scene, stationary, trace = _simulate(grid)
    rebuilt = _render(scene["grid"].shape, scene["marker_mask"], stationary)
    assert np.array_equal(out, rebuilt), "solver output must match the traced simulation"

    collisions = [event for event in trace if not event["boundary"]]
    assert len(collisions) == expected_groups, "each downstream group should consume exactly one relay step"
    if require_passthrough:
        assert any(len(event["group_colors"]) > 1 for event in collisions), "a passthrough sample must contain a contact group"
    else:
        assert all(len(event["group_colors"]) == 1 for event in collisions), "a plain sample must avoid contact groups"


def _gen_attempt() -> np.ndarray | None:
    layout, require_passthrough = _sample_layout()
    total_objects = 1 + sum(layout)
    colors = random.sample(PALETTE, total_objects + max(DISTRACTOR_COUNT_WEIGHTS))

    height = random.randint(*GRID_H_RANGE)
    lane = random.randint(2, height - 3)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    marker_w = random.randint(*MARKER_W_RANGE)

    chain_specs = []
    for _ in range(total_objects):
        obj_h = random.randint(*RECT_H_RANGE)
        obj_w = random.randint(*RECT_W_RANGE)
        top = _sample_band_top(lane, obj_h, height)
        if top is None:
            return None
        chain_specs.append((top, obj_h, obj_w))

    start_top, start_h, start_w = chain_specs[0]
    marker_h = random.randint(1, start_h)
    marker_top = random.randint(start_top, start_top + start_h - marker_h)

    gaps = [random.randint(*GAP_RANGE) for _ in range(len(layout))]
    final_gap = random.randint(*GAP_RANGE)

    cursor = left_margin + marker_w
    width_needed = cursor + start_w
    index = 0
    color_iter = iter(colors)
    chain_colors = [next(color_iter) for _ in range(total_objects)]

    for group_idx, group_size in enumerate((1,) + layout):
        if group_idx == 0:
            top, obj_h, obj_w = chain_specs[index]
            index += 1
            width_needed = cursor + obj_w
            cursor += obj_w
            continue

        cursor += gaps[group_idx - 1]
        for inner_idx in range(group_size):
            top, obj_h, obj_w = chain_specs[index]
            index += 1
            width_needed = max(width_needed, cursor + obj_w)
            cursor += obj_w

    width = width_needed + final_gap
    if width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    marker_left = left_margin
    grid[marker_top : marker_top + marker_h, marker_left : marker_left + marker_w] = MARKER

    start_left = left_margin + marker_w
    start_color = chain_colors[0]
    start_top, start_h, start_w = chain_specs[0]
    _paint_rect(grid, start_top, start_left, start_h, start_w, start_color)

    cursor = start_left + start_w
    spec_index = 1
    color_index = 1
    for group_idx, group_size in enumerate(layout):
        cursor += gaps[group_idx]
        for inner_idx in range(group_size):
            top, obj_h, obj_w = chain_specs[spec_index]
            spec_index += 1
            color = chain_colors[color_index]
            color_index += 1
            _paint_rect(grid, top, cursor, obj_h, obj_w, color)
            cursor += obj_w

    for _ in range(random.choice(DISTRACTOR_COUNT_WEIGHTS)):
        color = next(color_iter)
        placed = False
        for _ in range(50):
            h = random.randint(*DISTRACTOR_SIZE_RANGE)
            w = random.randint(*DISTRACTOR_SIZE_RANGE)
            if h >= height or w >= width:
                continue
            r0 = random.randint(0, height - h)
            c0 = random.randint(0, width - w)
            if _can_place_with_padding(grid, r0, c0, h, w, lane):
                grid[r0 : r0 + h, c0 : c0 + w] = color
                placed = True
                break
        if not placed:
            break

    rotated = _rotate_random(grid)
    try:
        _validate_generated_grid(
            rotated,
            expected_groups=len(layout),
            require_passthrough=require_passthrough,
        )
    except AssertionError:
        return None
    return rotated


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid momentum-relay input")
