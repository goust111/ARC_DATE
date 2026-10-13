import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import colors_present, find_components, get_bg


MARKER_COLOR = 9
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch the unique boundary-marked object, then propagate the slide only through objects hit on their marked face.

    Input structure:
    - The background is the dominant color.
    - Several disjoint monochromatic 4-connected foreground objects each carry exactly one maroon contact marker on one cardinal side.
    - Exactly one object's marker lies on the grid boundary; that object is the unique launcher.

    Rule:
    - The launcher slides in the direction away from its own marker until it becomes edge-adjacent to one other object or can move no farther without leaving the grid.
    - When an object is hit, it activates only if the incoming motion reaches the side where that object's marker sits; then the hit object slides away from its own marker.
    - If the hit side does not match that marked side, the struck object stays fixed and the chain stops. Unactivated objects never move, and each marker pixel moves with its own object.

    Key insight:
    - Motion is not determined by the previous mover alone: every object stores its own launch direction in its maroon side marker, and the chain survives only when the collision face matches that stored side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = int(get_bg(g))
    assert bg != MARKER_COLOR
    objects = _parse_objects(g, bg)
    final_objects = _simulate(objects, g.shape)
    return _render(final_objects, g.shape, bg)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _gen_attempt()
        if built is None:
            continue
        grid, meta = built
        if _validate_generated_case(grid, meta):
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")


def _parse_objects(grid: np.ndarray, bg: int) -> list[dict]:
    marker_components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[MARKER_COLOR],
        return_format="dict",
    )
    marker_coords = [tuple(map(int, comp["coords"][0])) for comp in marker_components]
    assert all(int(comp["area"]) == 1 for comp in marker_components), "each marker must be a singleton"
    marker_set = set(marker_coords)

    objects = []
    used_markers = set()
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color == MARKER_COLOR:
            continue

        body = comp["coords"].astype(int, copy=True)
        body_set = {tuple(map(int, cell)) for cell in body.tolist()}
        adjacent_markers = []
        for mr, mc in marker_set:
            for br, bc in body_set:
                if abs(mr - br) + abs(mc - bc) == 1:
                    adjacent_markers.append((mr, mc))
                    break

        assert len(adjacent_markers) == 1, "each object must touch exactly one marker"
        marker = adjacent_markers[0]
        assert marker not in used_markers, "markers must belong to unique objects"
        used_markers.add(marker)

        side_vectors = set()
        for nr, nc in _neighbors4(*marker):
            if (nr, nc) in body_set:
                side_vectors.add((marker[0] - nr, marker[1] - nc))
        assert len(side_vectors) == 1, "marker contact side must be unambiguous"
        marker_side = next(iter(side_vectors))

        objects.append(
            {
                "color": color,
                "body": body,
                "marker": marker,
                "marker_side": marker_side,
                "move_dir": (-marker_side[0], -marker_side[1]),
            }
        )

    assert objects, "expected at least one foreground object"
    assert len(used_markers) == len(marker_set), "every marker must belong to an object"
    start_count = sum(1 for obj in objects if _on_boundary(obj["marker"], grid.shape))
    assert start_count == 1, "expected exactly one boundary-touching launcher"
    return objects


def _simulate(objects: list[dict], shape: tuple[int, int], *, allow_any: bool = False, max_activated: int | None = None) -> list[dict]:
    objs = _clone_objects(objects)
    current = _start_index(objs, shape)
    activated = 1
    seen = {current}

    while True:
        target = _slide_until_stop(objs, current, shape)
        if target is None:
            break
        if max_activated is not None and activated >= max_activated:
            break

        current_dir = objs[current]["move_dir"]
        if not allow_any and objs[target]["marker_side"] != (-current_dir[0], -current_dir[1]):
            break

        assert target not in seen, "chain should not revisit an earlier object"
        current = target
        seen.add(current)
        activated += 1

    return objs


def _slide_until_stop(objects: list[dict], idx: int, shape: tuple[int, int]) -> int | None:
    mover = objects[idx]
    dr, dc = mover["move_dir"]
    while True:
        next_full = _shifted_full_set(mover, dr, dc)
        if not _all_in_bounds(next_full, shape):
            return None

        adjacent_targets = []
        for j, other in enumerate(objects):
            if j == idx:
                continue
            other_full = _full_set(other)
            assert not (next_full & other_full), "moving objects may not overlap"
            if _sets_adjacent(next_full, other_full):
                adjacent_targets.append(j)

        if adjacent_targets:
            assert len(adjacent_targets) == 1, "a slide must hit at most one object"
            _translate_in_place(mover, dr, dc)
            return adjacent_targets[0]

        _translate_in_place(mover, dr, dc)


def _render(objects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    occupied = {}
    for obj in objects:
        for r, c in obj["body"]:
            rr, cc = int(r), int(c)
            assert 0 <= rr < shape[0] and 0 <= cc < shape[1]
            prev = occupied.get((rr, cc))
            assert prev is None or prev == obj["color"]
            occupied[(rr, cc)] = obj["color"]
        mr, mc = obj["marker"]
        assert 0 <= mr < shape[0] and 0 <= mc < shape[1]
        assert (mr, mc) not in occupied
        occupied[(mr, mc)] = MARKER_COLOR

    for (r, c), color in occupied.items():
        out[r, c] = color
    return out


def _solve_array(grid: np.ndarray, *, allow_any: bool = False, max_activated: int | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    objects = _parse_objects(g, bg)
    final_objects = _simulate(objects, g.shape, allow_any=allow_any, max_activated=max_activated)
    return _render(final_objects, g.shape, bg)


def _validate_generated_case(grid: np.ndarray, meta: dict) -> bool:
    try:
        out = main(grid)
        bg = int(get_bg(grid))
        assert grid.ndim == 2 and out.ndim == 2
        assert max(grid.shape) <= 30 and max(out.shape) <= 30
        assert not np.array_equal(grid, out), "the puzzle must change the grid"

        moved_colors = _moved_colors(grid, out, bg)
        assert moved_colors == set(meta["active_colors"])

        start_only = _solve_array(grid, max_activated=1)
        assert not np.array_equal(out, start_only), "samples must require propagation beyond the launcher"

        if meta["has_dead"]:
            permissive = _solve_array(grid, allow_any=True)
            assert not np.array_equal(out, permissive), "samples with a dead hit must reject unconditional propagation"

        untouched_colors = set(colors_present(grid, exclude={bg, MARKER_COLOR})) - moved_colors
        assert untouched_colors, "at least one object should remain still"
    except AssertionError:
        return False
    return True


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    branch = random.choice(["dead", "boundary"])
    active_count = random.randint(2, 4)
    extra_after_dead = random.randint(0, 1) if branch == "dead" else 0
    decoy_count = random.randint(1, 2)
    corridor_count = active_count + (1 if branch == "dead" else 0) + extra_after_dead
    total_objects = corridor_count + decoy_count

    bg = random.choice([color for color in range(10) if color != MARKER_COLOR])
    body_colors = random.sample([color for color in range(10) if color not in {bg, MARKER_COLOR}], total_objects)

    widths = []
    heights = []
    for _ in range(corridor_count):
        h, w = _random_rect_dims()
        heights.append(h)
        widths.append(w)

    gaps = [random.randint(3, 5) for _ in range(corridor_count - 1)]
    corridor_span = 1 + sum(widths) + sum(gaps)
    w_min = corridor_span + 3
    if w_min > 28:
        return None
    width = random.randint(w_min, min(28, w_min + 3))

    max_h = max(heights) if heights else 3
    h_min = max(11, max_h + 8)
    if h_min > 20:
        return None
    height = random.randint(h_min, min(20, h_min + 3))

    contact_row = random.randint(3, height - 4)
    objects = []
    occupied = set()
    left = 1

    for i in range(corridor_count):
        h = heights[i]
        w = widths[i]
        top_lo = max(2, contact_row - h + 1)
        top_hi = min(contact_row, height - h - 2)
        if top_lo > top_hi:
            return None
        top = random.randint(top_lo, top_hi)
        color = body_colors[i]

        body = {(r, c) for r in range(top, top + h) for c in range(left, left + w)}
        if i < active_count:
            marker = (contact_row, left - 1)
        elif i == active_count and branch == "dead":
            marker = _dead_marker(top, left, h, w)
            if marker is None:
                return None
        else:
            marker = _random_marker(top, left, h, w, forbid_left=False)
            if marker is None:
                return None

        full = set(body)
        full.add(marker)
        if not _can_place(full, occupied):
            return None

        objects.append({"color": color, "body": body, "marker": marker})
        occupied |= full
        if i < corridor_count - 1:
            left += w + gaps[i]

    band_lo = contact_row - 2
    band_hi = contact_row + 2
    decoy_specs = []
    for j in range(decoy_count):
        color = body_colors[corridor_count + j]
        placed = False
        for _ in range(80):
            h, w = _random_rect_dims()
            valid_tops = [
                top
                for top in range(1, height - h - 1)
                if top + h - 1 < band_lo or top > band_hi
            ]
            if not valid_tops:
                continue
            top = random.choice(valid_tops)
            left = random.randint(2, width - w - 2)
            marker = _random_marker(top, left, h, w, forbid_left=False)
            if marker is None:
                continue
            body = {(r, c) for r in range(top, top + h) for c in range(left, left + w)}
            full = set(body)
            full.add(marker)
            if not _can_place(full, occupied):
                continue
            objects.append({"color": color, "body": body, "marker": marker})
            occupied |= full
            decoy_specs.append(color)
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), bg, dtype=int)
    for obj in objects:
        for r, c in obj["body"]:
            grid[r, c] = obj["color"]
        mr, mc = obj["marker"]
        grid[mr, mc] = MARKER_COLOR

    rotation_k = random.randint(0, 3)
    if rotation_k:
        grid = np.rot90(grid, k=rotation_k)

    meta = {
        "active_colors": body_colors[:active_count],
        "has_dead": branch == "dead",
    }
    return grid, meta


def _moved_colors(inp: np.ndarray, out: np.ndarray, bg: int) -> set[int]:
    moved = set()
    for color in colors_present(inp, exclude={bg, MARKER_COLOR}):
        if not np.array_equal(inp == color, out == color):
            moved.add(int(color))
    return moved


def _clone_objects(objects: list[dict]) -> list[dict]:
    return [
        {
            "color": int(obj["color"]),
            "body": obj["body"].copy(),
            "marker": tuple(map(int, obj["marker"])),
            "marker_side": tuple(map(int, obj["marker_side"])),
            "move_dir": tuple(map(int, obj["move_dir"])),
        }
        for obj in objects
    ]


def _start_index(objects: list[dict], shape: tuple[int, int]) -> int:
    starts = [i for i, obj in enumerate(objects) if _on_boundary(obj["marker"], shape)]
    assert len(starts) == 1
    return starts[0]


def _shifted_full_set(obj: dict, dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r) + dr, int(c) + dc) for r, c in obj["body"]} | {
        (obj["marker"][0] + dr, obj["marker"][1] + dc)
    }


def _translate_in_place(obj: dict, dr: int, dc: int) -> None:
    obj["body"][:, 0] += int(dr)
    obj["body"][:, 1] += int(dc)
    mr, mc = obj["marker"]
    obj["marker"] = (mr + int(dr), mc + int(dc))


def _full_set(obj: dict) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in obj["body"].tolist()} | {tuple(map(int, obj["marker"]))}


def _all_in_bounds(coords: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _sets_adjacent(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _can_place(candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    if candidate & occupied:
        return False
    for r, c in candidate:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _on_boundary(coord: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = map(int, coord)
    h, w = map(int, shape)
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _random_rect_dims() -> tuple[int, int]:
    while True:
        h = random.randint(1, 3)
        w = random.randint(1, 3)
        if h * w >= 2:
            return h, w


def _dead_marker(top: int, left: int, h: int, w: int) -> tuple[int, int] | None:
    choices = []
    if top >= 2:
        choices.append((top - 1, random.randint(left, left + w - 1)))
    if top + h <= 27:
        choices.append((top + h, random.randint(left, left + w - 1)))
    return random.choice(choices) if choices else None


def _random_marker(top: int, left: int, h: int, w: int, *, forbid_left: bool) -> tuple[int, int] | None:
    choices = []
    if top >= 1:
        choices.extend((top - 1, c) for c in range(left, left + w))
    choices.extend((top + h, c) for c in range(left, left + w))
    if not forbid_left and left >= 1:
        choices.extend((r, left - 1) for r in range(top, top + h))
    choices.extend((r, left + w) for r in range(top, top + h))
    choices = [coord for coord in choices if coord[0] >= 1 and coord[1] >= 1]
    return random.choice(choices) if choices else None
