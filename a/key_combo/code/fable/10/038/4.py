from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(10))
RECT_SHAPES = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
)
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _solid_color(cells: np.ndarray) -> int:
    vals = set(map(int, np.ravel(cells)))
    assert len(vals) == 1, "frame side must be monochrome"
    return vals.pop()


def _parse_frame(grid: np.ndarray) -> tuple[int, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5, "expected a framed 2D grid"
    bg = int(g[0, 0])
    assert {int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])} == {bg}
    side_colors = {
        "top": _solid_color(g[0, 1:-1]),
        "bottom": _solid_color(g[-1, 1:-1]),
        "left": _solid_color(g[1:-1, 0]),
        "right": _solid_color(g[1:-1, -1]),
    }
    assert len({bg, *side_colors.values()}) == 5, "frame colors must all differ"
    return bg, side_colors


def _parse_objects(grid: np.ndarray, bg: int) -> list[dict]:
    interior = np.asarray(grid[1:-1, 1:-1], dtype=int)
    comps = _find_components(interior != bg)
    assert comps, "expected interior objects"
    objects: list[dict] = []
    for comp in comps:
        colors = Counter(int(interior[r, c]) for r, c in comp)
        assert len(colors) == 2, "each object must contain one body color and one driver color"
        driver_colors = [color for color, count in colors.items() if count == 1]
        assert len(driver_colors) == 1, "each object must have exactly one driver pixel"
        driver_color = int(driver_colors[0])
        body_color = int([color for color in colors if color != driver_color][0])
        abs_coords = [(r + 1, c + 1) for r, c in comp]
        driver_cells = [(r + 1, c + 1) for r, c in comp if int(interior[r, c]) == driver_color]
        assert len(driver_cells) == 1
        r0, c0, r1, c1 = _bbox(abs_coords)
        local_pixels = []
        for r, c in abs_coords:
            color = driver_color if (r, c) == driver_cells[0] else body_color
            local_pixels.append((r - r0, c - c0, color))
        objects.append(
            {
                "coords": abs_coords,
                "bbox": (r0, c0, r1, c1),
                "driver_color": driver_color,
                "body_color": body_color,
                "driver_cell": driver_cells[0],
                "pixels": local_pixels,
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
            }
        )
    return objects


def _distance_to_side(obj: dict, side: str, shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    r0, c0, r1, c1 = obj["bbox"]
    if side == "top":
        return int(r0 - 1)
    if side == "bottom":
        return int((h - 2) - r1)
    if side == "left":
        return int(c0 - 1)
    return int((w - 2) - c1)


def _solve(grid: np.ndarray, *, steer_by: str = "driver", obey_collisions: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, side_colors = _parse_frame(g)
    side_by_color = {color: side for side, color in side_colors.items()}
    objects = _parse_objects(g, bg)

    movers: list[dict] = []
    out = g.copy()
    for obj in objects:
        steer_color = int(obj["driver_color"] if steer_by == "driver" else obj["body_color"])
        side = side_by_color.get(steer_color)
        if side is None:
            continue
        move = dict(obj)
        move["side"] = side
        move["distance"] = _distance_to_side(move, side, g.shape)
        movers.append(move)
        for r, c in obj["coords"]:
            out[r, c] = bg

    distances = [int(obj["distance"]) for obj in movers]
    assert len(distances) == len(set(distances)), "moving distances must be unique"
    occupied = out != bg

    for obj in sorted(movers, key=lambda item: int(item["distance"])):
        dr, dc = DIRS[obj["side"]]
        step = 0
        while True:
            blocked = False
            for lr, lc, _color in obj["pixels"]:
                nr = obj["bbox"][0] + lr + dr * (step + 1)
                nc = obj["bbox"][1] + lc + dc * (step + 1)
                if not (1 <= nr < g.shape[0] - 1 and 1 <= nc < g.shape[1] - 1):
                    blocked = True
                    break
                if obey_collisions and occupied[nr, nc]:
                    blocked = True
                    break
            if blocked:
                break
            step += 1

        for lr, lc, color in obj["pixels"]:
            nr = obj["bbox"][0] + lr + dr * step
            nc = obj["bbox"][1] + lc + dc * step
            out[nr, nc] = color
            if obey_collisions:
                occupied[nr, nc] = True

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each multi-cell object toward the frame side named by its embedded driver pixel.

    Input structure:
    - The grid has a one-cell-thick rectangular frame whose four corners are background and whose four sides are solid, pairwise distinct colors.
    - Inside the frame sit several separated multi-cell objects; each object is mostly one body color but contains exactly one differently colored embedded driver pixel.
    - Some body colors match frame-side colors as decoys, and some driver colors match no frame side at all.

    Rule:
    - Read each object's direction from the driver pixel, not from the body color: a driver matching the top, bottom, left, or right frame side makes the whole object slide orthogonally toward that side.
    - Process moving objects in ascending order of their current distance to their target side.
    - A moving object stops when one more step would hit the frame or an already settled object; objects whose driver matches no side stay where they are. The frame and background remain unchanged.

    Key insight:
    - The singleton embedded pixel is the steering instruction, while the body color can be a deliberate lure; earlier arrivals become later objects' new walls, so the final layout depends on both the driver decoding and the distance order.
    """
    return _solve(grid, steer_by="driver", obey_collisions=True)


def _driver_candidates(h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            deg = 0
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    deg += 1
            if deg >= 3:
                cells.append((r, c))
    return cells


def _make_patch(h: int, w: int, body_color: int, driver_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.full((h, w), int(body_color), dtype=int)
    driver = random.choice(_driver_candidates(h, w))
    patch[driver] = int(driver_color)
    return patch, driver


def _path_box(
    top: int,
    left: int,
    h: int,
    w: int,
    side: str,
    grid_shape: tuple[int, int],
    *,
    blocked_top: int | None = None,
    blocked_left: int | None = None,
) -> tuple[int, int, int, int]:
    gh, gw = map(int, grid_shape)
    if side == "top":
        final_top = 1 if blocked_top is None else int(blocked_top)
        return final_top, left, top + h - 1, left + w - 1
    if side == "bottom":
        final_top = gh - 1 - h if blocked_top is None else int(blocked_top)
        return top, left, final_top + h - 1, left + w - 1
    if side == "left":
        final_left = 1 if blocked_left is None else int(blocked_left)
        return top, final_left, top + h - 1, left + w - 1
    final_left = gw - 1 - w if blocked_left is None else int(blocked_left)
    return top, left, top + h - 1, final_left + w - 1


def _touches_reserved(mask: np.ndarray, box: tuple[int, int, int, int], *, pad: int = 1) -> bool:
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    bg = int(grid[0, 0])
    assert np.all(view == bg), "placement overlap"
    view[:] = patch


def _random_shape(
    *,
    force_width: int | None = None,
    force_height: int | None = None,
) -> tuple[int, int]:
    options = [
        (h, w)
        for h, w in RECT_SHAPES
        if (force_width is None or w == force_width) and (force_height is None or h == force_height)
    ]
    return random.choice(options)


def _render_frame(shape: tuple[int, int], bg: int, side_colors: dict[str, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), int(bg), dtype=int)
    grid[0, 1:-1] = int(side_colors["top"])
    grid[-1, 1:-1] = int(side_colors["bottom"])
    grid[1:-1, 0] = int(side_colors["left"])
    grid[1:-1, -1] = int(side_colors["right"])
    return grid


def _available_distance_values(h: int, w: int) -> list[int]:
    return list(range(1, max(2, min(h, w) - 3)))


def _build_core_pair(
    grid_shape: tuple[int, int],
    direction: str,
    driver_color: int,
    body_a: int,
    body_b: int,
    used_distances: set[int],
) -> tuple[list[dict], tuple[int, int, int, int]] | None:
    gh, gw = grid_shape
    vertical = direction in {"top", "bottom"}
    span = random.randint(2, 4)
    if vertical and span == 2:
        h_a = random.choice((3, 4))
        h_b = random.choice((3, 4))
    elif vertical:
        h_a = random.choice((2, 3, 4))
        h_b = random.choice((2, 3, 4))
    elif span == 2:
        w_a = random.choice((3, 4))
        w_b = random.choice((3, 4))
    else:
        w_a = random.choice((2, 3, 4))
        w_b = random.choice((2, 3, 4))

    if vertical:
        w_a = w_b = span
    else:
        h_a = h_b = span

    patch_a, _ = _make_patch(h_a, w_a, body_a, driver_color)
    patch_b, _ = _make_patch(h_b, w_b, body_b, driver_color)
    values = [d for d in _available_distance_values(gh, gw) if d not in used_distances]
    random.shuffle(values)

    for d_a in values:
        for d_b in values:
            if d_b <= d_a:
                continue
            if direction == "top":
                top_a = 1 + d_a
                top_b = 1 + d_b
                if d_b - d_a < h_a + 1 or top_b + h_b - 1 > gh - 2:
                    continue
                left = random.randint(1, gw - w_a - 1)
                lane = (1, left, gh - 2, left + w_a - 1)
            elif direction == "bottom":
                top_a = gh - 1 - h_a - d_a
                top_b = gh - 1 - h_b - d_b
                if top_a < 1 or top_b < 1 or d_b - d_a < h_a + 1:
                    continue
                if top_b + h_b >= top_a:
                    continue
                left = random.randint(1, gw - w_a - 1)
                lane = (1, left, gh - 2, left + w_a - 1)
            elif direction == "left":
                left_a = 1 + d_a
                left_b = 1 + d_b
                if d_b - d_a < w_a + 1 or left_b + w_b - 1 > gw - 2:
                    continue
                top = random.randint(1, gh - h_a - 1)
                lane = (top, 1, top + h_a - 1, gw - 2)
            else:
                left_a = gw - 1 - w_a - d_a
                left_b = gw - 1 - w_b - d_b
                if left_a < 1 or left_b < 1 or d_b - d_a < w_a + 1:
                    continue
                if left_b + w_b >= left_a:
                    continue
                top = random.randint(1, gh - h_a - 1)
                lane = (top, 1, top + h_a - 1, gw - 2)

            if direction in {"top", "bottom"}:
                obj_a = {
                    "patch": patch_a,
                    "top": top_a,
                    "left": left,
                    "direction": direction,
                    "distance": d_a,
                    "body_color": body_a,
                    "driver_color": driver_color,
                    "core": "front",
                }
                obj_b = {
                    "patch": patch_b,
                    "top": top_b,
                    "left": left,
                    "direction": direction,
                    "distance": d_b,
                    "body_color": body_b,
                    "driver_color": driver_color,
                    "core": "rear",
                }
            else:
                obj_a = {
                    "patch": patch_a,
                    "top": top,
                    "left": left_a,
                    "direction": direction,
                    "distance": d_a,
                    "body_color": body_a,
                    "driver_color": driver_color,
                    "core": "front",
                }
                obj_b = {
                    "patch": patch_b,
                    "top": top,
                    "left": left_b,
                    "direction": direction,
                    "distance": d_b,
                    "body_color": body_b,
                    "driver_color": driver_color,
                    "core": "rear",
                }
            used_distances.add(d_a)
            used_distances.add(d_b)
            return [obj_a, obj_b], lane
    return None


def _sample_free_box(
    occupied: np.ndarray,
    blocked: np.ndarray,
    h: int,
    w: int,
    *,
    pad: int = 1,
) -> tuple[int, int] | None:
    choices: list[tuple[int, int]] = []
    for top in range(1, occupied.shape[0] - h - 1):
        for left in range(1, occupied.shape[1] - w - 1):
            box = (top, left, top + h - 1, left + w - 1)
            if _touches_reserved(occupied, box, pad=pad):
                continue
            if _touches_reserved(blocked, box, pad=0):
                continue
            choices.append((top, left))
    if not choices:
        return None
    return random.choice(choices)


def _extra_mover(
    grid_shape: tuple[int, int],
    occupied: np.ndarray,
    blocked: np.ndarray,
    direction: str,
    driver_color: int,
    body_color: int,
    used_distances: set[int],
) -> dict | None:
    gh, gw = grid_shape
    for _ in range(100):
        h, w = _random_shape()
        distances = [d for d in _available_distance_values(gh, gw) if d not in used_distances]
        if not distances:
            return None
        d = random.choice(distances)
        if direction == "top":
            top = 1 + d
            if top + h - 1 > gh - 2:
                continue
            left = random.randint(1, gw - w - 1)
        elif direction == "bottom":
            top = gh - 1 - h - d
            if top < 1:
                continue
            left = random.randint(1, gw - w - 1)
        elif direction == "left":
            left = 1 + d
            if left + w - 1 > gw - 2:
                continue
            top = random.randint(1, gh - h - 1)
        else:
            left = gw - 1 - w - d
            if left < 1:
                continue
            top = random.randint(1, gh - h - 1)

        box = (top, left, top + h - 1, left + w - 1)
        if _touches_reserved(occupied, box, pad=1):
            continue
        path = _path_box(top, left, h, w, direction, grid_shape)
        if _touches_reserved(blocked, path, pad=0) or _touches_reserved(occupied, path, pad=0):
            continue
        used_distances.add(d)
        patch, _ = _make_patch(h, w, body_color, driver_color)
        return {
            "patch": patch,
            "top": top,
            "left": left,
            "direction": direction,
            "distance": d,
            "body_color": body_color,
            "driver_color": driver_color,
        }
    return None


def _static_object(
    occupied: np.ndarray,
    blocked: np.ndarray,
    body_color: int,
    driver_color: int,
) -> dict | None:
    for _ in range(100):
        h, w = _random_shape()
        pos = _sample_free_box(occupied, blocked, h, w, pad=1)
        if pos is None:
            continue
        top, left = pos
        patch, _ = _make_patch(h, w, body_color, driver_color)
        return {
            "patch": patch,
            "top": top,
            "left": left,
            "direction": None,
            "distance": None,
            "body_color": body_color,
            "driver_color": driver_color,
        }
    return None


def _render_input(
    shape: tuple[int, int],
    bg: int,
    side_colors: dict[str, int],
    objects: list[dict],
) -> np.ndarray:
    grid = _render_frame(shape, bg, side_colors)
    for obj in objects:
        _place_patch(grid, obj["patch"], int(obj["top"]), int(obj["left"]))
    return grid


def _body_steered_output(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, steer_by="body", obey_collisions=True)


def _no_collision_output(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, steer_by="driver", obey_collisions=False)


def _validate_semantics(
    inp: np.ndarray,
    objects: list[dict],
    side_colors: dict[str, int],
    core_direction: str,
) -> bool:
    moving = [obj for obj in objects if obj["direction"] is not None]
    static = [obj for obj in objects if obj["direction"] is None]
    if len(moving) < 3 or len(static) < 1:
        return False
    if len({int(obj["distance"]) for obj in moving}) != len(moving):
        return False

    side_set = set(map(int, side_colors.values()))
    decoys = [
        obj
        for obj in objects
        if int(obj["body_color"]) in side_set and int(obj["body_color"]) != int(obj["driver_color"])
    ]
    if not decoys:
        return False
    if not any(int(obj["driver_color"]) not in side_set for obj in static):
        return False

    try:
        out = main(inp)
        alt_body = _body_steered_output(inp)
        alt_noclip = _no_collision_output(inp)
    except AssertionError:
        return False

    if np.array_equal(out, inp):
        return False
    if np.array_equal(out, alt_body):
        return False
    if np.array_equal(out, alt_noclip):
        return False

    core_front = [obj for obj in moving if obj.get("core") == "front"]
    core_rear = [obj for obj in moving if obj.get("core") == "rear"]
    if len(core_front) != 1 or len(core_rear) != 1:
        return False
    front = core_front[0]
    rear = core_rear[0]
    front_h, front_w = front["patch"].shape
    rear_h, rear_w = rear["patch"].shape
    if core_direction == "top":
        front_top = 1
        front_left = int(front["left"])
        rear_top = front_top + front_h
        rear_left = int(rear["left"])
        if rear_left != front_left:
            return False
    elif core_direction == "bottom":
        front_top = inp.shape[0] - 1 - front_h
        front_left = int(front["left"])
        rear_top = front_top - rear_h
        rear_left = int(rear["left"])
        if rear_left != front_left:
            return False
    elif core_direction == "left":
        front_top = int(front["top"])
        front_left = 1
        rear_top = int(rear["top"])
        rear_left = front_left + front_w
        if rear_top != front_top:
            return False
    else:
        front_top = int(front["top"])
        front_left = inp.shape[1] - 1 - front_w
        rear_top = int(rear["top"])
        rear_left = front_left - rear_w
        if rear_top != front_top:
            return False

    if not np.array_equal(
        out[front_top : front_top + front_h, front_left : front_left + front_w],
        front["patch"],
    ):
        return False
    if not np.array_equal(
        out[rear_top : rear_top + rear_h, rear_left : rear_left + rear_w],
        rear["patch"],
    ):
        return False

    if core_direction == "top":
        if rear_top != front_top + front_h:
            return False
    elif core_direction == "bottom":
        if rear_top + rear_h != front_top:
            return False
    elif core_direction == "left":
        if rear_left != front_left + front_w:
            return False
    else:
        if rear_left + rear_w != front_left:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    bg, top_c, bottom_c, left_c, right_c = random.sample(PALETTE, 5)
    side_colors = {"top": top_c, "bottom": bottom_c, "left": left_c, "right": right_c}
    side_set = set(side_colors.values())
    non_side = [color for color in PALETTE if color not in side_set and color != bg]
    if len(non_side) < 2:
        return None

    core_direction = random.choice(SIDES)
    used_distances: set[int] = set()
    body_a = random.choice([c for c in side_set if c != side_colors[core_direction]])
    body_b = random.choice([c for c in PALETTE if c != bg and c != body_a])
    core = _build_core_pair((h, w), core_direction, side_colors[core_direction], body_a, body_b, used_distances)
    if core is None:
        return None
    objects, core_lane = core

    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    _mark(blocked, core_lane)
    for obj in objects:
        ph, pw = obj["patch"].shape
        _mark(occupied, (obj["top"], obj["left"], obj["top"] + ph - 1, obj["left"] + pw - 1))

    static_body = random.choice([c for c in PALETTE if c != bg])
    static_driver = random.choice(non_side)
    static_obj = _static_object(occupied, blocked, static_body, static_driver)
    if static_obj is None:
        return None
    ph, pw = static_obj["patch"].shape
    _mark(occupied, (static_obj["top"], static_obj["left"], static_obj["top"] + ph - 1, static_obj["left"] + pw - 1))
    objects.append(static_obj)

    extra_dirs = list(SIDES)
    random.shuffle(extra_dirs)
    extra_count = random.randint(1, 3)
    extra_movers: list[dict] = []
    for direction in extra_dirs:
        if len(extra_movers) >= extra_count:
            break
        driver_color = side_colors[direction]
        body_color = random.choice([c for c in PALETTE if c != bg])
        mover = _extra_mover((h, w), occupied, blocked, direction, driver_color, body_color, used_distances)
        if mover is None:
            continue
        extra_movers.append(mover)
        ph, pw = mover["patch"].shape
        box = (mover["top"], mover["left"], mover["top"] + ph - 1, mover["left"] + pw - 1)
        path = _path_box(mover["top"], mover["left"], ph, pw, direction, (h, w))
        _mark(occupied, box)
        _mark(blocked, path)
    if not extra_movers:
        return None
    objects.extend(extra_movers)

    if random.random() < 0.55:
        extra_static = _static_object(
            occupied,
            blocked,
            random.choice([c for c in side_set if c != bg]),
            random.choice(non_side),
        )
        if extra_static is not None:
            ph, pw = extra_static["patch"].shape
            _mark(occupied, (extra_static["top"], extra_static["left"], extra_static["top"] + ph - 1, extra_static["left"] + pw - 1))
            objects.append(extra_static)

    inp = _render_input((h, w), bg, side_colors, objects)
    if not _validate_semantics(inp, objects, side_colors, core_direction):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid driver-and-body ARC input")
