from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_RETRIES = 600
SHAPE_LIBRARY = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
]


def _coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask > 0).astype(int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0, keepdims=True)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _shape_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    norm = _normalize(coords)
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=int)
    mask[norm[:, 0], norm[:, 1]] = 1
    return mask


def _random_shape(*, stack_friendly: bool = False) -> np.ndarray:
    for _ in range(80):
        mask = SHAPE_LIBRARY[random.randrange(len(SHAPE_LIBRARY))].copy()
        mask = np.rot90(mask, k=random.randrange(4))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = _shape_mask_from_coords(_coords(mask))
        area = int(mask.sum())
        if not (3 <= area <= 6):
            continue
        h, w = mask.shape
        if stack_friendly and (h < 2 or w < 2):
            continue
        return mask
    raise RuntimeError("failed to sample shape")


def _extract_objects(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objs: list[dict] = []
    for r in range(1, h - 1):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 1 <= nr < h - 1 and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            b = _bbox(coords)
            objs.append({"color": color, "coords": coords, "bbox": b})
    return objs


def _render_objects(h: int, w: int, blue_objs: list[dict], red_objs: list[dict]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    out[0, :] = BLUE
    out[h - 1, :] = RED
    for obj in blue_objs + red_objs:
        coords = np.asarray(obj["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return out


def _slide_one(coords: np.ndarray, occupied: np.ndarray | None, dr: int, h: int) -> np.ndarray:
    moved = coords.copy()
    while True:
        trial = moved + np.array([dr, 0], dtype=int)
        if int(trial[:, 0].min()) < 1 or int(trial[:, 0].max()) > h - 2:
            return moved
        if occupied is not None and np.any(occupied[trial[:, 0], trial[:, 1]]):
            return moved
        moved = trial


def _slide_objects(blue_objs: list[dict], red_objs: list[dict], h: int, w: int, *, with_collision: bool) -> tuple[list[dict], list[dict]]:
    moved_blue: list[dict] = [None] * len(blue_objs)  # type: ignore[assignment]
    moved_red: list[dict] = [None] * len(red_objs)  # type: ignore[assignment]

    blue_occ = np.zeros((h, w), dtype=bool)
    red_occ = np.zeros((h, w), dtype=bool)

    blue_order = sorted(range(len(blue_objs)), key=lambda i: blue_objs[i]["bbox"][0])
    red_order = sorted(range(len(red_objs)), key=lambda i: red_objs[i]["bbox"][2], reverse=True)

    for idx in blue_order:
        coords = np.asarray(blue_objs[idx]["coords"], dtype=int)
        landed = _slide_one(coords, blue_occ if with_collision else None, -1, h)
        blue_occ[landed[:, 0], landed[:, 1]] = True
        moved_blue[idx] = {"color": BLUE, "coords": landed, "bbox": _bbox(landed)}

    for idx in red_order:
        coords = np.asarray(red_objs[idx]["coords"], dtype=int)
        landed = _slide_one(coords, red_occ if with_collision else None, 1, h)
        red_occ[landed[:, 0], landed[:, 1]] = True
        moved_red[idx] = {"color": RED, "coords": landed, "bbox": _bbox(landed)}

    return moved_blue, moved_red


def _marker_status(scene: np.ndarray, r: int, c: int) -> dict:
    if int(scene[r, c]) != BG:
        return {"active": False, "reason": "occupied"}

    up = None
    for rr in range(r - 1, -1, -1):
        value = int(scene[rr, c])
        if value not in (BG, GREEN):
            up = (rr, value)
            break

    down = None
    for rr in range(r + 1, scene.shape[0]):
        value = int(scene[rr, c])
        if value not in (BG, GREEN):
            down = (rr, value)
            break

    if up is None or down is None:
        return {"active": False, "reason": "missing_side"}

    if up[1] != down[1]:
        return {"active": False, "reason": "different_color", "up": up, "down": down}

    span = int(down[0] - up[0] + 1)
    if span > 5:
        return {"active": False, "reason": "too_far", "up": up, "down": down, "span": span}

    return {
        "active": True,
        "reason": "active",
        "color": int(up[1]),
        "top": int(up[0]),
        "bottom": int(down[0]),
        "segment": (int(c), int(up[0]), int(down[0]), int(up[1])),
    }


def _apply_markers(scene: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = scene.copy()
    active_segments: list[tuple[int, int, int]] = []
    for r, c in markers:
        status = _marker_status(scene, r, c)
        if status["active"]:
            active_segments.append((int(c), int(status["top"]), int(status["bottom"])))
    for c, r0, r1 in active_segments:
        out[r0 : r1 + 1, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide blue/red objects to their matching border bars, then use fixed green markers to bridge nearby same-color neighbors.

    Input structure:
    - The background is black.
    - Row 0 is a full-width blue bar and the last row is a full-width red bar.
    - Interior foreground objects are 4-connected blue or red sprites; isolated green single pixels are fixed markers.

    Rule:
    - Slide blue objects upward in bbox-top order until they touch row 1 or would collide with an already parked blue object; slide red objects downward in bbox-bottom order until they touch row H-2 or would collide with an already parked red object.
    - Green markers do not move.
    - After sliding, each green marker looks upward and downward in its own column for the nearest non-black, non-green pixels. If both exist, share the same color, and the inclusive span between them is at most 5 cells, paint that whole vertical span green; otherwise erase the marker.
    - The top and bottom border bars stay unchanged.

    Key insight:
    - The marker meaning is latent until after all same-color objects have been slid into their color-specific docking stacks; only the post-slide column geometry reveals which markers become bridges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 4
    assert np.all(g[0, :] == BLUE)
    assert np.all(g[h - 1, :] == RED)

    blue_objs = _extract_objects(g, BLUE)
    red_objs = _extract_objects(g, RED)
    moved_blue, moved_red = _slide_objects(blue_objs, red_objs, h, w, with_collision=True)
    base = _render_objects(h, w, moved_blue, moved_red)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    return _apply_markers(base, markers)


def _coords_to_object(coords: np.ndarray, color: int) -> dict:
    arr = np.asarray(coords, dtype=int)
    return {"color": color, "coords": arr, "bbox": _bbox(arr)}


def _shift_objects(objs: list[dict], dr: int) -> list[dict]:
    shifted = []
    for obj in objs:
        coords = np.asarray(obj["coords"], dtype=int) + np.array([dr, 0], dtype=int)
        shifted.append(_coords_to_object(coords, int(obj["color"])))
    return shifted


def _objects_separated(all_objs: list[dict]) -> bool:
    all_cells = []
    for idx, obj in enumerate(all_objs):
        for r, c in np.asarray(obj["coords"], dtype=int):
            all_cells.append((idx, int(r), int(c)))
    for i, r1, c1 in all_cells:
        for j, r2, c2 in all_cells:
            if i >= j:
                continue
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _canonical_landings(h: int, w: int, shape_specs: list[tuple[np.ndarray, int]], color: int) -> list[dict] | None:
    occupied = np.zeros((h, w), dtype=bool)
    objs: list[dict] = []
    for mask, left in shape_specs:
        cells = _coords(mask)
        start_r = h - 2 - mask.shape[0] if color == BLUE else 1
        coords = cells + np.array([start_r, left], dtype=int)
        landed = _slide_one(coords, occupied, -1 if color == BLUE else 1, h)
        obj = _coords_to_object(landed, color)
        if obj["bbox"][0] < 1 or obj["bbox"][2] > h - 2:
            return None
        occupied[landed[:, 0], landed[:, 1]] = True
        objs.append(obj)
    return objs


def _build_final_objects(h: int, w: int) -> tuple[list[dict], list[dict]] | None:
    blue_count = random.randint(2, 3)
    red_count = random.randint(2, 3)

    blue_specs: list[tuple[np.ndarray, int]] = []
    red_specs: list[tuple[np.ndarray, int]] = []

    for color_specs, count in ((blue_specs, blue_count), (red_specs, red_count)):
        first = _random_shape(stack_friendly=True)
        second = _random_shape(stack_friendly=True)
        w1, w2 = first.shape[1], second.shape[1]
        overlap_width = random.randint(1, min(w1, w2))
        left1 = random.randint(0, w - w1)
        left2_min = max(0, left1 - (w2 - overlap_width))
        left2_max = min(w - w2, left1 + w1 - overlap_width)
        if left2_min > left2_max:
            return None
        left2 = random.randint(left2_min, left2_max)
        color_specs.append((first, left1))
        color_specs.append((second, left2))
        for _ in range(count - 2):
            mask = _random_shape(stack_friendly=random.random() < 0.5)
            left = random.randint(0, w - mask.shape[1])
            color_specs.append((mask, left))

    blue_objs = _canonical_landings(h, w, blue_specs, BLUE)
    red_objs = _canonical_landings(h, w, red_specs, RED)
    if blue_objs is None or red_objs is None:
        return None

    all_objs = blue_objs + red_objs
    for obj in all_objs:
        r0, _c0, r1, _c1 = obj["bbox"]
        if r0 <= 0 or r1 >= h - 1:
            return None
    for i, a in enumerate(all_objs):
        for j, b in enumerate(all_objs):
            if i >= j:
                continue
            if np.any(
                (np.abs(a["coords"][:, None, 0] - b["coords"][None, :, 0]) <= 0)
                & (np.abs(a["coords"][:, None, 1] - b["coords"][None, :, 1]) <= 0)
            ):
                return None

    blue_bottom = max(obj["bbox"][2] for obj in blue_objs)
    red_top = min(obj["bbox"][0] for obj in red_objs)
    if red_top - blue_bottom < 3:
        return None
    return blue_objs, red_objs


def _search_blue_inputs(h: int, ordered_blue: list[dict], placed: list[dict], pos: int, prev_top: int) -> list[dict] | None:
    if pos == len(ordered_blue):
        return placed
    obj = ordered_blue[pos]
    max_shift = min(6, h - 3 - obj["bbox"][2])
    options = list(range(1, max_shift + 1))
    random.shuffle(options)
    options.sort(reverse=True)
    for shift in options:
        shifted = _shift_objects([obj], shift)[0]
        top = shifted["bbox"][0]
        if top <= prev_top or top < 2 or shifted["bbox"][2] > h - 3:
            continue
        if not _objects_separated(placed + [shifted]):
            continue
        found = _search_blue_inputs(h, ordered_blue, placed + [shifted], pos + 1, top)
        if found is not None:
            return found
    return None


def _search_red_inputs(h: int, ordered_red: list[dict], placed_fixed: list[dict], placed_red: list[dict], pos: int, prev_bottom: int) -> list[dict] | None:
    if pos == len(ordered_red):
        return placed_red
    obj = ordered_red[pos]
    max_shift = min(6, obj["bbox"][0] - 2)
    options = list(range(1, max_shift + 1))
    random.shuffle(options)
    options.sort(reverse=True)
    for shift in options:
        shifted = _shift_objects([obj], -shift)[0]
        bottom = shifted["bbox"][2]
        if bottom >= prev_bottom or shifted["bbox"][0] < 2 or bottom > h - 3:
            continue
        trial = placed_fixed + placed_red + [shifted]
        if not _objects_separated(trial):
            continue
        found = _search_red_inputs(h, ordered_red, placed_fixed, placed_red + [shifted], pos + 1, bottom)
        if found is not None:
            return found
    return None


def _find_input_objects(h: int, blue_final: list[dict], red_final: list[dict]) -> tuple[list[dict], list[dict]] | None:
    blue_order = sorted(blue_final, key=lambda obj: obj["bbox"][0])
    red_order = sorted(red_final, key=lambda obj: obj["bbox"][2], reverse=True)

    blue_placed = _search_blue_inputs(h, blue_order, [], 0, -1)
    if blue_placed is None:
        return None
    red_placed = _search_red_inputs(h, red_order, blue_placed, [], 0, h + 1)
    if red_placed is None:
        return None
    return blue_placed, red_placed


def _marker_is_isolated(scene: np.ndarray, markers: list[tuple[int, int]], r: int, c: int) -> bool:
    if int(scene[r, c]) != BG:
        return False
    marker_set = set(markers)
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if not (0 <= rr < scene.shape[0] and 0 <= cc < scene.shape[1]):
                continue
            if (rr, cc) in marker_set:
                return False
            if int(scene[rr, cc]) != BG:
                return False
    return True


def _select_markers(final_scene: np.ndarray, input_scene: np.ndarray) -> list[tuple[int, int]] | None:
    active_candidates = []
    inactive_by_reason: dict[str, list[tuple[int, int]]] = {}
    for r in range(2, final_scene.shape[0] - 2):
        for c in range(final_scene.shape[1]):
            if int(final_scene[r, c]) != BG or int(input_scene[r, c]) != BG:
                continue
            out_status = _marker_status(final_scene, r, c)
            in_status = _marker_status(input_scene, r, c)
            if out_status["active"] and not in_status["active"]:
                active_candidates.append((r, c, out_status))
            elif not out_status["active"]:
                inactive_by_reason.setdefault(str(out_status["reason"]), []).append((r, c))

    random.shuffle(active_candidates)
    chosen: list[tuple[int, int]] = []
    used_segments = set()
    colors_seen = set()
    for r, c, status in active_candidates:
        if status["segment"] in used_segments:
            continue
        if not _marker_is_isolated(input_scene, chosen, r, c):
            continue
        chosen.append((r, c))
        used_segments.add(status["segment"])
        colors_seen.add(int(status["color"]))
        if len(chosen) == 2 and len(colors_seen) == 2:
            break

    if len(chosen) < 2:
        for r, c, status in active_candidates:
            if status["segment"] in used_segments:
                continue
            if not _marker_is_isolated(input_scene, chosen, r, c):
                continue
            chosen.append((r, c))
            used_segments.add(status["segment"])
            if len(chosen) == 2:
                break

    if len(chosen) < 2:
        return None

    inactive_priority = ["different_color", "too_far", "missing_side"]
    reasons = inactive_priority[:]
    random.shuffle(reasons)
    reasons += [k for k in inactive_by_reason if k not in reasons]
    for reason in reasons:
        for r, c in random.sample(inactive_by_reason.get(reason, []), k=len(inactive_by_reason.get(reason, []))):
            if _marker_is_isolated(input_scene, chosen, r, c):
                return chosen + [(r, c)]
    return None


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, markers: list[tuple[int, int]]) -> bool:
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    blue_objs = _extract_objects(inp, BLUE)
    red_objs = _extract_objects(inp, RED)
    h, w = inp.shape

    moved_blue, moved_red = _slide_objects(blue_objs, red_objs, h, w, with_collision=True)
    moved_scene = _render_objects(h, w, moved_blue, moved_red)
    statuses = [_marker_status(moved_scene, r, c) for r, c in markers]
    if sum(int(s["active"]) for s in statuses) < 2:
        return False
    if sum(int(not s["active"]) for s in statuses) < 1:
        return False

    pre_scene = _render_objects(h, w, blue_objs, red_objs)
    pre_statuses = [_marker_status(pre_scene, r, c) for r, c in markers]
    if all(bool(s["active"]) for s in pre_statuses if s is not None):
        return False
    if np.array_equal(_apply_markers(pre_scene, markers), expected):
        return False

    no_collision_blue, no_collision_red = _slide_objects(blue_objs, red_objs, h, w, with_collision=False)
    no_collision_scene = _render_objects(h, w, no_collision_blue, no_collision_red)
    if np.array_equal(_apply_markers(no_collision_scene, markers), expected):
        return False

    if np.array_equal(inp, expected):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(13, 16)
    w = random.randint(13, 16)

    built = _build_final_objects(h, w)
    if built is None:
        return None
    blue_final, red_final = built

    found_inputs = _find_input_objects(h, blue_final, red_final)
    if found_inputs is None:
        return None
    blue_input, red_input = found_inputs
    input_scene = _render_objects(h, w, blue_input, red_input)
    final_scene = _render_objects(h, w, blue_final, red_final)

    markers = _select_markers(final_scene, input_scene)
    if markers is None:
        return None

    inp = input_scene.copy()
    for r, c in markers:
        inp[r, c] = GREEN

    expected = _apply_markers(final_scene, markers)
    if not _validate_semantics(inp, expected, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid slide-bridge ARC input")
