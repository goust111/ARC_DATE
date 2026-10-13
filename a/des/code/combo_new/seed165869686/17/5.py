from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ARROW_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _arrow_mask(direction: str, body_len: int) -> np.ndarray:
    assert body_len >= 3
    if direction in ("right", "left"):
        patch = np.zeros((3, body_len + 1), dtype=bool)
        if direction == "right":
            patch[0, :body_len] = True
            patch[2, :body_len] = True
            patch[1, 0] = True
            patch[1, body_len - 1] = True
            patch[1, body_len] = True
        else:
            patch[0, 1:] = True
            patch[2, 1:] = True
            patch[1, 0] = True
            patch[1, 1] = True
            patch[1, body_len] = True
        return patch

    patch = np.zeros((body_len + 1, 3), dtype=bool)
    if direction == "down":
        patch[:body_len, 0] = True
        patch[:body_len, 2] = True
        patch[0, :] = True
        patch[body_len - 1, :] = True
        patch[body_len, 1] = True
    else:
        patch[1:, 0] = True
        patch[1:, 2] = True
        patch[1, :] = True
        patch[body_len, :] = True
        patch[0, 1] = True
    return patch


def _decode_arrow(coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True

    matches: list[tuple[str, int]] = []
    if h == 3 and w >= 4:
        body_len = w - 1
        for direction in ("right", "left"):
            if np.array_equal(patch, _arrow_mask(direction, body_len)):
                matches.append((direction, body_len))
    if w == 3 and h >= 4:
        body_len = h - 1
        for direction in ("down", "up"):
            if np.array_equal(patch, _arrow_mask(direction, body_len)):
                matches.append((direction, body_len))

    assert len(matches) == 1, "each gray component must be a single arrow"
    direction, body_len = matches[0]

    if direction == "right":
        tip = (r0 + 1, c0 + body_len)
        inner = (r0 + 1, c0 + 1, 1, body_len - 2)
    elif direction == "left":
        tip = (r0 + 1, c0)
        inner = (r0 + 1, c0 + 2, 1, body_len - 2)
    elif direction == "down":
        tip = (r0 + body_len, c0 + 1)
        inner = (r0 + 1, c0 + 1, body_len - 2, 1)
    else:
        tip = (r0, c0 + 1)
        inner = (r0 + 2, c0 + 1, body_len - 2, 1)

    assert inner[2] >= 1 and inner[3] >= 1, "arrow body must have a non-empty interior"
    return {
        "direction": direction,
        "tip": tip,
        "step": DIRS[direction],
        "inner": inner,
    }


def _scan_first_object(active_mask: np.ndarray, tip: tuple[int, int], step: tuple[int, int]) -> tuple[int, int] | None:
    h, w = active_mask.shape
    r, c = tip[0] + step[0], tip[1] + step[1]
    while 0 <= r < h and 0 <= c < w:
        if active_mask[r, c]:
            return r, c
        r += step[0]
        c += step[1]
    return None


def _biased_origin(
    direction: str,
    inner: tuple[int, int, int, int],
    obj_shape: tuple[int, int],
) -> tuple[int, int]:
    inner_r, inner_c, inner_h, inner_w = inner
    obj_h, obj_w = map(int, obj_shape)
    assert obj_h <= inner_h and obj_w <= inner_w, "captured object must fit in the arrow body"

    extra_h = inner_h - obj_h
    extra_w = inner_w - obj_w
    top_gap = extra_h // 2
    left_gap = extra_w // 2

    if direction == "right":
        left_gap = (extra_w + 1) // 2
    elif direction == "down":
        top_gap = (extra_h + 1) // 2

    return inner_r + top_gap, inner_c + left_gap


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    view = dst[top : top + patch.shape[0], left : left + patch.shape[1]]
    view[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Capture each first ray-hit colored object into the hollow body of the gray arrow that points at it.

    Input structure:
    - The background is black.
    - Gray 4-connected components are arrows: a 3-pixel-wide hollow rectangular body with a pointed head on one short side.
    - Every arrow points up, down, left, or right, and its body interior is empty in the input.
    - On each arrow's center ray in front of the tip, the first non-black, non-gray 4-connected component is the target object.

    Rule:
    - Decode each gray arrow's pointing direction from its head and locate the tip plus the hollow body interior.
    - From the tip, scan forward along the arrow axis until the first colored object is reached.
    - Remove that whole object from its original location.
    - Paste the object's original shape and colors into the arrow body's interior, centered there; when odd leftover space cannot be split evenly, place the object one cell closer to the arrow head along the pointing axis.

    Key insight:
    - The arrow is both a direction cue and a container: the head determines which ray to search, while the hollow shaft is the destination slot for the first object on that ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    arrows = [_decode_arrow(coords) for coords in _components(g == ARROW_COLOR)]
    assert arrows, "expected at least one gray arrow"

    active_mask = (g != BG) & (g != ARROW_COLOR)
    object_coords = _components(active_mask)
    assert object_coords, "expected at least one colored object"

    object_data: list[dict] = []
    by_cell: dict[tuple[int, int], int] = {}
    for idx, coords in enumerate(object_coords):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        object_data.append({"coords": coords, "patch": patch})
        for r, c in coords:
            by_cell[(int(r), int(c))] = idx

    captures: list[tuple[int, tuple[int, int]]] = []
    for arrow in arrows:
        seed = _scan_first_object(active_mask, arrow["tip"], arrow["step"])
        assert seed is not None, "each arrow must see a colored object on its ray"
        obj_idx = by_cell[seed]
        patch = object_data[obj_idx]["patch"]
        origin = _biased_origin(arrow["direction"], arrow["inner"], patch.shape)
        captures.append((obj_idx, origin))

    out = g.copy()
    for obj_idx in {obj_idx for obj_idx, _origin in captures}:
        coords = object_data[obj_idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for obj_idx, (top, left) in captures:
        _paste_non_bg(out, object_data[obj_idx]["patch"], top, left)

    return out


def _scene_footprint(spec: dict) -> tuple[int, int]:
    span = spec["body_len"] + 1 + spec["gap"] + spec["target_len"]
    if spec["direction"] in ("right", "left"):
        return 3, span
    return span, 3


def _scene_positions(spec: dict, top: int, left: int) -> dict:
    direction = spec["direction"]
    body_len = spec["body_len"]
    gap = spec["gap"]
    target_len = spec["target_len"]

    if direction == "right":
        return {
            "arrow_top": top,
            "arrow_left": left,
            "obj_top": top + 1,
            "obj_left": left + body_len + 1 + gap,
        }
    if direction == "left":
        return {
            "arrow_top": top,
            "arrow_left": left + target_len + gap,
            "obj_top": top + 1,
            "obj_left": left,
        }
    if direction == "down":
        return {
            "arrow_top": top,
            "arrow_left": left,
            "obj_top": top + body_len + 1 + gap,
            "obj_left": left + 1,
        }
    return {
        "arrow_top": top + target_len + gap,
        "arrow_left": left,
        "obj_top": top,
        "obj_left": left + 1,
    }


def _can_place_box(occupied: np.ndarray, top: int, left: int, box_h: int, box_w: int, gap: int = 1) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + box_h > h or left + box_w > w:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + box_h + gap)
    c1 = min(w, left + box_w + gap)
    return not occupied[r0:r1, c0:c1].any()


def _render_scene(grid: np.ndarray, spec: dict) -> None:
    arrow_mask = _arrow_mask(spec["direction"], spec["body_len"])
    at, al = spec["arrow_top"], spec["arrow_left"]
    view = grid[at : at + arrow_mask.shape[0], al : al + arrow_mask.shape[1]]
    view[arrow_mask] = ARROW_COLOR

    if spec["direction"] in ("right", "left"):
        grid[spec["obj_top"], spec["obj_left"] : spec["obj_left"] + spec["target_len"]] = spec["color"]
    else:
        grid[spec["obj_top"] : spec["obj_top"] + spec["target_len"], spec["obj_left"]] = spec["color"]


def _expected_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        arrow_mask = _arrow_mask(spec["direction"], spec["body_len"])
        at, al = spec["arrow_top"], spec["arrow_left"]
        out_view = out[at : at + arrow_mask.shape[0], al : al + arrow_mask.shape[1]]
        out_view[arrow_mask] = ARROW_COLOR

    for spec in specs:
        if spec["direction"] in ("right", "left"):
            patch = np.full((1, spec["target_len"]), spec["color"], dtype=int)
        else:
            patch = np.full((spec["target_len"], 1), spec["color"], dtype=int)
        direction = spec["direction"]
        body_len = spec["body_len"]
        if direction == "right":
            inner = (spec["arrow_top"] + 1, spec["arrow_left"] + 1, 1, body_len - 2)
        elif direction == "left":
            inner = (spec["arrow_top"] + 1, spec["arrow_left"] + 2, 1, body_len - 2)
        elif direction == "down":
            inner = (spec["arrow_top"] + 1, spec["arrow_left"] + 1, body_len - 2, 1)
        else:
            inner = (spec["arrow_top"] + 2, spec["arrow_left"] + 1, body_len - 2, 1)
        top, left = _biased_origin(direction, inner, patch.shape)
        _paste_non_bg(out, patch, top, left)
    return out


def _sample_specs() -> list[dict]:
    count = random.randint(1, 4)
    colors = random.sample(COLOR_POOL, count)
    return [
        {
            "direction": random.choice(tuple(DIRS)),
            "body_len": random.randint(5, 10),
            "gap": random.randint(1, 6),
            "color": color,
        }
        for color in colors
    ]


def _finish_specs(specs: list[dict]) -> bool:
    for spec in specs:
        inner_len = spec["body_len"] - 2
        if inner_len <= 1:
            return False
        spec["target_len"] = random.randint(1, inner_len - 1)
        spec["foot_h"], spec["foot_w"] = _scene_footprint(spec)
    return True


def _choose_canvas(specs: list[dict]) -> tuple[int, int] | None:
    min_h = max(spec["foot_h"] for spec in specs) + 2
    min_w = max(spec["foot_w"] for spec in specs) + 2
    need_area = sum((spec["foot_h"] + 2) * (spec["foot_w"] + 2) for spec in specs)
    candidates = [
        (h, w)
        for h in range(max(14, min_h), 31)
        for w in range(max(14, min_w), 31)
        if h * w >= need_area
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _place_specs(specs: list[dict], shape: tuple[int, int]) -> bool:
    occupied = np.zeros(shape, dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["foot_h"] * specs[idx]["foot_w"], reverse=True)

    for idx in order:
        spec = specs[idx]
        placed = False
        for _ in range(400):
            top = random.randint(0, shape[0] - spec["foot_h"])
            left = random.randint(0, shape[1] - spec["foot_w"])
            if not _can_place_box(occupied, top, left, spec["foot_h"], spec["foot_w"]):
                continue
            occupied[top : top + spec["foot_h"], left : left + spec["foot_w"]] = True
            spec.update(_scene_positions(spec, top, left))
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if not _finish_specs(specs):
        return None
    canvas = _choose_canvas(specs)
    if canvas is None:
        return None
    if not _place_specs(specs, canvas):
        return None

    inp = np.zeros(canvas, dtype=int)
    for spec in specs:
        _render_scene(inp, spec)

    expected = _expected_from_specs(canvas, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-capture input")
