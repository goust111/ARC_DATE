from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components


BACKGROUND = Color.BLACK
ARROW = Color.GREY
BODY_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK]
DIRS = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}

_STRUCT4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_STRUCT8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _is_connected(mask: np.ndarray, *, connectivity: int) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(cells))]
    seen = set()
    nbrs = _STRUCT4 if connectivity == 4 else _STRUCT8
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _build_body_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[tuple[int, ...], ...]]] = set()
    for area in range(3, 7):
        for cells in combinations(range(9), area):
            mask = np.zeros((3, 3), dtype=bool)
            for idx in cells:
                mask[idx // 3, idx % 3] = True
            trimmed = _trim_mask(mask)
            if min(trimmed.shape) < 2:
                continue
            if not _is_connected(trimmed, connectivity=4):
                continue
            key = (trimmed.shape, tuple(tuple(int(v) for v in row) for row in trimmed.astype(int)))
            if key not in seen:
                seen.add(key)
                masks.append(trimmed)
    return masks


BODY_MASKS = _build_body_masks()

ARROW_PATTERNS = {
    "right": (
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
    ),
    "left": (
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
    ),
    "up": (
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
    ),
    "down": (
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
    ),
}


def _valid_attachment_offsets(mask: np.ndarray, direction: str) -> list[int]:
    h, w = mask.shape
    offsets: list[int] = []
    if direction == "right":
        side = w - 1
        for r0 in range(h - 1):
            if bool(mask[r0 : r0 + 2, side].any()):
                offsets.append(r0)
    elif direction == "left":
        for r0 in range(h - 1):
            if bool(mask[r0 : r0 + 2, 0].any()):
                offsets.append(r0)
    elif direction == "up":
        for c0 in range(w - 1):
            if bool(mask[0, c0 : c0 + 2].any()):
                offsets.append(c0)
    else:
        side = h - 1
        for c0 in range(w - 1):
            if bool(mask[side, c0 : c0 + 2].any()):
                offsets.append(c0)
    return offsets


def _make_object_patch(direction: str, color: int) -> np.ndarray:
    while True:
        body_mask = random.choice(BODY_MASKS)
        offsets = _valid_attachment_offsets(body_mask, direction)
        if offsets:
            break

    h, w = body_mask.shape
    offset = random.choice(offsets)
    arrow_mask = random.choice(ARROW_PATTERNS[direction])
    body_patch = np.where(body_mask, int(color), 0)

    if direction == "right":
        patch = np.zeros((h, w + 2), dtype=int)
        patch[:, :w] = body_patch
        box = patch[offset : offset + 2, w : w + 2]
    elif direction == "left":
        patch = np.zeros((h, w + 2), dtype=int)
        patch[:, 2:] = body_patch
        box = patch[offset : offset + 2, 0:2]
    elif direction == "up":
        patch = np.zeros((h + 2, w), dtype=int)
        patch[2:, :] = body_patch
        box = patch[0:2, offset : offset + 2]
    else:
        patch = np.zeros((h + 2, w), dtype=int)
        patch[:h, :] = body_patch
        box = patch[h : h + 2, offset : offset + 2]

    box[arrow_mask] = ARROW
    assert _is_connected(patch != 0, connectivity=8)
    return patch


def _boundary_distance(bbox: tuple[int, int, int, int], direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if direction == "left":
        return int(c0)
    if direction == "right":
        return int(w - 1 - c1)
    if direction == "up":
        return int(r0)
    return int(h - 1 - r1)


def _infer_direction(body_bbox: tuple[int, int, int, int], arrow_bbox: tuple[int, int, int, int]) -> str:
    br0, bc0, br1, bc1 = body_bbox
    ar0, ac0, ar1, ac1 = arrow_bbox
    if ac0 > bc1:
        return "right"
    if ac1 < bc0:
        return "left"
    if ar0 > br1:
        return "down"
    if ar1 < br0:
        return "up"
    raise AssertionError("arrow must sit on exactly one outer side of the body")


def _parse_objects(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    objects: list[dict] = []
    for component in components:
        coords = component["coords"]
        values = grid[coords[:, 0], coords[:, 1]].astype(int)
        colors = set(map(int, values))
        assert ARROW in colors, "each object must contain a grey arrow"
        body_colors = sorted(colors - {BACKGROUND, ARROW})
        assert len(body_colors) == 1, "each object must have exactly one body color"
        body_color = int(body_colors[0])

        body_coords = coords[values == body_color]
        arrow_coords = coords[values == ARROW]
        assert 3 <= int(body_coords.shape[0]) <= 6
        assert int(arrow_coords.shape[0]) == 3
        arrow_bbox = bbox_coords(arrow_coords)
        assert (arrow_bbox[2] - arrow_bbox[0] + 1, arrow_bbox[3] - arrow_bbox[1] + 1) == (2, 2)

        body_bbox = bbox_coords(body_coords)
        full_bbox = component["bbox"]
        origin = np.array(full_bbox[:2], dtype=int)
        objects.append(
            {
                "body_color": body_color,
                "body_bbox": body_bbox,
                "arrow_bbox": arrow_bbox,
                "full_bbox": full_bbox,
                "direction": _infer_direction(body_bbox, arrow_bbox),
                "rel_coords": coords - origin,
                "values": values,
            }
        )
    return objects


def _simulate(grid: np.ndarray, *, erase_arrows: bool, want_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _parse_objects(g)
    objects = sorted(
        objects,
        key=lambda obj: (
            _boundary_distance(obj["full_bbox"], obj["direction"], (h, w)),
            int(obj["body_bbox"][0]),
            int(obj["body_bbox"][1]),
        ),
    )

    occupied = np.zeros((h, w), dtype=int)
    meta: list[dict] = []
    for obj in objects:
        dr, dc = DIRS[obj["direction"]]
        rel = obj["rel_coords"]
        vals = obj["values"]
        start_r, start_c = map(int, obj["full_bbox"][:2])
        cur_r, cur_c = start_r, start_c
        steps = 0
        stop_reason = "boundary"

        while True:
            next_r = cur_r + dr
            next_c = cur_c + dc
            abs_rows = next_r + rel[:, 0]
            abs_cols = next_c + rel[:, 1]
            if (
                np.any(abs_rows < 0)
                or np.any(abs_rows >= h)
                or np.any(abs_cols < 0)
                or np.any(abs_cols >= w)
            ):
                stop_reason = "boundary"
                break
            if np.any(occupied[abs_rows, abs_cols] != BACKGROUND):
                stop_reason = "collision"
                break
            cur_r, cur_c = next_r, next_c
            steps += 1

        final_rows = cur_r + rel[:, 0]
        final_cols = cur_c + rel[:, 1]
        assert not np.any(occupied[final_rows, final_cols] != BACKGROUND)
        occupied[final_rows, final_cols] = vals
        meta.append(
            {
                "direction": obj["direction"],
                "start_top_left": (start_r, start_c),
                "final_top_left": (int(cur_r), int(cur_c)),
                "steps": int(steps),
                "stop_reason": stop_reason,
                "body_color": obj["body_color"],
            }
        )

    out = occupied.copy()
    if erase_arrows:
        out[out == ARROW] = BACKGROUND
    if want_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each arrow-marked object toward its own arrow side, let earlier arrivals block later ones, then erase the arrows.

    Input structure:
    - The grid is black background with 3 to 5 disjoint 8-connected composite objects.
    - Each object has one colored 4-connected body and one grey 3-pixel L-shaped arrow in a 2x2 box attached on exactly one side.
    - Different objects use different body colors, while every arrow uses the same grey color.

    Rule:
    - For each object, separate its colored body from its grey arrow and infer the motion direction from which side of the body the arrow occupies.
    - Sort objects by the distance from their full bounding box to the relevant grid edge, breaking ties by the body's top-left bbox corner.
    - Move each whole object one cell at a time in that direction until the next step would leave the grid or hit a previously placed object.
    - After all motion finishes, erase every grey arrow pixel to black and keep only the moved colored bodies.

    Key insight:
    - Direction is object-local, not global: the arrow side sets the slide direction, and sequential placement makes earlier moved objects become real blockers for later ones.
    """
    return _simulate(np.asarray(grid, dtype=int), erase_arrows=True)


def _distribute_extras(base_count: int, total: int) -> list[int]:
    counts = [1] * base_count
    for _ in range(total - base_count):
        counts[random.randrange(base_count)] += 1
    return counts


def _lane_rows(heights: list[int]) -> tuple[int, list[int]]:
    lane_count = len(heights)
    min_h = sum(heights) + (lane_count - 1) + 2
    canvas_h = random.randint(min_h, min(min_h + 5, 26))
    slack = canvas_h - min_h
    extras = [0] * (lane_count + 1)
    for _ in range(slack):
        extras[random.randrange(lane_count + 1)] += 1

    tops: list[int] = []
    r = 1 + extras[0]
    for idx, height in enumerate(heights):
        tops.append(r)
        r += height
        if idx < lane_count - 1:
            r += 1 + extras[idx + 1]
    return canvas_h, tops


def _lane_width_requirement(widths: list[int]) -> int:
    return 2 + sum(widths) + max(0, len(widths) - 1)


def _place_lane_objects(canvas: np.ndarray, lane_top: int, lane_objects: list[dict]) -> None:
    widths = [int(obj["patch"].shape[1]) for obj in lane_objects]
    required = _lane_width_requirement(widths)
    assert canvas.shape[1] >= required

    slack = canvas.shape[1] - required
    extras = [0] * (len(lane_objects) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1

    c = 1 + extras[0]
    for idx, obj in enumerate(lane_objects):
        patch = obj["patch"]
        ph, pw = patch.shape
        target = canvas[lane_top : lane_top + ph, c : c + pw]
        assert target.shape == patch.shape
        assert not np.any((target != BACKGROUND) & (patch != BACKGROUND))
        target[patch != BACKGROUND] = patch[patch != BACKGROUND]
        c += pw
        if idx < len(lane_objects) - 1:
            c += 1 + extras[idx + 1]


def _rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k).copy()


def _attempt_generate() -> np.ndarray | None:
    n_objects = random.randint(3, 5)
    require_collision = random.random() < 0.6
    max_lanes = min(4, n_objects)
    min_lanes = 2 if n_objects > 2 else 1
    lane_count = random.randint(min_lanes, max_lanes)
    if require_collision and lane_count == n_objects:
        lane_count -= 1

    lane_counts = _distribute_extras(lane_count, n_objects)
    colors = random.sample(BODY_COLORS, n_objects)

    lane_dirs = [random.choice(["left", "right"]) for _ in range(lane_count)]
    if lane_count > 1 and all(direction == lane_dirs[0] for direction in lane_dirs):
        lane_dirs[random.randrange(lane_count)] = "left" if lane_dirs[0] == "right" else "right"

    lanes: list[list[dict]] = []
    color_idx = 0
    for lane_idx in range(lane_count):
        direction = lane_dirs[lane_idx]
        lane_objs: list[dict] = []
        for _ in range(lane_counts[lane_idx]):
            color = colors[color_idx]
            color_idx += 1
            lane_objs.append({"patch": _make_object_patch(direction, color), "direction": direction})
        random.shuffle(lane_objs)
        lanes.append(lane_objs)

    heights = [max(int(obj["patch"].shape[0]) for obj in lane) for lane in lanes]
    canvas_h, lane_tops = _lane_rows(heights)
    min_w = max(_lane_width_requirement([int(obj["patch"].shape[1]) for obj in lane]) for lane in lanes)
    canvas_w = random.randint(max(12, min_w), min(max(12, min_w) + 6, 26))
    canvas = np.zeros((canvas_h, canvas_w), dtype=int)

    lane_order = list(range(lane_count))
    random.shuffle(lane_order)
    for lane_idx in lane_order:
        _place_lane_objects(canvas, lane_tops[lane_idx], lanes[lane_idx])

    rotated = _rotate_grid(canvas, random.randrange(4))
    if max(rotated.shape) > 30:
        return None

    try:
        solved, meta = _simulate(rotated, erase_arrows=False, want_meta=True)
    except Exception:
        return None

    if len(meta) != n_objects:
        return None
    if any(item["steps"] < 1 for item in meta):
        return None
    if require_collision and not any(item["stop_reason"] == "collision" for item in meta):
        return None
    if not require_collision and random.random() < 0.4 and all(item["stop_reason"] == "collision" for item in meta):
        return None
    if np.count_nonzero(rotated == ARROW) != 3 * n_objects:
        return None
    if np.any(main(rotated) == ARROW):
        return None
    if np.array_equal(main(rotated), rotated):
        return None
    if np.array_equal(solved, rotated):
        return None
    return rotated


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
