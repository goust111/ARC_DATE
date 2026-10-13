from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
LEFT_WALL = 1
RIGHT_WALL = 2
FRAME = 5

DIR_BY_COLOR = {LEFT_WALL: -1, RIGHT_WALL: 1}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = np.asarray(coords, dtype=int) - np.asarray(coords, dtype=int).min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _sig_dims(sig: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in sig]
    cols = [c for _r, c in sig]
    return max(rows) + 1, max(cols) + 1


def _coords_from_sig(sig: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    return np.asarray([(top + r, left + c) for r, c in sig], dtype=int)


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _frame_holes(frame_coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _bbox(frame_coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[frame_coords[:, 0] - r0, frame_coords[:, 1] - c0] = True
    empty = ~local

    outside = np.zeros_like(empty, dtype=bool)
    q = deque()
    h, w = empty.shape
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        cr, cc = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes_local = empty & ~outside
    return [_coords + np.array([r0, c0]) for _coords in _components(holes_local)]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w >= 5
    assert np.all(g[:, 0] == LEFT_WALL)
    assert np.all(g[:, w - 1] == RIGHT_WALL)

    objects = []
    for color, wall_col, direction in (
        (LEFT_WALL, 0, -1),
        (RIGHT_WALL, w - 1, 1),
    ):
        mask = g == color
        mask[:, wall_col] = False
        for coords in _components(mask):
            b = _bbox(coords)
            objects.append(
                {
                    "color": int(color),
                    "dir": int(direction),
                    "coords": coords,
                    "bbox": b,
                    "sig": _normalize(coords),
                }
            )

    frames = []
    for idx, coords in enumerate(_components(g == FRAME)):
        holes = _frame_holes(coords)
        assert len(holes) == 1
        hole = holes[0]
        frames.append(
            {
                "index": idx,
                "coords": coords,
                "bbox": _bbox(coords),
                "hole": hole,
                "hole_bbox": _bbox(hole),
                "sig": _normalize(hole),
                "hole_cells": frozenset(map(tuple, hole.tolist())),
            }
        )

    objects.sort(key=lambda item: item["bbox"])
    frames.sort(key=lambda item: item["bbox"])
    return {"shape": (h, w), "objects": objects, "frames": frames}


def _capture_frame_for_object(obj: dict, frames: list[dict]) -> dict | None:
    candidates = []
    for frame in frames:
        if obj["sig"] != frame["sig"]:
            continue
        dx = int(frame["hole_bbox"][1] - obj["bbox"][1])
        shifted = frozenset((int(r), int(c + dx)) for r, c in obj["coords"])
        if shifted != frame["hole_cells"]:
            continue
        if dx * obj["dir"] <= 0:
            continue
        candidates.append((abs(dx), frame["bbox"][1], frame))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _slide_object(coords: np.ndarray, direction: int, occupied: np.ndarray) -> np.ndarray:
    h, w = occupied.shape
    shift = 0
    while True:
        trial = coords + np.array([0, shift + direction], dtype=int)
        cols = trial[:, 1]
        if int(cols.min()) < 0 or int(cols.max()) >= w:
            break
        if np.any(occupied[trial[:, 0], trial[:, 1]]):
            break
        shift += direction
    return coords + np.array([0, shift], dtype=int)


def _distance_to_wall(obj: dict, width: int) -> int:
    if obj["dir"] == -1:
        return int(obj["bbox"][1] - 1)
    return int((width - 2) - obj["bbox"][3])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each blue/red object toward its same-colored wall, capturing only the same-shape trap holes that lie on that horizontal path.

    Input structure:
    - The background is black.
    - The left column is a solid blue wall and the right column is a solid red wall.
    - Interior blue and red 4-connected objects float away from the walls.
    - Grey 4-connected filled rectangles contain exactly one enclosed background hole each; the hole shape is the trap.

    Rule:
    - Blue objects slide left and red objects slide right.
    - Translation-normalize every movable object and every frame hole.
    - If an object can reach a same-shape hole by a pure horizontal translation in its slide direction, it is captured by the nearest such frame: erase the object, fill that hole with the object color, and recolor the frame border to that color.
    - Every uncaptured object is erased from its start and then slides horizontally, nearest-to-wall first, until the next step would overlap a wall or an already settled slid object.
    - Walls stay fixed; unmatched grey frames stay grey.

    Key insight:
    - Color determines direction, but shape and horizontal alignment determine whether the object gets trapped on the way or survives to slide into its wall.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    h, w = scene["shape"]

    out = g.copy()
    for obj in scene["objects"]:
        _paint(out, obj["coords"], BG)

    captures: dict[int, dict] = {}
    used_frames: set[int] = set()
    for idx, obj in enumerate(scene["objects"]):
        frame = _capture_frame_for_object(obj, scene["frames"])
        if frame is None:
            continue
        assert frame["index"] not in used_frames
        captures[idx] = frame
        used_frames.add(frame["index"])

    for idx, frame in captures.items():
        color = int(scene["objects"][idx]["color"])
        _paint(out, frame["coords"], color)
        _paint(out, frame["hole"], color)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, 0] = True
    occupied[:, w - 1] = True

    sliders = [obj for idx, obj in enumerate(scene["objects"]) if idx not in captures]
    sliders.sort(key=lambda obj: (_distance_to_wall(obj, w), obj["bbox"]))
    for obj in sliders:
        placed = _slide_object(obj["coords"], obj["dir"], occupied)
        occupied[placed[:, 0], placed[:, 1]] = True
        _paint(out, placed, obj["color"])

    return out


_BASE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)),
]


def _shape_variants(sig: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    variants = set()
    arr = np.asarray(sig, dtype=int)
    for transform in range(8):
        pts = []
        for r, c in arr:
            if transform == 0:
                nr, nc = r, c
            elif transform == 1:
                nr, nc = r, -c
            elif transform == 2:
                nr, nc = -r, c
            elif transform == 3:
                nr, nc = -r, -c
            elif transform == 4:
                nr, nc = c, r
            elif transform == 5:
                nr, nc = c, -r
            elif transform == 6:
                nr, nc = -c, r
            else:
                nr, nc = -c, -r
            pts.append((nr, nc))
        variants.add(_normalize(np.asarray(pts, dtype=int)))
    return variants


def _build_shape_pool() -> list[tuple[tuple[int, int], ...]]:
    pool = set()
    for base in _BASE_SHAPES:
        for variant in _shape_variants(base):
            h, w = _sig_dims(variant)
            area = len(variant)
            if 3 <= area <= 6 and h <= 3 and w <= 4:
                pool.add(variant)
    return sorted(pool, key=lambda sig: (len(sig), _sig_dims(sig), sig))


SHAPE_POOL = _build_shape_pool()
GEN_RETRIES = 400


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(rng.choice(np.arange(total + parts - 1), size=parts - 1, replace=False).tolist())
    values = []
    prev = -1
    for cut in [*cuts, total + parts - 1]:
        values.append(int(cut - prev - 1))
        prev = cut
    return values


def _build_expected_from_meta(inp: np.ndarray, meta: dict) -> np.ndarray:
    out = inp.copy()
    for obj in meta["objects"]:
        _paint(out, obj["coords"], BG)
    for obj in meta["objects"]:
        if obj["kind"] != "capture":
            continue
        frame = obj["frame"]
        _paint(out, frame["coords"], obj["color"])
        _paint(out, frame["hole"], obj["color"])

    occupied = np.zeros(inp.shape, dtype=bool)
    occupied[:, 0] = True
    occupied[:, inp.shape[1] - 1] = True
    sliders = [obj for obj in meta["objects"] if obj["kind"] != "capture"]
    sliders.sort(key=lambda obj: (_distance_to_wall(obj, inp.shape[1]), obj["bbox"]))
    for obj in sliders:
        placed = _slide_object(obj["coords"], obj["dir"], occupied)
        occupied[placed[:, 0], placed[:, 1]] = True
        _paint(out, placed, obj["color"])
    return out


def _validate_generated(inp: np.ndarray, meta: dict) -> bool:
    parsed = _parse_scene(inp)
    objects = parsed["objects"]
    frames = parsed["frames"]
    if not (2 <= len(frames) <= 3):
        return False
    blue_count = sum(obj["color"] == LEFT_WALL for obj in objects)
    red_count = sum(obj["color"] == RIGHT_WALL for obj in objects)
    if blue_count not in (2, 3) or red_count not in (2, 3):
        return False
    if len({frame["sig"] for frame in frames}) != len(frames):
        return False

    captures = []
    for parsed_obj, meta_obj in zip(objects, meta["objects"]):
        if parsed_obj["color"] != meta_obj["color"] or parsed_obj["sig"] != meta_obj["sig"]:
            return False
        frame = _capture_frame_for_object(parsed_obj, frames)
        captures.append(frame)
        if meta_obj["kind"] == "capture":
            if frame is None or frame["sig"] != meta_obj["frame"]["sig"] or frame["bbox"] != meta_obj["frame"]["bbox"]:
                return False
        else:
            if frame is not None:
                return False

    capture_count = sum(frame is not None for frame in captures)
    if capture_count not in (1, 2):
        return False
    if capture_count >= len(objects):
        return False

    if meta["has_wrong_side_decoy"]:
        found = False
        for obj in objects:
            for frame in frames:
                if obj["sig"] != frame["sig"]:
                    continue
                dx = int(frame["hole_bbox"][1] - obj["bbox"][1])
                shifted = frozenset((int(r), int(c + dx)) for r, c in obj["coords"])
                if shifted == frame["hole_cells"] and dx * obj["dir"] < 0:
                    found = True
        if not found:
            return False

    if meta["has_same_side_mismatch"]:
        found = False
        for meta_obj, parsed_obj in zip(meta["objects"], objects):
            frame = meta_obj.get("frame")
            if frame is None or meta_obj["kind"] != "mismatch_same_side":
                continue
            dx = int(frame["hole_bbox"][1] - parsed_obj["bbox"][1])
            if dx * parsed_obj["dir"] > 0 and frame["sig"] != parsed_obj["sig"]:
                found = True
        if not found:
            return False

    expected = _build_expected_from_meta(inp, meta)
    actual = main(inp)
    return np.array_equal(expected, actual)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    blue_count = int(rng.integers(2, 4))
    red_count = int(rng.integers(2, 4))
    lane_colors = [LEFT_WALL] * blue_count + [RIGHT_WALL] * red_count
    rng.shuffle(lane_colors)
    lane_count = len(lane_colors)

    object_shape_ids = rng.choice(len(SHAPE_POOL), size=lane_count, replace=False).tolist()
    frame_count = int(rng.integers(2, 4))
    capture_count = int(rng.integers(1, min(2, frame_count) + 1))

    lane_specs = []
    for color, shape_id in zip(lane_colors, object_shape_ids):
        lane_specs.append({"color": int(color), "obj_sig": SHAPE_POOL[int(shape_id)], "kind": "none"})

    frame_lanes = sorted(rng.choice(lane_count, size=frame_count, replace=False).tolist())
    capture_lanes = set(rng.choice(frame_lanes, size=capture_count, replace=False).tolist())
    remaining_lanes = [idx for idx in frame_lanes if idx not in capture_lanes]

    for idx in capture_lanes:
        lane_specs[idx]["kind"] = "capture"

    has_wrong_side_decoy = False
    if remaining_lanes:
        lane_specs[remaining_lanes[0]]["kind"] = "decoy"
        has_wrong_side_decoy = True
        remaining_lanes = remaining_lanes[1:]
    for idx in remaining_lanes:
        lane_specs[idx]["kind"] = "mismatch_same_side" if rng.random() < 0.65 else "mismatch_wrong_side"

    used_hole_shapes = set()
    has_same_side_mismatch = any(spec["kind"] == "mismatch_same_side" for spec in lane_specs)
    object_shapes = {spec["obj_sig"] for spec in lane_specs}
    available_mismatch = [sig for sig in SHAPE_POOL if sig not in object_shapes]
    rng.shuffle(available_mismatch)

    for spec in lane_specs:
        kind = spec["kind"]
        if kind == "none":
            spec["frame"] = None
            continue
        if kind in {"capture", "decoy"}:
            hole_sig = spec["obj_sig"]
        else:
            if not available_mismatch:
                return None
            hole_sig = available_mismatch.pop()
        if hole_sig in used_hole_shapes:
            return None
        used_hole_shapes.add(hole_sig)
        hole_h, hole_w = _sig_dims(hole_sig)
        spec["frame"] = {
            "sig": hole_sig,
            "hole_h": hole_h,
            "hole_w": hole_w,
            "frame_h": hole_h + 2,
            "frame_w": hole_w + 2,
        }

    lane_heights = []
    for spec in lane_specs:
        obj_h, _obj_w = _sig_dims(spec["obj_sig"])
        if spec["frame"] is None:
            lane_heights.append(obj_h)
        else:
            lane_heights.append(max(obj_h, spec["frame"]["frame_h"]))
    min_h = sum(lane_heights)
    if min_h > 26:
        return None
    grid_h = int(rng.integers(max(14, min_h), min(27, min_h + 5)))
    extras = _random_composition(grid_h - min_h, lane_count + 1, rng)

    row = extras[0]
    for idx, spec in enumerate(lane_specs):
        spec["lane_top"] = int(row)
        spec["lane_h"] = int(lane_heights[idx])
        row += lane_heights[idx] + extras[idx + 1]

    width_need = 15
    for spec in lane_specs:
        obj_h, obj_w = _sig_dims(spec["obj_sig"])
        if spec["frame"] is None:
            width_need = max(width_need, obj_w + 4)
        else:
            width_need = max(width_need, 5 + obj_w + spec["frame"]["frame_w"])
    if width_need > 20:
        return None
    grid_w = int(rng.integers(width_need, 21))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[:, 0] = LEFT_WALL
    grid[:, grid_w - 1] = RIGHT_WALL
    meta_objects = []

    for spec in lane_specs:
        color = int(spec["color"])
        direction = DIR_BY_COLOR[color]
        obj_sig = spec["obj_sig"]
        obj_h, obj_w = _sig_dims(obj_sig)
        lane_top = spec["lane_top"]
        lane_h = spec["lane_h"]
        frame_data = spec["frame"]

        if frame_data is None:
            obj_top = lane_top + (lane_h - obj_h) // 2
            left_min = 2
            left_max = grid_w - 2 - obj_w
            if left_min > left_max:
                return None
            obj_left = int(rng.integers(left_min, left_max + 1))
            frame_info = None
        else:
            frame_h = frame_data["frame_h"]
            frame_w = frame_data["frame_w"]
            frame_top = lane_top + (lane_h - frame_h) // 2
            same_rows = spec["kind"] in {"capture", "decoy"}
            obj_top = frame_top + 1 if same_rows else lane_top + (lane_h - obj_h) // 2

            frame_on_goal_side = spec["kind"] in {"capture", "mismatch_same_side"}
            frame_left_of_object = (direction == -1 and frame_on_goal_side) or (direction == 1 and not frame_on_goal_side)
            gap = int(rng.integers(1, 4))

            if frame_left_of_object:
                frame_left_min = 2
                frame_left_max = grid_w - 2 - obj_w - gap - frame_w
                if frame_left_min > frame_left_max:
                    return None
                frame_left = int(rng.integers(frame_left_min, frame_left_max + 1))
                obj_left = frame_left + frame_w + gap
            else:
                obj_left_min = 2
                obj_left_max = grid_w - 2 - obj_w - gap - frame_w
                if obj_left_min > obj_left_max:
                    return None
                obj_left = int(rng.integers(obj_left_min, obj_left_max + 1))
                frame_left = obj_left + obj_w + gap

            frame_coords = []
            for r in range(frame_top, frame_top + frame_h):
                for c in range(frame_left, frame_left + frame_w):
                    frame_coords.append((r, c))
            hole = _coords_from_sig(frame_data["sig"], frame_top + 1, frame_left + 1)
            hole_set = set(map(tuple, hole.tolist()))
            frame_coords = np.asarray([cell for cell in frame_coords if cell not in hole_set], dtype=int)
            frame_info = {
                "coords": frame_coords,
                "bbox": _bbox(frame_coords),
                "hole": hole,
                "hole_bbox": _bbox(hole),
                "sig": frame_data["sig"],
                "hole_cells": frozenset(map(tuple, hole.tolist())),
            }
            _paint(grid, frame_coords, FRAME)

        obj_coords = _coords_from_sig(obj_sig, obj_top, obj_left)
        if np.any(grid[obj_coords[:, 0], obj_coords[:, 1]] != BG):
            return None
        _paint(grid, obj_coords, color)
        meta_objects.append(
            {
                "color": color,
                "dir": direction,
                "coords": obj_coords,
                "bbox": _bbox(obj_coords),
                "sig": obj_sig,
                "kind": spec["kind"],
                "frame": frame_info,
            }
        )

    meta_objects.sort(key=lambda item: item["bbox"])
    meta = {
        "objects": meta_objects,
        "has_wrong_side_decoy": has_wrong_side_decoy,
        "has_same_side_mismatch": has_same_side_mismatch,
    }

    try:
        if not _validate_generated(grid, meta):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
