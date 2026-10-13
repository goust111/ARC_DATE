from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

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

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
FRAME_H_RANGE = (5, 8)
FRAME_W_RANGE = (5, 9)
GAP_MAX = 3
SPACE_MAX = 8
SCENE_RETRIES = 400
PLACEMENT_RETRIES = 120


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    pref = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    pref[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)
    return pref


def _rect_count(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(pref[r1 + 1, c1 + 1] - pref[r0, c1 + 1] - pref[r1 + 1, c0] + pref[r0, c0])


def _parse_side(seq: np.ndarray) -> tuple[bool, tuple[int, int] | None]:
    vals = set(map(int, seq.tolist()))
    if not vals <= {BLACK, GRAY}:
        return False, None
    black_idx = np.flatnonzero(seq == BLACK)
    if black_idx.size == 0:
        return True, None
    if black_idx.size > GAP_MAX:
        return False, None
    if int(black_idx[-1] - black_idx[0] + 1) != int(black_idx.size):
        return False, None
    return True, (int(black_idx[0]), int(black_idx[-1]))


def _bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _expand_bbox(
    bbox: tuple[int, int, int, int], pad: int, h: int, w: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _bbox_clear(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = occupied.shape
    er0, ec0, er1, ec1 = _expand_bbox(bbox, pad, h, w)
    return not bool(occupied[er0 : er1 + 1, ec0 : ec1 + 1].any())


def _mark_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _cells_clear(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    return all(0 <= r < h and 0 <= c < w and not occupied[r, c] for r, c in cells)


def _mark_cells(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _choose_gap(side_len: int) -> tuple[int, int]:
    width = random.randint(1, min(GAP_MAX, side_len))
    start = random.randint(0, side_len - width)
    return start, start + width - 1


def _frame_gap_cells(
    bbox: tuple[int, int, int, int], side: str, gap: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    g0, g1 = gap
    if side == "up":
        return [(r0, c0 + 1 + i) for i in range(g0, g1 + 1)]
    if side == "down":
        return [(r1, c0 + 1 + i) for i in range(g0, g1 + 1)]
    if side == "left":
        return [(r0 + 1 + i, c0) for i in range(g0, g1 + 1)]
    return [(r0 + 1 + i, c1) for i in range(g0, g1 + 1)]


def _ray_start_strip(
    bbox: tuple[int, int, int, int], side: str, gap: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    g0, g1 = gap
    if side == "up":
        return [(r0 - 1, c0 + 1 + i) for i in range(g0, g1 + 1)]
    if side == "down":
        return [(r1 + 1, c0 + 1 + i) for i in range(g0, g1 + 1)]
    if side == "left":
        return [(r0 + 1 + i, c0 - 1) for i in range(g0, g1 + 1)]
    return [(r0 + 1 + i, c1 + 1) for i in range(g0, g1 + 1)]


def _build_frame(
    frame_id: int,
    bbox: tuple[int, int, int, int],
    kind: str,
    gaps: dict[str, tuple[int, int] | None],
    color: int | None = None,
    block_bbox: tuple[int, int, int, int] | None = None,
) -> dict:
    return {
        "id": int(frame_id),
        "bbox": tuple(map(int, bbox)),
        "kind": kind,
        "gaps": {k: v for k, v in gaps.items()},
        "color": None if color is None else int(color),
        "block_bbox": None if block_bbox is None else tuple(map(int, block_bbox)),
    }


def _random_block_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    block_h = random.randint(1, inner_h - 2)
    block_w = random.randint(1, inner_w - 2)
    br0 = random.randint(r0 + 2, r1 - block_h - 1)
    bc0 = random.randint(c0 + 2, c1 - block_w - 1)
    return br0, bc0, br0 + block_h - 1, bc0 + block_w - 1


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    for side, gap in frame["gaps"].items():
        if gap is None:
            continue
        for r, c in _frame_gap_cells(frame["bbox"], side, gap):
            grid[r, c] = BLACK
    if frame["kind"] == "emitter":
        br0, bc0, br1, bc1 = frame["block_bbox"]
        grid[br0 : br1 + 1, bc0 : bc1 + 1] = int(frame["color"])


def _detect_frame_at(
    g: np.ndarray, gray_pref: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> dict | None:
    if int(g[r0, c0]) != GRAY or int(g[r0, c1]) != GRAY or int(g[r1, c0]) != GRAY or int(g[r1, c1]) != GRAY:
        return None
    if _rect_count(gray_pref, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
        return None

    ok_top, top_gap = _parse_side(g[r0, c0 + 1 : c1])
    ok_bottom, bottom_gap = _parse_side(g[r1, c0 + 1 : c1])
    ok_left, left_gap = _parse_side(g[r0 + 1 : r1, c0])
    ok_right, right_gap = _parse_side(g[r0 + 1 : r1, c1])
    if not (ok_top and ok_bottom and ok_left and ok_right):
        return None

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    inside_colors = [int(v) for v in np.unique(inner) if int(v) not in (BLACK, GRAY)]
    gaps = {
        "up": top_gap,
        "down": bottom_gap,
        "left": left_gap,
        "right": right_gap,
    }
    border_cells = set()
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr in (r0, r1) or cc in (c0, c1):
                if int(g[rr, cc]) == GRAY:
                    border_cells.add((int(rr), int(cc)))

    if not inside_colors:
        return {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "kind": "receiver",
            "gaps": gaps,
            "color": None,
            "gap_side": None,
            "block_bbox": None,
            "border_cells": border_cells,
        }

    if len(inside_colors) != 1:
        return None
    color = int(inside_colors[0])
    if bool(((inner != BLACK) & (inner != color)).any()):
        return None

    block_coords = np.argwhere(inner == color)
    if block_coords.size == 0:
        return None
    br0, bc0 = block_coords.min(axis=0)
    br1, bc1 = block_coords.max(axis=0)
    if int((br1 - br0 + 1) * (bc1 - bc0 + 1)) != int(block_coords.shape[0]):
        return None
    if int(br0) == 0 or int(bc0) == 0 or int(br1) == inner.shape[0] - 1 or int(bc1) == inner.shape[1] - 1:
        return None

    gap_sides = [side for side, gap in gaps.items() if gap is not None]
    if len(gap_sides) != 1:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "kind": "emitter",
        "gaps": gaps,
        "color": color,
        "gap_side": gap_sides[0],
        "block_bbox": (int(r0 + 1 + br0), int(c0 + 1 + bc0), int(r0 + 1 + br1), int(c0 + 1 + bc1)),
        "border_cells": border_cells,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    gray = g == GRAY
    gray_coords = [tuple(map(int, p)) for p in np.argwhere(gray)]
    gray_set = set(gray_coords)
    gray_pref = _prefix_sum(gray.astype(int))

    candidates: dict[tuple[int, int, int, int], dict] = {}
    for r0, c0 in gray_coords:
        for r1, c1 in gray_coords:
            if r1 <= r0 + 1 or c1 <= c0 + 1:
                continue
            if (r0, c1) not in gray_set or (r1, c0) not in gray_set:
                continue
            bbox = (int(r0), int(c0), int(r1), int(c1))
            if bbox in candidates:
                continue
            frame = _detect_frame_at(g, gray_pref, *bbox)
            if frame is not None:
                candidates[bbox] = frame

    frames = sorted(
        candidates.values(),
        key=lambda f: (int(f["bbox"][0]), int(f["bbox"][1]), int(f["bbox"][2]), int(f["bbox"][3])),
    )
    for idx, frame in enumerate(frames):
        frame["id"] = int(idx)
    return frames


def _cast_emitter_ray(
    g: np.ndarray, out: np.ndarray, frame: dict, border_to_frame: dict[tuple[int, int], int], frames: list[dict]
) -> tuple[int, int] | None:
    side = frame["gap_side"]
    gap = frame["gaps"][side]
    start_strip = _ray_start_strip(frame["bbox"], side, gap)
    dr, dc = DIRS[side]
    color = int(frame["color"])
    steps = 0
    strip = start_strip

    while True:
        if not strip or any(not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]) for r, c in strip):
            return None
        vals = [int(g[r, c]) for r, c in strip]
        if any(v != BLACK for v in vals):
            hit_ids = {border_to_frame[(r, c)] for r, c in strip if (r, c) in border_to_frame}
            if len(hit_ids) == 1 and all(v in (BLACK, GRAY) for v in vals):
                target_id = next(iter(hit_ids))
                if frames[target_id]["kind"] == "receiver":
                    return int(target_id), int(steps)
            return None
        for r, c in strip:
            out[r, c] = color
        steps += 1
        strip = [(r + dr, c + dc) for r, c in strip]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast colored rays outward from emitter frame gaps and fill hit receiver frames.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)

    border_to_frame: dict[tuple[int, int], int] = {}
    for frame in frames:
        for cell in frame["border_cells"]:
            border_to_frame[cell] = int(frame["id"])

    receiver_hits: dict[int, tuple[int, int]] = {}
    for frame in frames:
        if frame["kind"] != "emitter":
            continue
        hit = _cast_emitter_ray(g, out, frame, border_to_frame, frames)
        if hit is None:
            continue
        target_id, dist = hit
        prev = receiver_hits.get(int(target_id))
        if prev is None or int(dist) < int(prev[0]):
            receiver_hits[int(target_id)] = (int(dist), int(frame["color"]))

    for frame in frames:
        if frame["kind"] != "receiver":
            continue
        if frame["id"] not in receiver_hits:
            continue
        _, color = receiver_hits[int(frame["id"])]
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner == BLACK] = int(color)

    return out


def _incoming_segment(receiver_bbox: tuple[int, int, int, int], ray_dir: str, width: int) -> tuple[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = receiver_bbox
    if ray_dir in ("left", "right"):
        span = r1 - r0 - 1
        start = random.randint(0, span - width)
        rows = [r0 + 1 + i for i in range(start, start + width)]
        col = c0 if ray_dir == "right" else c1
        return start, [(rr, col) for rr in rows]
    span = c1 - c0 - 1
    start = random.randint(0, span - width)
    cols = [c0 + 1 + i for i in range(start, start + width)]
    row = r0 if ray_dir == "down" else r1
    return start, [(row, cc) for cc in cols]


def _ray_path_between(
    emitter_bbox: tuple[int, int, int, int], gap_side: str, gap: tuple[int, int], max_steps: int
) -> list[tuple[int, int]]:
    strip = _ray_start_strip(emitter_bbox, gap_side, gap)
    dr, dc = DIRS[gap_side]
    path: list[tuple[int, int]] = []
    for _ in range(max_steps):
        path.extend(strip)
        strip = [(r + dr, c + dc) for r, c in strip]
    return path


def _place_emitter_toward_receiver(
    occupied: np.ndarray,
    receiver: dict,
    ray_dir: str,
    color: int,
    frame_id: int,
    forced_space: int | None = None,
) -> tuple[dict, dict] | None:
    h, w = occupied.shape
    rr0, rc0, rr1, rc1 = receiver["bbox"]

    for _ in range(PLACEMENT_RETRIES):
        if ray_dir in ("left", "right"):
            width = random.randint(1, min(GAP_MAX, rr1 - rr0 - 1))
        else:
            width = random.randint(1, min(GAP_MAX, rc1 - rc0 - 1))
        recv_gap_ofs, _contact = _incoming_segment(receiver["bbox"], ray_dir, width)
        eh = random.randint(*FRAME_H_RANGE)
        ew = random.randint(*FRAME_W_RANGE)
        if ray_dir in ("left", "right") and eh - 2 < width:
            continue
        if ray_dir in ("up", "down") and ew - 2 < width:
            continue
        if eh < 5 or ew < 5:
            continue

        gap_ofs = random.randint(0, (eh - 2 - width) if ray_dir in ("left", "right") else (ew - 2 - width))
        space = forced_space if forced_space is not None else random.randint(1, SPACE_MAX)

        if ray_dir == "right":
            er0 = rr0 + recv_gap_ofs - gap_ofs
            ec1 = rc0 - space - 1
            ec0 = ec1 - ew + 1
            bbox = (er0, ec0, er0 + eh - 1, ec1)
        elif ray_dir == "left":
            er0 = rr0 + recv_gap_ofs - gap_ofs
            ec0 = rc1 + space + 1
            bbox = (er0, ec0, er0 + eh - 1, ec0 + ew - 1)
        elif ray_dir == "down":
            ec0 = rc0 + recv_gap_ofs - gap_ofs
            er1 = rr0 - space - 1
            bbox = (er1 - eh + 1, ec0, er1, ec0 + ew - 1)
        else:
            ec0 = rc0 + recv_gap_ofs - gap_ofs
            er0 = rr1 + space + 1
            bbox = (er0, ec0, er0 + eh - 1, ec0 + ew - 1)

        r0, c0, r1, c1 = bbox
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if not _bbox_clear(occupied, bbox, pad=1):
            continue

        gap = (int(gap_ofs), int(gap_ofs + width - 1))
        path = _ray_path_between(bbox, ray_dir, gap, int(space))
        if not path or not _cells_clear(occupied, path):
            continue

        frame = _build_frame(
            frame_id,
            bbox,
            "emitter",
            {"up": None, "down": None, "left": None, "right": None, ray_dir: gap},
            color=color,
            block_bbox=_random_block_bbox(bbox),
        )
        ray = {
            "source_id": int(frame_id),
            "target_id": int(receiver["id"]),
            "color": int(color),
            "distance": int(space),
            "path": path,
        }
        return frame, ray

    return None


def _random_receiver_gaps(
    bbox: tuple[int, int, int, int], protected_sides: set[str]
) -> dict[str, tuple[int, int] | None]:
    r0, c0, r1, c1 = bbox
    gaps: dict[str, tuple[int, int] | None] = {"up": None, "down": None, "left": None, "right": None}
    side_lens = {
        "up": c1 - c0 - 1,
        "down": c1 - c0 - 1,
        "left": r1 - r0 - 1,
        "right": r1 - r0 - 1,
    }
    for side in ("up", "down", "left", "right"):
        if side in protected_sides or side_lens[side] <= 0:
            continue
        if random.random() < 0.35:
            gaps[side] = _choose_gap(side_lens[side])
    return gaps


def _place_receiver_cluster(scene: dict) -> bool:
    h, w = scene["grid_shape"]
    used_colors: set[int] = scene["used_colors"]
    occupied = scene["occupied"]

    for _ in range(PLACEMENT_RETRIES):
        orientation = random.choice(["horizontal", "vertical"])
        hit_count = 2 if random.random() < 0.5 else 1
        ray_dirs = ["left", "right"] if orientation == "horizontal" else ["up", "down"]
        if hit_count == 1:
            ray_dirs = [random.choice(ray_dirs)]

        if len(PALETTE) - len(used_colors) < len(ray_dirs):
            return False

        rh = random.randint(*FRAME_H_RANGE)
        rw = random.randint(*FRAME_W_RANGE)
        bbox = (
            random.randint(1, h - rh - 2),
            random.randint(1, w - rw - 2),
            0,
            0,
        )
        bbox = (bbox[0], bbox[1], bbox[0] + rh - 1, bbox[1] + rw - 1)
        if not _bbox_clear(occupied, bbox, pad=1):
            continue

        local_occupied = occupied.copy()
        local_frames: list[dict] = []
        local_rays: list[dict] = []
        local_colors = set(used_colors)
        next_id = int(scene["next_id"])

        protected = {OPPOSITE[d] for d in ray_dirs}
        receiver = _build_frame(next_id, bbox, "receiver", _random_receiver_gaps(bbox, protected))
        next_id += 1
        _mark_bbox(local_occupied, receiver["bbox"])
        local_frames.append(receiver)

        spaces = random.sample(range(1, SPACE_MAX + 1), k=len(ray_dirs))
        random.shuffle(spaces)
        ok = True
        for ray_dir, space in zip(ray_dirs, spaces):
            choices = [c for c in PALETTE if c not in local_colors]
            color = random.choice(choices)
            placed = _place_emitter_toward_receiver(local_occupied, receiver, ray_dir, color, next_id, space)
            if placed is None:
                ok = False
                break
            emitter, ray = placed
            next_id += 1
            local_colors.add(int(color))
            _mark_bbox(local_occupied, emitter["bbox"])
            _mark_cells(local_occupied, ray["path"])
            local_frames.append(emitter)
            local_rays.append(ray)

        if not ok:
            continue

        scene["occupied"] = local_occupied
        scene["frames"].extend(local_frames)
        scene["rays"].extend(local_rays)
        scene["used_colors"] = local_colors
        scene["next_id"] = next_id
        return True

    return False


def _place_extra_receiver(scene: dict) -> bool:
    h, w = scene["grid_shape"]
    occupied = scene["occupied"]
    for _ in range(PLACEMENT_RETRIES):
        rh = random.randint(*FRAME_H_RANGE)
        rw = random.randint(*FRAME_W_RANGE)
        r0 = random.randint(1, h - rh - 2)
        c0 = random.randint(1, w - rw - 2)
        bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if not _bbox_clear(occupied, bbox, pad=1):
            continue
        frame = _build_frame(scene["next_id"], bbox, "receiver", _random_receiver_gaps(bbox, set()))
        scene["next_id"] += 1
        scene["frames"].append(frame)
        _mark_bbox(scene["occupied"], bbox)
        return True
    return False


def _path_to_edge(
    bbox: tuple[int, int, int, int], gap_side: str, gap: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIRS[gap_side]
    strip = _ray_start_strip(bbox, gap_side, gap)
    path: list[tuple[int, int]] = []
    while strip and all(0 <= r < h and 0 <= c < w for r, c in strip):
        path.extend(strip)
        strip = [(r + dr, c + dc) for r, c in strip]
    return path


def _place_stray_emitter(scene: dict) -> bool:
    h, w = scene["grid_shape"]
    occupied = scene["occupied"]
    choices = [c for c in PALETTE if c not in scene["used_colors"]]
    if not choices:
        return False

    for _ in range(PLACEMENT_RETRIES):
        eh = random.randint(*FRAME_H_RANGE)
        ew = random.randint(*FRAME_W_RANGE)
        gap_side = random.choice(["up", "down", "left", "right"])
        if gap_side == "up":
            r0 = random.randint(1, h - eh - 1)
            c0 = random.randint(1, w - ew - 2)
        elif gap_side == "down":
            r0 = random.randint(1, h - eh - 2)
            c0 = random.randint(1, w - ew - 2)
        elif gap_side == "left":
            r0 = random.randint(1, h - eh - 2)
            c0 = random.randint(1, w - ew - 1)
        else:
            r0 = random.randint(1, h - eh - 2)
            c0 = random.randint(1, w - ew - 2)
        bbox = (r0, c0, r0 + eh - 1, c0 + ew - 1)
        if not _bbox_clear(occupied, bbox, pad=1):
            continue

        side_len = (ew - 2) if gap_side in ("up", "down") else (eh - 2)
        gap = _choose_gap(side_len)
        path = _path_to_edge(bbox, gap_side, gap, scene["grid_shape"])
        if not path or not _cells_clear(occupied, path):
            continue

        color = random.choice(choices)
        frame = _build_frame(
            scene["next_id"],
            bbox,
            "emitter",
            {"up": None, "down": None, "left": None, "right": None, gap_side: gap},
            color=color,
            block_bbox=_random_block_bbox(bbox),
        )
        scene["next_id"] += 1
        scene["used_colors"].add(int(color))
        scene["frames"].append(frame)
        scene["rays"].append(
            {
                "source_id": int(frame["id"]),
                "target_id": None,
                "color": int(color),
                "distance": None,
                "path": path,
            }
        )
        _mark_bbox(scene["occupied"], bbox)
        _mark_cells(scene["occupied"], path)
        return True

    return False


def _render_scene(scene: dict) -> np.ndarray:
    h, w = scene["grid_shape"]
    grid = np.full((h, w), BLACK, dtype=int)
    for frame in scene["frames"]:
        _draw_frame(grid, frame)
    return grid


def _expected_from_scene(grid: np.ndarray, scene: dict) -> np.ndarray:
    out = grid.copy()
    hits: dict[int, tuple[int, int]] = {}
    for ray in scene["rays"]:
        for r, c in ray["path"]:
            out[r, c] = int(ray["color"])
        if ray["target_id"] is None:
            continue
        target_id = int(ray["target_id"])
        dist = int(ray["distance"])
        prev = hits.get(target_id)
        if prev is None or dist < prev[0]:
            hits[target_id] = (dist, int(ray["color"]))

    for frame in scene["frames"]:
        if frame["kind"] != "receiver":
            continue
        hit = hits.get(int(frame["id"]))
        if hit is None:
            continue
        _, color = hit
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner == BLACK] = color
    return out


def _sample_scene() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    scene = {
        "grid_shape": (int(h), int(w)),
        "occupied": np.zeros((h, w), dtype=bool),
        "frames": [],
        "rays": [],
        "used_colors": set(),
        "next_id": 0,
    }

    cluster_count = random.randint(1, 3)
    for _ in range(cluster_count):
        if not _place_receiver_cluster(scene):
            return None

    extra_receivers = random.randint(0, 1)
    for _ in range(extra_receivers):
        _place_extra_receiver(scene)

    if random.random() < 0.45:
        _place_stray_emitter(scene)

    if not any(ray["target_id"] is not None for ray in scene["rays"]):
        return None

    grid = _render_scene(scene)
    expected = _expected_from_scene(grid, scene)

    detected = _find_frames(grid)
    if len(detected) != len(scene["frames"]):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SCENE_RETRIES):
        grid = _sample_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scene")
