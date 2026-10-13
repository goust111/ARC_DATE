from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import permutations
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

BG = 0
BAR_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 800


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _extract_frame_colors(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h < 5 or w < 5:
        raise ValueError("grid too small for the four border bars")

    top = int(g[0, 0])
    bottom = int(g[h - 1, 0])
    left = int(g[1, 0])
    right = int(g[1, w - 1])

    if not np.all(g[0, :] == top):
        raise ValueError("top bar is not solid")
    if not np.all(g[h - 1, :] == bottom):
        raise ValueError("bottom bar is not solid")
    if not np.all(g[1 : h - 1, 0] == left):
        raise ValueError("left bar is not solid")
    if not np.all(g[1 : h - 1, w - 1] == right):
        raise ValueError("right bar is not solid")
    if len({top, bottom, left, right}) != 4:
        raise ValueError("the four border bars must use four distinct colors")

    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _render_frame(shape: tuple[int, int], frame_colors: dict[str, int], bg: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    out[0, :] = int(frame_colors["top"])
    out[h - 1, :] = int(frame_colors["bottom"])
    out[1 : h - 1, 0] = int(frame_colors["left"])
    out[1 : h - 1, w - 1] = int(frame_colors["right"])
    frame_mask = np.zeros((h, w), dtype=bool)
    frame_mask[0, :] = True
    frame_mask[h - 1, :] = True
    frame_mask[1 : h - 1, 0] = True
    frame_mask[1 : h - 1, w - 1] = True
    return out, frame_mask


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if visited[r, c] or int(g[r, c]) == int(bg):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) == int(bg):
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"coords": coords, "bbox": _bbox(coords), "area": int(coords.shape[0])})
    return comps


def _analyze_patch(
    patch: np.ndarray, bar_color_to_side: dict[int, str], bg: int
) -> tuple[int, int, tuple[int, int]]:
    vals, counts = np.unique(patch[patch != int(bg)], return_counts=True)
    if len(vals) != 2:
        raise ValueError("each object must have exactly one body color and one marker color")
    marker_candidates = [int(v) for v, cnt in zip(vals, counts) if int(cnt) == 1 and int(v) in bar_color_to_side]
    if len(marker_candidates) != 1:
        raise ValueError("object marker is not uniquely identifiable")
    marker_color = marker_candidates[0]
    body_colors = [int(v) for v in vals if int(v) != marker_color]
    if len(body_colors) != 1:
        raise ValueError("object body color is not uniquely identifiable")
    marker_coords = np.argwhere(patch == marker_color)
    if marker_coords.shape[0] != 1:
        raise ValueError("marker pixel must be unique")
    mr, mc = marker_coords[0]
    return body_colors[0], marker_color, (int(mr), int(mc))


def _distance_from_side(bbox: tuple[int, int, int, int], side: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if side == "top":
        return int(r0 - 1)
    if side == "bottom":
        return int((h - 2) - r1)
    if side == "left":
        return int(c0 - 1)
    if side == "right":
        return int((w - 2) - c1)
    raise ValueError(side)


def _marker_faces_side(patch: np.ndarray, marker_color: int, side: str, bg: int) -> bool:
    coords = np.argwhere(patch != int(bg))
    mr, mc = np.argwhere(patch == int(marker_color))[0]
    if side == "top":
        return int(mr) == int(coords[:, 0].min())
    if side == "bottom":
        return int(mr) == int(coords[:, 0].max())
    if side == "left":
        return int(mc) == int(coords[:, 1].min())
    if side == "right":
        return int(mc) == int(coords[:, 1].max())
    raise ValueError(side)


def _orient_patch(patch: np.ndarray, marker_color: int, side: str, bg: int) -> tuple[np.ndarray, int]:
    for k in range(4):
        rot = np.rot90(patch, k)
        if _marker_faces_side(rot, marker_color, side, bg):
            return rot, int(k)
    raise ValueError("no quarter-turn makes the marker face the target side")


def _fits_without_overlap(board: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    bh, bw = board.shape
    if r0 < 0 or c0 < 0 or r0 + h > bh or c0 + w > bw:
        return False
    mask = patch != int(bg)
    window = board[r0 : r0 + h, c0 : c0 + w]
    return not bool(((window != int(bg)) & mask).any())


def _place_patch(board: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != int(bg)
    window = board[r0 : r0 + h, c0 : c0 + w]
    window[mask] = patch[mask]


def _collision_kinds(
    board: np.ndarray, frame_mask: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int
) -> tuple[bool, bool]:
    h, w = patch.shape
    bh, bw = board.shape
    if r0 < 0 or c0 < 0 or r0 + h > bh or c0 + w > bw:
        return True, False
    mask = patch != int(bg)
    overlap = (board[r0 : r0 + h, c0 : c0 + w] != int(bg)) & mask
    if not bool(overlap.any()):
        return False, False
    hit_frame = bool((frame_mask[r0 : r0 + h, c0 : c0 + w] & overlap).any())
    hit_object = bool(((~frame_mask[r0 : r0 + h, c0 : c0 + w]) & overlap).any())
    return hit_frame, hit_object


def _slide_until_contact(board: np.ndarray, patch: np.ndarray, start_r: int, start_c: int, side: str, bg: int) -> tuple[int, int]:
    dr, dc = SIDE_TO_VEC[side]
    r, c = int(start_r), int(start_c)
    if not _fits_without_overlap(board, patch, r, c, bg):
        raise ValueError("rotated patch does not fit at its starting location")
    while _fits_without_overlap(board, patch, r + dr, c + dc, bg):
        r += dr
        c += dc
    return int(r), int(c)


def _parse_objects(grid: np.ndarray) -> tuple[list[dict], dict[str, int], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_colors = _extract_frame_colors(g)
    bar_color_to_side = {int(color): side for side, color in frame_colors.items()}
    objects = []
    for comp in _find_components(g, bg):
        bbox = comp["bbox"]
        patch = _crop(g, bbox)
        body_color, marker_color, marker_rc = _analyze_patch(patch, bar_color_to_side, bg)
        if not _marker_faces_side(patch, marker_color, "top", bg) and not _marker_faces_side(
            patch, marker_color, "bottom", bg
        ) and not _marker_faces_side(patch, marker_color, "left", bg) and not _marker_faces_side(
            patch, marker_color, "right", bg
        ):
            raise ValueError("marker pixel must lie on the component boundary")
        target_side = bar_color_to_side[marker_color]
        objects.append(
            {
                "bbox": bbox,
                "patch": patch,
                "body_color": int(body_color),
                "marker_color": int(marker_color),
                "marker_rc": marker_rc,
                "target_side": target_side,
                "distance": _distance_from_side(bbox, target_side, g.shape),
            }
        )
    return objects, frame_colors, int(bg)


def _simulate(
    grid: np.ndarray,
    *,
    edge_order: tuple[str, ...] = SIDES,
    rotate: bool = True,
    stack: bool = True,
    route_mode: str = "marker",
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    objects, frame_colors, bg = _parse_objects(g)
    bar_color_to_side = {int(color): side for side, color in frame_colors.items()}
    out, frame_mask = _render_frame(g.shape, frame_colors, bg)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for obj in objects:
        if route_mode == "marker":
            side = obj["target_side"]
        elif route_mode == "body_prefers_bar":
            side = bar_color_to_side.get(int(obj["body_color"]), obj["target_side"])
        else:
            raise ValueError(route_mode)

        if rotate:
            patch, rot_k = _orient_patch(obj["patch"], obj["marker_color"], side, bg)
        else:
            patch, rot_k = obj["patch"], 0

        entry = dict(obj)
        entry["route_side"] = side
        entry["oriented_patch"] = patch
        entry["rot_k"] = int(rot_k)
        entry["route_distance"] = _distance_from_side(entry["bbox"], side, g.shape)
        grouped[side].append(entry)

    for side in SIDES:
        grouped[side].sort(key=lambda obj: (obj["route_distance"], obj["bbox"][0], obj["bbox"][1]))

    traces: list[dict] = []
    if not stack:
        bare_frame, _ = _render_frame(g.shape, frame_colors, bg)
        for side in edge_order:
            for obj in grouped[side]:
                patch = obj["oriented_patch"]
                start_r, start_c = obj["bbox"][0], obj["bbox"][1]
                if not _fits_without_overlap(bare_frame, patch, start_r, start_c, bg):
                    raise ValueError("simplified solver cannot place the rotated patch at its source")
                final_r, final_c = _slide_until_contact(bare_frame, patch, start_r, start_c, side, bg)
                _place_patch(out, patch, final_r, final_c, bg)
        return out if not return_trace else (out, traces)

    for side in edge_order:
        for obj in grouped[side]:
            patch = obj["oriented_patch"]
            start_r, start_c = obj["bbox"][0], obj["bbox"][1]
            final_r, final_c = _slide_until_contact(out, patch, start_r, start_c, side, bg)
            dr, dc = SIDE_TO_VEC[side]
            hit_frame, hit_object = _collision_kinds(out, frame_mask, patch, final_r + dr, final_c + dc, bg)
            _place_patch(out, patch, final_r, final_c, bg)
            if return_trace:
                traces.append(
                    {
                        "side": side,
                        "body_color": obj["body_color"],
                        "marker_color": obj["marker_color"],
                        "distance": obj["route_distance"],
                        "rot_k": obj["rot_k"],
                        "source_bbox": obj["bbox"],
                        "source_top_left": (int(start_r), int(start_c)),
                        "final_top_left": (int(final_r), int(final_c)),
                        "final_shape": tuple(int(x) for x in patch.shape),
                        "stopped_on_bar": bool(hit_frame),
                        "stopped_on_object": bool(hit_object),
                    }
                )
    return out if not return_trace else (out, traces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each marked object so its marker faces the chosen border bar, then slide it there and stack by arrival order.

    Input structure:
    - A 1-cell frame surrounds the grid: the top row, bottom row, left interior column, and right interior column are four distinct solid border bars.
    - The interior background is uniform.
    - Each interior object is one 8-connected component made of a single body color plus exactly one different marker pixel whose color matches one border bar.

    Rule:
    - The marker color chooses the destination side.
    - Crop each object to its bbox and rotate it by the smallest `k in {0,1,2,3}` quarter-turns (`np.rot90(..., k)`) that puts the marker on the object's leading side for that route.
    - Redraw only the frame first, then for each side process its objects from smaller initial distance to larger initial distance and slide each one straight toward that bar until the next step would hit the bar or an already parked object.

    Key insight:
    - The single odd pixel carries two visible cues at once: its color selects the destination bar, and its position inside the shape fixes the landing orientation before stacking.
    """
    return _simulate(np.asarray(grid, dtype=int), return_trace=False)


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _side_cells(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if side == "top":
        edge = int(coords[:, 0].min())
        cells = [(int(r), int(c)) for r, c in coords if int(r) == edge]
    elif side == "bottom":
        edge = int(coords[:, 0].max())
        cells = [(int(r), int(c)) for r, c in coords if int(r) == edge]
    elif side == "left":
        edge = int(coords[:, 1].min())
        cells = [(int(r), int(c)) for r, c in coords if int(c) == edge]
    elif side == "right":
        edge = int(coords[:, 1].max())
        cells = [(int(r), int(c)) for r, c in coords if int(c) == edge]
    else:
        raise ValueError(side)
    if len(cells) >= 2:
        non_corner = []
        h, w = mask.shape
        for r, c in cells:
            if r in {0, h - 1} and c in {0, w - 1}:
                continue
            non_corner.append((r, c))
        if non_corner:
            cells = non_corner
    return cells


def _make_final_patch(side: str, body_color: int, marker_color: int) -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(max(4, h + w - 1), min(h * w, 7))
        mask = _random_connected_mask(h, w, area)
        if mask.sum() < 4:
            continue
        if len(np.unique(np.argwhere(mask)[:, 0])) == 1 or len(np.unique(np.argwhere(mask)[:, 1])) == 1:
            continue
        side_cells = _side_cells(mask, side)
        if not side_cells:
            continue
        marker_rc = random.choice(side_cells)
        patch = np.full((h, w), BG, dtype=int)
        patch[mask] = int(body_color)
        patch[marker_rc] = int(marker_color)
        if not _marker_faces_side(patch, marker_color, side, BG):
            continue
        if any(_marker_faces_side(patch, marker_color, other, BG) for other in SIDES if other != side):
            continue
        return patch
    raise RuntimeError("failed to build a nondegenerate patch")


def _group_depth(objects: list[dict], side: str) -> int:
    if side in {"top", "bottom"}:
        return int(sum(obj["final_patch"].shape[0] for obj in objects))
    return int(sum(obj["final_patch"].shape[1] for obj in objects))


def _pick_body_colors(obj_sides: list[str], frame_colors: dict[str, int]) -> list[int]:
    n = len(obj_sides)
    body_colors = [None] * n
    pair = random.sample(range(n), 2)
    while obj_sides[pair[0]] == obj_sides[pair[1]]:
        pair = random.sample(range(n), 2)
    spare_sides = [side for side in SIDES if side not in {obj_sides[pair[0]], obj_sides[pair[1]]}]
    decoy_side = random.choice(spare_sides)
    shared = int(frame_colors[decoy_side])
    body_colors[pair[0]] = shared
    body_colors[pair[1]] = shared

    for i, side in enumerate(obj_sides):
        if body_colors[i] is not None:
            continue
        choices = [color for color in BAR_PALETTE if color != int(frame_colors[side])]
        body_colors[i] = int(random.choice(choices))
    return [int(x) for x in body_colors]


def _max_input_offset(final_top_left: tuple[int, int], input_patch: np.ndarray, side: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = final_top_left
    ph, pw = input_patch.shape
    if side == "top":
        return int((h - 1 - ph) - r)
    if side == "bottom":
        return int(r - 1)
    if side == "left":
        return int((w - 1 - pw) - c)
    if side == "right":
        return int(c - 1)
    raise ValueError(side)


def _offset_top_left(final_top_left: tuple[int, int], side: str, offset: int) -> tuple[int, int]:
    r, c = final_top_left
    if side == "top":
        return int(r + offset), int(c)
    if side == "bottom":
        return int(r - offset), int(c)
    if side == "left":
        return int(r), int(c + offset)
    if side == "right":
        return int(r), int(c - offset)
    raise ValueError(side)


def _choose_offsets(side: str, objects: list[dict], shape: tuple[int, int]) -> list[int] | None:
    if len(objects) == 1:
        max_offset = _max_input_offset(objects[0]["final_top_left"], objects[0]["input_patch"], side, shape)
        if max_offset < 3:
            return None
        return [random.randint(3, max_offset)]

    if len(objects) != 2:
        return None

    max0 = _max_input_offset(objects[0]["final_top_left"], objects[0]["input_patch"], side, shape)
    max1 = _max_input_offset(objects[1]["final_top_left"], objects[1]["input_patch"], side, shape)
    if min(max0, max1) < 3 or max1 < 5:
        return None

    candidates = []
    for near in range(3, max0 + 1):
        for far in range(near + 2, max1 + 1):
            candidates.append((near, far))
    if not candidates:
        return None
    near, far = random.choice(candidates)
    return [int(near), int(far)]


def _build_input_from_specs(specs: list[dict], frame_colors: dict[str, int], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    frame_only, _ = _render_frame((h, w), frame_colors, BG)
    objects_by_side: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        objects_by_side[spec["side"]].append(spec)
    for side in SIDES:
        objects_by_side[side].sort(key=lambda obj: obj["rank"])

    for side in SIDES:
        offsets = _choose_offsets(side, objects_by_side[side], (h, w))
        if offsets is None:
            return None
        for spec, offset in zip(objects_by_side[side], offsets):
            spec["input_offset"] = int(offset)
            spec["input_top_left"] = _offset_top_left(spec["final_top_left"], side, offset)

    input_grid = frame_only.copy()
    for spec in specs:
        r0, c0 = spec["input_top_left"]
        if not _fits_without_overlap(input_grid, spec["input_patch"], r0, c0, BG):
            return None
        _place_patch(input_grid, spec["input_patch"], r0, c0, BG)
    if _component_count(input_grid) != len(specs):
        return None

    rotated_start_grid = frame_only.copy()
    for spec in specs:
        r0, c0 = spec["input_top_left"]
        if not _fits_without_overlap(rotated_start_grid, spec["final_patch"], r0, c0, BG):
            return None
        _place_patch(rotated_start_grid, spec["final_patch"], r0, c0, BG)
    if _component_count(rotated_start_grid) != len(specs):
        return None

    try:
        expected = _simulate(input_grid, return_trace=False)
    except Exception:
        return None
    return input_grid, expected


def _component_count(grid: np.ndarray) -> int:
    return len(_find_components(grid, BG))


def _validate_semantics(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        objects, frame_colors, bg = _parse_objects(grid)
    except Exception:
        return False

    if bg != BG:
        return False
    if len(objects) != 5:
        return False

    side_counts = Counter(obj["target_side"] for obj in objects)
    if sorted(side_counts.values()) != [1, 1, 1, 2]:
        return False
    for side in SIDES:
        dists = [obj["distance"] for obj in objects if obj["target_side"] == side]
        if len(dists) != len(set(dists)):
            return False

    if _component_count(grid) != len(objects):
        return False

    truth, trace = _simulate(grid, return_trace=True)
    if not np.array_equal(truth, expected):
        return False
    if not any(t["stopped_on_object"] for t in trace):
        return False
    if not any(t["rot_k"] > 0 for t in trace):
        return False

    body_to_sides: dict[int, set[str]] = defaultdict(set)
    for obj in objects:
        body_to_sides[int(obj["body_color"])].add(obj["target_side"])
    if not any(len(sides) >= 2 for sides in body_to_sides.values()):
        return False

    bar_colors = set(int(color) for color in frame_colors.values())
    if not any(int(obj["body_color"]) in bar_colors and int(obj["body_color"]) != int(obj["marker_color"]) for obj in objects):
        return False

    for order in permutations(SIDES):
        try:
            candidate = _simulate(grid, edge_order=order, return_trace=False)
        except Exception:
            return False
        if not np.array_equal(candidate, truth):
            return False

    try:
        if np.array_equal(_simulate(grid, rotate=False, return_trace=False), truth):
            return False
    except Exception:
        pass

    try:
        if np.array_equal(_simulate(grid, stack=False, return_trace=False), truth):
            return False
    except Exception:
        pass

    try:
        if np.array_equal(_simulate(grid, route_mode="body_prefers_bar", return_trace=False), truth):
            return False
    except Exception:
        pass

    if np.array_equal(grid, truth):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    frame_color_values = random.sample(BAR_PALETTE, 4)
    frame_colors = dict(zip(SIDES, frame_color_values))

    stack_side = random.choice(SIDES)
    obj_sides = list(SIDES) + [stack_side]
    body_colors = _pick_body_colors(obj_sides, frame_colors)

    specs = []
    per_side_rank = Counter()
    forced_rot_index = random.randrange(len(obj_sides))
    for idx, side in enumerate(obj_sides):
        body_color = body_colors[idx]
        marker_color = int(frame_colors[side])
        patch = _make_final_patch(side, body_color, marker_color)
        rot_back = random.randint(0, 3)
        if idx == forced_rot_index and rot_back == 0:
            rot_back = random.choice([1, 2, 3])
        specs.append(
            {
                "id": idx,
                "side": side,
                "rank": int(per_side_rank[side]),
                "body_color": int(body_color),
                "marker_color": int(marker_color),
                "final_patch": patch,
                "input_patch": np.rot90(patch, rot_back),
                "rot_back": int(rot_back),
            }
        )
        per_side_rank[side] += 1

    top_depth = _group_depth([spec for spec in specs if spec["side"] == "top"], "top")
    bottom_depth = _group_depth([spec for spec in specs if spec["side"] == "bottom"], "bottom")
    left_depth = _group_depth([spec for spec in specs if spec["side"] == "left"], "left")
    right_depth = _group_depth([spec for spec in specs if spec["side"] == "right"], "right")

    mid_h_need = max(spec["final_patch"].shape[0] for spec in specs if spec["side"] in {"left", "right"})
    mid_w_need = max(spec["final_patch"].shape[1] for spec in specs if spec["side"] in {"top", "bottom"})
    min_h = max(18, top_depth + bottom_depth + mid_h_need + 6)
    min_w = max(18, left_depth + right_depth + mid_w_need + 6)
    if min_h > 29 or min_w > 29:
        return None

    h = random.randint(min_h, 29)
    w = random.randint(min_w, 29)
    final_grid, _ = _render_frame((h, w), frame_colors, BG)

    row_min = top_depth + 2
    row_max = h - bottom_depth - 3
    col_min = left_depth + 2
    col_max = w - right_depth - 3
    if row_min > row_max or col_min > col_max:
        return None

    side_specs: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        side_specs[spec["side"]].append(spec)
    for side in SIDES:
        side_specs[side].sort(key=lambda obj: obj["rank"])

    for side in ("top", "bottom"):
        group = side_specs[side]
        lane_width = max(spec["final_patch"].shape[1] for spec in group)
        c_lo = col_min
        c_hi = w - right_depth - lane_width - 1
        if c_lo > c_hi:
            return None
        lane_c = random.randint(c_lo, c_hi)
        for spec in group:
            spec["lane_c"] = lane_c

    for side in ("left", "right"):
        group = side_specs[side]
        lane_height = max(spec["final_patch"].shape[0] for spec in group)
        r_lo = row_min
        r_hi = h - bottom_depth - lane_height - 1
        if r_lo > r_hi:
            return None
        lane_r = random.randint(r_lo, r_hi)
        for spec in group:
            spec["lane_r"] = lane_r

    for side in SIDES:
        for spec in side_specs[side]:
            patch = spec["final_patch"]
            ph, pw = patch.shape
            if side == "top":
                start_r = h - bottom_depth - ph - 2
                start_c = spec["lane_c"]
            elif side == "bottom":
                start_r = top_depth + 2
                start_c = spec["lane_c"]
            elif side == "left":
                start_r = spec["lane_r"]
                start_c = w - right_depth - pw - 2
            else:
                start_r = spec["lane_r"]
                start_c = left_depth + 2
            if not _fits_without_overlap(final_grid, patch, start_r, start_c, BG):
                return None
            final_r, final_c = _slide_until_contact(final_grid, patch, start_r, start_c, side, BG)
            spec["final_top_left"] = (int(final_r), int(final_c))
            _place_patch(final_grid, patch, final_r, final_c, BG)

    stacked_specs = side_specs[stack_side]
    if len(stacked_specs) != 2:
        return None

    built = _build_input_from_specs(specs, frame_colors, (h, w))
    if built is None:
        return None
    input_grid, expected = built

    if not np.array_equal(expected, final_grid):
        return None
    if not _validate_semantics(input_grid, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the border-bar rotation puzzle")
