from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("up", "down", "left", "right")
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(int(x) for x in mask.shape), mask.tobytes()


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = _trim(np.rot90(mask, k))
        key = _shape_key(rot)
        if key not in seen:
            seen.add(key)
            out.append(rot)
    return out


def _component_infos(grid: np.ndarray, color: int) -> list[dict]:
    infos = []
    for idx, coords in enumerate(_components(grid == color)):
        mask, box = _mask_from_coords(coords)
        infos.append(
            {
                "id": int(idx),
                "color": int(color),
                "coords": [(int(r), int(c)) for r, c in coords],
                "mask": mask,
                "bbox": box,
            }
        )
    return infos


def _frame_border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[int, list[dict]]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    comps_by_color = {color: _component_infos(grid, color) for color in colors}
    frames: list[dict] = []

    for color in colors:
        for comp in comps_by_color[color]:
            r0, c0, r1, c1 = comp["bbox"]
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue

            border = _frame_border_mask(h, w)
            if comp["mask"].shape != border.shape:
                continue
            if np.any(comp["mask"] & ~border):
                continue
            missing = np.argwhere(border & ~comp["mask"])
            if len(missing) != 1:
                continue

            mr, mc = int(missing[0, 0]), int(missing[0, 1])
            side_hits = int(mr in (0, h - 1)) + int(mc in (0, w - 1))
            if side_hits != 1:
                continue

            if mr == 0:
                side = "up"
            elif mr == h - 1:
                side = "down"
            elif mc == 0:
                side = "left"
            else:
                side = "right"

            if r1 - r0 <= 1 or c1 - c0 <= 1:
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            key_union = inner != bg
            if not key_union.any():
                continue
            key_comps = _components(key_union)
            if len(key_comps) != 1:
                continue
            key_mask, key_box = _mask_from_coords(key_comps[0])
            key_global = (r0 + 1 + key_box[0], c0 + 1 + key_box[1], r0 + 1 + key_box[2], c0 + 1 + key_box[3])

            frames.append(
                {
                    "color": int(color),
                    "component_id": int(comp["id"]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "gap": (int(r0 + mr), int(c0 + mc)),
                    "side": side,
                    "key_mask": key_mask,
                    "key_bbox": key_global,
                }
            )

    return frames, comps_by_color


def _project_top_left(
    frame_bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
    side: str,
    shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    gr, gc = gap
    h, w = shape
    if side == "up":
        return int(r0 - h), int(gc - (w // 2))
    if side == "down":
        return int(r1 + 1), int(gc - (w // 2))
    if side == "left":
        return int(gr - (h // 2)), int(c0 - w)
    if side == "right":
        return int(gr - (h // 2)), int(c1 + 1)
    raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames, comps_by_color = _find_frames(g, bg)
    out = g.copy()

    used_lock_ids: set[tuple[int, int]] = set()
    placements: list[tuple[tuple[int, int], np.ndarray, int]] = []

    for frame in frames:
        frame_color = int(frame["color"])
        key_mask = frame["key_mask"]
        matches: list[tuple[dict, int]] = []
        for comp in comps_by_color.get(frame_color, []):
            if int(comp["id"]) == int(frame["component_id"]):
                continue
            lock_mask = _trim(comp["mask"])
            rot_hits = []
            for k in range(4):
                if np.array_equal(_trim(np.rot90(key_mask, k)), lock_mask):
                    rot_hits.append(int(k))
            if rot_hits:
                matches.append((comp, rot_hits[0]))

        assert len(matches) == 1, "each frame must match exactly one same-color lock"
        lock_comp, rot_k = matches[0]
        used_lock_ids.add((frame_color, int(lock_comp["id"])))

        for r, c in lock_comp["coords"]:
            out[r, c] = bg

        rotated = _trim(np.rot90(key_mask, rot_k))
        top, left = _project_top_left(frame["bbox"], frame["gap"], frame["side"], rotated.shape)
        placements.append(((int(top), int(left)), rotated, frame_color))

    for (top, left), mask, color in placements:
        h, w = mask.shape
        assert 0 <= top and 0 <= left and top + h <= out.shape[0] and left + w <= out.shape[1]
        for dr in range(h):
            for dc in range(w):
                if mask[dr, dc]:
                    out[top + dr, left + dc] = int(color)

    return out


def _rotations_are_distinct(mask: np.ndarray) -> bool:
    return len(_unique_rotations(mask)) == 4


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target = random.randint(3, min(6, h * w - 1))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for dr, dc in DIR4:
            nr = start[0] + dr
            nc = start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < target and frontier:
            nr, nc = random.choice(tuple(frontier))
            frontier.remove((nr, nc))
            cells.add((nr, nc))
            for dr, dc in DIR4:
                rr = nr + dr
                cc = nc + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not _rotations_are_distinct(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a non-symmetric key")


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                grid[top + r, left + c] = int(color)


def _paint_frame_patch(
    grid: np.ndarray,
    top: int,
    left: int,
    frame_h: int,
    frame_w: int,
    frame_color: int,
    gap_local: tuple[int, int],
) -> None:
    border = _frame_border_mask(frame_h, frame_w)
    border[int(gap_local[0]), int(gap_local[1])] = False
    _paint_mask(grid, border, top, left, frame_color)


def _frame_spec(frame_color: int, key_color: int) -> dict:
    key_mask = _random_connected_mask()
    rot_k = random.randrange(4)
    lock_mask = _trim(np.rot90(key_mask, rot_k))
    side = random.choice(SIDES)

    inner_h = key_mask.shape[0] + random.randint(2, 4)
    inner_w = key_mask.shape[1] + random.randint(2, 4)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    key_r = 1 + random.randint(0, inner_h - key_mask.shape[0])
    key_c = 1 + random.randint(0, inner_w - key_mask.shape[1])

    if side in ("up", "down"):
        gap_local = (0 if side == "up" else frame_h - 1, random.randint(1, frame_w - 2))
    else:
        gap_local = (random.randint(1, frame_h - 2), 0 if side == "left" else frame_w - 1)

    rel_frame_box = (0, 0, frame_h - 1, frame_w - 1)
    rel_gap = (int(gap_local[0]), int(gap_local[1]))
    proj_top, proj_left = _project_top_left(rel_frame_box, rel_gap, side, lock_mask.shape)
    proj_box = (proj_top, proj_left, proj_top + lock_mask.shape[0] - 1, proj_left + lock_mask.shape[1] - 1)

    return {
        "frame_color": int(frame_color),
        "key_color": int(key_color),
        "key_mask": key_mask,
        "lock_mask": lock_mask,
        "rot_k": int(rot_k),
        "side": side,
        "frame_h": int(frame_h),
        "frame_w": int(frame_w),
        "key_top": int(key_r),
        "key_left": int(key_c),
        "gap_local": rel_gap,
        "proj_rel_box": proj_box,
        "union_area": int((_union_bbox(rel_frame_box, proj_box)[2] - _union_bbox(rel_frame_box, proj_box)[0] + 1)
                           * (_union_bbox(rel_frame_box, proj_box)[3] - _union_bbox(rel_frame_box, proj_box)[1] + 1)),
    }


def _projected_box_for_frame(spec: dict, top: int, left: int) -> tuple[int, int, int, int]:
    rel = spec["proj_rel_box"]
    return top + rel[0], left + rel[1], top + rel[2], left + rel[3]


def _place_frames(specs: list[dict], h: int, w: int) -> list[dict] | None:
    placed: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda item: -int(item["union_area"])):
        candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for top in range(h - spec["frame_h"] + 1):
            for left in range(w - spec["frame_w"] + 1):
                frame_box = (top, left, top + spec["frame_h"] - 1, left + spec["frame_w"] - 1)
                proj_box = _projected_box_for_frame(spec, top, left)
                if proj_box[0] < 0 or proj_box[1] < 0 or proj_box[2] >= h or proj_box[3] >= w:
                    continue
                expanded = _expand_bbox(_union_bbox(frame_box, proj_box), 1, h, w)
                if any(_bbox_overlap(expanded, other) for other in reserved):
                    continue
                candidates.append((int(top), int(left), expanded))

        if not candidates:
            return None
        top, left, expanded = random.choice(candidates)
        spec = dict(spec)
        spec["frame_top"] = int(top)
        spec["frame_left"] = int(left)
        spec["frame_bbox"] = (int(top), int(left), int(top + spec["frame_h"] - 1), int(left + spec["frame_w"] - 1))
        spec["proj_box"] = _projected_box_for_frame(spec, top, left)
        spec["gap_global"] = (int(top + spec["gap_local"][0]), int(left + spec["gap_local"][1]))
        placed.append(spec)
        reserved.append(expanded)

    return placed


def _place_locks(grid: np.ndarray, specs: list[dict], bg: int, reserved_boxes: list[tuple[int, int, int, int]]) -> bool:
    h, w = grid.shape
    for spec in sorted(specs, key=lambda item: -int(item["lock_mask"].sum())):
        mask = spec["lock_mask"]
        ph, pw = mask.shape
        candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for top in range(h - ph + 1):
            for left in range(w - pw + 1):
                box = (top, left, top + ph - 1, left + pw - 1)
                expanded = _expand_bbox(box, 1, h, w)
                if any(_bbox_overlap(expanded, other) for other in reserved_boxes):
                    continue
                if np.any(grid[top : top + ph, left : left + pw] != bg):
                    continue
                candidates.append((int(top), int(left), expanded))
        if not candidates:
            return False
        top, left, expanded = random.choice(candidates)
        spec["lock_top"] = int(top)
        spec["lock_left"] = int(left)
        spec["lock_bbox"] = (int(top), int(left), int(top + ph - 1), int(left + pw - 1))
        _paint_mask(grid, mask, top, left, spec["frame_color"])
        reserved_boxes.append(expanded)
    return True


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(20, 32)
    w = random.randint(20, 32)
    bg = random.randrange(10)
    frame_count = random.randint(2, 4)

    colors = [c for c in range(10) if c != bg]
    if len(colors) < 2 * frame_count:
        return None
    random.shuffle(colors)
    frame_colors = colors[:frame_count]
    key_colors = colors[frame_count : 2 * frame_count]

    specs = [_frame_spec(frame_colors[i], key_colors[i]) for i in range(frame_count)]
    placed = _place_frames(specs, h, w)
    if placed is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    reserved_boxes: list[tuple[int, int, int, int]] = []

    for spec in placed:
        top = spec["frame_top"]
        left = spec["frame_left"]
        _paint_frame_patch(inp, top, left, spec["frame_h"], spec["frame_w"], spec["frame_color"], spec["gap_local"])
        _paint_frame_patch(out, top, left, spec["frame_h"], spec["frame_w"], spec["frame_color"], spec["gap_local"])

        _paint_mask(inp, spec["key_mask"], top + spec["key_top"], left + spec["key_left"], spec["key_color"])
        _paint_mask(out, spec["key_mask"], top + spec["key_top"], left + spec["key_left"], spec["key_color"])

        proj_top, proj_left, _, _ = spec["proj_box"]
        _paint_mask(out, spec["lock_mask"], proj_top, proj_left, spec["frame_color"])
        reserved_boxes.append(_expand_bbox(_union_bbox(spec["frame_bbox"], spec["proj_box"]), 1, h, w))

    if not _place_locks(inp, placed, bg, reserved_boxes):
        return None

    values, counts = np.unique(inp, return_counts=True)
    dominant = int(values[int(np.argmax(counts))])
    if dominant != bg:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_example()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
