from __future__ import annotations

from collections import Counter, deque
import random
import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "right", "bottom", "left")
BG_COLOR = 8
GEN_RETRIES = 400
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
ROT_FOR_SIDE = {"left": 0, "bottom": 1, "right": 2, "top": 3}


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_points(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_true(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    return _bbox_from_points(points)


def _crop_true(mask: np.ndarray) -> np.ndarray:
    box = _bbox_true(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    start = tuple(map(int, points[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == int(mask.sum())


def _count_full_2x2(mask: np.ndarray) -> int:
    h, w = mask.shape
    cnt = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if bool(mask[r : r + 2, c : c + 2].all()):
                cnt += 1
    return cnt


def _shape_token(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_true(mask)
    return tuple(map(int, cropped.shape)), cropped.astype(np.uint8).tobytes()


def _canonical_rot_token(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_shape_token(np.rot90(mask, k)) for k in range(4))


def _component_infos(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if bool(seen[r, c]) or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            pts = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_points(pts)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
            comps.append(
                {
                    "color": int(color),
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    return comps


def _extract_key(mask: np.ndarray) -> dict | None:
    m = np.asarray(mask, dtype=bool)
    if not bool(m.any()) or not _is_connected(m):
        return None
    h, w = m.shape
    candidates = []
    for r in range(h - 1):
        for c in range(w - 1):
            if not bool(m[r : r + 2, c : c + 2].all()):
                continue
            teeth = m.copy()
            teeth[r : r + 2, c : c + 2] = False
            if not bool(teeth.any()) or not _is_connected(teeth):
                continue
            sides = []
            if c > 0 and bool(teeth[r : r + 2, c - 1].any()):
                sides.append("left")
            if c + 2 < w and bool(teeth[r : r + 2, c + 2].any()):
                sides.append("right")
            if r > 0 and bool(teeth[r - 1, c : c + 2].any()):
                sides.append("top")
            if r + 2 < h and bool(teeth[r + 2, c : c + 2].any()):
                sides.append("bottom")
            if len(sides) != 1:
                continue
            handle = np.zeros_like(m)
            handle[r : r + 2, c : c + 2] = True
            candidates.append(
                {
                    "mask": m.copy(),
                    "handle_mask": handle,
                    "teeth_mask": teeth,
                    "attach_side": sides[0],
                    "handle_bbox": (int(r), int(c), int(r + 1), int(c + 1)),
                    "teeth_bbox": _bbox_true(teeth),
                }
            )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _extract_lock(mask: np.ndarray) -> dict | None:
    m = np.asarray(mask, dtype=bool)
    if not bool(m.any()) or not _is_connected(m):
        return None
    hole = ~m
    if not bool(hole.any()) or not _is_connected(hole):
        return None
    touches = []
    if bool(hole[0, :].any()):
        touches.append("top")
    if bool(hole[:, -1].any()):
        touches.append("right")
    if bool(hole[-1, :].any()):
        touches.append("bottom")
    if bool(hole[:, 0].any()):
        touches.append("left")
    if len(touches) != 1:
        return None
    side = touches[0]
    if side != "top" and not bool(m[0, :].all()):
        return None
    if side != "right" and not bool(m[:, -1].all()):
        return None
    if side != "bottom" and not bool(m[-1, :].all()):
        return None
    if side != "left" and not bool(m[:, 0].all()):
        return None
    edge = {
        "top": hole[0, :],
        "right": hole[:, -1],
        "bottom": hole[-1, :],
        "left": hole[:, 0],
    }[side]
    edge_idx = np.flatnonzero(edge)
    if edge_idx.size == 0 or int(edge_idx[-1] - edge_idx[0] + 1) != int(edge_idx.size):
        return None
    return {
        "mask": m.copy(),
        "hole_mask": hole,
        "hole_bbox": _bbox_true(hole),
        "side": side,
    }


def _coords_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _inside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return int(r0) <= int(r) <= int(r1) and int(c0) <= int(c) <= int(c1)


def _handle_outside_lock(
    handle_cells: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
    lock_boxes: list[tuple[int, int, int, int]] | None = None,
) -> bool:
    r0, c0, r1, c1 = bbox
    if side == "left":
        if not handle_cells or max(c for _, c in handle_cells) != int(c0 - 1):
            return False
        if any(c >= c0 for _, c in handle_cells):
            return False
    elif side == "right":
        if not handle_cells or min(c for _, c in handle_cells) != int(c1 + 1):
            return False
        if any(c <= c1 for _, c in handle_cells):
            return False
    elif side == "top":
        if not handle_cells or max(r for r, _ in handle_cells) != int(r0 - 1):
            return False
        if any(r >= r0 for r, _ in handle_cells):
            return False
    else:
        if not handle_cells or min(r for r, _ in handle_cells) != int(r1 + 1):
            return False
        if any(r <= r1 for r, _ in handle_cells):
            return False
    if lock_boxes is not None:
        for cell in handle_cells:
            if any(_inside_bbox(cell, other) for other in lock_boxes):
                return False
    return True


def _match_candidates_for_pair(
    lock: dict,
    key: dict,
    cleared_grid: np.ndarray,
    bg: int,
    lock_boxes: list[tuple[int, int, int, int]],
) -> list[dict]:
    hole_crop = _crop_true(lock["hole_mask"])
    hole_bbox = lock["hole_bbox"]
    assert hole_bbox is not None
    lr0, lc0, _lr1, _lc1 = map(int, lock["bbox"])
    hole_global = {(lr0 + int(r), lc0 + int(c)) for r, c in np.argwhere(lock["hole_mask"])}
    out: list[dict] = []
    for rot in range(4):
        rot_mask = np.rot90(key["mask"], rot)
        rot_key = _extract_key(rot_mask)
        if rot_key is None:
            continue
        teeth_crop = _crop_true(rot_key["teeth_mask"])
        if teeth_crop.shape != hole_crop.shape or not np.array_equal(teeth_crop, hole_crop):
            continue
        tr0, tc0, _tr1, _tc1 = map(int, rot_key["teeth_bbox"])
        dest_r0 = int(lr0 + hole_bbox[0] - tr0)
        dest_c0 = int(lc0 + hole_bbox[1] - tc0)
        all_cells = {(dest_r0 + int(r), dest_c0 + int(c)) for r, c in np.argwhere(rot_mask)}
        teeth_cells = {(dest_r0 + int(r), dest_c0 + int(c)) for r, c in np.argwhere(rot_key["teeth_mask"])}
        handle_cells = {(dest_r0 + int(r), dest_c0 + int(c)) for r, c in np.argwhere(rot_key["handle_mask"])}
        if teeth_cells != hole_global:
            continue
        if not _handle_outside_lock(handle_cells, lock["bbox"], lock["side"], lock_boxes):
            continue
        ok = True
        for gr, gc in all_cells:
            if not (0 <= gr < cleared_grid.shape[0] and 0 <= gc < cleared_grid.shape[1]):
                ok = False
                break
            if int(cleared_grid[gr, gc]) != int(bg):
                ok = False
                break
        if not ok:
            continue
        out.append(
            {
                "key_idx": int(key["index"]),
                "lock_idx": int(lock["index"]),
                "color": int(key["color"]),
                "rot_steps": int(rot),
                "dest": (int(dest_r0), int(dest_c0)),
                "mask": rot_mask,
                "cells": all_cells,
            }
        )
    return out


def _solve_assignments(locks: list[dict], keys: list[dict], cleared_grid: np.ndarray, bg: int) -> dict[int, dict]:
    lock_boxes = [tuple(map(int, lock["bbox"])) for lock in locks]
    candidates: dict[int, list[dict]] = {}
    for lock in locks:
        cand_list: list[dict] = []
        for key in keys:
            cand_list.extend(_match_candidates_for_pair(lock, key, cleared_grid, bg, lock_boxes))
        candidates[int(lock["index"])] = cand_list
        assert cand_list, "every lock must have a matching key candidate"
    order = sorted((int(lock["index"]) for lock in locks), key=lambda i: len(candidates[i]))

    def dfs(pos: int, used_keys: set[int], occupied: set[tuple[int, int]], chosen: dict[int, dict]) -> dict[int, dict] | None:
        if pos == len(order):
            return dict(chosen)
        lock_idx = order[pos]
        for cand in candidates[lock_idx]:
            key_idx = int(cand["key_idx"])
            if key_idx in used_keys or bool(cand["cells"] & occupied):
                continue
            used_keys.add(key_idx)
            chosen[lock_idx] = cand
            result = dfs(pos + 1, used_keys, occupied | set(cand["cells"]), chosen)
            if result is not None:
                return result
            used_keys.remove(key_idx)
            del chosen[lock_idx]
        return None

    result = dfs(0, set(), set(), {})
    assert result is not None, "lock-key matching must be unambiguous"
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scattered key to the lock whose notch it complements, then insert the rotated key into that notch.

    Input structure:
    - The background is azure.
    - Foreground objects are monochrome 4-connected components of two kinds: larger lock blocks and smaller key sprites.
    - Each lock is a solid rectangle with one irregular azure notch carved from exactly one outer side.
    - Each key contains a solid 2x2 handle plus a connected tooth shape extending from one side of that handle.

    Rule:
    - Extract every lock notch shape and every key tooth shape.
    - For each key, try the four rotations and find the unique lock whose notch matches the rotated tooth exactly.
    - Clear every key from its original location.
    - Repaint each matched key at the lock so that its tooth fills the notch in the key's color while the 2x2 handle stays just outside the lock edge.

    Key insight:
    - Colors do not define the pairing; the solution comes from shape-complement matching between a lock's carved notch and a key's tooth under rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _component_infos(g, bg)
    locks: list[dict] = []
    keys: list[dict] = []
    for idx, comp in enumerate(comps):
        lock_info = _extract_lock(comp["mask"])
        key_info = _extract_key(comp["mask"])
        if lock_info is not None and key_info is None:
            item = dict(comp)
            item.update(lock_info)
            item["index"] = int(len(locks))
            locks.append(item)
        elif key_info is not None and lock_info is None:
            item = dict(comp)
            item.update(key_info)
            item["index"] = int(len(keys))
            keys.append(item)
        else:
            raise AssertionError("every component must be classified as exactly one key or one lock")
    assert locks and len(locks) == len(keys)

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[int(r), int(c)] = int(bg)

    placements = _solve_assignments(locks, keys, out, bg)
    for cand in placements.values():
        dest_r0, dest_c0 = map(int, cand["dest"])
        color = int(cand["color"])
        for r, c in np.argwhere(cand["mask"]):
            out[dest_r0 + int(r), dest_c0 + int(c)] = int(color)
    return out


def _build_key_mask_east(tooth: np.ndarray, handle_top: int) -> np.ndarray:
    tooth = np.asarray(tooth, dtype=bool)
    rows = np.flatnonzero(tooth[:, 0])
    assert rows.size == 1
    assert 0 <= int(handle_top) <= int(tooth.shape[0] - 2)
    assert int(rows[0]) in (int(handle_top), int(handle_top + 1))
    h, w = tooth.shape
    key = np.zeros((h, w + 2), dtype=bool)
    key[int(handle_top) : int(handle_top) + 2, 0:2] = True
    key[:, 2:] = tooth
    return key


def _render_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    arr = np.full(mask.shape, int(bg), dtype=int)
    arr[mask] = int(color)
    return arr


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in np.argwhere(mask):
        grid[int(top) + int(r), int(left) + int(c)] = int(color)


def _random_tooth_mask() -> tuple[np.ndarray, int] | None:
    for _ in range(300):
        h = random.randint(3, 5)
        w = random.randint(2, 4)
        handle_top = random.randint(0, h - 2)
        contact_row = int(handle_top + random.randint(0, 1))
        cells = {(contact_row, 0)}
        target_area = random.randint(max(4, w + 1), min(h * w - 1, h * w))
        while len(cells) < target_area or not any(c == w - 1 for _, c in cells):
            frontier: set[tuple[int, int]] = set()
            for r, c in list(cells):
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        if nc == 0 and nr != contact_row:
                            continue
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
            if len(cells) >= max(4, w + 1) and any(c == w - 1 for _, c in cells) and random.random() < 0.35:
                break
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True
        edge_rows = np.flatnonzero(mask[:, 0])
        if edge_rows.size != 1 or int(edge_rows[0]) != int(contact_row):
            continue
        if not bool(mask[:, -1].any()) or bool(mask.all()) or not _is_connected(mask):
            continue
        if len({_shape_token(np.rot90(mask, k)) for k in range(4)}) != 4:
            continue
        key_mask = _build_key_mask_east(mask, handle_top)
        if _count_full_2x2(key_mask) != 1 or _extract_key(key_mask) is None:
            continue
        return mask, int(handle_top)
    return None


def _build_lock_mask(tooth: np.ndarray, side: str) -> np.ndarray | None:
    tooth = np.asarray(tooth, dtype=bool)
    th, tw = tooth.shape
    for _ in range(200):
        if side in {"left", "right"}:
            h_min = max(4, th + 2)
            h_max = min(8, th + 4)
            w_min = max(6, tw + 3)
            w_max = min(8, tw + 5)
            if h_min > h_max or w_min > w_max:
                continue
            h = random.randint(h_min, h_max)
            w = random.randint(w_min, w_max)
            offset_r = random.randint(1, h - th - 1)
            offset_c = 0 if side == "left" else w - tw
        else:
            h_min = max(6, th + 3)
            h_max = min(8, th + 5)
            w_min = max(4, tw + 2)
            w_max = min(8, tw + 4)
            if h_min > h_max or w_min > w_max:
                continue
            h = random.randint(h_min, h_max)
            w = random.randint(w_min, w_max)
            offset_r = 0 if side == "top" else h - th
            offset_c = random.randint(1, w - tw - 1)
        hole = np.zeros((h, w), dtype=bool)
        patch = hole[offset_r : offset_r + th, offset_c : offset_c + tw]
        patch[tooth] = True
        hole[offset_r : offset_r + th, offset_c : offset_c + tw] = patch
        lock_mask = ~hole
        lock_info = _extract_lock(lock_mask)
        if lock_info is None or lock_info["side"] != side:
            continue
        if not np.array_equal(_crop_true(lock_info["hole_mask"]), tooth):
            continue
        return lock_mask
    return None


def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        int(min(a[0], b[0])),
        int(min(a[1], b[1])),
        int(max(a[2], b[2])),
        int(max(a[3], b[3])),
    )


def _pad_bbox(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return int(box[0] - pad), int(box[1] - pad), int(box[2] + pad), int(box[3] + pad)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _make_pair(existing_shapes: set[tuple[tuple[int, int], bytes]]) -> dict | None:
    for _ in range(300):
        result = _random_tooth_mask()
        if result is None:
            continue
        tooth, handle_top = result
        canon = _canonical_rot_token(tooth)
        if canon in existing_shapes:
            continue
        side = random.choice(SIDES)
        lock_mask = _build_lock_mask(tooth, side)
        if lock_mask is None:
            continue
        key_east = _build_key_mask_east(tooth, handle_top)
        input_rot = random.randrange(4)
        input_key_mask = np.rot90(key_east, input_rot)
        if _extract_key(input_key_mask) is None:
            continue
        final_key_mask = np.rot90(key_east, ROT_FOR_SIDE[side])
        lock_info = _extract_lock(lock_mask)
        final_key = _extract_key(final_key_mask)
        if lock_info is None or final_key is None:
            continue
        hole_bbox = lock_info["hole_bbox"]
        teeth_bbox = final_key["teeth_bbox"]
        assert hole_bbox is not None and teeth_bbox is not None
        final_offset = (int(hole_bbox[0] - teeth_bbox[0]), int(hole_bbox[1] - teeth_bbox[1]))
        placed_teeth = {(final_offset[0] + int(r), final_offset[1] + int(c)) for r, c in np.argwhere(final_key["teeth_mask"])}
        if placed_teeth != _coords_from_mask(lock_info["hole_mask"]):
            continue
        placed_handle = {(final_offset[0] + int(r), final_offset[1] + int(c)) for r, c in np.argwhere(final_key["handle_mask"])}
        if not _handle_outside_lock(
            placed_handle,
            (0, 0, lock_mask.shape[0] - 1, lock_mask.shape[1] - 1),
            side,
            [],
        ):
            continue
        lock_box = (0, 0, lock_mask.shape[0] - 1, lock_mask.shape[1] - 1)
        final_key_box = (
            int(final_offset[0]),
            int(final_offset[1]),
            int(final_offset[0] + final_key_mask.shape[0] - 1),
            int(final_offset[1] + final_key_mask.shape[1] - 1),
        )
        reserve_box = _pad_bbox(_union_bbox(lock_box, final_key_box), 1)
        return {
            "tooth": tooth,
            "handle_top": int(handle_top),
            "shape_token": canon,
            "side": side,
            "lock_mask": lock_mask,
            "input_key_mask": input_key_mask,
            "final_key_mask": final_key_mask,
            "final_key_offset": final_offset,
            "reserve_box": reserve_box,
        }
    return None


def _place_relative_box(
    rel_box: tuple[int, int, int, int],
    grid_h: int,
    grid_w: int,
    reserved: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    r_min = int(-rel_box[0])
    c_min = int(-rel_box[1])
    r_max = int(grid_h - rel_box[2] - 1)
    c_max = int(grid_w - rel_box[3] - 1)
    if r_min > r_max or c_min > c_max:
        return None
    choices = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
    random.shuffle(choices)
    for top, left in choices:
        placed = (top + rel_box[0], left + rel_box[1], top + rel_box[2], left + rel_box[3])
        if any(_boxes_intersect(placed, other) for other in reserved):
            continue
        return int(top), int(left)
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((1, 2, 3))
    grid_h = random.randint(max(18, 12 + 4 * pair_count), 30)
    grid_w = random.randint(max(18, 12 + 4 * pair_count), 30)
    pairs: list[dict] = []
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(pair_count):
        pair = _make_pair(used_shapes)
        if pair is None:
            return None
        used_shapes.add(pair["shape_token"])
        pairs.append(pair)

    colors = random.sample(COLOR_POOL, 2 * pair_count)
    for i, pair in enumerate(pairs):
        pair["lock_color"] = int(colors[2 * i])
        pair["key_color"] = int(colors[2 * i + 1])

    reserved: list[tuple[int, int, int, int]] = []
    for pair in sorted(pairs, key=lambda p: (p["reserve_box"][2] - p["reserve_box"][0]) * (p["reserve_box"][3] - p["reserve_box"][1]), reverse=True):
        pos = _place_relative_box(pair["reserve_box"], grid_h, grid_w, reserved)
        if pos is None:
            return None
        pair["lock_top"], pair["lock_left"] = map(int, pos)
        reserve_global = (
            int(pos[0] + pair["reserve_box"][0]),
            int(pos[1] + pair["reserve_box"][1]),
            int(pos[0] + pair["reserve_box"][2]),
            int(pos[1] + pair["reserve_box"][3]),
        )
        reserved.append(reserve_global)

    for pair in pairs:
        key_h, key_w = pair["input_key_mask"].shape
        key_rel = (-1, -1, int(key_h), int(key_w))
        pos = _place_relative_box(key_rel, grid_h, grid_w, reserved)
        if pos is None:
            return None
        pair["key_top"], pair["key_left"] = map(int, pos)
        reserved.append((int(pos[0] - 1), int(pos[1] - 1), int(pos[0] + key_h), int(pos[1] + key_w)))

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for pair in pairs:
        _paint_mask(grid, pair["lock_mask"], pair["lock_top"], pair["lock_left"], pair["lock_color"])
        _paint_mask(grid, pair["input_key_mask"], pair["key_top"], pair["key_left"], pair["key_color"])

    if int((grid == BG_COLOR).sum()) * 100 < 55 * grid.size:
        return None

    expected = grid.copy()
    for pair in pairs:
        for r, c in np.argwhere(pair["input_key_mask"]):
            expected[int(pair["key_top"]) + int(r), int(pair["key_left"]) + int(c)] = BG_COLOR
    for pair in pairs:
        final_top = int(pair["lock_top"] + pair["final_key_offset"][0])
        final_left = int(pair["lock_left"] + pair["final_key_offset"][1])
        _paint_mask(expected, pair["final_key_mask"], final_top, final_left, pair["key_color"])

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
