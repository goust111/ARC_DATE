from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
LOCK_COLOR = 5
KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDE_TO_ROT = {"left": 0, "up": 3, "right": 2, "down": 1}
DIM_CHOICES = [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2)]


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    max_count = int(counts.max())
    if int(counts[0]) == max_count:
        return 0
    return int(np.argmax(counts))


def _is_contiguous(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return bool(np.all(values[1:] == values[:-1] + 1))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1) + 1, int(c1) + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == int(bg):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows) + 1, max(cols) + 1),
                }
            )
    return comps


def _slice_patch(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0:r1, c0:c1], dtype=int).copy()


def _analyze_lock(patch: np.ndarray, color: int, bg: int) -> dict | None:
    mask = np.asarray(patch == int(color), dtype=bool)
    h, w = mask.shape
    if h < 3 or w < 3:
        return None

    side_segments: list[tuple[str, np.ndarray]] = []
    top_bg = np.where(~mask[0])[0]
    bot_bg = np.where(~mask[-1])[0]
    left_bg = np.where(~mask[:, 0])[0]
    right_bg = np.where(~mask[:, -1])[0]
    if top_bg.size:
        side_segments.append(("up", top_bg))
    if bot_bg.size:
        side_segments.append(("down", bot_bg))
    if left_bg.size:
        side_segments.append(("left", left_bg))
    if right_bg.size:
        side_segments.append(("right", right_bg))
    if len(side_segments) != 1:
        return None

    side, segment = side_segments[0]
    if not _is_contiguous(segment):
        return None

    expected = np.ones((h, w), dtype=bool)
    if side in ("up", "down"):
        start = int(segment[0])
        width = int(segment.size)
        if start == 0 or start + width == w:
            return None
        if side == "up":
            depth = 0
            while depth < h and bool(np.all(~mask[depth, start : start + width])):
                depth += 1
            if depth <= 0 or depth >= h:
                return None
            expected[:depth, start : start + width] = False
            notch_bbox = (0, start, depth, start + width)
        else:
            depth = 0
            while depth < h and bool(np.all(~mask[h - 1 - depth, start : start + width])):
                depth += 1
            if depth <= 0 or depth >= h:
                return None
            expected[h - depth :, start : start + width] = False
            notch_bbox = (h - depth, start, h, start + width)
    else:
        start = int(segment[0])
        width = int(segment.size)
        if start == 0 or start + width == h:
            return None
        if side == "left":
            depth = 0
            while depth < w and bool(np.all(~mask[start : start + width, depth])):
                depth += 1
            if depth <= 0 or depth >= w:
                return None
            expected[start : start + width, :depth] = False
            notch_bbox = (start, 0, start + width, depth)
        else:
            depth = 0
            while depth < w and bool(np.all(~mask[start : start + width, w - 1 - depth])):
                depth += 1
            if depth <= 0 or depth >= w:
                return None
            expected[start : start + width, w - depth :] = False
            notch_bbox = (start, w - depth, start + width, w)

    if not bool(np.array_equal(mask, expected)):
        return None

    return {
        "kind": "lock",
        "color": int(color),
        "patch": np.asarray(patch, dtype=int).copy(),
        "side": side,
        "notch_width": int(width),
        "notch_depth": int(depth),
        "notch_bbox": notch_bbox,
    }


def _analyze_key(patch: np.ndarray, color: int, bg: int) -> dict | None:
    mask = np.asarray(patch == int(color), dtype=bool)
    h, w = mask.shape
    if h < 2 or w < 3:
        return None

    full_cols = np.where(mask.all(axis=0))[0]
    if full_cols.size == 0 or int(full_cols[0]) != 0:
        return None

    body_w = 0
    while body_w < w and bool(mask[:, body_w].all()):
        body_w += 1
    if body_w <= 0 or body_w >= w:
        return None

    protr = mask[:, body_w:]
    ref = protr[:, 0]
    if not bool(np.all(protr == ref[:, None])):
        return None

    rows = np.where(ref)[0]
    if rows.size == 0 or rows.size == h or not _is_contiguous(rows):
        return None

    top = int(rows[0])
    height = int(rows.size)
    expected = np.zeros((h, w), dtype=bool)
    expected[:, :body_w] = True
    expected[top : top + height, body_w:] = True
    if not bool(np.array_equal(mask, expected)):
        return None

    protr_mask = np.zeros((h, w), dtype=bool)
    protr_mask[top : top + height, body_w:] = True
    return {
        "kind": "key",
        "color": int(color),
        "patch": np.asarray(patch, dtype=int).copy(),
        "protrusion_mask": protr_mask,
        "protrusion_height": int(height),
        "protrusion_width": int(w - body_w),
    }


def _with_global_lock_data(lock: dict, bbox: tuple[int, int, int, int]) -> dict:
    r0, c0, _r1, _c1 = bbox
    nr0, nc0, nr1, nc1 = lock["notch_bbox"]
    notch_cells = {
        (int(r0 + rr), int(c0 + cc))
        for rr in range(int(nr0), int(nr1))
        for cc in range(int(nc0), int(nc1))
    }
    out = dict(lock)
    out["bbox"] = tuple(map(int, bbox))
    out["r0"] = int(r0)
    out["c0"] = int(c0)
    out["notch_cells"] = notch_cells
    return out


def _rotate_key(key: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    k = int(SIDE_TO_ROT[side])
    return np.rot90(key["patch"], k), np.rot90(key["protrusion_mask"], k)


def _try_place_key(base: np.ndarray, lock: dict, key: dict, bg: int) -> dict | None:
    patch, protr_mask = _rotate_key(key, lock["side"])
    pb = _bbox_from_mask(protr_mask)
    if pb is None:
        return None
    pr0, pc0, pr1, pc1 = pb
    nr0, nc0, nr1, nc1 = lock["notch_bbox"]
    if (pr1 - pr0, pc1 - pc0) != (nr1 - nr0, nc1 - nc0):
        return None

    top = int(lock["r0"] + nr0 - pr0)
    left = int(lock["c0"] + nc0 - pc0)
    h, w = patch.shape
    H, W = base.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return None

    cells: set[tuple[int, int]] = set()
    protr_cells: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(patch != int(bg)):
        gr = int(top + rr)
        gc = int(left + cc)
        if int(base[gr, gc]) != int(bg):
            return None
        cells.add((gr, gc))
        if bool(protr_mask[int(rr), int(cc)]):
            protr_cells.add((gr, gc))

    if protr_cells != lock["notch_cells"]:
        return None

    return {"patch": patch, "top": int(top), "left": int(left), "cells": cells}


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[int(top) : int(top) + h, int(left) : int(left) + w]
    mask = patch != int(bg)
    sub[mask] = patch[mask]


def _solve_matching(base: np.ndarray, locks: list[dict], keys: list[dict], bg: int) -> list[tuple[int, dict]]:
    candidates: list[list[tuple[int, dict]]] = []
    for lock in locks:
        options: list[tuple[int, dict]] = []
        for ki, key in enumerate(keys):
            placement = _try_place_key(base, lock, key, bg)
            if placement is not None:
                options.append((int(ki), placement))
        assert options, "every lock should have at least one candidate key"
        candidates.append(options)

    order = sorted(range(len(locks)), key=lambda i: len(candidates[i]))
    solutions: list[list[tuple[int, dict]] | None] = []
    current: list[tuple[int, dict] | None] = [None] * len(locks)

    def dfs(idx: int, used_keys: set[int], occupied: set[tuple[int, int]]) -> None:
        if len(solutions) >= 2:
            return
        if idx == len(order):
            solutions.append(list(current))
            return
        li = int(order[idx])
        for ki, placement in candidates[li]:
            if int(ki) in used_keys or bool(occupied & placement["cells"]):
                continue
            current[li] = (int(ki), placement)
            used_keys.add(int(ki))
            dfs(idx + 1, used_keys, occupied | placement["cells"])
            used_keys.remove(int(ki))
            current[li] = None

    dfs(0, set(), set())
    assert len(solutions) == 1, "matching should be unique"
    solved = solutions[0]
    assert solved is not None
    return [item for item in solved if item is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the grey lock whose notch it fits, then rotate and insert the key into that notch.

    Input structure:
    - The background is the dominant color.
    - The grid contains several disjoint monochrome lock objects and several disjoint monochrome key objects.
    - Each lock is a solid rectangle with one centered rectangular notch carved from exactly one side.
    - Each key is a rectangle with one rectangular protrusion on its right side.

    Rule:
    - Parse every lock to recover its notch side and notch size, and parse every key to recover its protrusion size.
    - Pair locks and keys by testing which rotated key protrusion exactly fills each lock notch.
    - Rotate the matched key so its protrusion points into the lock, translate it so the protrusion fills the notch, and erase the key from its original position.
    - Keep the locks, keep the inserted keys, and leave all other cells unchanged.

    Key insight:
    The solution needs two coupled inferences: notch geometry determines which key fits which lock, and notch direction determines how that key must rotate before translation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(_get_bg(g))

    locks: list[dict] = []
    keys: list[dict] = []
    comps = _find_components(g, bg)
    assert comps, "expected at least one object"

    for comp in comps:
        patch = _slice_patch(g, comp["bbox"])
        color = int(comp["color"])
        lock = _analyze_lock(patch, color, bg)
        if lock is not None:
            lock = _with_global_lock_data(lock, comp["bbox"])
            lock["coords"] = list(comp["coords"])
            locks.append(lock)
            continue
        key = _analyze_key(patch, color, bg)
        assert key is not None, "every non-background object must be a lock or key"
        key["bbox"] = tuple(map(int, comp["bbox"]))
        key["coords"] = list(comp["coords"])
        keys.append(key)

    assert locks and keys and len(locks) == len(keys), "expected equal numbers of locks and keys"

    out = np.asarray(g, dtype=int).copy()
    for key in keys:
        for r, c in key["coords"]:
            out[int(r), int(c)] = int(bg)

    placements = _solve_matching(out, locks, keys, bg)
    for _ki, placement in placements:
        _paint_patch(out, placement["patch"], placement["top"], placement["left"], bg)
    return out


def _make_lock_patch(side: str, span: int, depth: int) -> np.ndarray:
    margin = random.randint(1, 2)
    if side in ("left", "right"):
        h = int(span + 2 * margin)
        w = int(depth + random.randint(2, 5))
        patch = np.full((h, w), LOCK_COLOR, dtype=int)
        if side == "left":
            patch[margin : margin + span, :depth] = BG
        else:
            patch[margin : margin + span, w - depth :] = BG
    else:
        w = int(span + 2 * margin)
        h = int(depth + random.randint(2, 5))
        patch = np.full((h, w), LOCK_COLOR, dtype=int)
        if side == "up":
            patch[:depth, margin : margin + span] = BG
        else:
            patch[h - depth :, margin : margin + span] = BG
    return patch


def _make_key_patch(color: int, span: int, depth: int) -> np.ndarray:
    is_t = bool(random.getrandbits(1))
    if is_t:
        top_extra = random.randint(1, 2)
        bottom_extra = random.randint(1, 2)
    else:
        extra = random.randint(1, 2)
        if bool(random.getrandbits(1)):
            top_extra, bottom_extra = 0, extra
        else:
            top_extra, bottom_extra = extra, 0
    body_h = int(span + top_extra + bottom_extra)
    body_w = int(random.randint(2, 5))
    patch = np.full((body_h, body_w + depth), BG, dtype=int)
    patch[:, :body_w] = int(color)
    patch[top_extra : top_extra + span, body_w:] = int(color)
    return patch


def _build_local_assembly(lock_patch: np.ndarray, key_patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    lock = _analyze_lock(lock_patch, LOCK_COLOR, BG)
    key = _analyze_key(key_patch, int(key_patch.max()), BG)
    assert lock is not None and key is not None

    pad = int(max(lock_patch.shape + key_patch.shape) + 3)
    temp_h = int(lock_patch.shape[0] + 2 * pad)
    temp_w = int(lock_patch.shape[1] + 2 * pad)
    base = np.full((temp_h, temp_w), BG, dtype=int)
    base[pad : pad + lock_patch.shape[0], pad : pad + lock_patch.shape[1]] = lock_patch
    lock = _with_global_lock_data(lock, (pad, pad, pad + lock_patch.shape[0], pad + lock_patch.shape[1]))
    placement = _try_place_key(base, lock, key, BG)
    assert placement is not None
    out = base.copy()
    _paint_patch(out, placement["patch"], placement["top"], placement["left"], BG)
    bbox = _bbox_from_mask(out != BG)
    assert bbox is not None
    r0, c0, r1, c1 = bbox
    crop = out[r0:r1, c0:c1].copy()
    return crop, (int(pad - r0), int(pad - c0))


def _can_place_bbox(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(H, int(top + h + pad))
    c1 = min(W, int(left + w + pad))
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_bbox(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    occupied[int(top) : int(top) + h, int(left) : int(left) + w] = True


def _sample_positions(
    canvas_shape: tuple[int, int], dims: tuple[int, int], tries: int = 250
) -> Iterable[tuple[int, int]]:
    H, W = canvas_shape
    h, w = dims
    for _ in range(int(tries)):
        yield random.randint(0, H - h), random.randint(0, W - w)


def _place_boxes(canvas_shape: tuple[int, int], boxes: list[tuple[int, int]], pad: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros(canvas_shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True)
    for idx in order:
        h, w = boxes[idx]
        placed = None
        for top, left in _sample_positions(canvas_shape, (h, w)):
            if _can_place_bbox(occupied, top, left, h, w, pad):
                _mark_bbox(occupied, top, left, h, w)
                placed = (int(top), int(left))
                break
        if placed is None:
            return None
        placements[idx] = placed
    return [p for p in placements if p is not None]


def _make_pair_data(color: int, side: str, span: int, depth: int) -> dict:
    lock_patch = _make_lock_patch(side, span, depth)
    key_patch = _make_key_patch(color, span, depth)
    assembly_patch, lock_offset = _build_local_assembly(lock_patch, key_patch)
    return {
        "lock_patch": lock_patch,
        "key_patch": key_patch,
        "assembly_patch": assembly_patch,
        "lock_offset": lock_offset,
    }


def _generate_attempt() -> np.ndarray | None:
    n_pairs = random.randint(2, 4)
    colors = random.sample(KEY_COLORS, n_pairs)
    sides = random.sample(["left", "up", "right", "down"], n_pairs)
    dims = random.sample(DIM_CHOICES, n_pairs)

    pairs = [
        _make_pair_data(int(color), side, int(span), int(depth))
        for color, side, (span, depth) in zip(colors, sides, dims)
    ]

    assembly_boxes = [tuple(pair["assembly_patch"].shape) for pair in pairs]
    key_boxes = [tuple(pair["key_patch"].shape) for pair in pairs]
    total_area = sum(h * w for h, w in assembly_boxes + key_boxes)
    max_h = max(max(h for h, _ in assembly_boxes), max(h for h, _ in key_boxes))
    max_w = max(max(w for _, w in assembly_boxes), max(w for _, w in key_boxes))
    base = max(int(np.ceil(np.sqrt(total_area))) + 5, max_h + 2, max_w + 2, 16)
    if base > 30:
        return None

    for _ in range(80):
        H = random.randint(base, 30)
        W = random.randint(base, 30)
        assembly_pos = _place_boxes((H, W), assembly_boxes, pad=1)
        if assembly_pos is None:
            continue

        occupied = np.zeros((H, W), dtype=bool)
        for (top, left), (h, w) in zip(assembly_pos, assembly_boxes):
            _mark_bbox(occupied, top, left, h, w)

        key_pos: list[tuple[int, int]] = []
        ok = True
        for h, w in key_boxes:
            placed = None
            for top, left in _sample_positions((H, W), (h, w), tries=300):
                if _can_place_bbox(occupied, top, left, h, w, pad=1):
                    _mark_bbox(occupied, top, left, h, w)
                    placed = (int(top), int(left))
                    break
            if placed is None:
                ok = False
                break
            key_pos.append(placed)
        if not ok:
            continue

        inp = np.full((H, W), BG, dtype=int)
        expected = np.full((H, W), BG, dtype=int)
        for pair, (atop, aleft), (ktop, kleft) in zip(pairs, assembly_pos, key_pos):
            ah, aw = pair["assembly_patch"].shape
            lr, lc = pair["lock_offset"]
            lock_patch = pair["lock_patch"]
            _paint_patch(expected, pair["assembly_patch"], atop, aleft, BG)
            _paint_patch(inp, lock_patch, atop + lr, aleft + lc, BG)
            _paint_patch(inp, pair["key_patch"], ktop, kleft, BG)
            assert 0 <= atop and 0 <= aleft and atop + ah <= H and aleft + aw <= W

        non_bg_cols = [int(col) for col in np.unique(inp) if int(col) != int(BG)]
        if non_bg_cols and int((inp == BG).sum()) <= max(int((inp == col).sum()) for col in non_bg_cols):
            continue
        try:
            produced = main(inp)
        except AssertionError:
            continue
        if not bool(np.array_equal(produced, expected)):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-key puzzle")
