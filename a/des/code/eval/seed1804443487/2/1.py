from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")
INWARD_DIR = {"top": "down", "bottom": "up", "left": "right", "right": "left"}
OPPOSITE_DIR = {"up": "down", "down": "up", "left": "right", "right": "left"}
ROTATE_DIR_CCW = {"up": "left", "left": "down", "down": "right", "right": "up"}
COLOR_POOL = tuple(range(1, 10))

MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200
GRID_MIN = 18
GRID_MAX = 30
PAIR_COUNT_RANGE = (1, 4)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_until(mask: np.ndarray, current_dir: str, target_dir: str) -> np.ndarray:
    out = np.array(mask, copy=True)
    direction = current_dir
    for _ in range(4):
        if direction == target_dir:
            return out
        out = np.rot90(out, 1)
        direction = ROTATE_DIR_CCW[direction]
    raise AssertionError("failed to align direction with target")


def _blit(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    assert not bool(np.any((target != BG) & mask)), "patch placement must not overlap"
    target[mask] = patch[mask]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _detect_lock(g: np.ndarray, color: int, cells: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    cell_set = set(cells)
    boundary = set()
    for c in range(c0, c1 + 1):
        boundary.add((r0, c))
        boundary.add((r1, c))
    for r in range(r0, r1 + 1):
        boundary.add((r, c0))
        boundary.add((r, c1))
    if not cell_set <= boundary:
        return None

    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != BG):
        return None

    side_segments = {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }

    gap_side = None
    gap_cells: list[tuple[int, int]] | None = None
    for side, seq in side_segments.items():
        missing = [i for i, (r, c) in enumerate(seq) if int(g[r, c]) != int(color)]
        if not missing:
            if any(int(g[r, c]) != int(color) for r, c in seq):
                return None
            continue
        if gap_side is not None:
            return None
        if len(missing) < 2:
            return None
        if missing != list(range(missing[0], missing[-1] + 1)):
            return None
        if any(int(g[r, c]) != BG for i, (r, c) in enumerate(seq) if i in missing):
            return None
        if any(int(g[r, c]) != int(color) for i, (r, c) in enumerate(seq) if i not in missing):
            return None
        gap_side = side
        gap_cells = [seq[i] for i in missing]

    if gap_side is None or gap_cells is None:
        return None

    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(g[r, c]) != int(color) for r, c in corners):
        return None

    expected = boundary - set(gap_cells)
    if cell_set != expected:
        return None

    if gap_side in {"top", "bottom"}:
        gap_span = (min(c for _, c in gap_cells), max(c for _, c in gap_cells))
    else:
        gap_span = (min(r for r, _ in gap_cells), max(r for r, _ in gap_cells))

    return {
        "color": int(color),
        "cells": cell_set,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": gap_side,
        "gap_span": gap_span,
        "gap_width": int(len(gap_cells)),
    }


def _find_locks(g: np.ndarray) -> tuple[list[dict], np.ndarray, dict[int, dict]]:
    locks: list[dict] = []
    lock_mask = np.zeros(g.shape, dtype=bool)
    by_color: dict[int, dict] = {}
    for color in sorted(set(map(int, np.unique(g))) - {BG}):
        for comp in _components(g == int(color)):
            parsed = _detect_lock(g, int(color), comp)
            if parsed is None:
                continue
            assert int(color) not in by_color, "lock colors must be unique"
            by_color[int(color)] = parsed
            locks.append(parsed)
            for r, c in parsed["cells"]:
                lock_mask[r, c] = True
    assert locks, "expected at least one lock"
    return locks, lock_mask, by_color


def _parse_key(g: np.ndarray, cells: list[tuple[int, int]], lock_by_color: dict[int, dict]) -> dict:
    cell_set = set(cells)
    marker_cells = [pos for pos in cell_set if int(g[pos]) in lock_by_color]
    assert len(marker_cells) == 1, "each key must contain exactly one lock-colored marker pixel"
    marker = marker_cells[0]
    marker_color = int(g[marker])
    body_cells = cell_set - {marker}
    assert body_cells, "key body must be non-empty"
    assert any((marker[0] + dr, marker[1] + dc) in body_cells for dr, dc in DIRS4), "marker must touch the key body"

    r0, c0, r1, c1 = _bbox(body_cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    body = np.zeros((h, w), dtype=bool)
    for r, c in body_cells:
        body[r - r0, c - c0] = True

    if w > h:
        full_slice = np.ones(h, dtype=bool)
        if marker[1] == c0 - 1:
            prefix = 0
            for idx in range(w):
                if np.array_equal(body[:, idx], full_slice):
                    prefix += 1
                else:
                    break
            assert 1 <= prefix < w, "horizontal key must have a non-empty handle and tooth region"
            tooth = body[:, prefix:].copy()
            current_dir = "right"
        elif marker[1] == c1 + 1:
            prefix = 0
            for idx in range(w - 1, -1, -1):
                if np.array_equal(body[:, idx], full_slice):
                    prefix += 1
                else:
                    break
            assert 1 <= prefix < w, "horizontal key must have a non-empty handle and tooth region"
            tooth = body[:, : w - prefix].copy()
            current_dir = "left"
        else:
            raise AssertionError("horizontal key marker must sit one step beyond a handle end")
    elif h > w:
        full_slice = np.ones(w, dtype=bool)
        if marker[0] == r0 - 1:
            prefix = 0
            for idx in range(h):
                if np.array_equal(body[idx, :], full_slice):
                    prefix += 1
                else:
                    break
            assert 1 <= prefix < h, "vertical key must have a non-empty handle and tooth region"
            tooth = body[prefix:, :].copy()
            current_dir = "down"
        elif marker[0] == r1 + 1:
            prefix = 0
            for idx in range(h - 1, -1, -1):
                if np.array_equal(body[idx, :], full_slice):
                    prefix += 1
                else:
                    break
            assert 1 <= prefix < h, "vertical key must have a non-empty handle and tooth region"
            tooth = body[: h - prefix, :].copy()
            current_dir = "up"
        else:
            raise AssertionError("vertical key marker must sit one step beyond a handle end")
    else:
        raise AssertionError("key body must be elongated along one axis")

    assert bool(tooth.any()), "tooth region must contain foreground pixels"
    return {
        "cells": cell_set,
        "lock_color": marker_color,
        "tooth": tooth,
        "current_dir": current_dir,
    }


def _paint_tooth(out: np.ndarray, lock: dict, tooth: np.ndarray) -> None:
    r0, c0, r1, c1 = lock["bbox"]
    gap0, gap1 = lock["gap_span"]
    color = int(lock["color"])
    side = lock["side"]

    if side in {"top", "bottom"}:
        assert tooth.shape[1] == int(lock["gap_width"]), "tooth width must match the lock gap width"
        depth = tooth.shape[0]
        interior_depth = r1 - r0 - 1
        assert depth <= interior_depth, "tooth must fit inside the lock interior"
        sr = r0 + 1 if side == "top" else r1 - depth
        sc = gap0
        target = out[sr : sr + depth, sc : gap1 + 1]
    else:
        assert tooth.shape[0] == int(lock["gap_width"]), "tooth width must match the lock gap width"
        depth = tooth.shape[1]
        interior_depth = c1 - c0 - 1
        assert depth <= interior_depth, "tooth must fit inside the lock interior"
        sr = gap0
        sc = c0 + 1 if side == "left" else c1 - depth
        target = out[sr : gap1 + 1, sc : sc + depth]

    assert target.shape == tooth.shape
    assert np.all(target[tooth] == BG), "lock interior should be empty before imprinting"
    target[tooth] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each key to the lock with the same marker color, strip off the straight handle,
    rotate the tooth pattern to the lock gap, and stamp that pattern into the lock interior
    using the lock's border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    locks, lock_mask, lock_by_color = _find_locks(g)
    key_mask = (g != BG) & (~lock_mask)
    key_components = _components(key_mask)
    assert len(key_components) == len(locks), "expected one key per lock"

    keys_by_color: dict[int, dict] = {}
    for comp in key_components:
        key = _parse_key(g, comp, lock_by_color)
        assert key["lock_color"] not in keys_by_color, "lock markers must identify keys unambiguously"
        keys_by_color[key["lock_color"]] = key

    assert set(keys_by_color) == set(lock_by_color), "every lock must have exactly one matching key"

    out = np.zeros_like(g)
    for lock in locks:
        for r, c in lock["cells"]:
            out[r, c] = int(lock["color"])

    for color, lock in lock_by_color.items():
        key = keys_by_color[color]
        rotated = _rotate_until(key["tooth"], key["current_dir"], INWARD_DIR[lock["side"]])
        _paint_tooth(out, lock, rotated)

    return out


def _build_lock_patch(color: int, side: str, gap_width: int) -> tuple[np.ndarray, dict]:
    if side in {"top", "bottom"}:
        interior_h = random.randint(3, 7)
        interior_w = random.randint(max(4, gap_width), 8)
        tooth_depth = random.randint(2, min(4, interior_h))
        gap_offset = random.randint(0, interior_w - gap_width)
        gap_span = (1 + gap_offset, 1 + gap_offset + gap_width - 1)
    else:
        interior_h = random.randint(max(4, gap_width), 8)
        interior_w = random.randint(3, 7)
        tooth_depth = random.randint(2, min(4, interior_w))
        gap_offset = random.randint(0, interior_h - gap_width)
        gap_span = (1 + gap_offset, 1 + gap_offset + gap_width - 1)

    patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    if side == "top":
        patch[0, gap_span[0] : gap_span[1] + 1] = BG
    elif side == "bottom":
        patch[-1, gap_span[0] : gap_span[1] + 1] = BG
    elif side == "left":
        patch[gap_span[0] : gap_span[1] + 1, 0] = BG
    else:
        patch[gap_span[0] : gap_span[1] + 1, -1] = BG

    meta = {
        "color": int(color),
        "side": side,
        "gap_width": int(gap_width),
        "gap_span": gap_span,
        "tooth_depth": int(tooth_depth),
    }
    return patch, meta


def _build_tooth_mask(width: int, depth: int) -> np.ndarray | None:
    for _ in range(80):
        first_interval = []
        for top in range(width):
            for bottom in range(top, width - 1):
                if top == 0 and bottom == width - 1:
                    continue
                first_interval.append((top, bottom))
        if not first_interval:
            return None

        intervals = [random.choice(first_interval)]
        for _col in range(1, depth):
            prev_top, prev_bottom = intervals[-1]
            candidates = []
            for dtop in (-1, 0, 1):
                for dbottom in (-1, 0, 1):
                    top = min(max(prev_top + dtop, 0), width - 1)
                    bottom = min(max(prev_bottom + dbottom, top), width - 1)
                    if max(top, prev_top) <= min(bottom, prev_bottom):
                        candidates.append((top, bottom))
            if not candidates:
                break
            intervals.append(random.choice(candidates))

        if len(intervals) != depth:
            continue
        if len(set(intervals)) == 1:
            continue

        tooth = np.zeros((width, depth), dtype=bool)
        for col, (top, bottom) in enumerate(intervals):
            tooth[top : bottom + 1, col] = True
        return tooth
    return None


def _build_key_patch(lock_color: int, body_color: int, width: int, tooth: np.ndarray, marker_side: str) -> np.ndarray:
    depth = tooth.shape[1]
    min_handle = max(2, width + 1 - depth)
    max_handle = max(min_handle, 5)
    handle_len = random.randint(min_handle, max_handle)
    marker_row = random.randint(0, width - 1)

    patch = np.zeros((width, 1 + handle_len + depth), dtype=int)
    patch[:, 1 : 1 + handle_len] = body_color
    tooth_region = patch[:, 1 + handle_len :]
    tooth_region[tooth] = body_color
    patch[marker_row, 0] = lock_color

    current_dir = "right"
    target_dir = OPPOSITE_DIR[INWARD_DIR[marker_side]]
    return _rotate_until(patch, current_dir, target_dir)


def _sample_grid_shape(total_bbox_area: int, max_obj_h: int, max_obj_w: int) -> tuple[int, int] | None:
    choices = []
    min_h = max(GRID_MIN, max_obj_h + 2)
    min_w = max(GRID_MIN, max_obj_w + 2)
    for h in range(min_h, GRID_MAX + 1):
        for w in range(min_w, GRID_MAX + 1):
            if h * w >= int(total_bbox_area * 1.7):
                choices.append((h, w))
    if not choices:
        return None
    return random.choice(choices)


def _place_patch(shape: tuple[int, int], patch_shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> tuple[int, int] | None:
    h, w = shape
    ph, pw = patch_shape
    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if any(_boxes_overlap(box, prev, pad=1) for prev in boxes):
            continue
        boxes.append(box)
        return r0, c0
    return None


def _render_expected(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for pair in pairs:
        _blit(out, pair["lock_patch"], pair["lock_pos"][0], pair["lock_pos"][1])

    for pair in pairs:
        side = pair["lock_meta"]["side"]
        rotated = _rotate_until(pair["tooth"], "right", INWARD_DIR[side])
        lock_r, lock_c = pair["lock_pos"]
        lock_patch = pair["lock_patch"]
        r0 = lock_r
        c0 = lock_c
        r1 = lock_r + lock_patch.shape[0] - 1
        c1 = lock_c + lock_patch.shape[1] - 1
        gap0, gap1 = pair["lock_meta"]["gap_span"]

        lock = {
            "color": int(pair["lock_meta"]["color"]),
            "bbox": (r0, c0, r1, c1),
            "side": side,
            "gap_span": (c0 + gap0, c0 + gap1) if side in {"top", "bottom"} else (r0 + gap0, r0 + gap1),
            "gap_width": int(pair["lock_meta"]["gap_width"]),
        }
        _paint_tooth(out, lock, rotated)

    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    if 2 * pair_count > len(COLOR_POOL):
        return None

    colors = random.sample(list(COLOR_POOL), 2 * pair_count)
    lock_colors = colors[:pair_count]
    body_colors = colors[pair_count:]

    pairs: list[dict] = []
    total_bbox_area = 0
    max_obj_h = 0
    max_obj_w = 0

    for idx in range(pair_count):
        side = random.choice(SIDES)
        gap_width = random.randint(2, 4)
        lock_patch, lock_meta = _build_lock_patch(lock_colors[idx], side, gap_width)
        tooth = _build_tooth_mask(gap_width, lock_meta["tooth_depth"])
        if tooth is None:
            return None
        marker_side = random.choice(SIDES)
        key_patch = _build_key_patch(lock_colors[idx], body_colors[idx], gap_width, tooth, marker_side)

        pairs.append(
            {
                "lock_patch": lock_patch,
                "lock_meta": lock_meta,
                "key_patch": key_patch,
                "tooth": tooth,
            }
        )
        total_bbox_area += int(np.prod(lock_patch.shape) + np.prod(key_patch.shape))
        max_obj_h = max(max_obj_h, lock_patch.shape[0], key_patch.shape[0])
        max_obj_w = max(max_obj_w, lock_patch.shape[1], key_patch.shape[1])

    shape = _sample_grid_shape(total_bbox_area, max_obj_h, max_obj_w)
    if shape is None:
        return None

    boxes: list[tuple[int, int, int, int]] = []
    for pair in pairs:
        lock_pos = _place_patch(shape, pair["lock_patch"].shape, boxes)
        if lock_pos is None:
            return None
        pair["lock_pos"] = lock_pos

    key_order = list(range(pair_count))
    random.shuffle(key_order)
    for idx in key_order:
        key_pos = _place_patch(shape, pairs[idx]["key_patch"].shape, boxes)
        if key_pos is None:
            return None
        pairs[idx]["key_pos"] = key_pos

    inp = np.zeros(shape, dtype=int)
    for pair in pairs:
        _blit(inp, pair["lock_patch"], pair["lock_pos"][0], pair["lock_pos"][1])
        _blit(inp, pair["key_patch"], pair["key_pos"][0], pair["key_pos"][1])

    expected = _render_expected(shape, pairs)
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
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
