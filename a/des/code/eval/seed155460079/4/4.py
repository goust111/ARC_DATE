import random
from collections import deque

import numpy as np


TOOTH_COLOR = 5
GEN_ATTEMPTS = 400
LOCK_SIDES = ("top", "bottom", "left", "right")


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [p[0] for p in coords]
            cs = [p[1] for p in coords]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "area": len(coords),
                }
            )
    return comps


def _lock_info(grid: np.ndarray, bg: int, comp: dict):
    r0, c0, r1, c1 = comp["bbox"]
    h, w = comp["h"], comp["w"]
    color = comp["color"]
    if h < 4 or w < 4:
        return None
    for r, c in comp["coords"]:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if (
        int(patch[0, 0]) != color
        or int(patch[0, -1]) != color
        or int(patch[-1, 0]) != color
        or int(patch[-1, -1]) != color
    ):
        return None

    sides = [
        ("top", patch[0, 1:-1]),
        ("bottom", patch[-1, 1:-1]),
        ("left", patch[1:-1, 0]),
        ("right", patch[1:-1, -1]),
    ]
    gaps = []
    for side, arr in sides:
        vals = np.asarray(arr, dtype=int)
        if np.any((vals != bg) & (vals != color)):
            return None
        gap_idx = np.flatnonzero(vals == bg)
        if len(gap_idx) == 0:
            if np.any(vals != color):
                return None
            continue
        if gap_idx[-1] - gap_idx[0] + 1 != len(gap_idx):
            return None
        gaps.append((side, len(gap_idx)))
    if len(gaps) != 1:
        return None

    gap_side, gap_len = gaps[0]
    if comp["area"] != 2 * (h + w) - 4 - gap_len:
        return None
    return {
        "color": color,
        "bbox": comp["bbox"],
        "gap_side": gap_side,
        "gap_len": gap_len,
    }


def _is_filled_rect(comp: dict) -> bool:
    return comp["h"] >= 2 and comp["w"] >= 2 and comp["area"] == comp["h"] * comp["w"]


def _is_tooth(comp: dict) -> bool:
    long_dim = max(comp["h"], comp["w"])
    short_dim = min(comp["h"], comp["w"])
    return short_dim == 1 and long_dim >= 2 and comp["area"] == long_dim


def _extract_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _find_components(g, bg)
    h, w = g.shape
    comp_id = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(comps):
        for r, c in comp["coords"]:
            comp_id[r, c] = idx

    locks = []
    lock_ids = set()
    for idx, comp in enumerate(comps):
        info = _lock_info(g, bg, comp)
        if info is not None:
            info["id"] = idx
            locks.append(info)
            lock_ids.add(idx)

    used_handles = set()
    key_map: dict[int, int] = {}
    for idx, comp in enumerate(comps):
        if idx in lock_ids or not _is_tooth(comp):
            continue
        adj = set()
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    nid = int(comp_id[nr, nc])
                    if nid >= 0 and nid != idx:
                        adj.add(nid)
        adj -= lock_ids
        if len(adj) != 1:
            continue
        handle_id = next(iter(adj))
        handle = comps[handle_id]
        if not _is_filled_rect(handle) or handle_id in used_handles:
            continue
        length = max(comp["h"], comp["w"])
        if length in key_map and key_map[length] != handle["color"]:
            raise AssertionError("ambiguous key lengths")
        used_handles.add(handle_id)
        key_map[length] = handle["color"]

    return bg, locks, key_map


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, locks, key_map = _extract_structure(g)
    assert locks, "expected at least one lock"
    assert len(locks) == len(key_map), "each lock should have one unique key"

    out = g.copy()
    for lock in locks:
        fill = key_map[lock["gap_len"]]
        r0, c0, r1, c1 = lock["bbox"]
        inside = out[r0 + 1 : r1, c0 + 1 : c1]
        inside[inside == bg] = fill
    return out


def _build_lock_patch(color: int, bg: int, gap_len: int, side: str, rng: random.Random) -> np.ndarray:
    if side in ("top", "bottom"):
        w = rng.randint(gap_len + 2, gap_len + 5)
        h = rng.randint(5, 9)
    else:
        h = rng.randint(gap_len + 2, gap_len + 5)
        w = rng.randint(5, 9)
    patch = np.full((h, w), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        start = rng.randint(1, w - 1 - gap_len)
        patch[0, start : start + gap_len] = bg
    elif side == "bottom":
        start = rng.randint(1, w - 1 - gap_len)
        patch[-1, start : start + gap_len] = bg
    elif side == "left":
        start = rng.randint(1, h - 1 - gap_len)
        patch[start : start + gap_len, 0] = bg
    else:
        start = rng.randint(1, h - 1 - gap_len)
        patch[start : start + gap_len, -1] = bg
    return patch


def _build_key_patch(handle_color: int, bg: int, tooth_len: int, rng: random.Random) -> np.ndarray:
    horizontal = rng.random() < 0.5
    if horizontal:
        handle_w = rng.randint(2, min(3, tooth_len))
        handle_h = rng.randint(3, 6)
        offset = rng.randint(0, tooth_len - handle_w)
        patch = np.full((handle_h + 1, tooth_len), bg, dtype=int)
        if rng.random() < 0.5:
            patch[handle_h, :] = TOOTH_COLOR
            patch[:handle_h, offset : offset + handle_w] = handle_color
        else:
            patch[0, :] = TOOTH_COLOR
            patch[1:, offset : offset + handle_w] = handle_color
        return patch

    handle_h = rng.randint(2, min(3, tooth_len))
    handle_w = rng.randint(3, 6)
    offset = rng.randint(0, tooth_len - handle_h)
    patch = np.full((tooth_len, handle_w + 1), bg, dtype=int)
    if rng.random() < 0.5:
        patch[:, handle_w] = TOOTH_COLOR
        patch[offset : offset + handle_h, :handle_w] = handle_color
    else:
        patch[:, 0] = TOOTH_COLOR
        patch[offset : offset + handle_h, 1:] = handle_color
    return patch


def _can_place(used: np.ndarray, r0: int, c0: int, shape: tuple[int, int], pad: int) -> bool:
    h, w = used.shape
    ph, pw = shape
    r1 = r0 + ph
    c1 = c0 + pw
    if r1 > h or c1 > w:
        return False
    rr0 = max(0, r0 - pad)
    rr1 = min(h, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w, c1 + pad)
    return not used[rr0:rr1, cc0:cc1].any()


def _place_patch(
    grid: np.ndarray, used: np.ndarray, patch: np.ndarray, bg: int, rng: random.Random, pad: int
):
    gh, gw = grid.shape
    ph, pw = patch.shape
    if gh - ph < 2 or gw - pw < 2:
        return None
    for _ in range(120):
        r0 = rng.randint(1, gh - ph - 1)
        c0 = rng.randint(1, gw - pw - 1)
        if not _can_place(used, r0, c0, patch.shape, pad):
            continue
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        sub[patch != bg] = patch[patch != bg]
        used[r0 : r0 + ph, c0 : c0 + pw] = True
        return r0, c0
    return None


def _expected_output(grid: np.ndarray, fills: list[tuple[tuple[int, int, int, int], int]]) -> np.ndarray:
    out = grid.copy()
    bg = _bg_color(grid)
    for (r0, c0, r1, c1), color in fills:
        inside = out[r0 + 1 : r1, c0 + 1 : c1]
        inside[inside == bg] = color
    return out


def _gen_attempt(rng: random.Random):
    bg = rng.choice([0, 1])
    n = rng.randint(2, 4)
    lengths = sorted(rng.sample(range(2, 7), n))
    palette = [c for c in range(10) if c not in (bg, TOOTH_COLOR)]
    rng.shuffle(palette)
    frame_colors = palette[:n]
    handle_colors = palette[n : 2 * n]

    locks = []
    keys = []
    for gap_len, frame_color, handle_color in zip(lengths, frame_colors, handle_colors):
        side = rng.choice(LOCK_SIDES)
        locks.append((_build_lock_patch(frame_color, bg, gap_len, side, rng), gap_len, handle_color))
        keys.append((_build_key_patch(handle_color, bg, gap_len, rng), gap_len, handle_color))

    objects = [("lock", *item) for item in locks] + [("key", *item) for item in keys]
    rng.shuffle(objects)

    grid_h = rng.randint(22 + 2 * (n - 2), 30)
    grid_w = rng.randint(22 + 2 * (n - 2), 30)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    used = np.zeros((grid_h, grid_w), dtype=bool)
    length_to_handle = {gap_len: handle_color for gap_len, handle_color in zip(lengths, handle_colors)}
    lock_boxes = []

    for kind, patch, gap_len, handle_color in sorted(objects, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True):
        placed = _place_patch(grid, used, patch, bg, rng, pad=1)
        if placed is None:
            return None
        if kind == "lock":
            r0, c0 = placed
            lock_boxes.append(((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1), gap_len))

    if _bg_color(grid) != bg:
        return None

    _, parsed_locks, key_map = _extract_structure(grid)
    if len(parsed_locks) != n or len(key_map) != n or sorted(key_map) != lengths:
        return None

    fills = [(bbox, length_to_handle[gap_len]) for bbox, gap_len in lock_boxes]
    expected = _expected_output(grid, fills)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
