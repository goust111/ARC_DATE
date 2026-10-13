from __future__ import annotations

from collections import deque
import random
import numpy as np


BG = 0
FRAME_COLOR = 5
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_ROTATIONS = {"top": 0, "left": 1, "bottom": 2, "right": 3}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
LOCK_COUNT_MIN = 2
LOCK_COUNT_MAX = 4
MAX_GENERATION_RETRIES = 500


def _shape(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


TOP_SHAPES = [
    _shape(["#", "#"]),
    _shape(["##"]),
    _shape(["###"]),
    _shape(["##", "#."]),
    _shape(["##", ".#"]),
    _shape([".#", "##"]),
    _shape(["##", ".#", ".#"]),
    _shape(["###", ".#."]),
    _shape(["###", "#.."]),
    _shape(["###", "..#"]),
    _shape([".##", "##."]),
    _shape([".##", "..#", "..#"]),
]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_monochrome_components(grid: np.ndarray, exclude: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in exclude or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
                    coords.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _extract_lock(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    coords = comp["coord_set"]
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    if any((r, c) not in expected for r, c in coords):
        return None

    missing = sorted(expected - coords)
    if not (1 <= len(missing) <= 3):
        return None

    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        return None

    if all(r == r0 for r, _ in missing):
        side = "top"
        positions = [c for _, c in missing]
        lo, hi = c0, c1
    elif all(r == r1 for r, _ in missing):
        side = "bottom"
        positions = [c for _, c in missing]
        lo, hi = c0, c1
    elif all(c == c0 for _, c in missing):
        side = "left"
        positions = [r for r, _ in missing]
        lo, hi = r0, r1
    elif all(c == c1 for _, c in missing):
        side = "right"
        positions = [r for r, _ in missing]
        lo, hi = r0, r1
    else:
        return None

    positions = sorted(positions)
    if positions != list(range(positions[0], positions[-1] + 1)):
        return None
    if positions[0] == lo or positions[-1] == hi:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap_cells": tuple(missing),
        "color": int(comp["color"]),
        "coords": comp["coords"],
    }


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _extract_key(comp: dict) -> dict:
    return {
        "color": int(comp["color"]),
        "coords": list(comp["coords"]),
        "bbox": comp["bbox"],
        "mask": _component_mask(comp),
        "area": int(comp["area"]),
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _find_monochrome_components(g, exclude={bg})

    locks: list[dict] = []
    lock_cells: set[tuple[int, int]] = set()
    key_comps: list[dict] = []
    for comp in comps:
        lock = _extract_lock(g, comp, bg)
        if lock is not None:
            locks.append(lock)
            lock_cells.update(comp["coords"])
        else:
            key_comps.append(comp)

    keys = []
    for comp in key_comps:
        if any(cell in lock_cells for cell in comp["coords"]):
            continue
        keys.append(_extract_key(comp))

    locks.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]))
    keys.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["color"]))
    return bg, locks, keys


def _side_signature(mask: np.ndarray, side: str) -> tuple[int, tuple[int, ...]] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        extreme = int(pts[:, 0].min())
        positions = sorted(int(c) for r, c in pts if int(r) == extreme)
    elif side == "bottom":
        extreme = int(pts[:, 0].max())
        positions = sorted(int(c) for r, c in pts if int(r) == extreme)
    elif side == "left":
        extreme = int(pts[:, 1].min())
        positions = sorted(int(r) for r, c in pts if int(c) == extreme)
    else:
        extreme = int(pts[:, 1].max())
        positions = sorted(int(r) for r, c in pts if int(c) == extreme)

    if len(positions) > 3:
        return None
    if positions != list(range(positions[0], positions[-1] + 1)):
        return None
    return extreme, tuple(positions)


def _key_fits_lock(lock: dict, key: dict) -> bool:
    side = lock["side"]
    sig = _side_signature(key["mask"], side)
    if sig is None:
        return False

    extreme, positions = sig
    gap = lock["gap_cells"]
    if side in ("top", "bottom"):
        gap_positions = tuple(sorted(c for _, c in gap))
    else:
        gap_positions = tuple(sorted(r for r, _ in gap))
    if len(positions) != len(gap_positions):
        return False

    kr0, kc0, kr1, kc1 = key["bbox"]
    lr0, lc0, lr1, lc1 = lock["bbox"]
    gap_start = gap_positions[0]
    pos_start = positions[0]

    if side == "top":
        dr = lr0 - kr0
        dc = gap_start - (kc0 + pos_start)
    elif side == "bottom":
        dr = lr1 - (kr0 + extreme)
        dc = gap_start - (kc0 + pos_start)
    elif side == "left":
        dr = gap_start - (kr0 + pos_start)
        dc = lc0 - kc0
    else:
        dr = gap_start - (kr0 + pos_start)
        dc = lc1 - (kc0 + extreme)

    gap_set = set(gap)
    translated = set()
    for r, c in key["coords"]:
        tr, tc = r + dr, c + dc
        if not (lr0 <= tr <= lr1 and lc0 <= tc <= lc1):
            return False
        on_border = tr == lr0 or tr == lr1 or tc == lc0 or tc == lc1
        if on_border and (tr, tc) not in gap_set:
            return False
        translated.add((tr, tc))

    return gap_set.issubset(translated)


def _candidate_map(locks: list[dict], keys: list[dict]) -> list[list[int]]:
    candidates: list[list[int]] = []
    for lock in locks:
        cur = [ki for ki, key in enumerate(keys) if _key_fits_lock(lock, key)]
        candidates.append(cur)
    return candidates


def _best_matchings(candidates: list[list[int]], key_count: int, limit: int = 2) -> tuple[int, list[dict[int, int]]]:
    lock_count = len(candidates)
    best_size = -1
    solutions: list[dict[int, int]] = []

    def recurse(remaining: set[int], used: set[int], chosen: dict[int, int]) -> None:
        nonlocal best_size, solutions
        if len(chosen) + len(remaining) < best_size:
            return
        if not remaining:
            size = len(chosen)
            if size > best_size:
                best_size = size
                solutions = [chosen.copy()]
            elif size == best_size and len(solutions) < limit:
                solutions.append(chosen.copy())
            return

        lock_idx = min(remaining, key=lambda idx: sum(1 for ki in candidates[idx] if ki not in used))
        next_remaining = set(remaining)
        next_remaining.remove(lock_idx)
        options = [ki for ki in candidates[lock_idx] if ki not in used]

        for key_idx in options:
            chosen[lock_idx] = key_idx
            used.add(key_idx)
            recurse(next_remaining, used, chosen)
            used.remove(key_idx)
            del chosen[lock_idx]
            if best_size == lock_count and len(solutions) >= limit:
                return

        recurse(next_remaining, used, chosen)

    recurse(set(range(len(candidates))), set(), {})
    return best_size, solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the unique frame gap it can enter, then use that key color to seal and flood the lock.

    Input structure:
    - The background is black.
    - Several small monochrome 4-connected components act as keys; each key is a solid polyomino with at most 6 cells.
    - Several one-pixel-thick rectangular frame components act as locks; each frame has exactly one contiguous 1-3-cell gap on one side, and its interior is otherwise background.

    Rule:
    - Detect every lock by its nearly complete rectangular border and record the side and cells of its gap.
    - For each lock, find the key that can be translated straight into that side so that the key stays inside the frame and only touches the border exactly on the missing gap cells; unmatched objects stay unchanged.
    - Clear every matched key from its original position.
    - For each matched lock, paint the gap cells with the matched key color and flood the entire frame interior with that same color.

    Key insight:
    - The color does not determine the pairing; the pairing comes from geometric insertion compatibility between a key's exposed side profile and a lock's gap, and that matched color is only applied afterward.
    """
    g = np.asarray(grid, dtype=int)
    bg, locks, keys = _parse_scene(g)
    out = g.copy()
    if not locks or not keys:
        return out

    candidates = _candidate_map(locks, keys)
    best_size, solutions = _best_matchings(candidates, len(keys), limit=1)
    if best_size <= 0 or not solutions:
        return out

    matching = solutions[0]
    matched_keys = {key_idx for key_idx in matching.values()}
    for key_idx in matched_keys:
        for r, c in keys[key_idx]["coords"]:
            out[r, c] = bg

    for lock_idx, key_idx in matching.items():
        color = int(keys[key_idx]["color"])
        r0, c0, r1, c1 = locks[lock_idx]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = color
        for r, c in locks[lock_idx]["gap_cells"]:
            out[r, c] = color

    return out


def _render_key_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _make_lock_spec(rng: random.Random, color: int) -> dict:
    side = rng.choice(SIDES)
    base = rng.choice(TOP_SHAPES)
    key_mask = np.rot90(base, SIDE_TO_ROTATIONS[side])
    kh, kw = key_mask.shape

    if side == "top":
        left = rng.randint(1, 3)
        right = rng.randint(1, 3)
        inward = max(rng.randint(1, 3), 4 - kh)
        frame_h = kh + inward
        frame_w = left + kw + right
        key_offset = (0, left)
        if frame_w < 4:
            right += 4 - frame_w
            frame_w = left + kw + right
    elif side == "bottom":
        left = rng.randint(1, 3)
        right = rng.randint(1, 3)
        inward = max(rng.randint(1, 3), 4 - kh)
        frame_h = inward + kh
        frame_w = left + kw + right
        key_offset = (inward, left)
        if frame_w < 4:
            right += 4 - frame_w
            frame_w = left + kw + right
    elif side == "left":
        top = rng.randint(1, 3)
        bottom = rng.randint(1, 3)
        inward = max(rng.randint(1, 3), 4 - kw)
        frame_h = top + kh + bottom
        frame_w = kw + inward
        key_offset = (top, 0)
        if frame_h < 4:
            bottom += 4 - frame_h
            frame_h = top + kh + bottom
    else:
        top = rng.randint(1, 3)
        bottom = rng.randint(1, 3)
        inward = max(rng.randint(1, 3), 4 - kw)
        frame_h = top + kh + bottom
        frame_w = inward + kw
        key_offset = (top, inward)
        if frame_h < 4:
            bottom += 4 - frame_h
            frame_h = top + kh + bottom

    lock_patch = np.full((frame_h, frame_w), BG, dtype=int)
    lock_patch[0, :] = FRAME_COLOR
    lock_patch[-1, :] = FRAME_COLOR
    lock_patch[:, 0] = FRAME_COLOR
    lock_patch[:, -1] = FRAME_COLOR

    placed = set()
    for r, c in np.argwhere(key_mask):
        rr = key_offset[0] + int(r)
        cc = key_offset[1] + int(c)
        placed.add((rr, cc))

    gap_cells = tuple(
        sorted(
            (r, c)
            for r, c in placed
            if r == 0 or r == frame_h - 1 or c == 0 or c == frame_w - 1
        )
    )
    for r, c in gap_cells:
        lock_patch[r, c] = BG

    return {
        "side": side,
        "key_mask": key_mask,
        "key_patch": _render_key_patch(key_mask, color, BG),
        "key_color": color,
        "lock_patch": lock_patch,
        "gap_cells_rel": gap_cells,
        "frame_shape": (frame_h, frame_w),
        "key_shape": key_mask.shape,
    }


def _can_place_rectangle(occupied: np.ndarray, top: int, left: int, height: int, width: int, gap: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + height + gap)
    c1 = min(w, left + width + gap)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _stamp_lock(grid: np.ndarray, spec: dict, top: int, left: int) -> None:
    patch = spec["lock_patch"]
    ph, pw = patch.shape
    mask = patch != BG
    grid[top : top + ph, left : left + pw][mask] = patch[mask]


def _stamp_key(grid: np.ndarray, spec: dict, top: int, left: int) -> None:
    patch = spec["key_patch"]
    ph, pw = patch.shape
    mask = patch != BG
    grid[top : top + ph, left : left + pw][mask] = patch[mask]


def _apply_expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        kt, kl = spec["key_pos"]
        for r, c in np.argwhere(spec["key_mask"]):
            out[kt + int(r), kl + int(c)] = BG

    for spec in specs:
        lt, ll = spec["lock_pos"]
        fh, fw = spec["frame_shape"]
        color = spec["key_color"]
        out[lt + 1 : lt + fh - 1, ll + 1 : ll + fw - 1] = color
        for r, c in spec["gap_cells_rel"]:
            out[lt + r, ll + c] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    lock_count = rng.randint(LOCK_COUNT_MIN, LOCK_COUNT_MAX)
    palette = [c for c in range(1, 10) if c != FRAME_COLOR]
    rng.shuffle(palette)
    colors = palette[:lock_count]
    specs = [_make_lock_spec(rng, color) for color in colors]

    min_h = max(
        GRID_SIDE_MIN,
        max(spec["frame_shape"][0] for spec in specs) + max(spec["key_shape"][0] for spec in specs) + 5,
    )
    min_w = max(
        GRID_SIDE_MIN,
        max(spec["frame_shape"][1] for spec in specs) + max(spec["key_shape"][1] for spec in specs) + 5,
    )
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None

    grid_h = rng.randint(min_h, GRID_SIDE_MAX)
    grid_w = rng.randint(min_w, GRID_SIDE_MAX)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in specs:
        fh, fw = spec["frame_shape"]
        positions = [
            (r, c)
            for r in range(grid_h - fh + 1)
            for c in range(grid_w - fw + 1)
            if _can_place_rectangle(occupied, r, c, fh, fw, gap=1)
        ]
        if not positions:
            return None
        lt, ll = rng.choice(positions)
        spec["lock_pos"] = (lt, ll)
        occupied[lt : lt + fh, ll : ll + fw] = True
        _stamp_lock(grid, spec, lt, ll)

    for spec in specs:
        kh, kw = spec["key_shape"]
        positions = [
            (r, c)
            for r in range(grid_h - kh + 1)
            for c in range(grid_w - kw + 1)
            if _can_place_rectangle(occupied, r, c, kh, kw, gap=1)
        ]
        if not positions:
            return None
        kt, kl = rng.choice(positions)
        spec["key_pos"] = (kt, kl)
        occupied[kt : kt + kh, kl : kl + kw] = True
        _stamp_key(grid, spec, kt, kl)

    expected = _apply_expected_output(grid, specs)
    bg, locks, keys = _parse_scene(grid)
    if bg != BG or len(locks) != lock_count or len(keys) != lock_count:
        return None
    if any(key["area"] > 6 for key in keys):
        return None

    candidates = _candidate_map(locks, keys)
    best_size, solutions = _best_matchings(candidates, len(keys), limit=2)
    if best_size != lock_count or len(solutions) != 1:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_RETRIES} attempts")
