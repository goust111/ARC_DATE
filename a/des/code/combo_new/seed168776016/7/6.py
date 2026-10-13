import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SOLVE_SHIFT = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
GEN_KEY_OFFSET = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
WORK_SIZE = 30
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.flat)).most_common(1)[0][0]


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
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _shift(coords: list[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in coords}


def _ring_coords(h: int, w: int, t: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if r < t or r >= h - t or c < t or c >= w - t:
                coords.add((int(r), int(c)))
    return coords


def _ring_thickness(coords: set[tuple[int, int]]) -> int | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    for t in range(1, min(h, w) // 2 + 1):
        inner_h = h - 2 * t
        inner_w = w - 2 * t
        if inner_h < 2 or inner_w < 2:
            continue
        ring = np.ones((h, w), dtype=bool)
        ring[t : h - t, t : w - t] = False
        if np.array_equal(local, ring):
            return int(t)
    return None


def _marker_direction(marker: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    mr, mc = marker
    r0, c0, r1, c1 = frame_bbox
    if mr < r0 and c0 <= mc <= c1:
        return "up"
    if mr > r1 and c0 <= mc <= c1:
        return "down"
    if mc < c0 and r0 <= mr <= r1:
        return "left"
    if mc > c1 and r0 <= mr <= r1:
        return "right"
    raise AssertionError("marker must sit outside exactly one frame side")


def _find_docked_key(
    frame: list[tuple[int, int]],
    key: list[tuple[int, int]],
    direction: str,
) -> set[tuple[int, int]]:
    frame_set = set(frame)
    frame_bbox = _bbox(frame)
    kr0, kc0, kr1, kc1 = _bbox(key)
    fr0, fc0, fr1, fc1 = frame_bbox
    valid: list[set[tuple[int, int]]] = []

    if direction in ("up", "down"):
        lo = fr0 - kr0
        hi = fr1 - kr1
        for dr in range(lo, hi + 1):
            if direction == "up" and dr <= 0:
                continue
            if direction == "down" and dr >= 0:
                continue
            cand = _shift(key, dr, 0)
            if cand & frame_set:
                continue
            if _bbox(cand | frame_set) != frame_bbox:
                continue
            if _ring_thickness(cand | frame_set) is not None:
                valid.append(cand)
    else:
        lo = fc0 - kc0
        hi = fc1 - kc1
        for dc in range(lo, hi + 1):
            if direction == "left" and dc <= 0:
                continue
            if direction == "right" and dc >= 0:
                continue
            cand = _shift(key, 0, dc)
            if cand & frame_set:
                continue
            if _bbox(cand | frame_set) != frame_bbox:
                continue
            if _ring_thickness(cand | frame_set) is not None:
                valid.append(cand)

    assert len(valid) == 1, "key placement must be unique"
    return valid[0]


def _solve_color(grid: np.ndarray, color: int) -> tuple[list[tuple[int, int]], tuple[int, int], set[tuple[int, int]]]:
    comps = _components(grid == int(color))
    assert len(comps) >= 3, "each color must have frame, key, and marker"
    comps.sort(key=len, reverse=True)
    frame = comps[0]
    others = comps[1:]
    markers = [comp for comp in others if len(comp) == 1]
    keys = [comp for comp in others if len(comp) > 1]
    assert len(markers) == 1, "direction marker must be a unique single pixel"
    assert len(keys) == 1, "key must be a unique non-marker component"
    marker = markers[0][0]
    key = keys[0]
    direction = _marker_direction(marker, _bbox(frame))
    docked = _find_docked_key(frame, key, direction)
    return key, marker, docked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored key into the matching door-frame gap using the outside direction marker.

    Input structure:
    - The background is black.
    - Each non-background color appears as a large rectangular door frame with a one-pixel-wide missing gap, one isolated same-color marker pixel outside one side of that frame, and one detached same-color key component elsewhere.
    - The key shape matches the missing frame cells exactly, and the marker sits above, below, left, or right of the frame without touching it.

    Rule:
    - Pair objects by color.
    - Use the marker position relative to the frame to decide the translation axis and direction: top marker means move the key downward, right marker means move it leftward, and so on.
    - Translate the key along that single axis until it uniquely completes a proper rectangular ring; erase the key's original location and erase the marker pixel.

    Key insight:
    - The missing cells are recovered by shape matching, but the hidden crucial clue is the lone same-color marker outside the frame, which fixes from which side the key must slide.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    for color in sorted(set(map(int, g.flat)) - {int(bg)}):
        key, marker, docked = _solve_color(g, int(color))
        for r, c in key:
            out[r, c] = int(bg)
        out[int(marker[0]), int(marker[1])] = int(bg)
        for r, c in docked:
            out[r, c] = int(color)
    return out


def _top_gap_path(length: int, thickness: int, rng: random.Random) -> set[tuple[int, int]]:
    if thickness == 1:
        return {(0, c) for c in range(length)}
    path: set[tuple[int, int]] = {(0, 0)}
    row = 0
    visited_inner = False
    for c in range(length - 1):
        if rng.random() < 0.5:
            row = 1 - row
            path.add((row, c))
            if row == 1:
                visited_inner = True
        path.add((row, c + 1))
        if row == 1:
            visited_inner = True
    if not visited_inner:
        path.add((1, length - 1))
    return path


def _gap_coords(h: int, w: int, thickness: int, side: str, rng: random.Random) -> set[tuple[int, int]]:
    side_len = w if side in ("top", "bottom") else h
    max_len = side_len - 2
    assert max_len >= 2
    length = rng.randint(2, max_len)
    start = rng.randint(1, side_len - length - 1)
    base = _top_gap_path(length, thickness, rng)
    if side == "top":
        return {(r, start + c) for r, c in base}
    if side == "bottom":
        return {(h - 1 - r, start + c) for r, c in base}
    if side == "left":
        return {(start + c, r) for r, c in base}
    return {(start + c, w - 1 - r) for r, c in base}


def _is_clear(pos: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    pr, pc = pos
    for r, c in occupied:
        if max(abs(pr - r), abs(pc - c)) <= 1:
            return False
    return True


def _pick_marker(
    frame: set[tuple[int, int]],
    key: set[tuple[int, int]],
    side: str,
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[int, int]:
    occupied = frame | key
    for dist in (2, 3, 4, 5):
        if side == "up":
            candidates = [(-dist, c) for c in range(w)]
        elif side == "down":
            candidates = [(h - 1 + dist, c) for c in range(w)]
        elif side == "left":
            candidates = [(r, -dist) for r in range(h)]
        else:
            candidates = [(r, w - 1 + dist) for r in range(h)]
        good = [pos for pos in candidates if _is_clear(pos, occupied)]
        if good:
            return rng.choice(good)
    raise AssertionError("marker must stay isolated")


def _normalize_pair(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    shifted = {(r - r0, c - c0) for r, c in coords}
    return shifted, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _build_pair(color: int, rng: random.Random) -> dict | None:
    thickness = rng.choice([1, 2])
    min_dim = 2 * thickness + 2
    h = rng.randint(max(6, min_dim), 11)
    w = rng.randint(max(6, min_dim), 11)
    frame = _ring_coords(h, w, thickness)
    gap_side = rng.choice(["top", "bottom", "left", "right"])
    gap = _gap_coords(h, w, thickness, gap_side, rng)
    frame -= gap
    if not frame or len(_components(_coords_to_mask(frame))[0]) != len(frame):
        return None

    direction = rng.choice(["up", "down", "left", "right"])
    offset = rng.randint(5, 7)
    dr0, dc0 = GEN_KEY_OFFSET[direction]
    key = _shift(gap, dr0 * offset, dc0 * offset)
    marker = _pick_marker(frame, key, direction, h, w, rng)
    all_coords = set(frame) | set(key) | {marker}
    norm, ph, pw = _normalize_pair(all_coords)
    return {
        "color": int(color),
        "coords": norm,
        "height": int(ph),
        "width": int(pw),
    }


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _place_pairs(pairs: list[dict], rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((WORK_SIZE, WORK_SIZE), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    for pair in sorted(pairs, key=lambda item: item["height"] * item["width"], reverse=True):
        ph = int(pair["height"])
        pw = int(pair["width"])
        placed = False
        for _ in range(200):
            r0 = rng.randint(0, WORK_SIZE - ph)
            c0 = rng.randint(0, WORK_SIZE - pw)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(not _bbox_separated(box, other, pad=1) for other in placed_boxes):
                continue
            for rr, cc in pair["coords"]:
                grid[r0 + rr, c0 + cc] = int(pair["color"])
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return grid


def _crop(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    pts = np.argwhere(grid != 0)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    top = rng.randint(0, min(4, int(r0)))
    left = rng.randint(0, min(4, int(c0)))
    bottom = rng.randint(0, min(4, int(WORK_SIZE - 1 - r1)))
    right = rng.randint(0, min(4, int(WORK_SIZE - 1 - c1)))
    return grid[int(r0 - top) : int(r1 + bottom) + 1, int(c0 - left) : int(c1 + right) + 1]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([1, 2, 3])
    colors = rng.sample(PALETTE, pair_count)
    pairs: list[dict] = []
    for color in colors:
        pair = None
        for _ in range(80):
            pair = _build_pair(int(color), rng)
            if pair is not None:
                break
        if pair is None:
            return None
        pairs.append(pair)
    grid = _place_pairs(pairs, rng)
    if grid is None:
        return None
    cropped = _crop(grid, rng)
    if max(cropped.shape) > 30:
        return None
    try:
        solved = main(cropped)
    except AssertionError:
        return None
    if solved.shape != cropped.shape:
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
