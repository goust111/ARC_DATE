import random

import numpy as np


BG = 8
YELLOW = 4
GREY = 5

COLOR_POOL = [0, 1, 2, 3, 6, 7, 9]
DIRECTIONS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400
PAIR_COUNT_RANGE = (2, 4)
GRID_MIN = 18
GRID_MAX = 28


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _center_offset(outer: int, inner: int) -> int:
    return (outer - inner + 1) // 2


def _fill_box(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _box_clear(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _lock_direction(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str | None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    candidates: list[str] = []
    if r0 > 0 and np.count_nonzero(grid[r0 - 1, c0 : c1 + 1] == YELLOW) == 1:
        candidates.append("up")
    if r1 + 1 < h and np.count_nonzero(grid[r1 + 1, c0 : c1 + 1] == YELLOW) == 1:
        candidates.append("down")
    if c0 > 0 and np.count_nonzero(grid[r0 : r1 + 1, c0 - 1] == YELLOW) == 1:
        candidates.append("left")
    if c1 + 1 < w and np.count_nonzero(grid[r0 : r1 + 1, c1 + 1] == YELLOW) == 1:
        candidates.append("right")
    return candidates[0] if len(candidates) == 1 else None


def _extract_locks(grid: np.ndarray, bg: int) -> dict[int, dict]:
    locks: dict[int, dict] = {}
    for color in sorted(set(map(int, grid.flat)) - {bg, GREY, YELLOW}):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(sub == color):
                continue
            direction = _lock_direction(grid, (r0, c0, r1, c1))
            if direction is None:
                continue
            if color in locks:
                raise AssertionError("each lock color must be unique")
            locks[color] = {
                "color": color,
                "lock_bbox": (r0, c0, r1, c1),
                "direction": direction,
            }
    return locks


def _extract_keys(grid: np.ndarray, bg: int) -> dict[int, dict]:
    keys: dict[int, dict] = {}
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            continue

        local = np.zeros((hh, ww), dtype=bool)
        for rr, cc in comp:
            local[rr - r0, cc - c0] = True

        frame = np.zeros((hh, ww), dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if not np.array_equal(local, frame):
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = sorted(set(map(int, interior.flat)) - {bg, GREY, YELLOW})
        if len(colors) != 1:
            continue
        color = colors[0]
        shape_mask = interior == color
        comps = _components(shape_mask)
        if len(comps) != 1:
            continue
        sr0, sc0, sr1, sc1 = _bbox(comps[0])
        key_mask = shape_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]
        if color in keys:
            raise AssertionError("each key color must be unique")
        keys[color] = {
            "color": color,
            "key_mask": key_mask,
            "key_frame_bbox": (r0, c0, r1, c1),
        }
    return keys


def _aligned_origin(lock_bbox: tuple[int, int, int, int], key_mask: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = lock_bbox
    lock_h = r1 - r0 + 1
    lock_w = c1 - c0 + 1
    key_h, key_w = key_mask.shape
    return r0 + _center_offset(lock_h, key_h), c0 + _center_offset(lock_w, key_w)


def _fragment_origin(lock_bbox: tuple[int, int, int, int], key_mask: np.ndarray, direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = lock_bbox
    key_h, key_w = key_mask.shape
    ar0, ac0 = _aligned_origin(lock_bbox, key_mask)
    if direction == "up":
        return r0 - key_h - 2, ac0
    if direction == "down":
        return r1 + 3, ac0
    if direction == "left":
        return ar0, c0 - key_w - 2
    if direction == "right":
        return ar0, c1 + 3
    raise AssertionError("unknown direction")


def _transform_pairs(grid: np.ndarray, pairs: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    for pair in pairs:
        fr0, fc0, fr1, fc1 = pair["key_frame_bbox"]
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg

    for pair in pairs:
        color = int(pair["color"])
        key_mask = np.asarray(pair["key_mask"], dtype=bool)
        lock_bbox = tuple(map(int, pair["lock_bbox"]))
        direction = str(pair["direction"])
        ar0, ac0 = _aligned_origin(lock_bbox, key_mask)
        fr0, fc0 = _fragment_origin(lock_bbox, key_mask, direction)

        key_h, key_w = key_mask.shape
        for rr in range(key_h):
            for cc in range(key_w):
                if not key_mask[rr, cc]:
                    continue
                out[ar0 + rr, ac0 + cc] = bg
                out[fr0 + rr, fc0 + cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    locks = _extract_locks(g, bg)
    keys = _extract_keys(g, bg)
    colors = sorted(set(locks) & set(keys))
    assert colors and len(colors) == len(locks) == len(keys), "expected one lock and one key per color"

    pairs = []
    for color in colors:
        pairs.append(
            {
                "color": color,
                "lock_bbox": locks[color]["lock_bbox"],
                "direction": locks[color]["direction"],
                "key_mask": keys[color]["key_mask"],
                "key_frame_bbox": keys[color]["key_frame_bbox"],
            }
        )
    return _transform_pairs(g, pairs, bg)


def _sample_lock_dims(rng: random.Random, direction: str) -> tuple[int, int]:
    if direction in {"up", "down"}:
        return rng.randint(3, 5), rng.choice([3, 5])
    return rng.choice([3, 5]), rng.randint(3, 5)


def _random_key_shape(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(200):
        target = rng.randint(2, 5)
        cells = {(0, 0)}
        while len(cells) < target:
            cr, cc = rng.choice(tuple(cells))
            dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            cells.add((cr + dr, cc + dc))

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        norm = {(r - r0, c - c0) for r, c in cells}
        rows = [r for r, _ in norm]
        cols = [c for _, c in norm]
        hh = max(rows) + 1
        ww = max(cols) + 1
        if hh > max_h or ww > max_w:
            continue
        if hh * ww == len(norm) and hh > 1 and ww > 1:
            continue

        mask = np.zeros((hh, ww), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        return mask
    return None


def _lock_reserved_bbox(
    top: int,
    left: int,
    lock_h: int,
    lock_w: int,
    key_mask: np.ndarray,
    direction: str,
) -> tuple[int, int, int, int]:
    lock_bbox = (top, left, top + lock_h - 1, left + lock_w - 1)
    frag_r0, frag_c0 = _fragment_origin(lock_bbox, key_mask, direction)
    key_h, key_w = key_mask.shape
    rr0 = min(top, frag_r0)
    cc0 = min(left, frag_c0)
    rr1 = max(top + lock_h - 1, frag_r0 + key_h - 1)
    cc1 = max(left + lock_w - 1, frag_c0 + key_w - 1)
    if direction == "up":
        rr0 = min(rr0, top - 1)
    elif direction == "down":
        rr1 = max(rr1, top + lock_h)
    elif direction == "left":
        cc0 = min(cc0, left - 1)
    else:
        cc1 = max(cc1, left + lock_w)
    return rr0, cc0, rr1, cc1


def _paint_lock(grid: np.ndarray, color: int, top: int, left: int, lock_h: int, lock_w: int, direction: str) -> None:
    grid[top : top + lock_h, left : left + lock_w] = color
    if direction == "up":
        grid[top - 1, left + lock_w // 2] = YELLOW
    elif direction == "down":
        grid[top + lock_h, left + lock_w // 2] = YELLOW
    elif direction == "left":
        grid[top + lock_h // 2, left - 1] = YELLOW
    else:
        grid[top + lock_h // 2, left + lock_w] = YELLOW


def _paint_key(grid: np.ndarray, color: int, top: int, left: int, key_mask: np.ndarray) -> tuple[int, int, int, int]:
    key_h, key_w = key_mask.shape
    frame_bbox = (top, left, top + key_h + 1, left + key_w + 1)
    grid[top, left : left + key_w + 2] = GREY
    grid[top + key_h + 1, left : left + key_w + 2] = GREY
    grid[top : top + key_h + 2, left] = GREY
    grid[top : top + key_h + 2, left + key_w + 1] = GREY
    for rr in range(key_h):
        for cc in range(key_w):
            if key_mask[rr, cc]:
                grid[top + 1 + rr, left + 1 + cc] = color
    return frame_bbox


def _place_lock(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    direction: str,
    rng: random.Random,
) -> dict | None:
    h, w = grid.shape
    for _ in range(250):
        lock_h, lock_w = _sample_lock_dims(rng, direction)
        key_mask = _random_key_shape(rng, lock_h, lock_w)
        if key_mask is None:
            continue

        key_h, key_w = key_mask.shape
        if direction == "up":
            min_top = key_h + 3
            max_top = h - lock_h - 2
            min_left = 1
            max_left = w - lock_w - 2
        elif direction == "down":
            min_top = 1
            max_top = h - (lock_h + key_h + 4)
            min_left = 1
            max_left = w - lock_w - 2
        elif direction == "left":
            min_top = 1
            max_top = h - lock_h - 2
            min_left = key_w + 3
            max_left = w - lock_w - 2
        else:
            min_top = 1
            max_top = h - lock_h - 2
            min_left = 1
            max_left = w - (lock_w + key_w + 4)
        if min_top > max_top or min_left > max_left:
            continue

        top = rng.randint(min_top, max_top)
        left = rng.randint(min_left, max_left)
        reserved = _lock_reserved_bbox(top, left, lock_h, lock_w, key_mask, direction)
        if not _box_clear(blocked, reserved):
            continue

        _paint_lock(grid, color, top, left, lock_h, lock_w, direction)
        _fill_box(blocked, reserved, pad=1)
        return {
            "color": color,
            "direction": direction,
            "key_mask": key_mask,
            "lock_bbox": (top, left, top + lock_h - 1, left + lock_w - 1),
        }
    return None


def _place_key(grid: np.ndarray, blocked: np.ndarray, spec: dict, rng: random.Random) -> bool:
    h, w = grid.shape
    key_mask = spec["key_mask"]
    key_h, key_w = key_mask.shape
    frame_h = key_h + 2
    frame_w = key_w + 2

    max_top = h - frame_h - 1
    max_left = w - frame_w - 1
    if max_top < 1 or max_left < 1:
        return False

    for _ in range(300):
        top = rng.randint(1, max_top)
        left = rng.randint(1, max_left)
        bbox = (top, left, top + frame_h - 1, left + frame_w - 1)
        if not _box_clear(blocked, bbox):
            continue
        spec["key_frame_bbox"] = _paint_key(grid, spec["color"], top, left, key_mask)
        _fill_box(blocked, bbox, pad=1)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    min_dim = GRID_MIN + 2 * max(0, n_pairs - 2)
    h = rng.randint(min_dim, GRID_MAX)
    w = rng.randint(min_dim, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = rng.sample(COLOR_POOL, n_pairs)
    directions = rng.sample(list(DIRECTIONS), n_pairs)
    specs: list[dict] = []

    for color, direction in zip(colors, directions):
        spec = _place_lock(grid, blocked, color, direction, rng)
        if spec is None:
            return None
        specs.append(spec)

    rng.shuffle(specs)
    for spec in specs:
        if not _place_key(grid, blocked, spec, rng):
            return None

    expected = _transform_pairs(grid, specs, BG)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
