from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
GEN_RETRIES = 400
LOCK_COUNT_RANGE = (1, 4)
LOCK_HEIGHT_RANGE = (5, 8)
LOCK_WIDTH_RANGE = (5, 8)
CANVAS_RANGE = (18, 28)
PLACEMENT_RETRIES = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_keys(grid: np.ndarray) -> list[dict[str, object]]:
    keys: list[dict[str, object]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in (BG, GREY)):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "key color must exist"
        rows = np.unique(coords[:, 0])
        assert rows.size == 1, "each key color must occupy exactly one row"
        row = int(rows[0])
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        span = grid[row, c0 : c1 + 1]
        assert np.all((span == BG) | (span == color)), "key span must contain only key color or black"
        pattern = tuple(int(v == color) for v in span.tolist())
        assert len(pattern) >= LOCK_HEIGHT_RANGE[0], "key must be long enough to encode a lock"
        assert any(pattern) and not all(pattern), "key must have both teeth and slots"
        keys.append(
            {
                "color": color,
                "row": row,
                "c0": c0,
                "c1": c1,
                "pattern": pattern,
            }
        )
    assert keys, "at least one key is required"
    return keys


def _parse_lock(sub: np.ndarray) -> tuple[tuple[int, ...], int, int] | None:
    h, w = sub.shape
    if h < 5 or w < 5:
        return None
    if not np.all(np.isin(sub, [BG, GREY])):
        return None
    if not np.all(sub[:, 1] == GREY):
        return None
    if not np.all(sub[:, -1] == GREY):
        return None
    if not np.all(sub[0, 1:] == GREY):
        return None
    if not np.all(sub[-1, 1:] == GREY):
        return None
    if sub.shape[1] > 3 and not np.all(sub[1:-1, 2:-1] == BG):
        return None

    allowed = np.zeros_like(sub, dtype=bool)
    allowed[:, 1] = True
    allowed[:, -1] = True
    allowed[0, 1:] = True
    allowed[-1, 1:] = True
    allowed[:, 0] = sub[:, 0] == GREY
    if np.any((sub == GREY) & ~allowed):
        return None

    pattern = tuple(int(v == GREY) for v in sub[:, 0].tolist())
    return pattern, 2, w - 1


def _extract_locks(grid: np.ndarray) -> list[dict[str, object]]:
    locks: list[dict[str, object]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        parsed = _parse_lock(sub)
        if parsed is None:
            continue
        pattern, fill_c0_local, fill_c1_local = parsed
        locks.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "pattern": pattern,
                "fill_c0": c0 + fill_c0_local,
                "fill_c1": c0 + fill_c1_local,
            }
        )
    assert locks, "at least one lock is required"
    return locks


def _complement(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return len(a) == len(b) and all(x + y == 1 for x, y in zip(a, b))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the grey lock with the complementary tooth pattern,
    erase the key, and fill the lock interior row-by-row with the key color where
    the key has a tooth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    keys = _extract_keys(g)
    locks = _extract_locks(g)
    assert len(keys) == len(locks), "keys and locks must be paired one-to-one"

    matches: dict[int, int] = {}
    used_locks: set[int] = set()
    for ki, key in enumerate(keys):
        candidates = [
            li
            for li, lock in enumerate(locks)
            if _complement(key["pattern"], lock["pattern"])
        ]
        assert len(candidates) == 1, "each key must match exactly one lock"
        li = candidates[0]
        assert li not in used_locks, "lock matched more than once"
        matches[ki] = li
        used_locks.add(li)
    assert len(matches) == len(locks), "all locks must be matched"

    out = g.copy()
    for key in keys:
        out[int(key["row"]), int(key["c0"]) : int(key["c1"]) + 1] = BG

    for ki, li in matches.items():
        key = keys[ki]
        lock = locks[li]
        color = int(key["color"])
        pattern = key["pattern"]
        r0 = int(lock["r0"])
        fill_c0 = int(lock["fill_c0"])
        fill_c1 = int(lock["fill_c1"])
        for i, bit in enumerate(pattern):
            if bit and 0 < i < len(pattern) - 1:
                out[r0 + i, fill_c0:fill_c1] = color

    return out


def _make_lock(lock_pattern: tuple[int, ...], width: int) -> np.ndarray:
    h = len(lock_pattern)
    arr = np.full((h, width), BG, dtype=int)
    arr[:, 1] = GREY
    arr[:, -1] = GREY
    arr[0, 1:] = GREY
    arr[-1, 1:] = GREY
    for r, bit in enumerate(lock_pattern):
        if bit:
            arr[r, 0] = GREY
    return arr


def _make_key(key_pattern: tuple[int, ...], color: int) -> np.ndarray:
    arr = np.full((1, len(key_pattern)), BG, dtype=int)
    for c, bit in enumerate(key_pattern):
        if bit:
            arr[0, c] = color
    return arr


def _reserve_ok(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    H, W = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _sample_lock_patterns(rng: random.Random, heights: list[int]) -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []
    used: set[tuple[int, ...]] = set()
    for h in heights:
        for _ in range(200):
            middle = [rng.randint(0, 1) for _ in range(h - 2)]
            if sum(middle) == 0 or sum(middle) == len(middle):
                continue
            lock_pattern = tuple([0] + middle + [0])
            if lock_pattern in used:
                continue
            used.add(lock_pattern)
            patterns.append(lock_pattern)
            break
        else:
            raise RuntimeError("failed to sample unique lock patterns")
    return patterns


def _direct_output(
    grid: np.ndarray,
    key_specs: list[dict[str, object]],
    lock_specs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for key in key_specs:
        row = int(key["row"])
        c0 = int(key["c0"])
        pattern = key["pattern"]
        out[row, c0 : c0 + len(pattern)] = BG

    for key, lock in zip(key_specs, lock_specs):
        color = int(key["color"])
        pattern = key["pattern"]
        r0 = int(lock["row"])
        c0 = int(lock["col"])
        width = int(lock["width"])
        for i, bit in enumerate(pattern):
            if bit and 0 < i < len(pattern) - 1:
                out[r0 + i, c0 + 2 : c0 + width - 1] = color
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n = rng.randint(*LOCK_COUNT_RANGE)
    heights = [rng.randint(*LOCK_HEIGHT_RANGE) for _ in range(n)]
    widths = [rng.randint(*LOCK_WIDTH_RANGE) for _ in range(n)]
    patterns = _sample_lock_patterns(rng, heights)

    palette = [c for c in range(1, 10) if c != GREY]
    if n > len(palette):
        return None
    colors = rng.sample(palette, n)

    H = rng.randint(max(CANVAS_RANGE[0], max(heights) + 4), CANVAS_RANGE[1])
    W = rng.randint(max(CANVAS_RANGE[0], max(max(widths), max(heights)) + 4), CANVAS_RANGE[1])
    grid = np.full((H, W), BG, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    lock_specs: list[dict[str, object]] = []
    key_specs: list[dict[str, object]] = []
    items = []
    for i in range(n):
        items.append(("lock", i, heights[i], widths[i]))
        items.append(("key", i, 1, heights[i]))
    rng.shuffle(items)

    placements: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, h, w in items:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(0, H - h)
            c0 = rng.randint(0, W - w)
            if not _reserve_ok(blocked, r0, c0, h, w):
                continue
            placements[(kind, idx)] = (r0, c0)
            _mark_blocked(blocked, r0, c0, h, w)
            placed = True
            break
        if not placed:
            return None

    for i in range(n):
        lock_pattern = patterns[i]
        key_pattern = tuple(1 - b for b in lock_pattern)

        lock_arr = _make_lock(lock_pattern, widths[i])
        lr, lc = placements[("lock", i)]
        grid[lr : lr + lock_arr.shape[0], lc : lc + lock_arr.shape[1]] = lock_arr
        lock_specs.append(
            {
                "row": lr,
                "col": lc,
                "width": widths[i],
                "pattern": lock_pattern,
            }
        )

        key_arr = _make_key(key_pattern, colors[i])
        kr, kc = placements[("key", i)]
        grid[kr : kr + 1, kc : kc + key_arr.shape[1]] = key_arr
        key_specs.append(
            {
                "row": kr,
                "c0": kc,
                "color": colors[i],
                "pattern": key_pattern,
            }
        )

    expected = _direct_output(grid, key_specs, lock_specs)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        made = _gen_attempt(rng)
        if made is None:
            continue
        grid, expected = made
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid key-lock puzzle")
