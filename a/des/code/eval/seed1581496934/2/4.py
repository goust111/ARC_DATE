import numpy as np
from functools import lru_cache


BLACK = 0
LOCK_COLOR = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()


def _build_lock_masks() -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for direction in DIRS:
        mask = np.ones((5, 5), dtype=bool)
        if direction == "up":
            mask[0, 1:4] = False
        elif direction == "down":
            mask[4, 1:4] = False
        elif direction == "left":
            mask[1:4, 0] = False
        else:
            mask[1:4, 4] = False
        masks[direction] = mask
    return masks


def _build_key_masks() -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}

    up = np.zeros((4, 3), dtype=bool)
    up[1:, :] = True
    up[0, 1] = True
    masks["up"] = up

    down = np.zeros((4, 3), dtype=bool)
    down[:3, :] = True
    down[3, 1] = True
    masks["down"] = down

    left = np.zeros((3, 4), dtype=bool)
    left[:, 1:] = True
    left[1, 0] = True
    masks["left"] = left

    right = np.zeros((3, 4), dtype=bool)
    right[:, :3] = True
    right[1, 3] = True
    masks["right"] = right

    return masks


LOCK_MASKS = _build_lock_masks()
KEY_MASKS = _build_key_masks()
HALF_BLOCKS = {
    "up": (((0, 2), (0, 2)), ((0, 2), (3, 5))),
    "down": (((3, 5), (0, 2)), ((3, 5), (3, 5))),
    "left": (((0, 2), (0, 2)), ((3, 5), (0, 2))),
    "right": (((0, 2), (3, 5)), ((3, 5), (3, 5))),
}


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0 = int(pts[:, 0].min())
            c0 = int(pts[:, 1].min())
            r1 = int(pts[:, 0].max())
            c1 = int(pts[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                    "center": pts.mean(axis=0),
                }
            )

    return comps


def _classify_component(comp: dict) -> tuple[str, str] | None:
    for direction, mask in LOCK_MASKS.items():
        if comp["mask"].shape == mask.shape and comp["area"] == int(mask.sum()) and np.array_equal(comp["mask"], mask):
            return ("lock", direction)
    for direction, mask in KEY_MASKS.items():
        if comp["mask"].shape == mask.shape and comp["area"] == int(mask.sum()) and np.array_equal(comp["mask"], mask):
            return ("key", direction)
    return None


def _find_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    locks: list[dict] = []
    keys: list[dict] = []

    for comp in _components(grid):
        classified = _classify_component(comp)
        assert classified is not None, "unexpected foreground component"
        kind, direction = classified
        obj = dict(comp)
        obj["direction"] = direction
        if kind == "lock":
            locks.append(obj)
        else:
            keys.append(obj)

    assert locks and len(locks) == len(keys), "expected equal counts of locks and keys"
    return locks, keys


def _distance_sq(a: dict, b: dict) -> float:
    delta = a["center"] - b["center"]
    return float(delta[0] * delta[0] + delta[1] * delta[1])


def _assign_group(locks: list[dict], keys: list[dict]) -> list[tuple[dict, dict]]:
    assert len(locks) == len(keys), "direction counts must match"
    locks = sorted(locks, key=lambda obj: (obj["bbox"], int(obj["color"])))
    keys = sorted(keys, key=lambda obj: (obj["bbox"], int(obj["color"])))
    n = len(locks)
    if n == 0:
        return []

    dist = [[_distance_sq(locks[i], keys[j]) for j in range(n)] for i in range(n)]

    @lru_cache(maxsize=None)
    def solve(i: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if i == n:
            return (0.0, ())
        best_cost = None
        best_assign = ()
        for j in range(n):
            if (used_mask >> j) & 1:
                continue
            tail_cost, tail_assign = solve(i + 1, used_mask | (1 << j))
            total_cost = dist[i][j] + tail_cost
            assignment = (j,) + tail_assign
            if best_cost is None or total_cost < best_cost or (total_cost == best_cost and assignment < best_assign):
                best_cost = total_cost
                best_assign = assignment
        assert best_cost is not None
        return (best_cost, best_assign)

    _, key_indexes = solve(0, 0)
    return [(locks[i], keys[key_indexes[i]]) for i in range(n)]


def _match_pairs(locks: list[dict], keys: list[dict]) -> list[tuple[dict, dict]]:
    pairs: list[tuple[dict, dict]] = []
    for direction in DIRS:
        dir_locks = [obj for obj in locks if obj["direction"] == direction]
        dir_keys = [obj for obj in keys if obj["direction"] == direction]
        pairs.extend(_assign_group(dir_locks, dir_keys))
    assert len(pairs) == len(locks), "every lock must be paired"
    return pairs


def _relative_side(lock: dict, key: dict) -> str:
    lr0, lc0, lr1, lc1 = map(int, lock["bbox"])
    kr0, kc0, kr1, kc1 = map(int, key["bbox"])
    if kr1 < lr0:
        return "up"
    if kr0 > lr1:
        return "down"
    if kc1 < lc0:
        return "left"
    if kc0 > lc1:
        return "right"

    dr = float(key["center"][0] - lock["center"][0])
    dc = float(key["center"][1] - lock["center"][1])
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _recolor_half(out: np.ndarray, lock: dict, fill_color: int, side: str) -> None:
    r0, c0, _, _ = map(int, lock["bbox"])
    for (rs0, rs1), (cs0, cs1) in HALF_BLOCKS[side]:
        patch = out[r0 + rs0 : r0 + rs1, c0 + cs0 : c0 + cs1]
        patch[patch == int(lock["color"])] = int(fill_color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    locks, keys = _find_objects(g)
    out = g.copy()

    for key in keys:
        rows = key["coords"][:, 0]
        cols = key["coords"][:, 1]
        out[rows, cols] = BLACK

    for lock, key in _match_pairs(locks, keys):
        _recolor_half(out, lock, int(key["color"]), _relative_side(lock, key))

    return out


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[top + rows, left + cols] = int(color)


def _key_anchor(lock_r: int, lock_c: int, key_h: int, key_w: int, side: str, gap: int, jitter: int) -> tuple[int, int]:
    if side == "up":
        return (lock_r - gap - key_h, lock_c + (5 - key_w) // 2 + jitter)
    if side == "down":
        return (lock_r + 5 + gap, lock_c + (5 - key_w) // 2 + jitter)
    if side == "left":
        return (lock_r + (5 - key_h) // 2 + jitter, lock_c - gap - key_w)
    return (lock_r + (5 - key_h) // 2 + jitter, lock_c + 5 + gap)


def _fits(h: int, w: int, top: int, left: int, mask: np.ndarray) -> bool:
    return 0 <= top and 0 <= left and top + mask.shape[0] <= h and left + mask.shape[1] <= w


def _buffered_union_bbox(
    h: int,
    w: int,
    lock_r: int,
    lock_c: int,
    key_r: int,
    key_c: int,
    key_mask: np.ndarray,
    pad: int = 1,
) -> tuple[int, int, int, int]:
    r0 = min(lock_r, key_r)
    c0 = min(lock_c, key_c)
    r1 = max(lock_r + 4, key_r + key_mask.shape[0] - 1)
    c1 = max(lock_c + 4, key_c + key_mask.shape[1] - 1)
    return (max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad))


def _sample_labels(n: int) -> list[str]:
    labels = [DIRS[int(RNG.integers(0, 4))] for _ in range(n)]
    for _ in range(12):
        if n <= 2 or len(set(labels)) >= 2:
            break
        labels = [DIRS[int(RNG.integers(0, 4))] for _ in range(n)]
    return labels


def _expected_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        key_mask = KEY_MASKS[pair["key_dir"]]
        rows, cols = np.nonzero(key_mask)
        out[pair["key_r"] + rows, pair["key_c"] + cols] = BLACK

    for pair in pairs:
        lock = {
            "bbox": (pair["lock_r"], pair["lock_c"], pair["lock_r"] + 4, pair["lock_c"] + 4),
            "color": LOCK_COLOR,
        }
        _recolor_half(out, lock, pair["color"], pair["side"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = int(RNG.integers(2, 6))
    grid_h = int(RNG.integers(16 + 4 * n_pairs, 23 + 5 * n_pairs))
    grid_w = int(RNG.integers(16 + 4 * n_pairs, 23 + 5 * n_pairs))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    directions = _sample_labels(n_pairs)
    sides = _sample_labels(n_pairs)
    colors = [int(color) for color in RNG.choice(np.array(KEY_COLORS), size=n_pairs, replace=False)]
    gaps = [int(RNG.integers(1, 5)) for _ in range(n_pairs)]

    pairs: list[dict] = []
    for i in range(n_pairs):
        key_dir = directions[i]
        side = sides[i]
        color = colors[i]
        gap = gaps[i]
        key_mask = KEY_MASKS[key_dir]

        placed = False
        for _ in range(300):
            lock_r = int(RNG.integers(0, grid_h - 4))
            lock_c = int(RNG.integers(0, grid_w - 4))
            jitter = int(RNG.integers(-1, 2))
            key_r, key_c = _key_anchor(lock_r, lock_c, key_mask.shape[0], key_mask.shape[1], side, gap, jitter)
            if not _fits(grid_h, grid_w, key_r, key_c, key_mask):
                continue
            if _relative_side(
                {"bbox": (lock_r, lock_c, lock_r + 4, lock_c + 4), "center": np.array([lock_r + 2.0, lock_c + 2.0])},
                {
                    "bbox": (key_r, key_c, key_r + key_mask.shape[0] - 1, key_c + key_mask.shape[1] - 1),
                    "center": np.array(
                        [
                            key_r + (key_mask.shape[0] - 1) / 2.0,
                            key_c + (key_mask.shape[1] - 1) / 2.0,
                        ]
                    ),
                },
            ) != side:
                continue

            br0, bc0, br1, bc1 = _buffered_union_bbox(grid_h, grid_w, lock_r, lock_c, key_r, key_c, key_mask)
            if occupied[br0 : br1 + 1, bc0 : bc1 + 1].any():
                continue

            _paint_mask(grid, lock_r, lock_c, LOCK_MASKS[key_dir], LOCK_COLOR)
            _paint_mask(grid, key_r, key_c, key_mask, color)
            occupied[br0 : br1 + 1, bc0 : bc1 + 1] = True
            pairs.append(
                {
                    "lock_r": lock_r,
                    "lock_c": lock_c,
                    "lock_dir": key_dir,
                    "key_r": key_r,
                    "key_c": key_c,
                    "key_dir": key_dir,
                    "color": color,
                    "side": side,
                }
            )
            placed = True
            break

        if not placed:
            return None

    expected = _expected_output(grid, pairs)
    if not np.array_equal(main(grid), expected):
        return None

    locks, keys = _find_objects(grid)
    if len(locks) != n_pairs or len(keys) != n_pairs:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
