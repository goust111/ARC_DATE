from __future__ import annotations

import numpy as np


BG = 0
LOCK_COLOR = 5
KEY_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

MIN_HEIGHT = 4
MAX_HEIGHT = 8
MIN_PAIRS = 2
MAX_PAIRS = 4
MAX_EXTRA_LOCKS = 2
PLACEMENT_PADDING = 1
PLACE_RETRIES = 300
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_lock(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    color = int(comp["color"])
    coords = comp["coords"]
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if w != 5 or h < MIN_HEIGHT:
        return None

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if set(map(int, np.unique(patch))) - {bg, color}:
        return None

    interior_rows = list(range(1, h - 1))
    if not interior_rows:
        return None

    side: str | None = None
    rows: tuple[int, ...] | None = None
    left_col = patch[1:-1, 0]
    right_col = patch[1:-1, -1]
    if np.all(right_col == color):
        left_missing = tuple(i + 1 for i, val in enumerate(left_col) if int(val) == bg)
        if 0 < len(left_missing) < len(interior_rows):
            side = "left"
            rows = left_missing
    if side is None and np.all(left_col == color):
        right_missing = tuple(i + 1 for i, val in enumerate(right_col) if int(val) == bg)
        if 0 < len(right_missing) < len(interior_rows):
            side = "right"
            rows = right_missing
    if side is None or rows is None:
        return None

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if side == "left":
        for rr in rows:
            expected[rr, 0] = False
    else:
        for rr in rows:
            expected[rr, -1] = False

    if not np.array_equal(patch == color, expected):
        return None

    return {
        "kind": "lock",
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "height": h,
        "side": side,
        "rows": rows,
    }


def _parse_key(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    color = int(comp["color"])
    coords = comp["coords"]
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if w != 2 or h < MIN_HEIGHT:
        return None

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if set(map(int, np.unique(patch))) - {bg, color}:
        return None

    interior_rows = list(range(1, h - 1))
    if not interior_rows:
        return None

    side: str | None = None
    rows: tuple[int, ...] | None = None
    left_col = patch[:, 0]
    right_col = patch[:, 1]
    if np.all(right_col == color):
        left_protrusions = tuple(i for i, val in enumerate(left_col) if int(val) == color)
        if left_protrusions and set(left_protrusions).issubset(interior_rows) and len(left_protrusions) < len(interior_rows):
            side = "left"
            rows = left_protrusions
    if side is None and np.all(left_col == color):
        right_protrusions = tuple(i for i, val in enumerate(right_col) if int(val) == color)
        if right_protrusions and set(right_protrusions).issubset(interior_rows) and len(right_protrusions) < len(interior_rows):
            side = "right"
            rows = right_protrusions
    if side is None or rows is None:
        return None

    expected = np.zeros((h, w), dtype=bool)
    if side == "left":
        expected[:, 1] = True
        for rr in rows:
            expected[rr, 0] = True
    else:
        expected[:, 0] = True
        for rr in rows:
            expected[rr, 1] = True

    if not np.array_equal(patch == color, expected):
        return None

    return {
        "kind": "key",
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "height": h,
        "side": side,
        "rows": rows,
    }


def _find_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    locks: list[dict] = []
    keys: list[dict] = []
    for comp in _components(grid, bg):
        lock = _parse_lock(grid, comp, bg)
        key = _parse_key(grid, comp, bg)
        if lock is not None and key is None:
            locks.append(lock)
        elif key is not None and lock is None:
            keys.append(key)
    return locks, keys


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    locks, keys = _find_objects(g, bg)

    lock_by_sig: dict[tuple[int, str, tuple[int, ...]], list[int]] = {}
    for idx, lock in enumerate(locks):
        sig = (int(lock["height"]), str(lock["side"]), tuple(lock["rows"]))
        lock_by_sig.setdefault(sig, []).append(idx)

    out = g.copy()
    used_locks: set[int] = set()
    for key in keys:
        sig = (int(key["height"]), str(key["side"]), tuple(key["rows"]))
        candidates = [idx for idx in lock_by_sig.get(sig, []) if idx not in used_locks]
        if len(candidates) != 1:
            continue
        lock = locks[candidates[0]]
        used_locks.add(candidates[0])

        for r, c in key["coords"]:
            out[int(r), int(c)] = bg

        r0, c0, r1, c1 = lock["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == bg] = int(key["color"])

    return out


def _build_lock(height: int, side: str, rows: tuple[int, ...], color: int = LOCK_COLOR, bg: int = BG) -> np.ndarray:
    patch = np.full((height, 5), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    col = 0 if side == "left" else 4
    for rr in rows:
        patch[rr, col] = bg
    return patch


def _build_key(height: int, side: str, rows: tuple[int, ...], color: int, bg: int = BG) -> np.ndarray:
    patch = np.full((height, 2), bg, dtype=int)
    if side == "left":
        patch[:, 1] = color
        for rr in rows:
            patch[rr, 0] = color
    else:
        patch[:, 0] = color
        for rr in rows:
            patch[rr, 1] = color
    return patch


def _sample_signature(rng: np.random.Generator, used: set[tuple[int, str, tuple[int, ...]]]) -> tuple[int, str, tuple[int, ...]] | None:
    for _ in range(200):
        height = int(rng.integers(MIN_HEIGHT, MAX_HEIGHT + 1))
        side = "left" if int(rng.integers(0, 2)) == 0 else "right"
        interior = np.arange(1, height - 1)
        count = int(rng.integers(1, len(interior)))
        rows = tuple(sorted(int(x) for x in rng.choice(interior, size=count, replace=False)))
        sig = (height, side, rows)
        if sig not in used:
            used.add(sig)
            return sig
    return None


def _fits(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    target[mask] = patch[mask]


def _random_crop(inp: np.ndarray, out: np.ndarray, rng: np.random.Generator, bg: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(inp != bg)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    top_pad = int(rng.integers(1, 4))
    bottom_pad = int(rng.integers(1, 4))
    left_pad = int(rng.integers(1, 4))
    right_pad = int(rng.integers(1, 4))
    rr0 = max(0, int(r0) - top_pad)
    cc0 = max(0, int(c0) - left_pad)
    rr1 = min(inp.shape[0] - 1, int(r1) + bottom_pad)
    cc1 = min(inp.shape[1] - 1, int(c1) + right_pad)
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1], out[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_pairs = int(rng.integers(MIN_PAIRS, MAX_PAIRS + 1))
    n_extra_locks = int(rng.integers(0, MAX_EXTRA_LOCKS + 1))
    canvas_h = int(rng.integers(22, 35))
    canvas_w = int(rng.integers(22, 37))

    key_colors = [int(x) for x in rng.choice(KEY_COLORS, size=n_pairs, replace=False)]
    used_signatures: set[tuple[int, str, tuple[int, ...]]] = set()
    objects: list[dict] = []

    for color in key_colors:
        sig = _sample_signature(rng, used_signatures)
        if sig is None:
            return None
        height, side, rows = sig
        lock_patch = _build_lock(height, side, rows)
        key_patch = _build_key(height, side, rows, color)
        filled_lock = lock_patch.copy()
        filled_lock[filled_lock == BG] = color
        objects.append({"in": lock_patch, "out": filled_lock, "area": int(lock_patch.shape[0] * lock_patch.shape[1])})
        objects.append({"in": key_patch, "out": np.full_like(key_patch, BG), "area": int(key_patch.shape[0] * key_patch.shape[1])})

    for _ in range(n_extra_locks):
        sig = _sample_signature(rng, used_signatures)
        if sig is None:
            return None
        height, side, rows = sig
        lock_patch = _build_lock(height, side, rows)
        objects.append({"in": lock_patch, "out": lock_patch.copy(), "area": int(lock_patch.shape[0] * lock_patch.shape[1])})

    reserved_area = sum((obj["in"].shape[0] + 2) * (obj["in"].shape[1] + 2) for obj in objects)
    if reserved_area > canvas_h * canvas_w:
        return None

    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    out = np.full((canvas_h, canvas_w), BG, dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)

    order = list(range(len(objects)))
    tie_break = rng.random(len(objects))
    order.sort(key=lambda idx: (-objects[idx]["area"], float(tie_break[idx])))
    for idx in order:
        obj = objects[idx]
        patch_in = obj["in"]
        patch_out = obj["out"]
        h, w = patch_in.shape
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = int(rng.integers(0, canvas_h - h + 1))
            c0 = int(rng.integers(0, canvas_w - w + 1))
            if not _fits(occupied, r0, c0, h, w, PLACEMENT_PADDING):
                continue
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            _paste(inp, patch_in, r0, c0, BG)
            _paste(out, patch_out, r0, c0, BG)
            placed = True
            break
        if not placed:
            return None

    inp, out = _random_crop(inp, out, rng, BG)
    if _most_common_color(inp) != BG:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
