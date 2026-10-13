from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BG_COLOR = 0
LOCK_COLOR = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _monochrome_components8(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors8(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    return comps


def _signature_from_key(mask: np.ndarray) -> tuple | None:
    h, w = mask.shape
    if h != w or h not in (2, 3):
        return None
    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return None

    missing = ~mask
    if h == 3 and bool(missing[1, 1]):
        return None
    for r, c in np.argwhere(missing):
        if r not in (0, h - 1) and c not in (0, w - 1):
            return None

    top = tuple(sorted(int(c) for r, c in np.argwhere(missing) if int(r) == 0))
    bottom = tuple(sorted(int(c) for r, c in np.argwhere(missing) if int(r) == h - 1))
    left = tuple(sorted(int(r) for r, c in np.argwhere(missing) if int(c) == 0))
    right = tuple(sorted(int(r) for r, c in np.argwhere(missing) if int(c) == w - 1))
    return (int(h), top, bottom, left, right)


def _signature_from_lock(mask: np.ndarray) -> tuple | None:
    h, w = mask.shape
    if h != w or h not in (4, 5):
        return None
    if not bool(mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
        return None

    interior = mask[1:-1, 1:-1]
    if bool(interior.all()):
        return None

    ih, iw = interior.shape
    for r, c in np.argwhere(interior):
        if r not in (0, ih - 1) and c not in (0, iw - 1):
            return None

    top = tuple(sorted(int(c) for r, c in np.argwhere(interior) if int(r) == 0))
    bottom = tuple(sorted(int(c) for r, c in np.argwhere(interior) if int(r) == ih - 1))
    left = tuple(sorted(int(r) for r, c in np.argwhere(interior) if int(c) == 0))
    right = tuple(sorted(int(r) for r, c in np.argwhere(interior) if int(c) == iw - 1))
    return (int(ih), top, bottom, left, right)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each notched key to the lock frame with the same edge-notch signature and fill that lock.

    Input structure:
    - The background is black.
    - Several small monochrome key sprites sit on the left side; each key is a 2x2 or 3x3 square with one or more 1-pixel edge notches removed.
    - Several grey hollow square lock frames sit on the right side; each lock has grey 1-pixel protrusions on its inner walls.
    - Keys and locks are separate monochrome connected components.

    Rule:
    - For every key, inspect the missing border cells inside its bounding square and record which side each notch touches and its relative position on that side.
    - For every lock, inspect the grey protrusions inside the frame and record the corresponding side and relative position on the inner wall.
    - Pair each key with the unique lock whose side-position signature matches exactly.
    - Fill the lock's empty interior cells with the key's color, leaving the key, the lock wall, and the protrusions unchanged.

    Key insight:
    The puzzle is solved by converting both keys and locks into the same side-relative notch signature, so spatial complement matching becomes a direct shape lookup.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    keys: list[dict] = []
    locks: list[dict] = []
    for comp in _monochrome_components8(g, bg):
        lock_sig = _signature_from_lock(comp["mask"])
        if lock_sig is not None:
            comp["signature"] = lock_sig
            locks.append(comp)
            continue

        key_sig = _signature_from_key(comp["mask"])
        if key_sig is not None:
            comp["signature"] = key_sig
            keys.append(comp)

    assert keys and locks and len(keys) == len(locks)

    lock_by_sig: dict[tuple, dict] = {}
    for lock in locks:
        sig = lock["signature"]
        assert sig not in lock_by_sig
        lock_by_sig[sig] = lock

    key_sigs = [key["signature"] for key in keys]
    assert len(set(key_sigs)) == len(key_sigs)
    assert set(key_sigs) == set(lock_by_sig)

    for key in keys:
        lock = lock_by_sig[key["signature"]]
        r0, c0, _r1, _c1 = lock["bbox"]
        interior = lock["mask"][1:-1, 1:-1]
        for rr, cc in np.argwhere(~interior):
            out[r0 + 1 + int(rr), c0 + 1 + int(cc)] = int(key["color"])

    return out


def _connected8(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c, h, w):
            if not bool(mask[nr, nc]) or (nr, nc) in seen:
                continue
            seen.add((int(nr), int(nc)))
            stack.append((int(nr), int(nc)))
    return len(seen) == int(mask.sum())


def _valid_missing_sets(size: int) -> list[tuple[tuple[int, int], ...]]:
    border = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if r in (0, size - 1) or c in (0, size - 1)
    ]
    if size == 2:
        return [((r, c),) for r, c in border]

    valid: list[tuple[tuple[int, int], ...]] = []
    for count in (1, 2, 3):
        for cells in combinations(border, count):
            mask = np.ones((size, size), dtype=bool)
            for r, c in cells:
                mask[r, c] = False
            if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                continue
            if not _connected8(mask):
                continue
            valid.append(tuple(sorted((int(r), int(c)) for r, c in cells)))
    return valid


SIGNATURE_POOLS = {
    size: [
        _signature_from_key(
            np.array(
                [
                    [False if (r, c) in missing else True for c in range(size)]
                    for r in range(size)
                ],
                dtype=bool,
            )
        )
        for missing in _valid_missing_sets(size)
    ]
    for size in (2, 3)
}


def _key_patch(size: int, color: int, signature: tuple, bg: int) -> np.ndarray:
    patch = np.full((size, size), int(color), dtype=int)
    _size, top, bottom, left, right = signature
    for c in top:
        patch[0, int(c)] = int(bg)
    for c in bottom:
        patch[size - 1, int(c)] = int(bg)
    for r in left:
        patch[int(r), 0] = int(bg)
    for r in right:
        patch[int(r), size - 1] = int(bg)
    return patch


def _lock_patch(size: int, signature: tuple, color: int, bg: int) -> np.ndarray:
    outer = size + 2
    patch = np.full((outer, outer), int(bg), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    _size, top, bottom, left, right = signature
    for c in top:
        patch[1, 1 + int(c)] = int(color)
    for c in bottom:
        patch[outer - 2, 1 + int(c)] = int(color)
    for r in left:
        patch[1 + int(r), 1] = int(color)
    for r in right:
        patch[1 + int(r), outer - 2] = int(color)
    return patch


def _fill_expected_lock(grid: np.ndarray, top: int, left: int, lock_patch: np.ndarray, key_color: int, bg: int) -> None:
    interior = lock_patch[1:-1, 1:-1]
    for rr, cc in np.argwhere(interior == int(bg)):
        grid[top + 1 + int(rr), left + 1 + int(cc)] = int(key_color)


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    size = random.choice((2, 3))
    signatures = random.sample(SIGNATURE_POOLS[size], random.randint(2, 4))
    key_colors = random.sample(KEY_COLORS, len(signatures))

    key_order = list(range(len(signatures)))
    lock_order = list(range(len(signatures)))
    random.shuffle(key_order)
    random.shuffle(lock_order)

    lock_size = size + 2
    band_gap = random.randint(1, 2)
    margin_y = random.randint(1, 2)
    grid_h = 2 * margin_y + len(signatures) * lock_size + (len(signatures) - 1) * band_gap

    left_zone_w = size + random.randint(3, 6)
    right_zone_w = lock_size + random.randint(3, 6)
    middle_gap = random.randint(3, 6)
    margin_x = random.randint(1, 2)
    grid_w = margin_x + left_zone_w + middle_gap + right_zone_w + margin_x
    assert grid_h <= 30 and grid_w <= 30

    left_x0 = margin_x
    right_x0 = margin_x + left_zone_w + middle_gap

    inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    out = inp.copy()

    key_positions: dict[int, tuple[int, int]] = {}
    lock_positions: dict[int, tuple[int, int]] = {}

    for band in range(len(signatures)):
        band_top = margin_y + band * (lock_size + band_gap)

        key_idx = key_order[band]
        key_col = left_x0 + random.randint(0, left_zone_w - size)
        key_row = band_top + random.randint(0, lock_size - size)
        key_positions[key_idx] = (int(key_row), int(key_col))

        lock_idx = lock_order[band]
        lock_col = right_x0 + random.randint(0, right_zone_w - lock_size)
        lock_row = band_top
        lock_positions[lock_idx] = (int(lock_row), int(lock_col))

    for idx, signature in enumerate(signatures):
        key_patch = _key_patch(size, key_colors[idx], signature, BG_COLOR)
        key_row, key_col = key_positions[idx]
        key_mask = key_patch != BG_COLOR
        inp[key_row : key_row + size, key_col : key_col + size][key_mask] = key_patch[key_mask]
        out[key_row : key_row + size, key_col : key_col + size][key_mask] = key_patch[key_mask]

    for idx, signature in enumerate(signatures):
        lock_patch = _lock_patch(size, signature, LOCK_COLOR, BG_COLOR)
        lock_row, lock_col = lock_positions[idx]
        lock_mask = lock_patch != BG_COLOR
        inp[lock_row : lock_row + lock_size, lock_col : lock_col + lock_size][lock_mask] = lock_patch[lock_mask]
        out[lock_row : lock_row + lock_size, lock_col : lock_col + lock_size][lock_mask] = lock_patch[lock_mask]
        _fill_expected_lock(out, lock_row, lock_col, lock_patch, key_colors[idx], BG_COLOR)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, expected = _build_example()
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
