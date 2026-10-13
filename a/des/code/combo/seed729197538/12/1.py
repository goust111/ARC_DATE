from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
LOCK = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _border_coords(h: int, w: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(w):
        border.add((0, c))
        border.add((h - 1, c))
    for r in range(h):
        border.add((r, 0))
        border.add((r, w - 1))
    return border


def _enclosed_bg(local: np.ndarray) -> list[tuple[int, int]]:
    h, w = local.shape
    bg_mask = local == BG
    seen = np.zeros((h, w), dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bool(bg_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(bg_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(bg_mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                queue.append((nr, nc))

    cavity = [(r, c) for r in range(h) for c in range(w) if bool(bg_mask[r, c]) and not bool(seen[r, c])]
    return cavity


def _extract_locks(grid: np.ndarray) -> list[dict[str, object]]:
    locks: list[dict[str, object]] = []
    for comp in _components(grid == LOCK):
        comp_set = set(comp)
        r0, c0, r1, c1 = _bbox(comp)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        border = {(r0 + rr, c0 + cc) for rr, cc in _border_coords(box_h, box_w)}
        gap = sorted(border - comp_set)
        assert gap, "each lock must have a missing border segment"

        gap_mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in gap:
            gap_mask[r - r0, c - c0] = True
        assert len(_components(gap_mask)) == 1, "lock gap must be a single connected notch"

        local = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)
        for r, c in gap:
            local[r - r0, c - c0] = LOCK
        cavity_local = _enclosed_bg(local)
        assert cavity_local, "patched lock must enclose a cavity"

        locks.append(
            {
                "gap": set(gap),
                "sig": _normalize(gap),
                "cavity": {(r0 + r, c0 + c) for r, c in cavity_local},
            }
        )
    return locks


def _extract_keys(grid: np.ndarray, sig_to_lock: dict[tuple[tuple[int, int], ...], int]) -> list[dict[str, object]]:
    keys: list[dict[str, object]] = []
    claimed: set[int] = set()
    for comp in _components((grid != BG) & (grid != LOCK)):
        colors = sorted({int(grid[r, c]) for r, c in comp})
        assert len(colors) == 2, "each key must use exactly two colors"

        matches: list[tuple[int, int]] = []
        for color in colors:
            shape = [(r, c) for r, c in comp if int(grid[r, c]) == color]
            sig = _normalize(shape)
            if sig in sig_to_lock:
                matches.append((color, sig_to_lock[sig]))

        assert len(matches) == 1, "exactly one key color-region should match a lock notch"
        outline_color, lock_index = matches[0]
        fill_color = next(color for color in colors if color != outline_color)
        assert lock_index not in claimed, "each lock should be claimed by exactly one key"
        claimed.add(lock_index)
        keys.append({"cells": comp, "lock_index": lock_index, "fill_color": fill_color})

    assert len(keys) == len(sig_to_lock), "number of keys must match number of locks"
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each two-color key's outline shape to patch the matching lock gap, then
    fill that lock's enclosed cavity with the key's interior color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    locks = _extract_locks(g)
    assert locks, "expected at least one gray lock"

    sig_to_lock: dict[tuple[tuple[int, int], ...], int] = {}
    for i, lock in enumerate(locks):
        sig = lock["sig"]
        assert sig not in sig_to_lock, "lock notch shapes must be unique"
        sig_to_lock[sig] = i

    keys = _extract_keys(g, sig_to_lock)

    out = np.array(g, copy=True)
    for key in keys:
        for r, c in key["cells"]:
            out[r, c] = BG

    for key in keys:
        lock = locks[int(key["lock_index"])]
        for r, c in lock["gap"]:
            out[r, c] = LOCK
        for r, c in lock["cavity"]:
            out[r, c] = int(key["fill_color"])

    return out


# --------------- generator ---------------

_SPEC_DEFS = [
    {"name": "top_h3", "anchor": "top", "outline": ((0, 0), (0, 1), (0, 2)), "fill": ((1, 1),)},
    {"name": "bottom_h3", "anchor": "bottom", "outline": ((0, 0), (0, 1), (0, 2)), "fill": ((-1, 1),)},
    {"name": "top_h4", "anchor": "top", "outline": ((0, 0), (0, 1), (0, 2), (0, 3)), "fill": ((1, 1), (1, 2))},
    {"name": "bottom_h4", "anchor": "bottom", "outline": ((0, 0), (0, 1), (0, 2), (0, 3)), "fill": ((-1, 1), (-1, 2))},
    {"name": "left_v3", "anchor": "left", "outline": ((0, 0), (1, 0), (2, 0)), "fill": ((1, 1),)},
    {"name": "right_v3", "anchor": "right", "outline": ((0, 0), (1, 0), (2, 0)), "fill": ((1, -1),)},
    {"name": "left_v4", "anchor": "left", "outline": ((0, 0), (1, 0), (2, 0), (3, 0)), "fill": ((1, 1), (2, 1))},
    {"name": "right_v4", "anchor": "right", "outline": ((0, 0), (1, 0), (2, 0), (3, 0)), "fill": ((1, -1), (2, -1))},
    {"name": "tl3", "anchor": "tl", "outline": ((0, 0), (0, 1), (1, 0)), "fill": ((1, 1),)},
    {"name": "tr3", "anchor": "tr", "outline": ((0, 0), (0, 1), (1, 1)), "fill": ((1, 0),)},
    {"name": "bl3", "anchor": "bl", "outline": ((0, 0), (1, 0), (1, 1)), "fill": ((0, 1),)},
    {"name": "br3", "anchor": "br", "outline": ((0, 1), (1, 0), (1, 1)), "fill": ((0, 0),)},
    {"name": "tl4", "anchor": "tl", "outline": ((0, 0), (0, 1), (0, 2), (1, 0)), "fill": ((1, 1),)},
    {"name": "tr4", "anchor": "tr", "outline": ((0, 0), (0, 1), (0, 2), (1, 2)), "fill": ((1, 1),)},
    {"name": "bl4", "anchor": "bl", "outline": ((0, 0), (1, 0), (1, 1), (1, 2)), "fill": ((0, 1),)},
    {"name": "br4", "anchor": "br", "outline": ((0, 2), (1, 0), (1, 1), (1, 2)), "fill": ((0, 1),)},
]

GAP_SPECS = [
    {
        **spec,
        "sig": _normalize(spec["outline"]),
        "shape_h": max(r for r, _ in spec["outline"]) + 1,
        "shape_w": max(c for _, c in spec["outline"]) + 1,
    }
    for spec in _SPEC_DEFS
]

GEN_RETRIES = 400
LOCK_COUNT_RANGE = (1, 3)
LOCK_HEIGHT_RANGE = (6, 11)
LOCK_WIDTH_RANGE = (6, 12)
GRID_MIN_BASE = 16
GRID_PER_LOCK = 5
GRID_EXTRA = 4
PLACEMENT_PADDING = 1
EDGE_MARGIN = 1


def _make_lock_patch(spec: dict[str, object]) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    min_h = max(LOCK_HEIGHT_RANGE[0], int(spec["shape_h"]) + 4)
    min_w = max(LOCK_WIDTH_RANGE[0], int(spec["shape_w"]) + 4)
    max_h = max(min_h, LOCK_HEIGHT_RANGE[1])
    max_w = max(min_w, LOCK_WIDTH_RANGE[1])
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = LOCK
    patch[-1, :] = LOCK
    patch[:, 0] = LOCK
    patch[:, -1] = LOCK

    shape_h = int(spec["shape_h"])
    shape_w = int(spec["shape_w"])
    anchor = str(spec["anchor"])

    if anchor == "top":
        r_off = 0
        c_off = random.randint(1, w - shape_w - 1)
    elif anchor == "bottom":
        r_off = h - shape_h
        c_off = random.randint(1, w - shape_w - 1)
    elif anchor == "left":
        r_off = random.randint(1, h - shape_h - 1)
        c_off = 0
    elif anchor == "right":
        r_off = random.randint(1, h - shape_h - 1)
        c_off = w - shape_w
    elif anchor == "tl":
        r_off = 0
        c_off = 0
    elif anchor == "tr":
        r_off = 0
        c_off = w - shape_w
    elif anchor == "bl":
        r_off = h - shape_h
        c_off = 0
    else:
        r_off = h - shape_h
        c_off = w - shape_w

    gap = {(r_off + r, c_off + c) for r, c in spec["outline"]}
    for r, c in gap:
        patch[r, c] = BG

    cavity = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
    return patch, gap, cavity


def _make_key_patch(spec: dict[str, object], outline_color: int, fill_color: int) -> np.ndarray:
    outline = list(spec["outline"])
    fill = list(spec["fill"])
    all_cells = outline + fill
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)

    patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c in outline:
        patch[r - min_r, c - min_c] = outline_color
    for r, c in fill:
        patch[r - min_r, c - min_c] = fill_color
    return patch


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = PLACEMENT_PADDING) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    for dr, dc in np.argwhere(patch != BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != BG):
            return False
    return True


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for dr, dc in np.argwhere(patch != BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        canvas[rr, cc] = int(patch[dr, dc])
        cells.append((rr, cc))
    return cells


def _random_positions(canvas: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    rows = list(range(EDGE_MARGIN, H - h - EDGE_MARGIN + 1))
    cols = list(range(EDGE_MARGIN, W - w - EDGE_MARGIN + 1))
    random.shuffle(rows)
    random.shuffle(cols)
    return [(r, c) for r in rows for c in cols]


def _expected_output(inp: np.ndarray, locks: list[dict[str, object]], keys: list[dict[str, object]]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for key in keys:
        for r, c in key["cells"]:
            out[r, c] = BG
    for lock, key in zip(locks, keys, strict=False):
        for r, c in lock["gap"]:
            out[r, c] = LOCK
        for r, c in lock["cavity"]:
            out[r, c] = int(key["fill_color"])
    return out


def _pick_specs(n_locks: int) -> list[dict[str, object]]:
    specs = GAP_SPECS[:]
    random.shuffle(specs)
    chosen: list[dict[str, object]] = []
    used_sigs: set[tuple[tuple[int, int], ...]] = set()
    for spec in specs:
        sig = spec["sig"]
        if sig in used_sigs:
            continue
        chosen.append(spec)
        used_sigs.add(sig)
        if len(chosen) == n_locks:
            return chosen
    raise RuntimeError("failed to pick enough unique gap shapes")


def _gen_attempt() -> np.ndarray | None:
    n_locks = random.randint(*LOCK_COUNT_RANGE)
    side_base = GRID_MIN_BASE + GRID_PER_LOCK * n_locks
    H = random.randint(side_base, side_base + GRID_EXTRA)
    W = random.randint(side_base, side_base + GRID_EXTRA)
    canvas = np.zeros((H, W), dtype=int)

    specs = _pick_specs(n_locks)
    color_pool = random.sample(PALETTE, 2 * n_locks)
    random.shuffle(color_pool)

    locks_meta: list[dict[str, object]] = []
    for spec in specs:
        lock_patch, gap_local, cavity_local = _make_lock_patch(spec)
        positions = _random_positions(canvas, lock_patch)
        placed = False
        for r0, c0 in positions:
            if not _can_place(canvas, lock_patch, r0, c0):
                continue
            _place_patch(canvas, lock_patch, r0, c0)
            locks_meta.append(
                {
                    "gap": {(r0 + r, c0 + c) for r, c in gap_local},
                    "cavity": {(r0 + r, c0 + c) for r, c in cavity_local},
                    "sig": spec["sig"],
                }
            )
            placed = True
            break
        if not placed:
            return None

    keys_meta: list[dict[str, object]] = []
    for spec, lock in zip(specs, locks_meta, strict=False):
        outline_color = color_pool.pop()
        fill_color = color_pool.pop()
        key_patch = _make_key_patch(spec, outline_color, fill_color)
        positions = _random_positions(canvas, key_patch)
        placed = False
        for r0, c0 in positions:
            if not _can_place(canvas, key_patch, r0, c0):
                continue
            cells = _place_patch(canvas, key_patch, r0, c0)
            keys_meta.append({"cells": cells, "fill_color": fill_color, "sig": lock["sig"]})
            placed = True
            break
        if not placed:
            return None

    inp = canvas
    expected = _expected_output(inp, locks_meta, keys_meta)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
