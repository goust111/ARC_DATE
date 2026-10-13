import random
from collections import Counter, deque

import numpy as np


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells):
    r0, c0, _, _ = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _component_mask(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _nonbg_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_connected(cells) -> bool:
    cells = set(cells)
    if not cells:
        return False
    q = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _enclosed_bg(bbox, occupied):
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    occ = np.zeros((h, w), dtype=bool)
    for r, c in occupied:
        occ[r - r0, c - c0] = True

    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if occ[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not occ[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    enclosed = set()
    for r in range(h):
        for c in range(w):
            if not occ[r, c] and not seen[r, c]:
                enclosed.add((r0 + r, c0 + c))
    return enclosed


def _classify_components(grid: np.ndarray, bg: int):
    keys = {}
    locks = {}
    for cells in _nonbg_components(grid, bg):
        values = [int(grid[r, c]) for r, c in cells]
        counts = Counter(values)
        r0, c0, r1, c1 = _bbox(cells)

        if len(counts) == 2 and sorted(counts.values()) == [1, len(cells) - 1]:
            major_color, major_count = counts.most_common(1)[0]
            assert major_count >= 2
            marker_color = next(color for color, cnt in counts.items() if cnt == 1)
            marker_cell = next((r, c) for r, c in cells if int(grid[r, c]) == marker_color)
            keys[major_color] = {
                "cells": set(cells),
                "shape": _normalize(cells),
                "bbox": (r0, c0, r1, c1),
                "marker_color": marker_color,
                "marker_cell": marker_cell,
            }
            continue

        if len(counts) != 1:
            continue

        color = next(iter(counts))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        cell_set = set(cells)
        perimeter = _perimeter_cells(r0, c0, r1, c1)
        if not cell_set <= perimeter:
            continue
        missing = perimeter - cell_set
        if not missing or not _is_connected(missing):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        if np.any(interior != bg):
            continue
        locks[color] = {
            "cells": cell_set,
            "bbox": (r0, c0, r1, c1),
            "perimeter": perimeter,
            "missing": missing,
        }
    return keys, locks


def _find_placement(grid: np.ndarray, bg: int, lock, key):
    r0, c0, r1, c1 = lock["bbox"]
    lock_cells = lock["cells"]
    missing = lock["missing"]
    key_shape = list(key["shape"])
    kh = 1 + max(r for r, _ in key_shape)
    kw = 1 + max(c for _, c in key_shape)

    candidates = []
    for tr in range(r0, r1 - kh + 2):
        for tc in range(c0, c1 - kw + 2):
            placed = {(tr + dr, tc + dc) for dr, dc in key_shape}
            if not missing <= placed:
                continue
            if placed & lock_cells:
                continue
            if any(int(grid[r, c]) != bg for r, c in placed):
                continue
            occupied = lock_cells | placed
            if not lock["perimeter"] <= occupied:
                continue
            if not any(
                any((nr, nc) in lock_cells for nr, nc in neighbors4(r, c))
                for r, c in placed
            ):
                continue
            enclosed = _enclosed_bg(lock["bbox"], occupied)
            if not enclosed:
                continue
            candidates.append((placed, enclosed))

    assert len(candidates) == 1, "expected a unique key placement"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each colored key with the lock frame of the same color, insert the key
    into the lock's notch, then fill the enclosed interior with the key's marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = 0

    keys, locks = _classify_components(g, bg)
    assert keys and locks and set(keys) == set(locks)

    out = g.copy()
    for color in sorted(keys):
        key = keys[color]
        lock = locks[color]
        placed, enclosed = _find_placement(g, bg, lock, key)

        for r, c in key["cells"]:
            out[r, c] = bg
        for r, c in placed:
            out[r, c] = color
        for r, c in enclosed:
            if out[r, c] == bg:
                out[r, c] = key["marker_color"]

    return out


GEN_RETRIES = 400
GRID_H_BY_COUNT = {1: (16, 22), 2: (20, 28), 3: (24, 30)}
GRID_W_BY_COUNT = {1: (16, 22), 2: (20, 28), 3: (24, 30)}
PAIR_COUNT_CHOICES = [1, 2, 3]
FRAME_H_RANGE = (6, 11)
FRAME_W_RANGE = (6, 11)
SPAN_RANGE = (3, 5)
DEPTH_RANGE = (2, 4)
PLACE_TRIES = 80
PADDING = 1
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ["top", "bottom", "left", "right"]


def _canonical_key(depth: int, span: int, rng: random.Random):
    mask = np.zeros((depth, span), dtype=bool)
    mask[0, :] = True
    left, right = 0, span - 1
    for r in range(1, depth):
        new_left = rng.randint(max(0, left - 1), min(left + 1, right))
        new_right = rng.randint(max(new_left, right - 1), min(span - 1, right + 1))
        mask[r, new_left : new_right + 1] = True
        left, right = new_left, new_right

    marker_candidates = list(map(tuple, np.argwhere(mask)))
    marker_candidates = [pt for pt in marker_candidates if pt[0] > 0]
    preferred = [
        pt
        for pt in marker_candidates
        if 0 < pt[0] < depth - 1 and 0 < pt[1] < span - 1
    ]
    marker = rng.choice(preferred or marker_candidates)
    return mask, marker


def _orient(mask: np.ndarray, marker, side: str):
    marker_mask = np.zeros_like(mask, dtype=bool)
    marker_mask[marker] = True
    contact_mask = np.zeros_like(mask, dtype=bool)
    contact_mask[0, :] = mask[0, :]

    if side == "top":
        out_mask = mask
        out_marker = marker_mask
        out_contact = contact_mask
    elif side == "bottom":
        out_mask = np.flipud(mask)
        out_marker = np.flipud(marker_mask)
        out_contact = np.flipud(contact_mask)
    elif side == "left":
        out_mask = np.rot90(mask, 1)
        out_marker = np.rot90(marker_mask, 1)
        out_contact = np.rot90(contact_mask, 1)
    else:
        out_mask = np.rot90(mask, -1)
        out_marker = np.rot90(marker_mask, -1)
        out_contact = np.rot90(contact_mask, -1)

    marker_pos = tuple(map(int, np.argwhere(out_marker)[0]))
    return out_mask, marker_pos, out_contact


def _expanded_slices(h: int, w: int, r0: int, c0: int, rh: int, cw: int, pad: int):
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + rh + pad)
    cc1 = min(w, c0 + cw + pad)
    return rr0, rr1, cc0, cc1


def _can_place(occ: np.ndarray, r0: int, c0: int, rh: int, cw: int, pad: int):
    h, w = occ.shape
    rr0, rr1, cc0, cc1 = _expanded_slices(h, w, r0, c0, rh, cw, pad)
    return not occ[rr0:rr1, cc0:cc1].any()


def _reserve(occ: np.ndarray, r0: int, c0: int, rh: int, cw: int, pad: int):
    h, w = occ.shape
    rr0, rr1, cc0, cc1 = _expanded_slices(h, w, r0, c0, rh, cw, pad)
    occ[rr0:rr1, cc0:cc1] = True


def _draw_lock(grid: np.ndarray, r0: int, c0: int, fh: int, fw: int, color: int, missing_rel):
    r1 = r0 + fh - 1
    c1 = c0 + fw - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    for dr, dc in missing_rel:
        grid[r0 + dr, c0 + dc] = 0


def _draw_key(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int, marker, marker_color: int):
    pts = np.argwhere(mask)
    for dr, dc in pts:
        grid[r0 + int(dr), c0 + int(dc)] = color
    mr, mc = marker
    grid[r0 + mr, c0 + mc] = marker_color


def _sample_pair_spec(rng: random.Random):
    side = rng.choice(SIDES)
    for _ in range(PLACE_TRIES):
        fh = rng.randint(*FRAME_H_RANGE)
        fw = rng.randint(*FRAME_W_RANGE)
        side_len = fw if side in ("top", "bottom") else fh
        orth_len = fh if side in ("top", "bottom") else fw
        max_span = min(SPAN_RANGE[1], side_len - 2)
        max_depth = min(DEPTH_RANGE[1], orth_len - 2)
        if max_span < SPAN_RANGE[0] or max_depth < DEPTH_RANGE[0]:
            continue
        span = rng.randint(SPAN_RANGE[0], max_span)
        depth = rng.randint(DEPTH_RANGE[0], max_depth)
        mask, marker = _canonical_key(depth, span, rng)
        oriented, oriented_marker, contact = _orient(mask, marker, side)
        kh, kw = oriented.shape
        if side in ("top", "bottom"):
            if fw - kw - 1 < 1:
                continue
            offset = rng.randint(1, fw - kw - 1)
            key_rel_r = 0 if side == "top" else fh - kh
            key_rel_c = offset
        else:
            if fh - kh - 1 < 1:
                continue
            offset = rng.randint(1, fh - kh - 1)
            key_rel_r = offset
            key_rel_c = 0 if side == "left" else fw - kw

        missing_rel = {
            (key_rel_r + int(dr), key_rel_c + int(dc))
            for dr, dc in np.argwhere(contact)
        }
        placed_rel = {
            (key_rel_r + int(dr), key_rel_c + int(dc))
            for dr, dc in np.argwhere(oriented)
        }
        bbox = (0, 0, fh - 1, fw - 1)
        perimeter = _perimeter_cells(*bbox)
        lock_cells = perimeter - missing_rel
        enclosed = _enclosed_bg(bbox, lock_cells | placed_rel)
        if enclosed:
            return {
                "frame_h": fh,
                "frame_w": fw,
                "mask": oriented,
                "marker": oriented_marker,
                "missing_rel": missing_rel,
            }
    return None


def _gen_attempt(rng: random.Random):
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    h = rng.randint(*GRID_H_BY_COUNT[pair_count])
    w = rng.randint(*GRID_W_BY_COUNT[pair_count])
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    colors = rng.sample(PALETTE, 2 * pair_count)
    lock_colors = colors[:pair_count]
    marker_colors = colors[pair_count:]

    for idx in range(pair_count):
        spec = _sample_pair_spec(rng)
        if spec is None:
            return None

        fh = spec["frame_h"]
        fw = spec["frame_w"]
        kh, kw = spec["mask"].shape

        lock_pos = None
        for _ in range(PLACE_TRIES):
            lr0 = rng.randint(0, h - fh)
            lc0 = rng.randint(0, w - fw)
            if _can_place(occ, lr0, lc0, fh, fw, PADDING):
                lock_pos = (lr0, lc0)
                break
        if lock_pos is None:
            return None

        key_pos = None
        for _ in range(PLACE_TRIES):
            kr0 = rng.randint(0, h - kh)
            kc0 = rng.randint(0, w - kw)
            if _can_place(occ, kr0, kc0, kh, kw, PADDING):
                key_pos = (kr0, kc0)
                break
        if key_pos is None:
            return None

        lock_color = lock_colors[idx]
        marker_color = marker_colors[idx]
        _draw_lock(grid, lock_pos[0], lock_pos[1], fh, fw, lock_color, spec["missing_rel"])
        _draw_key(grid, key_pos[0], key_pos[1], spec["mask"], lock_color, spec["marker"], marker_color)
        _reserve(occ, lock_pos[0], lock_pos[1], fh, fw, PADDING)
        _reserve(occ, key_pos[0], key_pos[1], kh, kw, PADDING)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
