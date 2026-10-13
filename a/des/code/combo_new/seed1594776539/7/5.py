import random
from collections import deque

import numpy as np


BG = 0
DIVIDER = 5
NOISE_LIMIT = 3
RETENTION_NUM = 7
RETENTION_DEN = 10
QUADRANT_ORDERS = (0, 1, 2, 3)  # TL, TR, BR, BL
FG_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _tight_bbox(mask: np.ndarray):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(r1), int(c0), int(c1)


def _quadrant_slices(side: int):
    return (
        (slice(0, side), slice(0, side)),
        (slice(0, side), slice(side + 1, 2 * side + 1)),
        (slice(side + 1, 2 * side + 1), slice(side + 1, 2 * side + 1)),
        (slice(side + 1, 2 * side + 1), slice(0, side)),
    )


def _find_dividers(grid: np.ndarray, bg: int):
    h, w = grid.shape
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    for r, color_r in row_candidates:
        for c, color_c in col_candidates:
            if color_r == color_c:
                return int(r), int(c), int(color_r)
    raise AssertionError("expected a full-row/full-column divider cross")


def _extract_patch_mask(patch: np.ndarray, bg: int):
    colors = [int(v) for v in np.unique(patch) if int(v) != bg]
    assert len(colors) == 1, "each quadrant must contain exactly one foreground color"
    color = colors[0]
    return patch == color, color


def _rot180(rc, h: int, w: int):
    r, c = rc
    return h - 1 - r, w - 1 - c


def _rot90(rc, n: int):
    r, c = rc
    return c, n - 1 - r


def _orbit(rc, order: int, h: int, w: int):
    if order == 2:
        return frozenset((rc, _rot180(rc, h, w)))
    if order == 4:
        assert h == w
        pts = set()
        cur = rc
        for _ in range(4):
            pts.add(cur)
            cur = _rot90(cur, h)
        return frozenset(pts)
    raise ValueError(order)


def _complete_orbits(inside: set[tuple[int, int]], order: int, h: int, w: int):
    keep = set()
    seen = set()
    for cell in inside:
        if cell in seen:
            continue
        orb = _orbit(cell, order, h, w)
        seen.update(orb)
        if orb.issubset(inside):
            keep.update(orb)
    return keep


def _best_symmetric_subset(mask: np.ndarray, order: int):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None

    total = int(len(pts))
    rows = sorted({int(r) for r in pts[:, 0]})
    cols = sorted({int(c) for c in pts[:, 1]})
    occupied = [(int(r), int(c)) for r, c in pts]
    best = None

    for i_r0, r0 in enumerate(rows):
        for r1 in rows[i_r0:]:
            h = r1 - r0 + 1
            for i_c0, c0 in enumerate(cols):
                for c1 in cols[i_c0:]:
                    w = c1 - c0 + 1
                    if order == 4 and h != w:
                        continue

                    inside = {
                        (r - r0, c - c0)
                        for r, c in occupied
                        if r0 <= r <= r1 and c0 <= c <= c1
                    }
                    if not inside:
                        continue

                    keep_rel = _complete_orbits(inside, order, h, w)
                    if not keep_rel:
                        continue

                    rel_rows = [r for r, _ in keep_rel]
                    rel_cols = [c for _, c in keep_rel]
                    if min(rel_rows) != 0 or max(rel_rows) != h - 1:
                        continue
                    if min(rel_cols) != 0 or max(rel_cols) != w - 1:
                        continue

                    keep_abs = {(r0 + r, c0 + c) for r, c in keep_rel}
                    count = len(keep_abs)
                    area = h * w
                    candidate = {
                        "cells": keep_abs,
                        "count": count,
                        "deletions": total - count,
                        "bbox_area": area,
                    }
                    if best is None:
                        best = candidate
                        continue
                    if count > best["count"]:
                        best = candidate
                        continue
                    if count == best["count"] and area < best["bbox_area"]:
                        best = candidate

    return best


def _render_cells(shape: tuple[int, int], cells: set[tuple[int, int]], color: int, bg: int):
    out = np.full(shape, bg, dtype=int)
    for r, c in cells:
        out[r, c] = color
    return out


def _candidate_ok(candidate, total: int):
    if candidate is None:
        return False
    if candidate["deletions"] > NOISE_LIMIT:
        return False
    return candidate["count"] * RETENTION_DEN >= total * RETENTION_NUM


def _classify_patch(patch: np.ndarray, bg: int):
    mask, color = _extract_patch_mask(patch, bg)
    total = int(mask.sum())

    best4 = _best_symmetric_subset(mask, 4)
    if _candidate_ok(best4, total):
        clean = _render_cells(mask.shape, best4["cells"], color, bg)
        return clean, 4

    best2 = _best_symmetric_subset(mask, 2)
    if _candidate_ok(best2, total):
        clean = _render_cells(mask.shape, best2["cells"], color, bg)
        return clean, 2

    return patch.copy(), 1


def main(grid: np.ndarray) -> np.ndarray:
    """Remove asymmetric noise from quadrant motifs, then copy each motif clockwise according to its rotational order.

    Input structure:
    - The background is black.
    - A grey full row and a grey full column cross at the grid center and split the canvas into four equal square quadrants.
    - Each quadrant contains one monochrome motif in a single non-background, non-grey color.
    - A motif may be a slightly corrupted version of a shape with 90-degree rotational symmetry, 180-degree rotational symmetry, or no non-trivial rotational symmetry.

    Rule:
    - For each quadrant independently, remove the smallest visible set of extra pixels that restores the highest plausible rotational symmetry of the motif around the center of its tight bounding box.
    - Classify the cleaned motif by its rotational order K: 4 for 90-degree symmetry, 2 for 180-degree symmetry, or 1 otherwise.
    - Build an output grid of the same size on a black background, without the grey divider. Keep the cleaned motif in its original quadrant and copy it clockwise into the next K-1 quadrants.
    - Each clockwise copy is rotated by 90 degrees relative to the previous one, and later source quadrants overwrite earlier ones in TL, TR, BR, BL order.

    Key insight:
    - The same rotational-consistency test serves two purposes: it filters out a few extra pixels inside each quadrant, and the recovered symmetry order directly determines how many rotated quadrant copies must be stamped into the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    mid_r, mid_c, _ = _find_dividers(g, bg)
    assert mid_r == mid_c, "quadrants must be square for 90-degree copying"
    side = mid_r
    assert g.shape == (2 * side + 1, 2 * side + 1)

    out = np.full_like(g, bg)
    quads = _quadrant_slices(side)

    for source_idx in QUADRANT_ORDERS:
        rs, cs = quads[source_idx]
        clean_patch, order = _classify_patch(g[rs, cs], bg)
        for step in range(order):
            target_idx = (source_idx + step) % 4
            trs, tcs = quads[target_idx]
            rotated = np.rot90(clean_patch, -step)
            target = out[trs, tcs]
            target[rotated != bg] = rotated[rotated != bg]

    return out


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(mask: np.ndarray):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(int(v) for v in pts[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray):
    bbox = _tight_bbox(mask)
    assert bbox is not None
    r0, r1, c0, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _center_connector(h: int, w: int):
    rows = [h // 2] if h % 2 == 1 else [h // 2 - 1, h // 2]
    cols = [w // 2] if w % 2 == 1 else [w // 2 - 1, w // 2]
    return {(r, c) for r in rows for c in cols}


def _paint_cells(mask: np.ndarray, cells):
    for r, c in cells:
        mask[r, c] = True


def _walk_to_top(size: int, mid: int):
    r, c = mid, mid
    cells = {(r, c)}
    while r > 0:
        choices = [(-1, 0)]
        if c < size - 1:
            choices.append((0, 1))
        if c > mid:
            choices.append((0, -1))
        dr, dc = random.choice(choices)
        nr, nc = r + dr, c + dc
        if 0 <= nr <= mid and mid <= nc < size:
            r, c = nr, nc
        else:
            r -= 1
        cells.add((r, c))
        if c > mid and random.random() < 0.35:
            cells.add((r, c - 1))
    return cells


def _make_order4_mask(side: int):
    max_size = min(side - 2, 9)
    sizes = [s for s in range(5, max_size + 1, 2)]
    if not sizes:
        sizes = [3]
    for _ in range(120):
        size = random.choice(sizes)
        mid = size // 2
        seed = _walk_to_top(size, mid)
        for _ in range(random.randint(0, size // 2)):
            sr, sc = random.choice(tuple(seed))
            candidates = [
                (nr, nc)
                for nr, nc in _neighbors4(sr, sc)
                if 0 <= nr <= mid and mid <= nc < size
            ]
            if candidates:
                seed.add(random.choice(candidates))

        full = set()
        for cell in seed:
            full.update(_orbit(cell, 4, size, size))
        mask = np.zeros((size, size), dtype=bool)
        _paint_cells(mask, full)
        if not _is_connected(mask):
            continue
        if mask.sum() < max(7, size + 2):
            continue
        return mask
    raise RuntimeError("failed to build an order-4 motif")


def _walk_axis_path(h: int, w: int, start: tuple[int, int], axis: str):
    r, c = start
    cells = {(r, c)}
    if axis == "top":
        while r > 0:
            moves = [(-1, 0)]
            if c > 0 and random.random() < 0.45:
                moves.append((0, -1))
            dr, dc = random.choice(moves)
            nr, nc = max(0, r + dr), max(0, c + dc)
            r, c = nr, nc
            cells.add((r, c))
    else:
        while c > 0:
            moves = [(0, -1)]
            if r > 0 and random.random() < 0.45:
                moves.append((-1, 0))
            dr, dc = random.choice(moves)
            nr, nc = max(0, r + dr), max(0, c + dc)
            r, c = nr, nc
            cells.add((r, c))
    return cells


def _make_order2_mask(side: int):
    max_h = min(side - 2, 8)
    max_w = min(side - 2, 9)
    for _ in range(160):
        h = random.randint(4, max_h)
        w = random.randint(5, max_w)
        if h == w:
            continue

        core = _center_connector(h, w)
        start = min(core)
        seed = set(core)
        seed.update(_walk_axis_path(h, w, start, "top"))
        seed.update(_walk_axis_path(h, w, start, "left"))

        for _ in range(random.randint(0, max(2, min(h, w) // 2))):
            sr, sc = random.choice(tuple(seed))
            candidates = [
                (nr, nc)
                for nr, nc in _neighbors4(sr, sc)
                if 0 <= nr < h and 0 <= nc < w and nr <= h // 2 and nc <= w // 2
            ]
            if candidates:
                seed.add(random.choice(candidates))

        full = set(seed)
        for cell in tuple(seed):
            full.update(_orbit(cell, 2, h, w))
        mask = np.zeros((h, w), dtype=bool)
        _paint_cells(mask, full)
        if not _is_connected(mask):
            continue
        if mask.sum() < 8:
            continue
        if _best_symmetric_subset(mask, 4) is not None and _best_symmetric_subset(mask, 4)["count"] == int(mask.sum()):
            continue
        return mask
    raise RuntimeError("failed to build an order-2 motif")


def _make_order1_mask(side: int):
    max_h = min(side - 2, 8)
    max_w = min(side - 2, 8)
    for _ in range(200):
        h = random.randint(4, max_h)
        w = random.randint(4, max_w)
        raw = np.zeros((h, w), dtype=bool)
        r = random.randrange(h)
        c = random.randrange(w)
        raw[r, c] = True
        target = random.randint(7, min(h * w, 12))
        while int(raw.sum()) < target:
            pts = np.argwhere(raw)
            sr, sc = random.choice(pts.tolist())
            candidates = [
                (nr, nc)
                for nr, nc in _neighbors4(sr, sc)
                if 0 <= nr < h and 0 <= nc < w and not raw[nr, nc]
            ]
            if not candidates:
                break
            nr, nc = random.choice(candidates)
            raw[nr, nc] = True
        mask = _crop_mask(raw)
        if min(mask.shape) < 3 or mask.sum() < 7:
            continue
        if not _is_connected(mask):
            continue
        return mask
    raise RuntimeError("failed to build an order-1 motif")


def _place_base_mask(base_mask: np.ndarray, side: int):
    h, w = base_mask.shape
    r0 = random.randint(1, side - h - 1)
    c0 = random.randint(1, side - w - 1)
    placed = np.zeros((side, side), dtype=bool)
    placed[r0 : r0 + h, c0 : c0 + w] = base_mask
    return placed


def _render_patch(mask: np.ndarray, color: int):
    return np.where(mask, color, BG).astype(int)


def _noise_candidates(clean_mask: np.ndarray):
    bbox = _tight_bbox(clean_mask)
    assert bbox is not None
    r0, r1, c0, c1 = bbox
    side = clean_mask.shape[0]
    cells = []
    for r in range(max(0, r0 - 1), min(side, r1 + 2)):
        for c in range(max(0, c0 - 1), min(side, c1 + 2)):
            if not clean_mask[r, c]:
                cells.append((r, c))
    random.shuffle(cells)
    return cells


def _add_noise(clean_patch: np.ndarray, order: int):
    color = int(clean_patch[clean_patch != BG][0])
    noisy = clean_patch.copy()
    clean_mask = noisy == color
    target_noise = random.randint(1, 2)
    added = 0

    for r, c in _noise_candidates(clean_mask):
        trial = noisy.copy()
        trial[r, c] = color
        recovered, got_order = _classify_patch(trial, BG)
        if got_order == order and np.array_equal(recovered, clean_patch):
            noisy = trial
            clean_mask[r, c] = True
            added += 1
            if added >= target_noise:
                break

    return noisy if added >= 1 else None


def _generate_quadrant(side: int, order: int, color: int):
    for _ in range(120):
        if order == 4:
            base = _make_order4_mask(side)
        elif order == 2:
            base = _make_order2_mask(side)
        else:
            base = _make_order1_mask(side)

        placed = _place_base_mask(base, side)
        clean_patch = _render_patch(placed, color)
        recovered, got_order = _classify_patch(clean_patch, BG)
        if got_order != order or not np.array_equal(recovered, clean_patch):
            continue

        if order == 1:
            return clean_patch

        noisy = _add_noise(clean_patch, order)
        if noisy is not None:
            return noisy

    raise RuntimeError(f"failed to build quadrant for order {order}")


def _generate_orders():
    orders = [1, 2, 4, random.choice([1, 2, 4])]
    random.shuffle(orders)
    return orders


def _gen_attempt():
    side = random.randint(8, 14)
    full = np.full((2 * side + 1, 2 * side + 1), BG, dtype=int)
    full[side, :] = DIVIDER
    full[:, side] = DIVIDER

    orders = _generate_orders()
    colors = random.sample(FG_POOL, 4)
    quads = _quadrant_slices(side)

    for idx, ((rs, cs), order, color) in enumerate(zip(quads, orders, colors, strict=False)):
        patch = _generate_quadrant(side, order, color)
        full[rs, cs] = patch

        recovered, got_order = _classify_patch(patch, BG)
        assert got_order == order
        if order != 1:
            assert not np.array_equal(patch, recovered)
        else:
            assert np.array_equal(patch, recovered)

    out = main(full)
    assert out.shape == full.shape
    assert max(full.shape) <= 30 and max(out.shape) <= 30
    return full


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except RuntimeError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
