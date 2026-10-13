import random
from collections import deque

import numpy as np


BG = 0
ORANGE = 7
PALETTE = tuple(c for c in range(10) if c not in (BG, ORANGE))
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _normalize(coords):
    r0, c0, _, _ = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _rect_cells(bbox):
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _lock_from_component(grid: np.ndarray, color: int, pts):
    r0, c0, r1, c1 = _bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((sub != BG) & (sub != color)):
        return None

    hole_mask = sub == BG
    if not np.any(hole_mask):
        return None

    holes = _components(hole_mask)
    if len(holes) != 1:
        return None

    hole = holes[0]
    hr0, hc0, hr1, hc1 = _bbox(hole)
    if len(hole) != (hr1 - hr0 + 1) * (hc1 - hc0 + 1):
        return None
    if len(pts) + len(hole) != h * w:
        return None

    touches = []
    if hr0 == 0:
        touches.append("top")
    if hr1 == h - 1:
        touches.append("bottom")
    if hc0 == 0:
        touches.append("left")
    if hc1 == w - 1:
        touches.append("right")
    if len(touches) != 1:
        return None

    side = touches[0]
    if side in ("top", "bottom"):
        if hc0 == 0 or hc1 == w - 1:
            return None
    else:
        if hr0 == 0 or hr1 == h - 1:
            return None

    return {
        "color": int(color),
        "cells": tuple((int(r), int(c)) for r, c in pts),
        "cell_set": frozenset((int(r), int(c)) for r, c in pts),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "notch_bbox": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
    }


def _find_locks(grid: np.ndarray):
    locks = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, ORANGE):
            continue
        for pts in _components(grid == color):
            lock = _lock_from_component(grid, color, pts)
            if lock is not None:
                locks.append(lock)

    by_color = {}
    for lock in locks:
        assert lock["color"] not in by_color, "lock colors must be unique"
        by_color[lock["color"]] = lock
    assert by_color, "expected at least one lock"
    return locks, by_color


def _find_indicators(grid: np.ndarray):
    indicators = []
    for pts in _components(grid == ORANGE):
        if len(pts) != 3:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        indicators.append(
            {
                "cells": tuple((int(r), int(c)) for r, c in pts),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return indicators


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _indicator_matches(lock, indicator) -> bool:
    ir0, ic0, ir1, ic1 = indicator["bbox"]
    r0, c0, r1, c1 = lock["bbox"]
    nr0, nc0, nr1, nc1 = lock["notch_bbox"]
    side = lock["side"]
    if side == "top":
        return ir1 == r0 - 2 and _ranges_overlap(ic0, ic1, nc0 - 1, nc1 + 1)
    if side == "bottom":
        return ir0 == r1 + 2 and _ranges_overlap(ic0, ic1, nc0 - 1, nc1 + 1)
    if side == "left":
        return ic1 == c0 - 2 and _ranges_overlap(ir0, ir1, nr0 - 1, nr1 + 1)
    return ic0 == c1 + 2 and _ranges_overlap(ir0, ir1, nr0 - 1, nr1 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    locks, lock_by_color = _find_locks(g)
    indicators = _find_indicators(g)
    assert len(indicators) == len(locks), "expected one indicator per lock"

    used_indicators = set()
    for lock in locks:
        matches = [i for i, ind in enumerate(indicators) if _indicator_matches(lock, ind)]
        assert len(matches) == 1, "each lock must have one nearby orange L indicator"
        used_indicators.add(matches[0])
    assert len(used_indicators) == len(indicators), "each indicator must belong to a lock"

    lock_sets = {lock["cell_set"] for lock in locks}
    parts = []
    cell_to_part = {}
    for color in np.unique(g):
        color = int(color)
        if color in (BG, ORANGE):
            continue
        for pts in _components(g == color):
            pt_set = frozenset((int(r), int(c)) for r, c in pts)
            if pt_set in lock_sets:
                continue
            part = {
                "color": color,
                "coords": tuple((int(r), int(c)) for r, c in pts),
                "area": len(pts),
            }
            part["kind"] = "marker" if len(pts) == 1 and color in lock_by_color else "key"
            idx = len(parts)
            parts.append(part)
            for cell in part["coords"]:
                cell_to_part[cell] = idx

    markers = [part for part in parts if part["kind"] == "marker"]
    keys = [part for part in parts if part["kind"] == "key"]
    assert len(markers) == len(locks), "expected one marker per lock"

    out = g.copy()
    out[g == ORANGE] = BG

    used_keys = set()
    matched_locks = set()
    for marker in markers:
        (mr, mc) = marker["coords"][0]
        key_ids = set()
        for nr, nc in _neighbors4(mr, mc):
            idx = cell_to_part.get((nr, nc))
            if idx is None:
                continue
            if parts[idx]["kind"] != "key":
                continue
            key_ids.add(idx)
        assert len(key_ids) == 1, "each marker must touch exactly one key"

        key_id = next(iter(key_ids))
        assert key_id not in used_keys, "keys must not be shared"
        key = parts[key_id]
        lock_color = marker["color"]
        lock = lock_by_color[lock_color]
        assert lock_color not in matched_locks, "locks must not be reused"

        notch_shape = _normalize(_rect_cells(lock["notch_bbox"]))
        assert _normalize(key["coords"]) == notch_shape, "key shape must match lock notch"

        for r, c in key["coords"]:
            out[r, c] = BG
        out[mr, mc] = BG

        r0, c0, r1, c1 = lock["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = lock_color
        out[r0 + 1 : r1, c0 + 1 : c1] = key["color"]
        nr0, nc0, nr1, nc1 = lock["notch_bbox"]
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = key["color"]

        used_keys.add(key_id)
        matched_locks.add(lock_color)

    assert len(used_keys) == len(keys), "all keys must be used"
    assert len(matched_locks) == len(locks), "all locks must be filled"
    return out


def _make_lock_spec(rng: random.Random, lock_color: int, key_color: int):
    h = rng.randint(5, 8)
    w = rng.randint(5, 8)
    side = rng.choice(SIDES)

    if side in ("top", "bottom"):
        notch_w = rng.randint(1, min(2, w - 2))
        notch_d = rng.randint(1, min(3, h - 2))
        notch_c0 = rng.randint(1, w - notch_w - 1)
        if side == "top":
            notch_bbox = (0, notch_c0, notch_d - 1, notch_c0 + notch_w - 1)
        else:
            notch_bbox = (h - notch_d, notch_c0, h - 1, notch_c0 + notch_w - 1)
        key_shape = (notch_d, notch_w)
        ind_c = notch_c0
        if side == "top":
            indicator = [(-3, ind_c), (-2, ind_c), (-2, ind_c + 1)]
        else:
            indicator = [(h + 1, ind_c), (h + 1, ind_c + 1), (h + 2, ind_c + 1)]
    else:
        notch_h = rng.randint(1, min(2, h - 2))
        notch_d = rng.randint(1, min(3, w - 2))
        notch_r0 = rng.randint(1, h - notch_h - 1)
        if side == "left":
            notch_bbox = (notch_r0, 0, notch_r0 + notch_h - 1, notch_d - 1)
        else:
            notch_bbox = (notch_r0, w - notch_d, notch_r0 + notch_h - 1, w - 1)
        key_shape = (notch_h, notch_d)
        ind_r = notch_r0
        if side == "left":
            indicator = [(ind_r, -3), (ind_r + 1, -3), (ind_r + 1, -2)]
        else:
            indicator = [(ind_r, w + 1), (ind_r, w + 2), (ind_r + 1, w + 2)]

    nr0, nc0, nr1, nc1 = notch_bbox
    lock_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not (nr0 <= r <= nr1 and nc0 <= c <= nc1)
    ]

    return {
        "lock_color": lock_color,
        "key_color": key_color,
        "side": side,
        "lock_cells_rel": lock_cells,
        "indicator_rel": indicator,
        "key_shape": key_shape,
        "union_rel": lock_cells + indicator,
    }


def _make_key_spec(rng: random.Random, key_shape, key_color: int, marker_color: int):
    h, w = key_shape
    key_cells = [(r, c) for r in range(h) for c in range(w)]
    marker_side = rng.choice(SIDES)
    if marker_side == "top":
        marker = (-1, rng.randrange(w))
    elif marker_side == "bottom":
        marker = (h, rng.randrange(w))
    elif marker_side == "left":
        marker = (rng.randrange(h), -1)
    else:
        marker = (rng.randrange(h), w)
    return {
        "key_color": key_color,
        "marker_color": marker_color,
        "key_cells_rel": key_cells,
        "marker_rel": marker,
        "union_rel": key_cells + [marker],
    }


def _place_shape(rng: random.Random, blocked: np.ndarray, rel_cells):
    h, w = blocked.shape
    min_r, min_c, max_r, max_c = _bbox(rel_cells)
    lo_r = -min_r
    hi_r = h - 1 - max_r
    lo_c = -min_c
    hi_c = w - 1 - max_c
    if lo_r > hi_r or lo_c > hi_c:
        return None

    for _ in range(160):
        off_r = rng.randint(lo_r, hi_r)
        off_c = rng.randint(lo_c, hi_c)
        abs_cells = [(r + off_r, c + off_c) for r, c in rel_cells]
        r0, c0, r1, c1 = _bbox(abs_cells)
        er0 = max(0, r0 - 1)
        ec0 = max(0, c0 - 1)
        er1 = min(h - 1, r1 + 1)
        ec1 = min(w - 1, c1 + 1)
        if blocked[er0 : er1 + 1, ec0 : ec1 + 1].any():
            continue
        blocked[er0 : er1 + 1, ec0 : ec1 + 1] = True
        return off_r, off_c
    return None


def _size_range(num_locks: int):
    ranges = {
        1: (18, 24),
        2: (22, 27),
        3: (24, 29),
        4: (27, 30),
    }
    return ranges[num_locks]


def _gen_attempt(rng: random.Random):
    num_locks = rng.randint(1, 4)
    lo, hi = _size_range(num_locks)
    h = rng.randint(lo, hi)
    w = rng.randint(lo, hi)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors = list(PALETTE)
    rng.shuffle(colors)
    lock_colors = colors[:num_locks]
    key_colors = colors[num_locks : 2 * num_locks]

    lock_specs = [_make_lock_spec(rng, lock_colors[i], key_colors[i]) for i in range(num_locks)]
    lock_specs.sort(key=lambda spec: len(spec["union_rel"]), reverse=True)

    for spec in lock_specs:
        offset = _place_shape(rng, blocked, spec["union_rel"])
        if offset is None:
            return None
        off_r, off_c = offset
        for r, c in spec["lock_cells_rel"]:
            grid[r + off_r, c + off_c] = spec["lock_color"]
        for r, c in spec["indicator_rel"]:
            grid[r + off_r, c + off_c] = ORANGE

    rng.shuffle(lock_specs)
    for spec in lock_specs:
        key_spec = _make_key_spec(rng, spec["key_shape"], spec["key_color"], spec["lock_color"])
        offset = _place_shape(rng, blocked, key_spec["union_rel"])
        if offset is None:
            return None
        off_r, off_c = offset
        for r, c in key_spec["key_cells_rel"]:
            grid[r + off_r, c + off_c] = key_spec["key_color"]
        mr, mc = key_spec["marker_rel"]
        grid[mr + off_r, mc + off_c] = key_spec["marker_color"]

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
