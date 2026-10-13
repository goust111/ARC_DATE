from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_COLOR_IDS = tuple(range(10))
_GEN_RETRIES = 500
_SHAPE_RETRIES = 200
_GRID_SIDE_MAX = 24
_OBJECT_COUNT_RANGE = (2, 5)
_BBOX_H_RANGE = (2, 5)
_BBOX_W_RANGE = (2, 5)
_FULL_RECT_PROB = 0.35
_TIE_DISTANCE_PROB = 0.5


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(g: np.ndarray) -> int:
    counts = Counter(map(int, g.reshape(-1)))
    return int(max(counts.items(), key=lambda kv: (int(kv[1]), -int(kv[0])))[0])


def _components_of_color(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _sort_key(obj: dict) -> tuple[int, int, int, int, int, int]:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    return (int(obj["dist"]), -int(obj["area"]), int(r0), int(c0), int(r1), int(c1))


def _extract_objects(g: np.ndarray) -> tuple[int, list[dict]]:
    bg = int(_most_common_color(g))
    h, w = g.shape
    objects: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _components_of_color(g, int(color)):
            rows = [int(r) for r, _c in comp]
            cols = [int(c) for _r, c in comp]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            dist = min(int(r0), int(c0), int(h - 1 - r1), int(w - 1 - c1))
            objects.append(
                {
                    "color": int(color),
                    "area": int(len(comp)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "dist": int(dist),
                }
            )

    objects.sort(key=_sort_key)
    return int(bg), objects


def _render_rings(objects: list[dict], bg: int) -> np.ndarray:
    n = int(len(objects))
    assert n > 0, "expected at least one object"
    side = int(2 * n + 1)
    out = np.full((side, side), int(bg), dtype=int)

    for layer, obj in enumerate(objects):
        lo = int(layer)
        hi = int(side - 1 - layer)
        color = int(obj["color"])
        out[lo, lo : hi + 1] = int(color)
        out[hi, lo : hi + 1] = int(color)
        out[lo : hi + 1, lo] = int(color)
        out[lo : hi + 1, hi] = int(color)

    out[n, n] = int(objects[-1]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Layer objects by nearest-edge distance, then redraw them as concentric square rings.

    Rule:
    - For each foreground object, compute the minimum distance from its bbox to any grid edge.
    - Sort by distance ascending; ties are broken by object area descending.
    - Emit a (2*N+1)x(2*N+1) square, drawing each object color as a 1px square ring from outside to inside.
    - Fill the center pixel with the innermost object's color.
    """
    g = np.asarray(grid, dtype=int)
    _bg, objects = _extract_objects(g)
    assert objects, "input must contain at least one foreground object"
    return _render_rings(objects, _bg)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) in cells and (int(nr), int(nc)) not in seen:
                seen.add((int(nr), int(nc)))
                q.append((int(nr), int(nc)))
    return len(seen) == len(cells)


def _touches_all_bbox_sides(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    rows = {int(r) for r, _c in cells}
    cols = {int(c) for _r, c in cells}
    return 0 in rows and int(h - 1) in rows and 0 in cols and int(w - 1) in cols


def _make_shape(h: int, w: int) -> tuple[tuple[int, int], ...]:
    cells = {(int(r), int(c)) for r in range(h) for c in range(w)}
    if random.random() < _FULL_RECT_PROB:
        return tuple(sorted(cells))

    area_min = max(3, int((11 * h * w + 19) // 20))
    target = random.randint(area_min, int(h * w))

    while len(cells) > target:
        removable: list[tuple[int, int]] = []
        for cell in tuple(cells):
            nxt = set(cells)
            nxt.remove(cell)
            if not _touches_all_bbox_sides(nxt, int(h), int(w)):
                continue
            if not _is_connected(nxt):
                continue
            removable.append((int(cell[0]), int(cell[1])))
        if not removable:
            break

        interior = [cell for cell in removable if 0 < int(cell[0]) < int(h - 1) and 0 < int(cell[1]) < int(w - 1)]
        pool = interior if interior and random.random() < 0.75 else removable
        cells.remove(random.choice(pool))

    return tuple(sorted(cells))


def _candidate_origins(h: int, w: int, box_h: int, box_w: int, dist: int) -> list[tuple[int, int]]:
    origins: list[tuple[int, int]] = []
    for r0 in range(1, int(h - box_h)):
        r1 = int(r0 + box_h - 1)
        top = int(r0)
        bottom = int(h - 1 - r1)
        if top < dist or bottom < dist:
            continue
        for c0 in range(1, int(w - box_w)):
            c1 = int(c0 + box_w - 1)
            left = int(c0)
            right = int(w - 1 - c1)
            if min(top, left, bottom, right) == int(dist):
                origins.append((int(r0), int(c0)))
    return origins


def _choose_distances(n_obj: int, max_dist: int) -> tuple[list[int], bool] | None:
    if max_dist < 1:
        return None

    tie_mode = bool(n_obj > 1 and random.random() < _TIE_DISTANCE_PROB)
    values = list(range(1, int(max_dist) + 1))

    if not tie_mode:
        if len(values) < n_obj:
            return None
        dists = random.sample(values, n_obj)
        random.shuffle(dists)
        return [int(d) for d in dists], False

    uniq_count = random.randint(1, int(min(len(values), n_obj - 1)))
    uniq = random.sample(values, uniq_count)
    dists = list(uniq)
    while len(dists) < n_obj:
        dists.append(int(random.choice(uniq)))
    random.shuffle(dists)
    return [int(d) for d in dists], True


def _padded_bbox_cells(r0: int, c0: int, box_h: int, box_w: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    rr0 = max(0, int(r0 - 1))
    rr1 = min(int(h - 1), int(r0 + box_h))
    cc0 = max(0, int(c0 - 1))
    cc1 = min(int(w - 1), int(c0 + box_w))
    for rr in range(rr0, rr1 + 1):
        for cc in range(cc0, cc1 + 1):
            cells.append((int(rr), int(cc)))
    return tuple(cells)


def _build_specs(n_obj: int, h: int, w: int, bg: int) -> tuple[list[dict], bool] | None:
    max_dist = int(min(h, w) // 2 - 3)
    picked = _choose_distances(int(n_obj), int(max_dist))
    if picked is None:
        return None
    dists, tie_mode = picked

    colors = random.sample([int(c) for c in _COLOR_IDS if int(c) != int(bg)], n_obj)
    used_areas: set[int] = set()
    specs: list[dict] = []

    for color, dist in zip(colors, dists, strict=False):
        for _ in range(_SHAPE_RETRIES):
            box_h = random.randint(*_BBOX_H_RANGE)
            box_w = random.randint(*_BBOX_W_RANGE)
            origins = _candidate_origins(int(h), int(w), int(box_h), int(box_w), int(dist))
            if not origins:
                continue
            shape = _make_shape(int(box_h), int(box_w))
            area = int(len(shape))
            if area in used_areas:
                continue
            specs.append(
                {
                    "color": int(color),
                    "dist": int(dist),
                    "box_h": int(box_h),
                    "box_w": int(box_w),
                    "shape": shape,
                    "area": int(area),
                    "origins": origins,
                }
            )
            used_areas.add(int(area))
            break
        else:
            return None

    same_dist_exists = len({int(spec["dist"]) for spec in specs}) < len(specs)
    if tie_mode != same_dist_exists:
        return None
    return specs, tie_mode


def _place_specs(specs: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    reserved: set[tuple[int, int]] = set()
    placements: list[tuple[int, int] | None] = [None] * len(specs)

    order = list(range(len(specs)))
    order.sort(
        key=lambda idx: (
            len(specs[idx]["origins"]),
            -(int(specs[idx]["box_h"]) * int(specs[idx]["box_w"])),
            -int(specs[idx]["area"]),
        )
    )

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True

        idx = int(order[pos])
        spec = specs[idx]
        origins = list(spec["origins"])
        random.shuffle(origins)

        for r0, c0 in origins:
            buffer_cells = _padded_bbox_cells(int(r0), int(c0), int(spec["box_h"]), int(spec["box_w"]), int(h), int(w))
            if any((int(rr), int(cc)) in reserved for rr, cc in buffer_cells):
                continue
            for cell in buffer_cells:
                reserved.add((int(cell[0]), int(cell[1])))
            placements[idx] = (int(r0), int(c0))
            if backtrack(int(pos + 1)):
                return True
            placements[idx] = None
            for cell in buffer_cells:
                reserved.remove((int(cell[0]), int(cell[1])))
        return False

    if not backtrack(0):
        return None
    return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]


def _materialize(specs: list[dict], placements: list[tuple[int, int]], h: int, w: int, bg: int) -> tuple[np.ndarray, list[dict]]:
    g = np.full((h, w), int(bg), dtype=int)
    objects: list[dict] = []

    for spec, (r0, c0) in zip(specs, placements, strict=False):
        for rr, cc in spec["shape"]:
            g[int(r0 + rr), int(c0 + cc)] = int(spec["color"])
        objects.append(
            {
                "color": int(spec["color"]),
                "area": int(spec["area"]),
                "bbox": (int(r0), int(c0), int(r0 + spec["box_h"] - 1), int(c0 + spec["box_w"] - 1)),
                "dist": int(spec["dist"]),
            }
        )
    return g, objects


def _gen_attempt() -> np.ndarray | None:
    n_obj = random.randint(*_OBJECT_COUNT_RANGE)
    min_side = int(2 * n_obj + 9)
    h = random.randint(min_side, _GRID_SIDE_MAX)
    w = random.randint(min_side, _GRID_SIDE_MAX)
    bg = random.choice(_COLOR_IDS)

    built = _build_specs(int(n_obj), int(h), int(w), int(bg))
    if built is None:
        return None
    specs, _tie_mode = built

    placements = _place_specs(specs, int(h), int(w))
    if placements is None or len(placements) != len(specs):
        return None

    g, objects = _materialize(specs, placements, int(h), int(w), int(bg))
    if int((g == int(bg)).sum()) <= int((g != int(bg)).sum()):
        return None

    expected = _render_rings(sorted(objects, key=_sort_key), int(bg))
    got = main(g)
    if not np.array_equal(got, expected):
        return None

    _, extracted = _extract_objects(g)
    if len(extracted) != len(specs):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input failed after repeated retries")
