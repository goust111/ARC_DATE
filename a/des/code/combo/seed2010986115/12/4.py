from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 400
OBJECT_BUILD_ATTEMPTS = 240
LOCAL_PATH_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 100
SEED_AREA_MIN = 2

OBJECT_COUNT_CHOICES = (1, 2, 3)
OBJECT_COUNT_WEIGHTS = (3, 3, 2)
OBJECT_KIND_WEIGHTS = {
    "h": 3,
    "v": 3,
    "d1": 2,
    "d2": 2,
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not arr[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_axis(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[str, int]:
    pts = set(coords)
    r0, c0, r1, c1 = _bbox(pts)
    axes: list[tuple[str, int]] = []

    if all((r0 + r1 - r, c) in pts for r, c in pts):
        axes.append(("h", r0 + r1))
    if all((r, c0 + c1 - c) in pts for r, c in pts):
        axes.append(("v", c0 + c1))
    if r1 - r0 == c1 - c0:
        k = r0 - c0
        if all((c + k, r - k) in pts for r, c in pts):
            axes.append(("d1", k))
        s = r0 + c1
        if all((s - c, s - r) in pts for r, c in pts):
            axes.append(("d2", s))

    assert len(axes) == 1, "each fence must have a unique reflection axis"
    return axes[0]


def _reflect_point(
    point: tuple[int, int],
    axis: tuple[str, int],
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    r, c = point
    _r0, _c0, _r1, _c1 = box
    kind, value = axis
    if kind == "h":
        return value - r, c
    if kind == "v":
        return r, value - c
    if kind == "d1":
        return c + value, r - value
    if kind == "d2":
        return value - c, value - r
    raise AssertionError("unknown axis kind")


def _flood_fill_black(out: np.ndarray, start: tuple[int, int], color: int) -> None:
    sr, sc = start
    if int(out[sr, sc]) != BLACK:
        return
    h, w = out.shape
    q = deque([(sr, sc)])
    out[sr, sc] = int(color)
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(out[nr, nc]) == BLACK:
                out[nr, nc] = int(color)
                q.append((nr, nc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each seed color to paint its reflected same-shape black region across the fence's unique axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    h, w = g.shape
    for comp in _components(g == GREY):
        axis = _detect_axis(comp)
        r0, c0, r1, c1 = _bbox(comp)
        seeds: list[tuple[int, int, int]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                val = int(g[r, c])
                if val not in (BLACK, GREY):
                    seeds.append((r, c, val))

        assert seeds, "each fence must contain at least one seed"
        for r, c, color in sorted(seeds):
            rr, cc = _reflect_point((r, c), axis, (r0, c0, r1, c1))
            assert 0 <= rr < h and 0 <= cc < w, "reflected seed must stay in bounds"
            _flood_fill_black(out, (rr, cc), color)
    return out


def _touches_edge(coords: list[tuple[int, int]] | set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _profile_to_path(rows: list[int]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    r = rows[0]
    path.append((r, 0))
    for nc in range(1, len(rows)):
        path.append((r, nc))
        target = rows[nc]
        step = 1 if target > r else -1
        while r != target:
            r += step
            path.append((r, nc))
    return path


def _bounded_pairs(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    arr = np.asarray(mask, dtype=bool)
    axis = _detect_axis([tuple(map(int, rc)) for rc in np.argwhere(arr)])
    box = (0, 0, arr.shape[0] - 1, arr.shape[1] - 1)

    comps = []
    for comp in _components(~arr):
        if _touches_edge(comp, arr.shape):
            continue
        comps.append(sorted(comp))

    lookup = {frozenset(comp): i for i, comp in enumerate(comps)}
    pairs: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    used: set[int] = set()
    for i, comp in enumerate(comps):
        if i in used:
            continue
        reflected = frozenset(_reflect_point(cell, axis, box) for cell in comp)
        j = lookup.get(reflected)
        if j is None or j == i or j in used:
            continue
        if min(len(comp), len(comps[j])) < SEED_AREA_MIN:
            continue
        used.add(i)
        used.add(j)
        pairs.append((comp, comps[j]))
    pairs.sort(key=lambda pair: min(len(pair[0]), len(pair[1])), reverse=True)
    return pairs


def _make_object(mask: np.ndarray) -> dict | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    try:
        axis = _detect_axis(coords)
    except AssertionError:
        return None

    pairs = _bounded_pairs(mask)
    if not pairs:
        return None

    local = np.full(mask.shape, BLACK, dtype=int)
    local[mask] = GREY
    return {"grid": local, "axis": axis, "pairs": pairs}


def _horizontal_candidate(height: int, width: int) -> np.ndarray:
    axis = height // 2
    rows = [random.randint(1, axis - 1)]
    for _ in range(width - 1):
        options = [nr for nr in (rows[-1] - 1, rows[-1], rows[-1] + 1) if 1 <= nr <= axis - 1]
        rows.append(random.choice(options))

    cells = set(_profile_to_path(rows))
    cells |= {(height - 1 - r, c) for r, c in list(cells)}
    for r in range(rows[0], height - rows[0]):
        cells.add((r, 0))
        cells.add((r, width - 1))
    for c in range(width):
        cells.add((axis, c))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _build_horizontal_object(compact: bool) -> dict | None:
    widths = (9, 13) if compact else (9, 15)
    for _ in range(OBJECT_BUILD_ATTEMPTS):
        height = random.choice((7, 9, 11))
        width = random.randint(widths[0], widths[1])
        obj = _make_object(_horizontal_candidate(height, width))
        if obj is not None and obj["axis"][0] == "h":
            return obj
    return None


def _build_vertical_object(compact: bool) -> dict | None:
    widths = (9, 13) if compact else (9, 15)
    for _ in range(OBJECT_BUILD_ATTEMPTS):
        height = random.randint(widths[0], widths[1])
        width = random.choice((7, 9, 11))
        obj = _make_object(_horizontal_candidate(width, height).T)
        if obj is not None and obj["axis"][0] == "v":
            return obj
    return None


def _main_diag_candidate(size: int) -> np.ndarray | None:
    max_anchor = (size - 3) // 2
    if max_anchor < 2:
        return None
    anchor = random.randint(2, max_anchor)
    rights = size - 1 - anchor
    downs = anchor

    path: list[tuple[int, int]] | None = None
    for _ in range(LOCAL_PATH_ATTEMPTS):
        steps = ["R"] * rights + ["D"] * downs
        random.shuffle(steps)
        r, c = 0, anchor
        cur_path = [(r, c)]
        ok = True
        for step in steps:
            if step == "R":
                c += 1
            else:
                r += 1
            if c - r < 1 or r >= size or c >= size:
                ok = False
                break
            cur_path.append((r, c))
        if ok and (r, c) == (anchor, size - 1):
            path = cur_path
            break
    if path is None:
        return None

    cells = set(path)
    cells |= {(c, r) for r, c in list(cells)}
    for x in range(anchor + 1):
        cells.add((0, x))
        cells.add((x, 0))
    for x in range(anchor, size):
        cells.add((size - 1, x))
        cells.add((x, size - 1))
    for i in range(size):
        cells.add((i, i))

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _build_main_diag_object(compact: bool) -> dict | None:
    sizes = (9, 11) if compact else (9, 11, 13)
    for _ in range(OBJECT_BUILD_ATTEMPTS):
        mask = _main_diag_candidate(random.choice(sizes))
        if mask is None:
            continue
        obj = _make_object(mask)
        if obj is not None and obj["axis"][0] == "d1":
            return obj
    return None


def _build_anti_diag_object(compact: bool) -> dict | None:
    sizes = (9, 11) if compact else (9, 11, 13)
    for _ in range(OBJECT_BUILD_ATTEMPTS):
        mask = _main_diag_candidate(random.choice(sizes))
        if mask is None:
            continue
        obj = _make_object(np.fliplr(mask))
        if obj is not None and obj["axis"][0] == "d2":
            return obj
    return None


def _build_object(kind: str, compact: bool) -> dict | None:
    if kind == "h":
        return _build_horizontal_object(compact)
    if kind == "v":
        return _build_vertical_object(compact)
    if kind == "d1":
        return _build_main_diag_object(compact)
    if kind == "d2":
        return _build_anti_diag_object(compact)
    raise AssertionError("unknown object kind")


def _reserve(blocked: np.ndarray, r0: int, c0: int, height: int, width: int, padding: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + height + padding)
    cc1 = min(w, c0 + width + padding)
    blocked[rr0:rr1, cc0:cc1] = True


def _can_place(blocked: np.ndarray, r0: int, c0: int, height: int, width: int, padding: int) -> bool:
    h, w = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + height > h or c0 + width > w:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + height + padding)
    cc1 = min(w, c0 + width + padding)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _sample_canvas_size(n_objects: int) -> tuple[int, int]:
    if n_objects == 1:
        h = random.randint(18, 26)
        w = random.randint(18, 26)
    elif n_objects == 2:
        h = random.randint(24, 30)
        w = random.randint(24, 30)
    else:
        h = random.randint(28, 30)
        w = random.randint(28, 30)
    return h, w


def _place_objects(shape: tuple[int, int], objects: list[dict]) -> list[tuple[int, int, dict]] | None:
    h, w = shape
    blocked = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int, dict]] = []
    ordered = sorted(objects, key=lambda obj: obj["grid"].shape[0] * obj["grid"].shape[1], reverse=True)
    for obj in ordered:
        oh, ow = obj["grid"].shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, h - oh)
            c0 = random.randint(0, w - ow)
            if not _can_place(blocked, r0, c0, oh, ow, padding=2):
                continue
            _reserve(blocked, r0, c0, oh, ow, padding=2)
            placements.append((r0, c0, obj))
            placed = True
            break
        if not placed:
            return None
    return placements


def _choose_seed_sources(pairs: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]) -> list[list[tuple[int, int]]]:
    count = 1
    if len(pairs) >= 2 and random.random() < 0.45:
        count = 2
    chosen_pairs = random.sample(pairs, count)
    sources: list[list[tuple[int, int]]] = []
    for comp_a, comp_b in chosen_pairs:
        sources.append(random.choice([comp_a, comp_b]))
    return sources


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_objects = random.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
        kinds = random.choices(tuple(OBJECT_KIND_WEIGHTS), weights=tuple(OBJECT_KIND_WEIGHTS.values()), k=n_objects)
        compact = n_objects >= 2

        objects: list[dict] = []
        ok = True
        for kind in kinds:
            obj = _build_object(kind, compact)
            if obj is None:
                ok = False
                break
            objects.append(obj)
        if not ok:
            continue

        shape = _sample_canvas_size(n_objects)
        placements = _place_objects(shape, objects)
        if placements is None:
            continue

        grid = np.full(shape, BLACK, dtype=int)
        seed_spots: list[tuple[int, int]] = []
        for r0, c0, obj in placements:
            local = obj["grid"]
            oh, ow = local.shape
            patch = grid[r0 : r0 + oh, c0 : c0 + ow]
            assert not bool((patch != BLACK).any())
            patch[local == GREY] = GREY

            sources = _choose_seed_sources(obj["pairs"])
            for comp in sources:
                sr, sc = random.choice(comp)
                seed_spots.append((r0 + sr, c0 + sc))

        if len(seed_spots) > len(SEED_COLORS):
            continue

        for color, (r, c) in zip(random.sample(SEED_COLORS, len(seed_spots)), seed_spots, strict=False):
            grid[r, c] = color

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
