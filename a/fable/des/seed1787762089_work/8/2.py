from __future__ import annotations

from collections import defaultdict, deque
from functools import lru_cache
import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
GENERATE_MAX_RETRIES = 400


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _canonical_shape(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    variants = []
    for rsgn in (1, -1):
        for csgn in (1, -1):
            variants.append(_normalize_coords((rsgn * r, csgn * c) for r, c in pts))
            variants.append(_normalize_coords((rsgn * c, csgn * r) for r, c in pts))
    return min(variants)


def _shape_variants(coords) -> list[tuple[tuple[int, int], ...]]:
    pts = [(int(r), int(c)) for r, c in coords]
    seen = set()
    variants = []
    for rsgn in (1, -1):
        for csgn in (1, -1):
            for transformed in (
                [(rsgn * r, csgn * c) for r, c in pts],
                [(rsgn * c, csgn * r) for r, c in pts],
            ):
                norm = _normalize_coords(transformed)
                if norm in seen:
                    continue
                seen.add(norm)
                variants.append(norm)
    return variants


BASE_SHAPES = {
    "domino": ((0, 0), (0, 1)),
    "line3": ((0, 0), (0, 1), (0, 2)),
    "el3": ((0, 0), (0, 1), (1, 0)),
    "line4": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "el4": ((0, 0), (1, 0), (2, 0), (2, 1)),
}
SHAPE_CANONS = {name: _canonical_shape(coords) for name, coords in BASE_SHAPES.items()}
CANON_TO_NAME = {canon: name for name, canon in SHAPE_CANONS.items()}
SHAPE_VARIANTS = {name: _shape_variants(coords) for name, coords in BASE_SHAPES.items()}
ALL_SHAPE_NAMES = tuple(BASE_SHAPES)


def _bbox(coords) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _is_connected(coords) -> bool:
    pts = {tuple(map(int, p)) for p in coords}
    if not pts:
        return False
    q = deque([next(iter(pts))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nb = (r + dr, c + dc)
            if nb in pts and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(pts)


@lru_cache(maxsize=None)
def _border_coords(h: int, w: int) -> tuple[tuple[int, int], ...]:
    pts = []
    for c in range(w):
        pts.append((0, c))
    for r in range(1, h - 1):
        pts.append((r, w - 1))
    for c in range(w - 1, -1, -1):
        pts.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        pts.append((r, 0))
    return tuple(pts)


@lru_cache(maxsize=None)
def _gap_options(h: int, w: int) -> dict[str, tuple[tuple[tuple[int, int], ...], ...]]:
    cycle = _border_coords(h, w)
    n = len(cycle)
    buckets: dict[str, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    seen = set()
    for length in (2, 3, 4):
        for start in range(n):
            seg = tuple(cycle[(start + k) % n] for k in range(length))
            key = frozenset(seg)
            if key in seen:
                continue
            seen.add(key)
            canon = _canonical_shape(seg)
            if canon not in CANON_TO_NAME:
                continue
            buckets[CANON_TO_NAME[canon]].append(tuple(sorted(seg)))
    return {name: tuple(buckets.get(name, ())) for name in ALL_SHAPE_NAMES}


def _classify_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    locks = []
    keys = []
    for comp in _components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        rel = tuple((r - r0, c - c0) for r, c in comp["coords"])
        rel_set = set(rel)
        if h >= 4 and w >= 4:
            border = set(_border_coords(h, w))
            gap = tuple(sorted(border - rel_set))
            if rel_set <= border and 2 <= len(gap) <= 4 and _is_connected(gap):
                interior = g[r0 + 1 : r1, c0 + 1 : c1]
                if interior.size == 0 or np.all(interior == bg):
                    locks.append(
                        {
                            "color": int(comp["color"]),
                            "bbox": (r0, c0, r1, c1),
                            "gap_rel": gap,
                            "gap_abs": tuple((r0 + r, c0 + c) for r, c in gap),
                            "shape": _canonical_shape(gap),
                        }
                    )
                    continue
        keys.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "shape": _canonical_shape(rel),
            }
        )
    return bg, locks, keys


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, locks, keys = _classify_objects(g)
    out = np.full_like(g, int(bg))

    locks_by_shape: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    keys_by_shape: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for lock in locks:
        locks_by_shape[lock["shape"]].append(lock)
    for key in keys:
        keys_by_shape[key["shape"]].append(key)

    for shape, lock_group in locks_by_shape.items():
        key_group = keys_by_shape.get(shape, [])
        if len(lock_group) != 1 or len(key_group) != 1:
            continue
        lock = lock_group[0]
        key = key_group[0]
        r0, c0, r1, c1 = lock["bbox"]
        lock_color = int(lock["color"])
        key_color = int(key["color"])
        out[r0, c0 : c1 + 1] = lock_color
        out[r1, c0 : c1 + 1] = lock_color
        out[r0 : r1 + 1, c0] = lock_color
        out[r0 : r1 + 1, c1] = lock_color
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            out[r0 + 1 : r1, c0 + 1 : c1] = key_color
        for gr, gc in lock["gap_abs"]:
            out[int(gr), int(gc)] = key_color
    return out


def _render_coords(coords, color: int, bg: int = BG) -> np.ndarray:
    norm = _normalize_coords(coords)
    h = 1 + max(r for r, _ in norm)
    w = 1 + max(c for _, c in norm)
    patch = np.full((h, w), int(bg), dtype=int)
    for r, c in norm:
        patch[r, c] = int(color)
    return patch


def _build_lock_patch(h: int, w: int, color: int, gap_rel) -> np.ndarray:
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for r, c in gap_rel:
        patch[int(r), int(c)] = BG
    return patch


def _choose(seq, rng):
    return seq[int(rng.integers(len(seq)))]


def _sample_without_replacement(seq, k: int, rng) -> list:
    if k == 0:
        return []
    idx = rng.choice(len(seq), size=k, replace=False)
    return [seq[int(i)] for i in np.atleast_1d(idx)]


def _shuffle(seq, rng) -> list:
    order = rng.permutation(len(seq))
    return [seq[int(i)] for i in order]


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int = BG) -> None:
    mask = patch != bg
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _bbox_fits(occupied: np.ndarray, h: int, w: int, r0: int, c0: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _reserve_bbox(occupied: np.ndarray, h: int, w: int, r0: int, c0: int, pad: int = 1) -> None:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _candidate_positions(grid_shape: tuple[int, int], patch_shape: tuple[int, int], margin: int = 1) -> list[tuple[int, int]]:
    h, w = grid_shape
    ph, pw = patch_shape
    row_lo = margin
    col_lo = margin
    row_hi = h - ph - margin
    col_hi = w - pw - margin
    if row_hi < row_lo or col_hi < col_lo:
        return []
    return [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]


def _choose_scenario(rng) -> tuple[str, int, int, int]:
    scenario = _choose(("perfect", "all_locks", "mixed"), rng)
    if scenario == "perfect":
        return scenario, 3, 3, 3
    if scenario == "all_locks":
        num_locks = int(_choose((2, 3), rng))
        min_keys = max(3, num_locks + 1)
        num_keys = int(_choose(tuple(range(min_keys, 6)), rng))
        return scenario, num_locks, num_locks, num_keys
    candidates = []
    for num_locks in (2, 3):
        for matched in range(1, num_locks):
            max_keys = min(5, matched + (len(ALL_SHAPE_NAMES) - num_locks))
            for num_keys in range(3, max_keys + 1):
                if num_keys > matched:
                    candidates.append((num_locks, matched, num_keys))
    num_locks, matched, num_keys = _choose(tuple(candidates), rng)
    return scenario, int(num_locks), int(matched), int(num_keys)


def _expected_output(shape: tuple[int, int], locks: list[dict], key_color_by_shape: dict[str, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for lock in locks:
        match_color = key_color_by_shape.get(lock["shape_name"])
        if match_color is None:
            continue
        r0 = int(lock["r0"])
        c0 = int(lock["c0"])
        h = int(lock["h"])
        w = int(lock["w"])
        lock_color = int(lock["color"])
        out[r0, c0 : c0 + w] = lock_color
        out[r0 + h - 1, c0 : c0 + w] = lock_color
        out[r0 : r0 + h, c0] = lock_color
        out[r0 : r0 + h, c0 + w - 1] = lock_color
        out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = int(match_color)
        for gr, gc in lock["gap_rel"]:
            out[r0 + int(gr), c0 + int(gc)] = int(match_color)
    return out


def _gen_attempt(rng) -> np.ndarray | None:
    _scenario, num_locks, matched_count, num_keys = _choose_scenario(rng)

    lock_shapes = _sample_without_replacement(ALL_SHAPE_NAMES, num_locks, rng)
    matched_shapes = _sample_without_replacement(lock_shapes, matched_count, rng)
    extra_pool = [shape for shape in ALL_SHAPE_NAMES if shape not in lock_shapes]
    extra_count = num_keys - matched_count
    extra_shapes = _sample_without_replacement(extra_pool, extra_count, rng)
    key_shapes = _shuffle(matched_shapes + extra_shapes, rng)

    total_colors = num_locks + num_keys
    colors = _sample_without_replacement(tuple(range(1, 10)), total_colors, rng)
    lock_colors = colors[:num_locks]
    key_colors = colors[num_locks:]

    min_side = 16 + (1 if num_locks == 3 else 0) + (1 if num_keys >= 4 else 0)
    h = int(rng.integers(min_side, 23))
    w = int(rng.integers(min_side, 23))
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    lock_specs = []
    for shape_name, color in zip(lock_shapes, lock_colors):
        max_dim = 8 if num_locks == 2 else 7
        lock_h = int(rng.integers(5, max_dim + 1))
        lock_w = int(rng.integers(5, max_dim + 1))
        options = _gap_options(lock_h, lock_w)[shape_name]
        if not options:
            return None
        gap_rel = _choose(options, rng)
        patch = _build_lock_patch(lock_h, lock_w, int(color), gap_rel)
        lock_specs.append(
            {
                "shape_name": str(shape_name),
                "color": int(color),
                "h": lock_h,
                "w": lock_w,
                "gap_rel": gap_rel,
                "patch": patch,
            }
        )

    lock_specs = sorted(lock_specs, key=lambda item: item["h"] * item["w"], reverse=True)
    placed_locks = []
    for spec in lock_specs:
        positions = _candidate_positions(grid.shape, spec["patch"].shape, margin=1)
        if not positions:
            return None
        for r0, c0 in _shuffle(positions, rng):
            if not _bbox_fits(occupied, spec["h"], spec["w"], int(r0), int(c0), pad=1):
                continue
            _place_patch(grid, spec["patch"], int(r0), int(c0), bg=BG)
            _reserve_bbox(occupied, spec["h"], spec["w"], int(r0), int(c0), pad=1)
            spec = dict(spec)
            spec["r0"] = int(r0)
            spec["c0"] = int(c0)
            placed_locks.append(spec)
            break
        else:
            return None

    key_specs = []
    for shape_name, color in zip(key_shapes, key_colors):
        variant = _choose(SHAPE_VARIANTS[shape_name], rng)
        patch = _render_coords(variant, int(color), bg=BG)
        key_specs.append(
            {
                "shape_name": str(shape_name),
                "color": int(color),
                "patch": patch,
            }
        )

    key_specs = sorted(key_specs, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)
    for spec in key_specs:
        positions = _candidate_positions(grid.shape, spec["patch"].shape, margin=1)
        if not positions:
            return None
        ph, pw = spec["patch"].shape
        for r0, c0 in _shuffle(positions, rng):
            if not _bbox_fits(occupied, ph, pw, int(r0), int(c0), pad=1):
                continue
            _place_patch(grid, spec["patch"], int(r0), int(c0), bg=BG)
            _reserve_bbox(occupied, ph, pw, int(r0), int(c0), pad=1)
            break
        else:
            return None

    key_color_by_shape = {str(shape): int(color) for shape, color in zip(key_shapes, key_colors)}
    expected = _expected_output(grid.shape, placed_locks, key_color_by_shape)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
