from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _mask_from_coords(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in coords:
        out[r, c] = True
    return out


def _band_mask(h: int, w: int, thickness: int) -> np.ndarray:
    band = np.zeros((h, w), dtype=bool)
    band[:thickness, :] = True
    band[h - thickness :, :] = True
    band[:, :thickness] = True
    band[:, w - thickness :] = True
    return band


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _all_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    comp_id = 0
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            bbox = _bbox(coords)
            comps.append(
                {
                    "id": comp_id,
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                }
            )
            comp_id += 1
    return comps


def _find_locks(grid: np.ndarray, bg: int, comps: list[dict]) -> list[dict]:
    locks: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        comp_mask = np.zeros((h, w), dtype=bool)
        for rr, cc in comp["coords"]:
            comp_mask[rr - r0, cc - c0] = True

        matches: list[dict] = []
        for thickness in range(1, min(h, w) // 2 + 1):
            if h - 2 * thickness < 1 or w - 2 * thickness < 1:
                continue
            band = _band_mask(h, w, thickness)
            if np.any(comp_mask & ~band):
                continue
            if np.any((sub == color) & ~band):
                continue

            hole_mask = band & ~comp_mask
            hole_size = int(hole_mask.sum())
            if hole_size < 2 or hole_size > 4:
                continue
            if np.any(sub[hole_mask] != bg):
                continue

            hole_comps = _components(hole_mask)
            if len(hole_comps) != 1:
                continue
            hole_rel = hole_comps[0]

            touched = set()
            for rr, cc in hole_rel:
                if rr == 0:
                    touched.add("top")
                if rr == h - 1:
                    touched.add("bottom")
                if cc == 0:
                    touched.add("left")
                if cc == w - 1:
                    touched.add("right")
            if len(touched) != 1:
                continue
            side = next(iter(touched))
            if side == "top" and any(rr >= thickness for rr, _ in hole_rel):
                continue
            if side == "bottom" and any(rr < h - thickness for rr, _ in hole_rel):
                continue
            if side == "left" and any(cc >= thickness for _, cc in hole_rel):
                continue
            if side == "right" and any(cc < w - thickness for _, cc in hole_rel):
                continue

            matches.append(
                {
                    "comp_id": int(comp["id"]),
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "thickness": int(thickness),
                    "side": side,
                    "hole_rel": hole_rel,
                    "hole_abs": [(r0 + rr, c0 + cc) for rr, cc in hole_rel],
                    "hole_shape": _normalize(hole_rel),
                    "interior_bbox": (r0 + thickness, c0 + thickness, r1 - thickness, c1 - thickness),
                }
            )

        if len(matches) == 1:
            locks.append(matches[0])

    assert locks, "expected at least one lock frame"
    return locks


def _find_keys(grid: np.ndarray, bg: int, comps: list[dict], locks: list[dict]) -> list[dict]:
    lock_ids = {int(lock["comp_id"]) for lock in locks}
    keys: list[dict] = []
    for comp in comps:
        if int(comp["id"]) in lock_ids:
            continue
        containing: list[int] = []
        for idx, lock in enumerate(locks):
            ir0, ic0, ir1, ic1 = lock["interior_bbox"]
            if ir0 > ir1 or ic0 > ic1:
                continue
            if all(ir0 <= r <= ir1 and ic0 <= c <= ic1 for r, c in comp["coords"]):
                containing.append(idx)
        assert len(containing) == 1, "every non-frame component must be a key inside exactly one lock"
        keys.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "container_lock": int(containing[0]),
                "shape": _normalize(comp["coords"]),
            }
        )

    assert keys, "expected at least one key"
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each interior key shape to the lockhole with the same normalized outline,
    erase the key, and repaint that hole with the matching lock's border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    comps = _all_monochrome_components(g, bg)
    locks = _find_locks(g, bg, comps)
    keys = _find_keys(g, bg, comps, locks)

    hole_to_lock: dict[tuple[tuple[int, int], ...], dict] = {}
    for lock in locks:
        shape = lock["hole_shape"]
        assert shape not in hole_to_lock, "lockhole shapes must be unique"
        hole_to_lock[shape] = lock

    key_to_shape: dict[tuple[tuple[int, int], ...], dict] = {}
    for key in keys:
        shape = key["shape"]
        assert shape not in key_to_shape, "key shapes must be unique"
        key_to_shape[shape] = key

    assert len(locks) == len(keys), "expected one key per lock"
    assert set(hole_to_lock) == set(key_to_shape), "key shapes must exactly match lockhole shapes"

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = bg
    for shape, lock in hole_to_lock.items():
        _ = key_to_shape[shape]
        for r, c in lock["hole_abs"]:
            out[r, c] = int(lock["color"])
    return out


# --------------- generator ---------------

BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
LOCK_COUNT_RANGE = (2, 4)
THICKNESS_CHOICES = [2, 3]
FRAME_SIDE_MIN = 8
FRAME_SIDE_MAX = 14
FRAME_PADDING = 2
PLACEMENT_TRIES = 200
GEN_RETRIES = 400

BASE_HOLE_PATCHES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
]
SIDES = ["top", "bottom", "left", "right"]
ROT_K = {"top": 0, "left": 1, "bottom": 2, "right": 3}


def _rotated_patch(base: np.ndarray, side: str) -> np.ndarray:
    return np.rot90(base, ROT_K[side]).copy()


def _shape_from_patch(patch: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = [tuple(int(v) for v in rc) for rc in np.argwhere(patch)]
    return tuple(sorted(coords))


def _hole_offset(
    frame_h: int,
    frame_w: int,
    thickness: int,
    side: str,
    patch: np.ndarray,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    if side in ("top", "bottom"):
        lo = thickness
        hi = frame_w - thickness - pw
        if hi < lo:
            return None
        c0 = random.randint(lo, hi)
        r0 = 0 if side == "top" else frame_h - ph
        return r0, c0
    lo = thickness
    hi = frame_h - thickness - ph
    if hi < lo:
        return None
    r0 = random.randint(lo, hi)
    c0 = 0 if side == "left" else frame_w - pw
    return r0, c0


def _frame_patch(
    frame_h: int,
    frame_w: int,
    thickness: int,
    color: int,
    hole_patch: np.ndarray,
    hole_origin: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    patch = np.zeros((frame_h, frame_w), dtype=int)
    band = _band_mask(frame_h, frame_w, thickness)
    patch[band] = color
    hr0, hc0 = hole_origin
    hole_coords: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(hole_patch):
        r = hr0 + int(rr)
        c = hc0 + int(cc)
        patch[r, c] = BG
        hole_coords.append((r, c))
    return patch, hole_coords


def _bbox_with_padding_ok(
    bbox: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], padding: int
) -> bool:
    r0, c0, r1, c1 = bbox
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + padding < pr0 or pr1 + padding < r0 or c1 + padding < pc0 or pc1 + padding < c0):
            return False
    return True


def _place_frames(specs: list[dict], grid_h: int, grid_w: int) -> list[dict] | None:
    order = sorted(range(len(specs)), key=lambda i: specs[i]["frame_h"] * specs[i]["frame_w"], reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_specs: list[dict | None] = [None] * len(specs)
    for idx in order:
        spec = dict(specs[idx])
        h = int(spec["frame_h"])
        w = int(spec["frame_w"])
        if h + 2 > grid_h or w + 2 > grid_w:
            return None
        success = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(1, grid_h - h - 1)
            c0 = random.randint(1, grid_w - w - 1)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if not _bbox_with_padding_ok(bbox, placed_boxes, FRAME_PADDING):
                continue
            placed_boxes.append(bbox)
            spec["bbox"] = bbox
            placed_specs[idx] = spec
            success = True
            break
        if not success:
            return None
    return [spec for spec in placed_specs if spec is not None]


def _key_origin(inner_h: int, inner_w: int, key_h: int, key_w: int) -> tuple[int, int]:
    assert 0 <= key_h <= inner_h and 0 <= key_w <= inner_w
    extra_h = inner_h - key_h
    extra_w = inner_w - key_w
    if extra_h >= 2:
        r = random.randint(1, extra_h - 1)
    else:
        r = random.randint(0, extra_h)
    if extra_w >= 2:
        c = random.randint(1, extra_w - 1)
    else:
        c = random.randint(0, extra_w)
    return r, c


def _random_permutation(n: int) -> list[int]:
    order = list(range(n))
    random.shuffle(order)
    return order


def _random_derangement(n: int) -> list[int]:
    assert n >= 2
    for _ in range(100):
        order = _random_permutation(n)
        if all(i != order[i] for i in range(n)):
            return order
    return list(range(1, n)) + [0]


def _specs_for_case(lock_count: int) -> list[dict] | None:
    specs: list[dict] = []
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(lock_count):
        for _ in range(100):
            thickness = random.choice(THICKNESS_CHOICES)
            side = random.choice(SIDES)
            base = random.choice(BASE_HOLE_PATCHES)
            hole_patch = _rotated_patch(base, side)
            shape = _shape_from_patch(hole_patch)
            if shape in used_shapes:
                continue

            hole_h, hole_w = hole_patch.shape
            min_h = max(FRAME_SIDE_MIN, 2 * thickness + hole_h + 2)
            min_w = max(FRAME_SIDE_MIN, 2 * thickness + hole_w + 2)
            max_h = max(min_h, FRAME_SIDE_MAX)
            max_w = max(min_w, FRAME_SIDE_MAX)
            frame_h = random.randint(min_h, max_h)
            frame_w = random.randint(min_w, max_w)

            hole_origin = _hole_offset(frame_h, frame_w, thickness, side, hole_patch)
            if hole_origin is None:
                continue

            inner_h = frame_h - 2 * thickness
            inner_w = frame_w - 2 * thickness
            if inner_h < hole_h or inner_w < hole_w:
                continue

            specs.append(
                {
                    "thickness": int(thickness),
                    "side": side,
                    "hole_patch": hole_patch,
                    "hole_shape": shape,
                    "hole_origin": hole_origin,
                    "frame_h": int(frame_h),
                    "frame_w": int(frame_w),
                    "key_h": int(hole_h),
                    "key_w": int(hole_w),
                }
            )
            used_shapes.add(shape)
            break
        else:
            return None
    return specs


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    lock_count = random.randint(*LOCK_COUNT_RANGE)
    specs = _specs_for_case(lock_count)
    if specs is None:
        return None

    colors = random.sample(COLOR_POOL, 2 * lock_count)
    frame_colors = colors[:lock_count]
    key_colors = colors[lock_count:]

    total_area = sum(spec["frame_h"] * spec["frame_w"] for spec in specs)
    max_h = max(spec["frame_h"] for spec in specs)
    max_w = max(spec["frame_w"] for spec in specs)
    min_grid_h = max(GRID_H_RANGE[0], max_h + 4, int(np.ceil(np.sqrt(total_area))) + 6)
    min_grid_w = max(GRID_W_RANGE[0], max_w + 4, int(np.ceil(np.sqrt(total_area))) + 6)
    if min_grid_h > GRID_H_RANGE[1] or min_grid_w > GRID_W_RANGE[1]:
        return None
    grid_h = random.randint(min_grid_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_grid_w, GRID_W_RANGE[1])

    placed_specs = _place_frames(specs, grid_h, grid_w)
    if placed_specs is None:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = inp.copy()

    if lock_count >= 2 and random.random() < 0.5:
        assignment = _random_derangement(lock_count)
    else:
        assignment = _random_permutation(lock_count)

    for idx, spec in enumerate(placed_specs):
        color = int(frame_colors[idx])
        bbox = spec["bbox"]
        r0, c0, r1, c1 = bbox
        frame_patch, hole_rel = _frame_patch(
            spec["frame_h"],
            spec["frame_w"],
            int(spec["thickness"]),
            color,
            spec["hole_patch"],
            spec["hole_origin"],
        )
        inp[r0 : r1 + 1, c0 : c1 + 1] = frame_patch
        out[r0 : r1 + 1, c0 : c1 + 1] = frame_patch
        for hr, hc in hole_rel:
            out[r0 + hr, c0 + hc] = color
        spec["hole_abs"] = [(r0 + hr, c0 + hc) for hr, hc in hole_rel]

    for dst_idx, src_idx in enumerate(assignment):
        dst = placed_specs[dst_idx]
        key_patch = specs[src_idx]["hole_patch"]
        key_h, key_w = key_patch.shape
        key_color = int(key_colors[dst_idx])
        thickness = int(dst["thickness"])
        r0, c0, r1, c1 = dst["bbox"]
        inner_h = (r1 - r0 + 1) - 2 * thickness
        inner_w = (c1 - c0 + 1) - 2 * thickness
        off_r, off_c = _key_origin(inner_h, inner_w, key_h, key_w)
        base_r = r0 + thickness + off_r
        base_c = c0 + thickness + off_c
        key_mask = _mask_from_coords(key_patch.shape, [tuple(int(v) for v in rc) for rc in np.argwhere(key_patch)])
        for rr, cc in np.argwhere(key_mask):
            inp[base_r + int(rr), base_c + int(cc)] = key_color

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
