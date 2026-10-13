from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ("up", "down", "left", "right")
RNG = np.random.default_rng()

GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
PAIR_COUNTS = (1, 2, 3)
MARGIN = 1
GAP_MIN = 2
GAP_MAX = 4
MAX_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _apply_transform(mask: np.ndarray, tid: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if tid >= 4:
        arr = np.fliplr(arr)
    arr = np.rot90(arr, tid % 4)
    return _trim_mask(arr)


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    arr = np.asarray(mask, dtype=bool)
    return int(arr.shape[0]), int(arr.shape[1]), tuple(int(v) for v in arr.ravel())


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    out: list[np.ndarray] = []
    for tid in range(8):
        arr = _apply_transform(mask, tid)
        key = _mask_key(arr)
        if key not in seen:
            seen.add(key)
            out.append(arr)
    return out


def _canonical_mask(mask: np.ndarray) -> np.ndarray:
    return min(_unique_transforms(mask), key=_mask_key)


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(np.asarray(mask, dtype=bool), np.ones((scale, scale), dtype=bool))


def _primitive_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    arr = _trim_mask(np.asarray(mask, dtype=bool))
    h, w = arr.shape
    max_k = min(h, w)
    for k in range(max_k, 1, -1):
        if h % k != 0 or w % k != 0:
            continue
        hh, ww = h // k, w // k
        blocks = arr.reshape(hh, k, ww, k).transpose(0, 2, 1, 3)
        base = blocks[:, :, 0, 0]
        if np.all(blocks == base[:, :, None, None]):
            return _trim_mask(base), int(k)
    return arr, 1


def _component_center(comp: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = comp["bbox"]
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            primitive, primitive_scale = _primitive_mask(mask)
            canonical = _canonical_mask(primitive)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "primitive": primitive,
                    "primitive_scale": int(primitive_scale),
                    "canonical": canonical,
                    "key": _mask_key(canonical),
                    "area": int(mask.sum()),
                }
            )
    return comps


def _direction_from_centers(small: dict, large: dict) -> str:
    sr, sc = _component_center(small)
    lr, lc = _component_center(large)
    dr = sr - lr
    dc = sc - lc
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _oriented_stamp_mask(small: dict, large: dict) -> np.ndarray:
    base = small["canonical"]
    for arr in _unique_transforms(base):
        if np.array_equal(_scale_mask(arr, large["primitive_scale"]), large["mask"]):
            return _scale_mask(arr, small["primitive_scale"])
    raise AssertionError("failed to align canonical shape with the canvas orientation")


def _best_stamp_translation(large_mask: np.ndarray, stamp_mask: np.ndarray, direction: str) -> tuple[int, int]:
    lh, lw = large_mask.shape
    sh, sw = stamp_mask.shape
    assert sh <= lh and sw <= lw

    center_r = (lh - sh) / 2.0
    center_c = (lw - sw) / 2.0
    best_score = None
    best_pos = None

    for r0 in range(lh - sh + 1):
        for c0 in range(lw - sw + 1):
            window = large_mask[r0 : r0 + sh, c0 : c0 + sw]
            if not np.all(window[stamp_mask]):
                continue

            if direction == "up":
                score = (r0, abs(c0 - center_c), abs(r0 - center_r), c0)
            elif direction == "down":
                score = (-r0, abs(c0 - center_c), abs(r0 - center_r), c0)
            elif direction == "left":
                score = (c0, abs(r0 - center_r), abs(c0 - center_c), r0)
            else:
                score = (-c0, abs(r0 - center_r), abs(c0 - center_c), r0)

            if best_score is None or score < best_score:
                best_score = score
                best_pos = (r0, c0)

    if best_pos is None:
        raise AssertionError("stamp mask does not fit inside the canvas")
    return best_pos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shape regions across scales, erase the smaller one, and stamp its color into the larger one.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected foreground components"

    groups: dict[tuple[int, int, tuple[int, ...]], list[dict]] = defaultdict(list)
    for comp in comps:
        groups[comp["key"]].append(comp)

    out = g.copy()

    for pair in groups.values():
        assert len(pair) == 2, "each normalized shape must appear exactly twice"
        pair = sorted(pair, key=lambda comp: comp["area"])
        small, large = pair
        assert small["area"] < large["area"], "paired objects must have different sizes"
        assert small["color"] != large["color"], "paired objects must have different colors"

        for r, c in small["coords"]:
            out[r, c] = bg

        direction = _direction_from_centers(small, large)
        stamp_mask = _oriented_stamp_mask(small, large)
        inner_r0, inner_c0 = _best_stamp_translation(large["mask"], stamp_mask, direction)
        base_r, base_c = large["bbox"][:2]
        for rr, cc in np.argwhere(stamp_mask):
            out[base_r + inner_r0 + int(rr), base_c + inner_c0 + int(cc)] = int(small["color"])

    return out


def _has_hole(mask: np.ndarray) -> bool:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if arr[r, c] or outside[r, c]:
                continue
            outside[r, c] = True
            stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not arr[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any((~arr) & (~outside)))


def _random_primitive_shape() -> np.ndarray:
    for _ in range(200):
        h = int(RNG.integers(2, 5))
        w = int(RNG.integers(2, 5))
        target_area = int(RNG.integers(3, min(h * w, 7) + 1))
        start = (int(RNG.integers(h)), int(RNG.integers(w)))
        cells = {start}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(RNG.integers(len(frontier)))]
            cells.add(pick)
        if len(cells) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        primitive, primitive_scale = _primitive_mask(mask)
        if primitive_scale != 1:
            continue
        if _has_hole(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a primitive shape")


def _direction_clear(
    large_r0: int,
    large_c0: int,
    large_mask: np.ndarray,
    small_r0: int,
    small_c0: int,
    small_mask: np.ndarray,
    direction: str,
) -> bool:
    lh, lw = large_mask.shape
    sh, sw = small_mask.shape
    dr = (small_r0 + (sh - 1) / 2.0) - (large_r0 + (lh - 1) / 2.0)
    dc = (small_c0 + (sw - 1) / 2.0) - (large_c0 + (lw - 1) / 2.0)
    if direction == "up":
        return dr < 0 and abs(dr) > abs(dc)
    if direction == "down":
        return dr > 0 and abs(dr) > abs(dc)
    if direction == "left":
        return dc < 0 and abs(dc) > abs(dr)
    return dc > 0 and abs(dc) > abs(dr)


def _can_place(mask: np.ndarray, r0: int, c0: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for rr, cc in np.argwhere(mask):
        r = r0 + int(rr)
        c = c0 + int(cc)
        if not (0 <= r < h and 0 <= c < w):
            return False
        rs = slice(max(0, r - 1), min(h, r + 2))
        cs = slice(max(0, c - 1), min(w, c + 2))
        if np.any(occupied[rs, cs]):
            return False
    return True


def _mark(mask: np.ndarray, r0: int, c0: int, occupied: np.ndarray) -> None:
    for rr, cc in np.argwhere(mask):
        occupied[r0 + int(rr), c0 + int(cc)] = True


def _paint(mask: np.ndarray, r0: int, c0: int, color: int, grid: np.ndarray) -> None:
    for rr, cc in np.argwhere(mask):
        grid[r0 + int(rr), c0 + int(cc)] = int(color)


def _candidate_large_ranges(
    direction: str,
    small_mask: np.ndarray,
    large_mask: np.ndarray,
    H: int,
    W: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    sh, sw = small_mask.shape
    lh, lw = large_mask.shape

    min_r = MARGIN
    max_r = H - MARGIN - lh
    min_c = MARGIN
    max_c = W - MARGIN - lw

    if direction == "up":
        min_r = max(min_r, MARGIN + sh + GAP_MIN)
    elif direction == "down":
        max_r = min(max_r, H - MARGIN - lh - sh - GAP_MIN)
    elif direction == "left":
        min_c = max(min_c, MARGIN + sw + GAP_MIN)
    else:
        max_c = min(max_c, W - MARGIN - lw - sw - GAP_MIN)

    if min_r > max_r or min_c > max_c:
        return None
    return (min_r, max_r), (min_c, max_c)


def _small_candidates(
    direction: str,
    large_r0: int,
    large_c0: int,
    large_mask: np.ndarray,
    small_mask: np.ndarray,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    gap = int(RNG.integers(GAP_MIN, GAP_MAX + 1))
    lh, lw = large_mask.shape
    sh, sw = small_mask.shape
    out: list[tuple[int, int]] = []

    if direction in {"up", "down"}:
        center_c = large_c0 + (lw - sw) / 2.0
        c_start = max(MARGIN, int(np.floor(center_c)) - 2)
        c_end = min(W - MARGIN - sw, int(np.ceil(center_c)) + 2)
        if direction == "up":
            r0 = large_r0 - gap - sh
        else:
            r0 = large_r0 + lh + gap
        if not (MARGIN <= r0 <= H - MARGIN - sh):
            return []
        cols = list(range(c_start, c_end + 1))
        RNG.shuffle(cols)
        cols.sort(key=lambda c0: (abs(c0 - center_c), c0))
        out = [(r0, c0) for c0 in cols]
    else:
        center_r = large_r0 + (lh - sh) / 2.0
        r_start = max(MARGIN, int(np.floor(center_r)) - 2)
        r_end = min(H - MARGIN - sh, int(np.ceil(center_r)) + 2)
        if direction == "left":
            c0 = large_c0 - gap - sw
        else:
            c0 = large_c0 + lw + gap
        if not (MARGIN <= c0 <= W - MARGIN - sw):
            return []
        rows = list(range(r_start, r_end + 1))
        RNG.shuffle(rows)
        rows.sort(key=lambda r0: (abs(r0 - center_r), r0))
        out = [(r0, c0) for r0 in rows]

    return out


def _make_pairs(bg: int) -> list[dict] | None:
    pair_count = int(RNG.choice(PAIR_COUNTS))
    palette = [c for c in range(10) if c != bg]
    if 2 * pair_count > len(palette):
        return None
    colors = [int(c) for c in RNG.choice(palette, size=2 * pair_count, replace=False)]

    if pair_count == 1:
        directions = [str(RNG.choice(DIR_NAMES))]
    else:
        directions = [str(d) for d in RNG.choice(DIR_NAMES, size=pair_count, replace=False)]
        RNG.shuffle(directions)

    pairs: list[dict] = []
    used_keys: set[tuple[int, int, tuple[int, ...]]] = set()

    for idx in range(pair_count):
        for _ in range(120):
            primitive = _random_primitive_shape()
            canonical = _canonical_mask(primitive)
            key = _mask_key(canonical)
            if key in used_keys:
                continue

            max_big_dim = 12 if pair_count == 1 else 10 if pair_count == 2 else 8
            small_scales = [1, 2]
            if max(primitive.shape) >= 4 or pair_count == 3:
                small_scales = [1]
            small_scale = int(RNG.choice(small_scales))

            large_choices = []
            for large_scale in range(small_scale + 1, 5):
                if max(primitive.shape) * large_scale > max_big_dim:
                    continue
                if int(primitive.sum()) * large_scale * large_scale > 80:
                    continue
                large_choices.append(large_scale)
            if not large_choices:
                continue

            large_scale = int(RNG.choice(large_choices))
            small_tid = int(RNG.integers(8))
            large_tid = int(RNG.integers(8))
            small_mask = _scale_mask(_apply_transform(primitive, small_tid), small_scale)
            large_mask = _scale_mask(_apply_transform(primitive, large_tid), large_scale)
            pair = {
                "direction": directions[idx],
                "small_mask": small_mask,
                "large_mask": large_mask,
                "small_color": colors[2 * idx],
                "large_color": colors[2 * idx + 1],
                "canonical": canonical,
            }

            small_comp = {
                "canonical": canonical,
                "primitive_scale": small_scale,
            }
            large_comp = {
                "mask": large_mask,
                "primitive_scale": large_scale,
            }
            try:
                stamp = _oriented_stamp_mask(small_comp, large_comp)
                _best_stamp_translation(large_mask, stamp, directions[idx])
            except AssertionError:
                continue

            used_keys.add(key)
            pairs.append(pair)
            break
        else:
            return None
    return pairs


def _place_pairs(pairs: list[dict], H: int, W: int, bg: int) -> np.ndarray | None:
    grid = np.full((H, W), int(bg), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    order = sorted(range(len(pairs)), key=lambda i: int(pairs[i]["large_mask"].sum()), reverse=True)

    for idx in order:
        pair = pairs[idx]
        direction = pair["direction"]
        small_mask = pair["small_mask"]
        large_mask = pair["large_mask"]
        ranges = _candidate_large_ranges(direction, small_mask, large_mask, H, W)
        if ranges is None:
            return None
        (min_r, max_r), (min_c, max_c) = ranges
        if min_r > max_r or min_c > max_c:
            return None

        placed = False
        for _ in range(250):
            large_r0 = int(RNG.integers(min_r, max_r + 1))
            large_c0 = int(RNG.integers(min_c, max_c + 1))
            if not _can_place(large_mask, large_r0, large_c0, occupied):
                continue

            temp_occ = occupied.copy()
            _mark(large_mask, large_r0, large_c0, temp_occ)
            for small_r0, small_c0 in _small_candidates(
                direction, large_r0, large_c0, large_mask, small_mask, H, W
            ):
                if not _can_place(small_mask, small_r0, small_c0, temp_occ):
                    continue
                if not _direction_clear(
                    large_r0, large_c0, large_mask, small_r0, small_c0, small_mask, direction
                ):
                    continue

                _paint(large_mask, large_r0, large_c0, pair["large_color"], grid)
                _paint(small_mask, small_r0, small_c0, pair["small_color"], grid)
                _mark(large_mask, large_r0, large_c0, occupied)
                _mark(small_mask, small_r0, small_c0, occupied)
                placed = True
                break
            if placed:
                break

        if not placed:
            return None

    return grid


def _valid_instance(grid: np.ndarray, pair_count: int, bg: int) -> bool:
    if int(get_bg(grid)) != int(bg):
        return False
    comps = _find_components(grid, bg)
    if len(comps) != 2 * pair_count:
        return False
    grouped: dict[tuple[int, int, tuple[int, ...]], list[dict]] = defaultdict(list)
    for comp in comps:
        grouped[comp["key"]].append(comp)
    if len(grouped) != pair_count or any(len(v) != 2 for v in grouped.values()):
        return False
    for pair in grouped.values():
        if len({comp["color"] for comp in pair}) != 2:
            return False
        areas = sorted(int(comp["area"]) for comp in pair)
        if not areas[0] < areas[1]:
            return False
    out = main(grid)
    return out.shape == grid.shape and not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        bg = int(RNG.integers(10))
        pairs = _make_pairs(bg)
        if not pairs:
            continue
        H = int(RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
        W = int(RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
        grid = _place_pairs(pairs, H, W, bg)
        if grid is None:
            continue
        if _valid_instance(grid, len(pairs), bg):
            return grid
    raise RuntimeError("failed to generate a valid sample")
