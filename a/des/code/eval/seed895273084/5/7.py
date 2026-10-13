from __future__ import annotations

from collections import defaultdict, deque
import itertools
import numpy as np


BG = 0
GREY = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}

_TEMPLATE_CACHE: dict[str, dict[tuple[int, ...], list[np.ndarray]]] | None = None


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _edge_profile(mask: np.ndarray, side: str) -> tuple[int, ...]:
    if side == "top":
        line = mask[0, :]
    elif side == "bottom":
        line = mask[-1, :]
    elif side == "left":
        line = mask[:, 0]
    else:
        line = mask[:, -1]
    return tuple(int(x) for x in line.tolist())


def _is_contiguous(bits: tuple[int, ...]) -> bool:
    idx = [i for i, x in enumerate(bits) if x]
    return bool(idx) and idx == list(range(idx[0], idx[-1] + 1))


def _parse_lock(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None
    coord_set = set(coords)
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    if int(grid[r0, c0]) != GREY or int(grid[r0, c1]) != GREY or int(grid[r1, c0]) != GREY or int(grid[r1, c1]) != GREY:
        return None
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.all(interior == BG):
        return None
    side_lines = {
        "top": grid[r0, c0 + 1 : c1],
        "bottom": grid[r1, c0 + 1 : c1],
        "left": grid[r0 + 1 : r1, c0],
        "right": grid[r0 + 1 : r1, c1],
    }
    gap_side = None
    gap_profile = None
    for side, line in side_lines.items():
        vals = [int(x) for x in line.tolist()]
        if any(v not in (BG, GREY) for v in vals):
            return None
        miss = tuple(1 if v == BG else 0 for v in vals)
        miss_count = sum(miss)
        if miss_count == 0:
            if any(v != GREY for v in vals):
                return None
            continue
        if gap_side is not None:
            return None
        if miss_count not in (1, 2) or not _is_contiguous(miss):
            return None
        gap_side = side
        gap_profile = miss
    if gap_side is None or gap_profile is None:
        return None
    expected_border = set()
    for rr in range(r0, r1 + 1):
        expected_border.add((rr, c0))
        expected_border.add((rr, c1))
    for cc in range(c0, c1 + 1):
        expected_border.add((r0, cc))
        expected_border.add((r1, cc))
    missing = expected_border - coord_set
    expected_missing = set()
    if gap_side == "top":
        for i, bit in enumerate(gap_profile):
            if bit:
                expected_missing.add((r0, c0 + 1 + i))
    elif gap_side == "bottom":
        for i, bit in enumerate(gap_profile):
            if bit:
                expected_missing.add((r1, c0 + 1 + i))
    elif gap_side == "left":
        for i, bit in enumerate(gap_profile):
            if bit:
                expected_missing.add((r0 + 1 + i, c0))
    else:
        for i, bit in enumerate(gap_profile):
            if bit:
                expected_missing.add((r0 + 1 + i, c1))
    if missing != expected_missing:
        return None
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": gap_side,
        "signature": (gap_side, gap_profile),
    }


def _find_locks(grid: np.ndarray) -> list[dict]:
    locks = []
    for coords in _components(grid == GREY):
        info = _parse_lock(grid, coords)
        if info is not None:
            locks.append(info)
    locks.sort(key=lambda d: d["bbox"])
    return locks


def _find_keys(grid: np.ndarray) -> list[dict]:
    keys = []
    mask = (grid != BG) & (grid != GREY)
    for coords in _components(mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "key components must be monochrome"
        key_mask, bbox = _tight_mask(coords)
        keys.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": key_mask,
                "color": int(next(iter(colors))),
            }
        )
    keys.sort(key=lambda d: d["bbox"])
    return keys


def _key_matches(lock_sig: tuple[str, tuple[int, ...]], key_mask: np.ndarray) -> bool:
    lock_side, gap_profile = lock_sig
    edge_side = OPPOSITE[lock_side]
    for k in range(4):
        rotated = np.rot90(key_mask, k)
        if _edge_profile(rotated, edge_side) == gap_profile:
            return True
    return False


def _solve_matchings(locks: list[dict], keys: list[dict], limit: int = 2) -> list[list[int]]:
    if len(locks) != len(keys):
        return []
    candidates = []
    for lock in locks:
        opts = [ki for ki, key in enumerate(keys) if _key_matches(lock["signature"], key["mask"])]
        candidates.append(opts)
    if any(not opts for opts in candidates):
        return []

    order = sorted(range(len(locks)), key=lambda i: (len(candidates[i]), locks[i]["bbox"]))
    solutions: list[list[int]] = []
    assign = [-1] * len(locks)
    used = [False] * len(keys)

    def dfs(pos: int) -> None:
        if len(solutions) >= limit:
            return
        if pos == len(order):
            solutions.append(assign.copy())
            return
        li = order[pos]
        for ki in sorted(candidates[li], key=lambda idx: keys[idx]["bbox"]):
            if used[ki]:
                continue
            used[ki] = True
            assign[li] = int(ki)
            dfs(pos + 1)
            assign[li] = -1
            used[ki] = False

    dfs(0)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match rotated keys to lock gaps, fill lock interiors with key colors, and erase the keys.
    """
    g = np.asarray(grid, dtype=int)
    locks = _find_locks(g)
    keys = _find_keys(g)
    solutions = _solve_matchings(locks, keys, limit=2)
    assert len(solutions) == 1, "expected a unique key-lock matching"

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = BG
    for lock_idx, key_idx in enumerate(solutions[0]):
        color = int(keys[key_idx]["color"])
        r0, c0, r1, c1 = locks[lock_idx]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _build_template_cache() -> dict[str, dict[tuple[int, ...], list[np.ndarray]]]:
    library: dict[str, dict[tuple[int, ...], list[np.ndarray]]] = {
        side: defaultdict(list) for side in SIDES
    }
    unique_masks: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for h in range(1, 5):
        for w in range(1, 5):
            cell_count = h * w
            coords = [(r, c) for r in range(h) for c in range(w)]
            for area in range(3, 6):
                if area > cell_count:
                    continue
                for subset in itertools.combinations(range(cell_count), area):
                    mask = np.zeros((h, w), dtype=bool)
                    for idx in subset:
                        rr, cc = coords[idx]
                        mask[rr, cc] = True
                    if len(_components(mask)) != 1:
                        continue
                    cropped = _crop_bool(mask)
                    if min(cropped.shape) < 2:
                        continue
                    unique_masks[(tuple(map(int, cropped.shape)), cropped.tobytes())] = cropped

    for mask in unique_masks.values():
        for lock_side in SIDES:
            edge_side = OPPOSITE[lock_side]
            profile = _edge_profile(mask, edge_side)
            if len(profile) not in (3, 4):
                continue
            if sum(profile) not in (1, 2) or not _is_contiguous(profile):
                continue
            library[lock_side][profile].append(mask.copy())
    return library


def _template_cache() -> dict[str, dict[tuple[int, ...], list[np.ndarray]]]:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = _build_template_cache()
    return _TEMPLATE_CACHE


def _lock_patch(side: str, profile: tuple[int, ...], other_dim: int) -> np.ndarray:
    if side in ("top", "bottom"):
        inner_h = int(other_dim)
        inner_w = int(len(profile))
    else:
        inner_h = int(len(profile))
        inner_w = int(other_dim)
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    if side == "top":
        for i, bit in enumerate(profile):
            if bit:
                patch[0, 1 + i] = BG
    elif side == "bottom":
        for i, bit in enumerate(profile):
            if bit:
                patch[-1, 1 + i] = BG
    elif side == "left":
        for i, bit in enumerate(profile):
            if bit:
                patch[1 + i, 0] = BG
    else:
        for i, bit in enumerate(profile):
            if bit:
                patch[1 + i, -1] = BG
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _reserve_ok(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    hh, ww = reserved.shape
    r0 = max(0, top - pad)
    r1 = min(hh, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(ww, left + w + pad)
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve_box(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    hh, ww = reserved.shape
    r0 = max(0, top - pad)
    r1 = min(hh, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(ww, left + w + pad)
    reserved[r0:r1, c0:c1] = True


def _random_location(
    rng: np.random.Generator,
    reserved: np.ndarray,
    shape: tuple[int, int],
    pad: int,
    tries: int = 200,
) -> tuple[int, int] | None:
    h, w = map(int, shape)
    hh, ww = reserved.shape
    if h > hh or w > ww:
        return None
    for _ in range(tries):
        top = int(rng.integers(0, hh - h + 1))
        left = int(rng.integers(0, ww - w + 1))
        if _reserve_ok(reserved, top, left, h, w, pad):
            return top, left
    return None


def _expected_output(
    inp: np.ndarray,
    lock_records: list[dict],
    key_records: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for rec in key_records:
        top = int(rec["top"])
        left = int(rec["left"])
        mask = rec["patch"] != BG
        out[top : top + rec["patch"].shape[0], left : left + rec["patch"].shape[1]][mask] = BG
    for rec in lock_records:
        r0, c0 = int(rec["top"]), int(rec["left"])
        h, w = rec["patch"].shape
        color = int(rec["color"])
        out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = color
    return out


def _sample_sides(rng: np.random.Generator, n: int) -> list[str]:
    sides: list[str] = []
    while len(sides) < n:
        batch = list(SIDES)
        rng.shuffle(batch)
        sides.extend(batch)
    return sides[:n]


def _sample_widths(rng: np.random.Generator, n: int) -> list[int]:
    widths = ([1, 2] * ((n + 1) // 2))[:n]
    rng.shuffle(widths)
    return widths


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    library = _template_cache()
    n_pairs = int(rng.integers(2, 5))
    sides = _sample_sides(rng, n_pairs)
    widths = _sample_widths(rng, n_pairs)

    palette = [c for c in range(1, 10) if c != GREY]
    rng.shuffle(palette)
    colors = palette[:n_pairs]

    lock_specs = []
    used_signatures: set[tuple[str, tuple[int, ...]]] = set()
    for side, gap_width, color in zip(sides, widths, colors, strict=False):
        profiles = [
            profile
            for profile in library[side]
            if sum(profile) == gap_width and (side, profile) not in used_signatures
        ]
        if not profiles:
            return None
        profile = profiles[int(rng.integers(0, len(profiles)))]
        used_signatures.add((side, profile))
        key_templates = library[side][profile]
        key_patch = key_templates[int(rng.integers(0, len(key_templates)))].copy()
        key_patch = np.rot90(key_patch, int(rng.integers(0, 4)))
        painted_key = np.full(key_patch.shape, BG, dtype=int)
        painted_key[key_patch] = int(color)
        lock_other_dim = int(rng.integers(2, 5))
        lock_specs.append(
            {
                "side": side,
                "profile": profile,
                "color": int(color),
                "key_patch": painted_key,
                "lock_patch": _lock_patch(side, profile, lock_other_dim),
            }
        )

    total_area = sum(spec["lock_patch"].size + spec["key_patch"].size for spec in lock_specs)
    base_h = max(18, int(np.ceil(np.sqrt(total_area * 2.4))))
    base_w = max(18, int(np.ceil(np.sqrt(total_area * 2.8))))
    h = int(rng.integers(base_h, base_h + 8))
    w = int(rng.integers(base_w, base_w + 10))
    h = min(h, 30)
    w = min(w, 30)
    if h < 18 or w < 18:
        return None

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    lock_records: list[dict] = []
    key_records: list[dict] = []

    for spec in sorted(lock_specs, key=lambda d: -d["lock_patch"].size):
        patch = spec["lock_patch"]
        loc = _random_location(rng, reserved, patch.shape, pad=1)
        if loc is None:
            return None
        top, left = loc
        _place_patch(grid, patch, top, left)
        _reserve_box(reserved, top, left, patch.shape[0], patch.shape[1], pad=1)
        lock_records.append(
            {
                "top": int(top),
                "left": int(left),
                "patch": patch,
                "side": spec["side"],
                "profile": spec["profile"],
                "color": spec["color"],
            }
        )

    for spec in sorted(lock_specs, key=lambda d: -d["key_patch"].size):
        patch = spec["key_patch"]
        loc = _random_location(rng, reserved, patch.shape, pad=1)
        if loc is None:
            return None
        top, left = loc
        _place_patch(grid, patch, top, left)
        _reserve_box(reserved, top, left, patch.shape[0], patch.shape[1], pad=1)
        key_records.append(
            {
                "top": int(top),
                "left": int(left),
                "patch": patch,
                "color": spec["color"],
            }
        )

    expected = _expected_output(grid, lock_records, key_records)
    parsed_locks = _find_locks(grid)
    parsed_keys = _find_keys(grid)
    if len(parsed_locks) != n_pairs or len(parsed_keys) != n_pairs:
        return None
    if len(_solve_matchings(parsed_locks, parsed_keys, limit=2)) != 1:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-key puzzle")
