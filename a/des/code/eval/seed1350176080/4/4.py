from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
COLOR_TO_LIGHT = {
    1: 8,  # blue -> azure
    2: 7,  # red -> orange
    3: 4,  # green -> yellow
    6: 9,  # magenta -> maroon
}
KEY_COLORS = tuple(COLOR_TO_LIGHT.keys())
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (22, 32)
NUM_LOCKS_RANGE = (2, 4)
LOCK_MARGIN_RANGE = (1, 3)
LOCK_PAD = 2
KEY_PAD = 1
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _rotate_coords(coords, k: int) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    k %= 4
    if k == 0:
        rot = pts
    elif k == 1:
        rot = [(c, -r) for r, c in pts]
    elif k == 2:
        rot = [(-r, -c) for r, c in pts]
    else:
        rot = [(-c, r) for r, c in pts]
    return _normalize_coords(rot)


def _canonical_rotation(coords) -> tuple[tuple[int, int], ...]:
    pts = _normalize_coords(coords)
    return min(_rotate_coords(pts, k) for k in range(4))


def _coords_to_mask(coords) -> np.ndarray:
    pts = [(int(r), int(c)) for r, c in coords]
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    coords = np.argwhere(mask)
    sig = _normalize_coords((int(r), int(c)) for r, c in coords)
    return sig, tuple(int(x) for x in mask.shape)


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = _mask_signature(rot)
        if key in seen:
            continue
        seen.add(key)
        variants.append(rot.copy())
    return variants


def _build_shape_catalog() -> tuple[tuple[tuple[int, int], ...], ...]:
    shapes_by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 7):
        grown: set[tuple[tuple[int, int], ...]] = set()
        for shape in shapes_by_size[size - 1]:
            cells = set(shape)
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            for cell in frontier:
                new_shape = set(cells)
                new_shape.add(cell)
                grown.add(_canonical_rotation(new_shape))
        shapes_by_size[size] = grown
    catalog: list[tuple[tuple[int, int], ...]] = []
    for size in range(3, 7):
        catalog.extend(sorted(shapes_by_size[size]))
    return tuple(catalog)


SHAPE_CATALOG = _build_shape_catalog()


def _extract_locks(grid: np.ndarray) -> list[dict]:
    locks: list[dict] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((sub != GREY) & (sub != BG)):
            continue
        holes = []
        for rel in _components(sub == BG):
            touches = (
                np.any(rel[:, 0] == 0)
                or np.any(rel[:, 0] == sub.shape[0] - 1)
                or np.any(rel[:, 1] == 0)
                or np.any(rel[:, 1] == sub.shape[1] - 1)
            )
            if not touches:
                holes.append(rel)
        if len(holes) != 1:
            continue
        hole_rel = holes[0]
        hole_area = int(len(hole_rel))
        if not (3 <= hole_area <= 6):
            continue
        if int(np.count_nonzero(sub == GREY)) + hole_area != int(sub.size):
            continue
        hole_shape = _normalize_coords((int(r), int(c)) for r, c in hole_rel)
        hole_abs = [(r0 + int(r), c0 + int(c)) for r, c in hole_rel]
        grey_abs = [(int(r), int(c)) for r, c in coords]
        center_row = (r0 + r1) / 2.0
        locks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "grey_coords": grey_abs,
                "hole_coords": hole_abs,
                "hole_mask": _coords_to_mask(hole_shape),
                "area": hole_area,
                "is_upper": center_row < (grid.shape[0] / 2.0),
            }
        )
    locks.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return locks


def _extract_keys(grid: np.ndarray) -> list[dict]:
    keys: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)):
        for coords in _components(grid == color):
            area = int(len(coords))
            if not (3 <= area <= 6):
                continue
            shape = _normalize_coords((int(r), int(c)) for r, c in coords)
            keys.append(
                {
                    "bbox": _bbox(coords),
                    "coords": [(int(r), int(c)) for r, c in coords],
                    "mask": _coords_to_mask(shape),
                    "area": area,
                    "color": color,
                }
            )
    keys.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return keys


def _match_keys_to_locks(keys: list[dict], locks: list[dict]) -> list[int]:
    candidates: list[list[int]] = []
    for key in keys:
        matches: list[int] = []
        for lock_index, lock in enumerate(locks):
            if key["area"] != lock["area"]:
                continue
            for rot in _unique_rotations(key["mask"]):
                if rot.shape == lock["hole_mask"].shape and np.array_equal(rot, lock["hole_mask"]):
                    matches.append(lock_index)
                    break
        candidates.append(matches)
    assert all(candidates), "every key must match at least one lock"

    order = sorted(range(len(keys)), key=lambda i: (len(candidates[i]), keys[i]["bbox"]))
    assignment = [-1] * len(keys)
    used: set[int] = set()

    def dfs(depth: int) -> bool:
        if depth == len(order):
            return True
        key_index = order[depth]
        for lock_index in candidates[key_index]:
            if lock_index in used:
                continue
            used.add(lock_index)
            assignment[key_index] = lock_index
            if dfs(depth + 1):
                return True
            assignment[key_index] = -1
            used.remove(lock_index)
        return False

    assert dfs(0), "matching must be one-to-one"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each colored key into the matching grey lock-hole and recolor upper locks.

    Locks are grey filled rectangles with one enclosed black hole. Keys are small
    monochrome shapes elsewhere on the grid. Each key matches exactly one hole
    under rotation. After insertion, upper-half lock regions recolor their
    remaining grey pixels with the key's paired light color.
    """
    g = np.asarray(grid, dtype=int)
    locks = _extract_locks(g)
    keys = _extract_keys(g)
    assert 2 <= len(locks) <= 4, "expected 2-4 lock regions"
    assert len(keys) == len(locks), "number of keys must equal number of locks"

    assignment = _match_keys_to_locks(keys, locks)
    out = g.copy()

    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = BG

    for key_index, lock_index in enumerate(assignment):
        key = keys[key_index]
        lock = locks[lock_index]
        color = int(key["color"])
        assert color in COLOR_TO_LIGHT, "unsupported key color"
        for r, c in lock["hole_coords"]:
            out[r, c] = color
        if lock["is_upper"]:
            fill = int(COLOR_TO_LIGHT[color])
            for r, c in lock["grey_coords"]:
                out[r, c] = fill

    return out


def _rect_is_free(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(mask[r0:r1, c0:c1].any())


def _mark_rect(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    mask[r0:r1, c0:c1] = True


def _candidate_positions(
    reserved: np.ndarray,
    height: int,
    width: int,
    row_values,
    pad: int,
) -> list[tuple[int, int]]:
    grid_h, grid_w = reserved.shape
    positions = []
    for top in row_values:
        for left in range(1, grid_w - width):
            if _rect_is_free(reserved, int(top), int(left), height, width, pad):
                positions.append((int(top), int(left)))
    random.shuffle(positions)
    return positions


def _make_lock_patch(shape) -> tuple[np.ndarray, np.ndarray]:
    hole_mask = _coords_to_mask(shape)
    hh, hw = hole_mask.shape
    top = random.randint(*LOCK_MARGIN_RANGE)
    bottom = random.randint(*LOCK_MARGIN_RANGE)
    left = random.randint(*LOCK_MARGIN_RANGE)
    right = random.randint(*LOCK_MARGIN_RANGE)
    patch = np.full((hh + top + bottom, hw + left + right), GREY, dtype=int)
    hole_rows, hole_cols = np.nonzero(hole_mask)
    patch[top + hole_rows, left + hole_cols] = BG
    return patch, hole_mask


def _render_expected(grid: np.ndarray, pair_records: list[dict]) -> np.ndarray:
    out = grid.copy()
    for rec in pair_records:
        for r, c in rec["key_coords"]:
            out[r, c] = BG
    for rec in pair_records:
        color = int(rec["color"])
        for r, c in rec["hole_coords"]:
            out[r, c] = color
        if rec["branch"] == "upper":
            fill = int(COLOR_TO_LIGHT[color])
            for r, c in rec["grey_coords"]:
                out[r, c] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    num_pairs = random.randint(*NUM_LOCKS_RANGE)

    if num_pairs % 2 == 0:
        upper_count = num_pairs // 2
    else:
        upper_count = random.choice([num_pairs // 2, num_pairs // 2 + 1])
    branches = ["upper"] * upper_count + ["lower"] * (num_pairs - upper_count)
    random.shuffle(branches)

    shapes = random.sample(SHAPE_CATALOG, num_pairs)
    colors = random.sample(KEY_COLORS, num_pairs)

    specs: list[dict] = []
    for shape, color, branch in zip(shapes, colors, branches):
        lock_patch, hole_mask = _make_lock_patch(shape)
        specs.append(
            {
                "shape": shape,
                "color": int(color),
                "branch": branch,
                "lock_patch": lock_patch,
                "hole_mask": hole_mask,
            }
        )
    specs.sort(key=lambda item: item["lock_patch"].size, reverse=True)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    pair_records: list[dict] = []

    for spec in specs:
        patch = spec["lock_patch"]
        lock_h, lock_w = patch.shape
        mid = grid_h // 2
        if spec["branch"] == "upper":
            row_values = range(1, mid - lock_h)
        else:
            row_values = range(mid + 1, grid_h - lock_h)
        positions = _candidate_positions(reserved, lock_h, lock_w, row_values, LOCK_PAD)
        if not positions:
            return None
        top, left = random.choice(positions)
        grid[top : top + lock_h, left : left + lock_w] = patch
        _mark_rect(reserved, top, left, lock_h, lock_w, LOCK_PAD)

        grey_local = np.argwhere(patch == GREY)
        hole_local = np.argwhere(patch == BG)
        pair_records.append(
            {
                "branch": spec["branch"],
                "color": spec["color"],
                "hole_mask": spec["hole_mask"],
                "hole_coords": [(top + int(r), left + int(c)) for r, c in hole_local],
                "grey_coords": [(top + int(r), left + int(c)) for r, c in grey_local],
            }
        )

    key_records = list(pair_records)
    random.shuffle(key_records)
    key_reserved = reserved.copy()

    for rec in key_records:
        key_mask = np.rot90(rec["hole_mask"], random.randrange(4)).copy()
        key_h, key_w = key_mask.shape
        positions = _candidate_positions(key_reserved, key_h, key_w, range(1, grid_h - key_h), KEY_PAD)
        if not positions:
            return None
        top, left = random.choice(positions)
        rows, cols = np.nonzero(key_mask)
        for rr, cc in zip(rows, cols):
            grid[top + int(rr), left + int(cc)] = int(rec["color"])
        _mark_rect(key_reserved, top, left, key_h, key_w, KEY_PAD)
        rec["key_coords"] = [(top + int(r), left + int(c)) for r, c in zip(rows, cols)]

    expected = _render_expected(grid, pair_records)
    try:
        parsed_locks = _extract_locks(grid)
        parsed_keys = _extract_keys(grid)
        if len(parsed_locks) != num_pairs or len(parsed_keys) != num_pairs:
            return None
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
