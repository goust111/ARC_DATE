import random
from collections import deque

import numpy as np


BG = 0
LOCK_COLOR = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator bounds.
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
OBJECT_PADDING = 1
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 36
GEN_RETRIES = 400

# Shape generation.
CORE_H_RANGE = (5, 8)
CORE_W_RANGE = (5, 8)
EXTRUSION_THICK_RANGE = (1, 3)
HOLE_COUNT_OPTIONS = (1, 2, 3)
HOLE_RECT_SIZES = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
SHAPE_RETRIES = 200
HOLE_RETRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _transform_mask(mask: np.ndarray, rot_k: int, flipped: bool) -> np.ndarray:
    arr = np.fliplr(mask) if flipped else mask
    return _tight(np.rot90(arr, rot_k))


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    tight = _tight(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in tight)


def _canonical_signature(mask: np.ndarray) -> tuple[str, ...]:
    variants = []
    for flipped in (False, True):
        for rot_k in range(4):
            variants.append(_mask_key(_transform_mask(mask, rot_k, flipped)))
    return min(variants)


def _unique_orientations(mask: np.ndarray) -> list[tuple[np.ndarray, int, bool]]:
    seen: set[tuple[str, ...]] = set()
    out: list[tuple[np.ndarray, int, bool]] = []
    for flipped in (False, True):
        for rot_k in range(4):
            oriented = _transform_mask(mask, rot_k, flipped)
            key = _mask_key(oriented)
            if key in seen:
                continue
            seen.add(key)
            out.append((oriented, int(rot_k), bool(flipped)))
    return out


def _hole_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    bg_mask = ~mask.astype(bool)
    seen = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            if not touches_edge:
                holes.append(comp)
    return holes


def _filled_mask(mask: np.ndarray) -> np.ndarray:
    filled = mask.copy().astype(bool)
    for hole in _hole_regions(mask):
        for r, c in hole:
            filled[r, c] = True
    return filled


def _interior_cells(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    interior = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                interior[r, c] = True
    return interior


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask.astype(bool))
    return len(comps) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each solid colored key to the lock with the same filled silhouette
    under a dihedral transform, delete the keys, and color every hole inside
    the matched grey lock with the key color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    components = _components(g != BG)
    assert components, "expected non-background objects"

    locks: dict[tuple[str, ...], dict[str, object]] = {}
    keys: dict[tuple[str, ...], dict[str, object]] = {}

    for comp in components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "objects must be monochrome"
        color = colors.pop()
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True

        holes = _hole_regions(mask)
        if holes:
            filled = _filled_mask(mask)
            sig = _canonical_signature(filled)
            assert sig not in locks, "lock silhouettes must be unique"
            locks[sig] = {
                "bbox": (int(r0), int(c0)),
                "holes": holes,
                "color": int(color),
            }
        else:
            sig = _canonical_signature(mask)
            assert sig not in keys, "key silhouettes must be unique"
            keys[sig] = {
                "coords": comp,
                "color": int(color),
            }

    assert locks, "expected at least one lock"
    assert len(locks) == len(keys), "locks and keys must be paired one-to-one"
    assert set(locks) == set(keys), "lock and key silhouettes must match exactly"

    for key_info in keys.values():
        for r, c in key_info["coords"]:
            out[r, c] = BG

    for sig, lock_info in locks.items():
        fill_color = int(keys[sig]["color"])
        base_r, base_c = lock_info["bbox"]
        for hole in lock_info["holes"]:
            for r, c in hole:
                out[base_r + r, base_c + c] = fill_color

    return out


def _build_outer_shape(required_holes: int) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        core_h = random.randint(*CORE_H_RANGE)
        core_w = random.randint(*CORE_W_RANGE)
        canvas = np.zeros((core_h + 8, core_w + 8), dtype=bool)
        r0 = 4
        c0 = 4
        canvas[r0 : r0 + core_h, c0 : c0 + core_w] = True

        side_order = ["top", "bottom", "left", "right"]
        random.shuffle(side_order)
        extrusion_count = random.randint(2, 4)
        for side in side_order[:extrusion_count]:
            if side in ("top", "bottom"):
                ext_h = random.randint(*EXTRUSION_THICK_RANGE)
                ext_w = random.randint(2, core_w)
                start_c = c0 + random.randint(0, core_w - ext_w)
                if side == "top":
                    canvas[r0 - ext_h : r0, start_c : start_c + ext_w] = True
                else:
                    canvas[r0 + core_h : r0 + core_h + ext_h, start_c : start_c + ext_w] = True
            else:
                ext_w = random.randint(*EXTRUSION_THICK_RANGE)
                ext_h = random.randint(2, core_h)
                start_r = r0 + random.randint(0, core_h - ext_h)
                if side == "left":
                    canvas[start_r : start_r + ext_h, c0 - ext_w : c0] = True
                else:
                    canvas[start_r : start_r + ext_h, c0 + core_w : c0 + core_w + ext_w] = True

        if random.random() < 0.6:
            side = random.choice(side_order)
            if side in ("top", "bottom"):
                ext_h = random.randint(*EXTRUSION_THICK_RANGE)
                ext_w = random.randint(2, core_w)
                start_c = c0 + random.randint(0, core_w - ext_w)
                if side == "top":
                    canvas[r0 - ext_h : r0, start_c : start_c + ext_w] = True
                else:
                    canvas[r0 + core_h : r0 + core_h + ext_h, start_c : start_c + ext_w] = True
            else:
                ext_w = random.randint(*EXTRUSION_THICK_RANGE)
                ext_h = random.randint(2, core_h)
                start_r = r0 + random.randint(0, core_h - ext_h)
                if side == "left":
                    canvas[start_r : start_r + ext_h, c0 - ext_w : c0] = True
                else:
                    canvas[start_r : start_r + ext_h, c0 + core_w : c0 + core_w + ext_w] = True

        mask = _tight(canvas)
        if not _is_connected(mask):
            continue
        if _hole_regions(mask):
            continue
        if len(_unique_orientations(mask)) != 8:
            continue
        area = int(mask.sum())
        if area < 18 or area > 60:
            continue
        if int(_interior_cells(mask).sum()) < required_holes + 2:
            continue
        return mask
    return None


def _carve_holes(filled: np.ndarray, hole_count: int) -> np.ndarray | None:
    interior = _interior_cells(filled)
    h, w = filled.shape
    candidates: list[tuple[int, int, int, int]] = []
    for hole_h, hole_w in HOLE_RECT_SIZES:
        if hole_h > h or hole_w > w:
            continue
        for r0 in range(h - hole_h + 1):
            for c0 in range(w - hole_w + 1):
                if interior[r0 : r0 + hole_h, c0 : c0 + hole_w].all():
                    candidates.append((int(r0), int(c0), int(hole_h), int(hole_w)))
    if len(candidates) < hole_count:
        return None

    for _ in range(HOLE_RETRIES):
        random.shuffle(candidates)
        taken: list[tuple[int, int, int, int]] = []
        blocked = np.zeros_like(filled, dtype=bool)
        for r0, c0, hole_h, hole_w in candidates:
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + hole_h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + hole_w + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            taken.append((r0, c0, hole_h, hole_w))
            blocked[rr0:rr1, cc0:cc1] = True
            if len(taken) == hole_count:
                break
        if len(taken) != hole_count:
            continue

        lock = filled.copy()
        for r0, c0, hole_h, hole_w in taken:
            lock[r0 : r0 + hole_h, c0 : c0 + hole_w] = False

        if not _is_connected(lock):
            continue
        if len(_hole_regions(lock)) != hole_count:
            continue
        return lock
    return None


def _make_pair(existing_signatures: set[tuple[str, ...]]) -> dict[str, object] | None:
    hole_count = random.choice(HOLE_COUNT_OPTIONS)
    for _ in range(SHAPE_RETRIES):
        filled = _build_outer_shape(hole_count)
        if filled is None:
            continue
        sig = _canonical_signature(filled)
        if sig in existing_signatures:
            continue
        lock = _carve_holes(filled, hole_count)
        if lock is None:
            continue
        return {
            "filled": filled,
            "lock": lock,
            "signature": sig,
            "hole_count": hole_count,
        }
    return None


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _sample_canvas(objects: list[np.ndarray]) -> tuple[int, int]:
    max_h = max(obj.shape[0] for obj in objects)
    max_w = max(obj.shape[1] for obj in objects)
    footprint = sum((obj.shape[0] + 2 * OBJECT_PADDING) * (obj.shape[1] + 2 * OBJECT_PADDING) for obj in objects)
    density = random.uniform(0.26, 0.38)
    target_area = int(np.ceil(footprint / density))
    aspect = random.choice((0.8, 1.0, 1.25, 1.5))
    h = int(round((target_area / aspect) ** 0.5))
    w = int(round(h * aspect))
    if random.random() < 0.5:
        h, w = w, h
    h = max(GRID_SIDE_MIN, max_h + 2, h)
    w = max(GRID_SIDE_MIN, max_w + 2, w)
    h = min(GRID_SIDE_MAX, h)
    w = min(GRID_SIDE_MAX, w)
    return int(h), int(w)


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        rr0 = max(0, r0 - OBJECT_PADDING)
        rr1 = min(h, r0 + ph + OBJECT_PADDING)
        cc0 = max(0, c0 - OBJECT_PADDING)
        cc1 = min(w, c0 + pw + OBJECT_PADDING)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw][patch != BG] = patch[patch != BG]
        occupied[rr0:rr1, cc0:cc1] = True
        return int(r0), int(c0)
    return None


def _expected_from_pairs(grid: np.ndarray, pairs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        key_r, key_c = pair["key_pos"]
        key_patch = pair["key_patch"]
        out[key_r : key_r + key_patch.shape[0], key_c : key_c + key_patch.shape[1]][key_patch != BG] = BG

        lock_r, lock_c = pair["lock_pos"]
        for hole in pair["lock_holes"]:
            for hr, hc in hole:
                out[lock_r + hr, lock_c + hc] = int(pair["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    colors = random.sample(KEY_COLORS, pair_count)
    signatures: set[tuple[str, ...]] = set()
    pairs: list[dict[str, object]] = []

    for color in colors:
        pair = _make_pair(signatures)
        if pair is None:
            return None
        signatures.add(pair["signature"])

        lock_mask = _transform_mask(pair["lock"], random.randint(0, 3), bool(random.getrandbits(1)))
        key_mask = _transform_mask(pair["filled"], random.randint(0, 3), bool(random.getrandbits(1)))
        pair["color"] = int(color)
        pair["lock_patch"] = _render_patch(lock_mask, LOCK_COLOR)
        pair["key_patch"] = _render_patch(key_mask, color)
        pair["lock_holes"] = _hole_regions(lock_mask)
        pairs.append(pair)

    objects = [pair["lock_patch"] for pair in pairs] + [pair["key_patch"] for pair in pairs]
    h, w = _sample_canvas(objects)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    entries: list[tuple[str, int, np.ndarray]] = []
    for idx, pair in enumerate(pairs):
        entries.append(("lock", idx, pair["lock_patch"]))
        entries.append(("key", idx, pair["key_patch"]))
    random.shuffle(entries)

    for kind, idx, patch in entries:
        pos = _place_patch(grid, occupied, patch)
        if pos is None:
            return None
        if kind == "lock":
            pairs[idx]["lock_pos"] = pos
        else:
            pairs[idx]["key_pos"] = pos

    expected = _expected_from_pairs(grid, pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock ARC input")
