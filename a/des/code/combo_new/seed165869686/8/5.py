from __future__ import annotations

from collections import Counter
import random

import numpy as np


OUTLINE_COLOR = 5
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[dict]:
    g = np.asarray(mask, dtype=bool)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = _NEIGHBORS8 if int(connectivity) == 8 else _NEIGHBORS4
    comps: list[dict] = []

    for r, c in np.argwhere(g):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((int(cr), int(cc2)))
            for dr, dc in dirs:
                nr = int(cr + dr)
                nc = int(cc2 + dc)
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        rs = [p[0] for p in coords]
        cs = [p[1] for p in coords]
        comps.append(
            {
                "coords": coords,
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "area": len(coords),
            }
        )
    return comps


def _mask_from_coords(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in coords:
        out[int(r), int(c)] = True
    return out


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rs = [int(r) for r, _c in coords]
    cs = [int(c) for _r, c in coords]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(np.asarray(arr), (-int(turns)) % 4)


def _has_hole(mask: np.ndarray) -> bool:
    bg_comps = _components(~np.asarray(mask, dtype=bool), connectivity=4)
    h, w = mask.shape
    for comp in bg_comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        return True
    return False


def _is_rotation_asymmetric(mask: np.ndarray) -> bool:
    base = np.asarray(mask, dtype=bool)
    for turns in (1, 2, 3):
        rot = _rotate_cw(base, turns)
        if rot.shape == base.shape and np.array_equal(rot, base):
            return False
    return True


def _marker_side(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = int(marker[0]), int(marker[1])
    r0, c0, r1, c1 = map(int, bbox)
    if r < r0:
        return 0
    if c > c1:
        return 1
    if r > r1:
        return 2
    if c < c0:
        return 3
    raise AssertionError("marker should lie outside the lock bbox")


def _extract_cavity(grid: np.ndarray, lock_coords: list[tuple[int, int]], bg: int) -> dict:
    ring_mask, ring_bbox = _tight_mask(lock_coords)
    r0, c0, _r1, _c1 = ring_bbox
    sub = np.asarray(grid[r0 : ring_bbox[2] + 1, c0 : ring_bbox[3] + 1], dtype=int)
    bg_comps = _components(sub == int(bg), connectivity=4)

    enclosed = []
    for comp in bg_comps:
        br0, bc0, br1, bc1 = map(int, comp["bbox"])
        if br0 == 0 or bc0 == 0 or br1 == sub.shape[0] - 1 or bc1 == sub.shape[1] - 1:
            continue
        enclosed.append(comp)
    assert enclosed, "expected a background cavity inside each lock"

    cavity_comp = max(enclosed, key=lambda d: int(d["area"]))
    cavity_global = [(int(r0 + r), int(c0 + c)) for r, c in cavity_comp["coords"]]
    cavity_mask, cavity_bbox = _tight_mask(cavity_global)
    return {
        "coords": cavity_global,
        "mask": cavity_mask,
        "bbox": cavity_bbox,
        "ring_bbox": ring_bbox,
        "ring_mask": ring_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored key by the direction encoded by its lock marker, then insert it into the matching grey cavity.

    Input structure:
    - The background is the dominant color.
    - Grey 1-pixel outlines form closed lockhole shapes whose interiors are background-colored cavities.
    - Each lockhole has one adjacent single-pixel colored marker outside the outline.
    - Separate solid monochrome key components share colors with the markers.

    Rule:
    - Detect each grey outline and extract its enclosed cavity shape.
    - Use marker adjacency to attach a colored marker to exactly one lockhole.
    - Match each key to the lockhole with the same marker color.
    - Interpret marker position as a clockwise rotation instruction:
      above = 0, right = 90, below = 180, left = 270.
    - Clear the original key and the marker, rotate the key accordingly, and
      paint it into the cavity while keeping the grey outline unchanged.

    Key insight: The marker color identifies the target lockhole, and the marker side independently encodes the exact rotation needed to make the key match the cavity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = g.copy()

    lock_comps = _components(g == int(OUTLINE_COLOR), connectivity=8)
    assert lock_comps, "expected at least one grey lock"

    locks: list[dict] = []
    lock_id_by_cell: dict[tuple[int, int], int] = {}
    for comp in lock_comps:
        lock = {
            "coords": comp["coords"],
            "bbox": comp["bbox"],
            "cavity": _extract_cavity(g, comp["coords"], bg),
        }
        lock_idx = len(locks)
        locks.append(lock)
        for cell in comp["coords"]:
            lock_id_by_cell[(int(cell[0]), int(cell[1]))] = lock_idx

    color_comps = _components((g != int(bg)) & (g != int(OUTLINE_COLOR)), connectivity=4)
    keys_by_color: dict[int, dict] = {}
    marker_by_lock: dict[int, dict] = {}

    for comp in color_comps:
        colors = {int(g[r, c]) for r, c in comp["coords"]}
        assert len(colors) == 1, "keys and markers should be monochrome"
        color = int(next(iter(colors)))
        if int(comp["area"]) == 1:
            mr, mc = comp["coords"][0]
            adjacent_locks = {
                lock_id_by_cell[(nr, nc)]
                for dr, dc in _NEIGHBORS4
                for nr, nc in [(int(mr + dr), int(mc + dc))]
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in lock_id_by_cell
            }
            assert len(adjacent_locks) == 1, "each marker should identify one lock"
            lock_idx = int(next(iter(adjacent_locks)))
            marker_by_lock[lock_idx] = {
                "color": color,
                "coord": (int(mr), int(mc)),
            }
        else:
            assert color not in keys_by_color, "expected one key per color"
            keys_by_color[color] = comp

    assert len(marker_by_lock) == len(locks), "each lock needs one marker"

    for comp in keys_by_color.values():
        for r, c in comp["coords"]:
            out[int(r), int(c)] = int(bg)
    for info in marker_by_lock.values():
        r, c = info["coord"]
        out[int(r), int(c)] = int(bg)

    for lock_idx, lock in enumerate(locks):
        marker = marker_by_lock[lock_idx]
        color = int(marker["color"])
        assert color in keys_by_color, "marker color should match a key"
        key_mask, _key_bbox = _tight_mask(keys_by_color[color]["coords"])
        turns = _marker_side(marker["coord"], lock["bbox"])
        rotated = _rotate_cw(key_mask, turns)
        cavity = np.asarray(lock["cavity"]["mask"], dtype=bool)
        assert rotated.shape == cavity.shape and np.array_equal(rotated, cavity), "rotated key should fit its cavity"
        cr0, cc0, _cr1, _cc1 = lock["cavity"]["bbox"]
        out[cr0 : cr0 + cavity.shape[0], cc0 : cc0 + cavity.shape[1]][cavity] = int(color)

    return out


# --------------- generator ---------------

_GEN_BG_CHOICES = [0, 8]
_GEN_PAIR_COUNT_CHOICES = [2, 3, 4]
_GEN_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_MIN_SIDE = {2: 16, 3: 20, 4: 24}
_GEN_MAX_RETRIES = 500


def _abs_coords(rel_coords: list[tuple[int, int]], origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, origin)
    return [(int(r0 + r), int(c0 + c)) for r, c in rel_coords]


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _mark_halo(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in _NEIGHBORS8 + [(0, 0)]:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _find_origin(rel_coords: list[tuple[int, int]], blocked: np.ndarray) -> tuple[int, int] | None:
    h, w = blocked.shape
    min_r = min(int(r) for r, _c in rel_coords)
    max_r = max(int(r) for r, _c in rel_coords)
    min_c = min(int(c) for _r, c in rel_coords)
    max_c = max(int(c) for _r, c in rel_coords)
    row_lo = -min_r
    row_hi = h - 1 - max_r
    col_lo = -min_c
    col_hi = w - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    origins = [(r0, c0) for r0 in range(row_lo, row_hi + 1) for c0 in range(col_lo, col_hi + 1)]
    random.shuffle(origins)
    for r0, c0 in origins:
        ok = True
        for rr, cc in rel_coords:
            ar = int(r0 + rr)
            ac = int(c0 + cc)
            if blocked[ar, ac]:
                ok = False
                break
        if ok:
            return int(r0), int(c0)
    return None


def _gen_shape() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        target = random.randint(4, min(8, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}

        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        degree = sum((nr + xr, nc + xc) in cells for xr, xc in _NEIGHBORS4)
                        frontier.extend([(nr, nc)] * max(1, degree))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target:
            continue
        rs = [int(r) for r, _c in cells]
        cs = [int(c) for _r, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[int(r) - r0, int(c) - c0] = True

        if min(mask.shape) < 2 or int(mask.sum()) < 4:
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if _has_hole(mask):
            continue
        if not _is_rotation_asymmetric(mask):
            continue
        return _rotate_cw(mask, random.randint(0, 3))
    raise RuntimeError("failed to generate an asymmetric cavity shape")


def _lock_geometry(shape: np.ndarray) -> dict:
    cavity = {(int(r + 1), int(c + 1)) for r, c in np.argwhere(shape)}
    ph, pw = int(shape.shape[0] + 2), int(shape.shape[1] + 2)
    ring = set()
    for r, c in cavity:
        for dr, dc in _NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in cavity:
                ring.add((nr, nc))

    ring_mask = _mask_from_coords(list(ring), (ph, pw))
    if len(_components(ring_mask, connectivity=8)) != 1:
        raise RuntimeError("lock outline should stay connected")

    marker_options = {
        0: [(int(r - 1), int(c)) for r, c in ring if r == 0],
        1: [(int(r), int(c + 1)) for r, c in ring if c == pw - 1],
        2: [(int(r + 1), int(c)) for r, c in ring if r == ph - 1],
        3: [(int(r), int(c - 1)) for r, c in ring if c == 0],
    }
    assert all(marker_options[side] for side in range(4)), "each side should allow one marker"
    return {
        "shape": np.asarray(shape, dtype=bool),
        "ring": sorted((int(r), int(c)) for r, c in ring),
        "cavity": sorted((int(r), int(c)) for r, c in cavity),
        "marker_options": marker_options,
    }


def _build_case() -> np.ndarray | None:
    pair_count = random.choice(_GEN_PAIR_COUNT_CHOICES)
    side_min = _GEN_MIN_SIDE[pair_count]
    h = random.randint(side_min, 30)
    w = random.randint(side_min, 30)
    bg = random.choice(_GEN_BG_CHOICES)
    palette = [color for color in _GEN_COLOR_POOL if color not in {bg, OUTLINE_COLOR}]
    colors = random.sample(palette, pair_count)
    directions = random.sample([0, 1, 2, 3], pair_count)

    inp = np.full((h, w), int(bg), dtype=int)
    expected = np.full((h, w), int(bg), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    pairs = []
    for color, turns in zip(colors, directions):
        shape = _gen_shape()
        geom = _lock_geometry(shape)
        pairs.append(
            {
                "color": int(color),
                "turns": int(turns),
                "shape": shape,
                "geom": geom,
                "reserve_area": len(geom["ring"]) + len(geom["cavity"]) + 1,
            }
        )
    pairs.sort(key=lambda d: int(d["reserve_area"]), reverse=True)

    for spec in pairs:
        marker_rel = random.choice(spec["geom"]["marker_options"][spec["turns"]])
        reserve_rel = list(spec["geom"]["ring"]) + list(spec["geom"]["cavity"]) + [marker_rel]
        origin = _find_origin(reserve_rel, blocked)
        if origin is None:
            return None

        ring_abs = _abs_coords(spec["geom"]["ring"], origin)
        cavity_abs = _abs_coords(spec["geom"]["cavity"], origin)
        marker_abs = _abs_coords([marker_rel], origin)[0]

        _paint(inp, ring_abs, OUTLINE_COLOR)
        inp[int(marker_abs[0]), int(marker_abs[1])] = int(spec["color"])
        _paint(expected, ring_abs, OUTLINE_COLOR)
        _paint(expected, cavity_abs, int(spec["color"]))
        _mark_halo(blocked, ring_abs + cavity_abs + [marker_abs])

        key_mask = np.rot90(np.asarray(spec["shape"], dtype=bool), int(spec["turns"]))
        key_rel = [(int(r), int(c)) for r, c in np.argwhere(key_mask)]
        key_origin = _find_origin(key_rel, blocked)
        if key_origin is None:
            return None
        key_abs = _abs_coords(key_rel, key_origin)
        _paint(inp, key_abs, int(spec["color"]))
        _mark_halo(blocked, key_abs)

    if _get_bg(inp) != int(bg) or _get_bg(expected) != int(bg):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
