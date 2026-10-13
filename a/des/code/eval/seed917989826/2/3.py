from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Generator colors.
GEN_BG = 0
GEN_LOCK = 5
GEN_MARKER = 4
GEN_KEY_COLORS = (1, 2, 3, 6, 7, 8, 9)

# Generator bounds.
GEN_ATTEMPTS = 400
GEN_GRID_H = (22, 32)
GEN_GRID_W = (22, 32)
GEN_LOCK_COUNT = (2, 4)
GEN_RECT_MARGIN = (1, 2)
GEN_PLACE_ATTEMPTS = 160
GEN_PADDING = 2


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = DIRS
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _rotate_cw(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, -int(k))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _paint_edge(out: np.ndarray, bbox: tuple[int, int, int, int], side: int, color: int) -> None:
    r0, c0, r1, c1 = bbox
    if side == 0:
        out[r0, c0 : c1 + 1] = int(color)
    elif side == 1:
        out[r0 : r1 + 1, c1] = int(color)
    elif side == 2:
        out[r1, c0 : c1 + 1] = int(color)
    else:
        out[r0 : r1 + 1, c0] = int(color)


def _mask_is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask, connectivity=4)
    return len(comps) == 1


def _is_lock_component(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    color = int(comp["color"])
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if patch.shape[0] < 4 or patch.shape[1] < 4:
        return None
    if not np.all((patch == color) | (patch == bg)):
        return None
    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return None
    hole_mask = patch == bg
    if not hole_mask.any():
        return None
    if (
        bool(hole_mask[0, :].any())
        or bool(hole_mask[-1, :].any())
        or bool(hole_mask[:, 0].any())
        or bool(hole_mask[:, -1].any())
    ):
        return None
    hole_comps = _components(hole_mask, connectivity=4)
    if len(hole_comps) != 1:
        return None

    hole_local = hole_comps[0]
    hole_tight, (hr0, hc0, hr1, hc1) = _tight_mask_from_coords(hole_local)
    hole_abs = [(r0 + int(rr), c0 + int(cc)) for rr, cc in hole_local]
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "hole_bbox_local": (hr0, hc0, hr1, hc1),
        "hole_mask": hole_tight,
        "hole_coords": hole_abs,
    }


def _find_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict], list[dict]]:
    comps: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color, connectivity=4):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )

    locks: list[dict] = []
    markers: list[dict] = []
    keys: list[dict] = []
    for comp in comps:
        lock_info = _is_lock_component(grid, comp, bg)
        if lock_info is not None:
            locks.append(lock_info)
        elif comp["area"] == 1:
            cell = tuple(map(int, comp["coords"][0]))
            markers.append({"color": int(comp["color"]), "coord": cell})
        else:
            mask, _ = _tight_mask_from_coords(comp["coords"])
            keys.append(
                {
                    "color": int(comp["color"]),
                    "coords": [tuple(map(int, xy)) for xy in comp["coords"]],
                    "bbox": comp["bbox"],
                    "mask": mask,
                }
            )
    return locks, keys, markers


def _marker_side_for_key(key_bbox: tuple[int, int, int, int], marker: tuple[int, int]) -> int:
    r0, c0, r1, c1 = key_bbox
    mr, mc = marker
    if mr == r0 - 1 and c0 <= mc <= c1:
        return 0
    if mc == c1 + 1 and r0 <= mr <= r1:
        return 1
    if mr == r1 + 1 and c0 <= mc <= c1:
        return 2
    if mc == c0 - 1 and r0 <= mr <= r1:
        return 3
    raise AssertionError("marker must sit immediately outside one key side")


def _attach_markers(keys: list[dict], markers: list[dict]) -> None:
    used: set[int] = set()
    for key in keys:
        key_cells = set(key["coords"])
        matches: list[tuple[int, dict]] = []
        for idx, marker in enumerate(markers):
            if idx in used:
                continue
            if any(_manhattan(marker["coord"], cell) == 1 for cell in key_cells):
                matches.append((idx, marker))
        assert len(matches) == 1, "each key must have exactly one adjacent marker"
        idx, marker = matches[0]
        used.add(idx)
        key["marker_coord"] = marker["coord"]
        key["marker_color"] = int(marker["color"])
        key["marker_dir"] = _marker_side_for_key(key["bbox"], marker["coord"])
    assert len(used) == len(markers), "all markers must belong to a key"


def _match_keys_to_locks(keys: list[dict], locks: list[dict]) -> list[tuple[int, int]]:
    assert len(keys) == len(locks), "expected one key per lock"
    candidates: list[list[tuple[int, int]]] = []
    for key in keys:
        matches: list[tuple[int, int]] = []
        for lock_idx, lock in enumerate(locks):
            for rot in range(4):
                if np.array_equal(_rotate_cw(key["mask"], rot), lock["hole_mask"]):
                    matches.append((lock_idx, rot))
        assert matches, "every key must match at least one lock"
        candidates.append(matches)

    order = sorted(range(len(keys)), key=lambda i: len(candidates[i]))
    solutions: list[list[tuple[int, int] | None]] = []
    current: list[tuple[int, int] | None] = [None] * len(keys)
    used_locks: set[int] = set()

    def dfs(depth: int) -> None:
        if len(solutions) > 1:
            return
        if depth == len(order):
            solutions.append(current.copy())
            return
        key_idx = order[depth]
        for lock_idx, rot in candidates[key_idx]:
            if lock_idx in used_locks:
                continue
            used_locks.add(lock_idx)
            current[key_idx] = (lock_idx, rot)
            dfs(depth + 1)
            current[key_idx] = None
            used_locks.remove(lock_idx)

    dfs(0)
    assert len(solutions) == 1, "shape matching must be unambiguous"
    return [item for item in solutions[0] if item is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the lock-hole with the same shape under rotation.
    Fill the hole with the key color, rotate the adjacent marker direction by the
    same amount, color the corresponding outer edge of the grey rectangle, and
    erase the original key-plus-marker pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    locks, keys, markers = _find_objects(g, bg)
    assert locks and keys and markers
    _attach_markers(keys, markers)
    assignments = _match_keys_to_locks(keys, locks)

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = bg
        mr, mc = key["marker_coord"]
        out[mr, mc] = bg

    for key_idx, (lock_idx, rot) in enumerate(assignments):
        key = keys[key_idx]
        lock = locks[lock_idx]
        for r, c in lock["hole_coords"]:
            out[r, c] = int(key["color"])
        edge_side = (int(key["marker_dir"]) + int(rot)) % 4
        _paint_edge(out, lock["bbox"], edge_side, int(key["marker_color"]))

    return out


def _rows_to_mask(rows: tuple[str, ...]) -> np.ndarray:
    mask = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
    rr = np.where(mask.any(axis=1))[0]
    cc = np.where(mask.any(axis=0))[0]
    return mask[rr[0] : rr[-1] + 1, cc[0] : cc[-1] + 1]


def _rot_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = [(_rotate_cw(mask, k).shape, _rotate_cw(mask, k).tobytes()) for k in range(4)]
    return min(variants)


def _distinct_rotations(mask: np.ndarray) -> int:
    return len({(_rotate_cw(mask, k).shape, _rotate_cw(mask, k).tobytes()) for k in range(4)})


def _shape_library() -> list[np.ndarray]:
    raw = [
        ("111", "100", "110"),
        ("111", "001", "011"),
        ("111", "010", "110"),
        ("111", "010", "011"),
        ("1110", "0011", "0010"),
        ("0111", "1100", "0100"),
        ("111", "001", "010"),
        ("111", "100", "010"),
        ("1110", "0110", "0100"),
        ("0111", "0110", "0010"),
    ]
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for rows in raw:
        mask = _rows_to_mask(rows)
        if not _mask_is_connected(mask):
            continue
        if _distinct_rotations(mask) != 4:
            continue
        key = _rot_key(mask)
        if key in seen:
            continue
        seen.add(key)
        out.append(mask)
    assert len(out) >= 8
    return out


SHAPES = _shape_library()


def _render_lock_patch(
    hole_mask: np.ndarray,
    key_color: int,
    marker_color: int,
    final_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    hole_h, hole_w = hole_mask.shape
    top = random.randint(*GEN_RECT_MARGIN)
    bottom = random.randint(*GEN_RECT_MARGIN)
    left = random.randint(*GEN_RECT_MARGIN)
    right = random.randint(*GEN_RECT_MARGIN)
    rect_h = top + hole_h + bottom
    rect_w = left + hole_w + right

    inp = np.full((rect_h, rect_w), GEN_LOCK, dtype=int)
    out = np.full((rect_h, rect_w), GEN_LOCK, dtype=int)
    hole_rows, hole_cols = np.where(hole_mask)
    inp[top + hole_rows, left + hole_cols] = GEN_BG
    out[top + hole_rows, left + hole_cols] = int(key_color)
    _paint_edge(out, (0, 0, rect_h - 1, rect_w - 1), final_side, marker_color)
    return inp, out


def _build_key_patch(mask: np.ndarray, key_color: int, marker_dir: int) -> np.ndarray:
    key_cells = [tuple(map(int, xy)) for xy in np.argwhere(mask)]
    if marker_dir == 0:
        choices = [(r, c) for r, c in key_cells if r == 0]
        kr, kc = random.choice(choices)
        marker = (kr - 1, kc)
    elif marker_dir == 1:
        max_c = mask.shape[1] - 1
        choices = [(r, c) for r, c in key_cells if c == max_c]
        kr, kc = random.choice(choices)
        marker = (kr, kc + 1)
    elif marker_dir == 2:
        max_r = mask.shape[0] - 1
        choices = [(r, c) for r, c in key_cells if r == max_r]
        kr, kc = random.choice(choices)
        marker = (kr + 1, kc)
    else:
        choices = [(r, c) for r, c in key_cells if c == 0]
        kr, kc = random.choice(choices)
        marker = (kr, kc - 1)

    all_cells = key_cells + [marker]
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), GEN_BG, dtype=int)
    for r, c in key_cells:
        patch[r - min_r, c - min_c] = int(key_color)
    patch[marker[0] - min_r, marker[1] - min_c] = GEN_MARKER
    return patch


def _can_place(occupied: np.ndarray, reserve_mask: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = reserve_mask.shape
    H, W = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h >= H or c0 + w >= W:
        return False
    cells = np.argwhere(reserve_mask)
    for dr, dc in cells:
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if bool(occupied[r1:r2, c1:c2].any()):
            return False
    return True


def _stamp(occupied: np.ndarray, reserve_mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = reserve_mask.shape
    block = occupied[r0 : r0 + h, c0 : c0 + w]
    block[reserve_mask] = True


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != GEN_BG
    target[mask] = patch[mask]


def _sample_specs() -> list[dict]:
    n = random.randint(*GEN_LOCK_COUNT)
    shapes = random.sample(SHAPES, n)
    key_colors = random.sample(GEN_KEY_COLORS, n)
    specs: list[dict] = []
    for shape, key_color in zip(shapes, key_colors):
        hole_rot = random.randrange(4)
        key_rot = random.randrange(4)
        marker_dir = random.randrange(4)
        hole_mask = _rotate_cw(shape, hole_rot)
        key_mask = _rotate_cw(shape, key_rot)
        rot_needed = (hole_rot - key_rot) % 4
        final_side = (marker_dir + rot_needed) % 4
        lock_inp, lock_out = _render_lock_patch(hole_mask, key_color, GEN_MARKER, final_side)
        key_patch = _build_key_patch(key_mask, key_color, marker_dir)
        specs.append(
            {
                "lock_inp": lock_inp,
                "lock_out": lock_out,
                "lock_reserve": np.ones(lock_inp.shape, dtype=bool),
                "key_patch": key_patch,
                "key_reserve": key_patch != GEN_BG,
            }
        )
    return specs


def _attempt() -> np.ndarray | None:
    H = random.randint(*GEN_GRID_H)
    W = random.randint(*GEN_GRID_W)
    specs = _sample_specs()

    inp = np.full((H, W), GEN_BG, dtype=int)
    expected = np.full((H, W), GEN_BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    lock_order = sorted(range(len(specs)), key=lambda i: specs[i]["lock_inp"].size, reverse=True)
    for idx in lock_order:
        spec = specs[idx]
        placed = False
        h, w = spec["lock_reserve"].shape
        for _ in range(GEN_PLACE_ATTEMPTS):
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            if not _can_place(occupied, spec["lock_reserve"], r0, c0, GEN_PADDING):
                continue
            _stamp(occupied, spec["lock_reserve"], r0, c0)
            _paste_patch(inp, spec["lock_inp"], r0, c0)
            _paste_patch(expected, spec["lock_out"], r0, c0)
            placed = True
            break
        if not placed:
            return None

    key_order = sorted(range(len(specs)), key=lambda i: specs[i]["key_patch"].size, reverse=True)
    for idx in key_order:
        spec = specs[idx]
        placed = False
        h, w = spec["key_patch"].shape
        for _ in range(GEN_PLACE_ATTEMPTS):
            r0 = random.randint(1, H - h - 1)
            c0 = random.randint(1, W - w - 1)
            if not _can_place(occupied, spec["key_reserve"], r0, c0, GEN_PADDING):
                continue
            _stamp(occupied, spec["key_reserve"], r0, c0)
            _paste_patch(inp, spec["key_patch"], r0, c0)
            placed = True
            break
        if not placed:
            return None

    if np.count_nonzero(inp == GEN_BG) <= max(np.count_nonzero(inp == GEN_LOCK), np.count_nonzero(inp == GEN_MARKER)):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock input")
