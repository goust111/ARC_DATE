from __future__ import annotations

import random
import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
COLORS = tuple(range(1, 10))

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
PAIR_COUNT_RANGE = (2, 4)
BODY_H_RANGE = (4, 7)
BODY_W_RANGE = (4, 8)
SHAPE_DEPTH_RANGE = (2, 3)
SHAPE_SPAN_MAX = 5
PLACEMENT_PAD = 1
GEN_RETRIES = 400
SHAPE_RETRIES = 120
PLACE_RETRIES = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _mask_component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and _mask_component_count(mask) == 1


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = r in (0, h - 1) or c in (0, w - 1)
            while stack:
                cr, cc = stack.pop()
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        touches_edge |= nr in (0, h - 1) or nc in (0, w - 1)
                        stack.append((nr, nc))
            if not touches_edge:
                return True
    return False


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return comps


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = (tuple(rotated.shape), rotated.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rotated)
    return variants


def _matches_under_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    ta = _tight_mask(a)
    tb = _tight_mask(b)
    for rotated in _rotations(tb):
        if ta.shape == rotated.shape and np.array_equal(ta, rotated):
            return True
    return False


def _valid_protrusion(mask: np.ndarray, contact: str) -> bool:
    if not mask.any() or mask.all() or not _is_connected(mask):
        return False
    if contact == "bottom":
        return bool(np.any(mask[-1, :]))
    if contact == "top":
        return bool(np.any(mask[0, :]))
    if contact == "right":
        return bool(np.any(mask[:, -1]))
    return bool(np.any(mask[:, 0]))


def _decompose_lock(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    results: list[dict] = []

    for cut in range(1, h):
        body = mask[cut:, :]
        protrusion = mask[:cut, :]
        if body.shape[0] >= 2 and body.shape[1] >= 2 and body.all() and _valid_protrusion(protrusion, "bottom"):
            results.append(
                {
                    "side": "top",
                    "cut_coords": np.argwhere(protrusion),
                    "key_mask": _tight_mask(protrusion),
                }
            )
            break

    for cut in range(1, h):
        body = mask[: h - cut, :]
        protrusion = mask[h - cut :, :]
        if body.shape[0] >= 2 and body.shape[1] >= 2 and body.all() and _valid_protrusion(protrusion, "top"):
            results.append(
                {
                    "side": "bottom",
                    "cut_coords": np.argwhere(protrusion) + np.array([h - cut, 0]),
                    "key_mask": _tight_mask(protrusion),
                }
            )
            break

    for cut in range(1, w):
        body = mask[:, cut:]
        protrusion = mask[:, :cut]
        if body.shape[0] >= 2 and body.shape[1] >= 2 and body.all() and _valid_protrusion(protrusion, "right"):
            results.append(
                {
                    "side": "left",
                    "cut_coords": np.argwhere(protrusion),
                    "key_mask": _tight_mask(protrusion),
                }
            )
            break

    for cut in range(1, w):
        body = mask[:, : w - cut]
        protrusion = mask[:, w - cut :]
        if body.shape[0] >= 2 and body.shape[1] >= 2 and body.all() and _valid_protrusion(protrusion, "left"):
            results.append(
                {
                    "side": "right",
                    "cut_coords": np.argwhere(protrusion) + np.array([0, w - cut]),
                    "key_mask": _tight_mask(protrusion),
                }
            )
            break

    unique: list[dict] = []
    seen: set[tuple[bytes, bytes]] = set()
    for item in results:
        coords = item["cut_coords"]
        key = (coords.tobytes(), item["key_mask"].tobytes())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each same-color lock/key pair, erase the key sprite, and cut the matching protrusion off the lock.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    components = _monochrome_components(g, bg)
    assert components

    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    out = g.copy()
    matched_colors = 0

    for color, comps in by_color.items():
        candidate_locks: list[tuple[int, dict, dict]] = []
        for idx, comp in enumerate(comps):
            for decomp in _decompose_lock(comp["mask"]):
                candidate_locks.append((idx, comp, decomp))

        candidate_locks.sort(key=lambda item: -int(item[1]["area"]))
        used: set[int] = set()
        pair_count = 0
        progress = True
        while progress:
            progress = False
            for lock_idx, lock_comp, decomp in candidate_locks:
                if lock_idx in used:
                    continue
                key_idx = None
                for idx, comp in enumerate(comps):
                    if idx == lock_idx or idx in used or int(comp["area"]) >= int(lock_comp["area"]):
                        continue
                    if _matches_under_rotation(decomp["key_mask"], comp["mask"]):
                        key_idx = idx
                        break
                if key_idx is None:
                    continue

                lr0, lc0, _, _ = lock_comp["bbox"]
                cut = decomp["cut_coords"]
                out[lr0 + cut[:, 0], lc0 + cut[:, 1]] = bg

                key_coords = comps[key_idx]["coords"]
                out[key_coords[:, 0], key_coords[:, 1]] = bg

                used.add(lock_idx)
                used.add(key_idx)
                pair_count += 1
                progress = True
                break

        assert pair_count >= 1, f"color {color} must yield at least one lock/key pair"
        matched_colors += 1

    assert matched_colors == len(by_color)
    return out


def _contact_ok(mask: np.ndarray, contact: str) -> bool:
    if contact == "bottom":
        return bool(np.any(mask[-1, :]))
    if contact == "top":
        return bool(np.any(mask[0, :]))
    if contact == "right":
        return bool(np.any(mask[:, -1]))
    return bool(np.any(mask[:, 0]))


def _grow_shape(h: int, w: int, contact: str) -> np.ndarray | None:
    cells_by_contact = {
        "bottom": [(h - 1, c) for c in range(w)],
        "top": [(0, c) for c in range(w)],
        "right": [(r, w - 1) for r in range(h)],
        "left": [(r, 0) for r in range(h)],
    }
    min_area = max(3, (h * w) // 2)
    max_area = h * w - 1
    if min_area > max_area:
        return None

    for _ in range(SHAPE_RETRIES):
        target = random.randint(min_area, max_area)
        start = random.choice(cells_by_contact[contact])
        chosen = {start}
        frontier = set()
        for dr, dc in DIRS4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while frontier and len(chosen) < target:
            cell = random.choice(list(frontier))
            frontier.discard(cell)
            if cell in chosen:
                continue
            chosen.add(cell)
            for dr, dc in DIRS4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                    frontier.add((nr, nc))

        if len(chosen) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        mask = _tight_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if mask.all() or not _is_connected(mask) or _has_hole(mask) or not _contact_ok(mask, contact):
            continue
        return mask
    return None


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _build_pair_spec(color: int) -> dict | None:
    side = random.choice(SIDES)
    body_h = random.randint(*BODY_H_RANGE)
    body_w = random.randint(*BODY_W_RANGE)

    if side in {"top", "bottom"}:
        shape_h = random.randint(*SHAPE_DEPTH_RANGE)
        max_shape_w = min(SHAPE_SPAN_MAX, body_w - 2)
        if max_shape_w < 2:
            return None
        shape_w = random.randint(2, max_shape_w)
        shape = _grow_shape(shape_h, shape_w, "bottom" if side == "top" else "top")
        if shape is None:
            return None
        offset = random.randint(1, body_w - shape.shape[1] - 1)
        lock_h = shape.shape[0] + body_h
        lock_w = body_w
        lock_mask = np.zeros((lock_h, lock_w), dtype=bool)
        if side == "top":
            lock_mask[shape.shape[0] :, :] = True
            lock_mask[: shape.shape[0], offset : offset + shape.shape[1]] |= shape
            body_offset = (shape.shape[0], 0)
        else:
            lock_mask[:body_h, :] = True
            lock_mask[body_h:, offset : offset + shape.shape[1]] |= shape
            body_offset = (0, 0)
    else:
        max_shape_h = min(SHAPE_SPAN_MAX, body_h - 2)
        if max_shape_h < 2:
            return None
        shape_h = random.randint(2, max_shape_h)
        shape_w = random.randint(*SHAPE_DEPTH_RANGE)
        shape = _grow_shape(shape_h, shape_w, "right" if side == "left" else "left")
        if shape is None:
            return None
        offset = random.randint(1, body_h - shape.shape[0] - 1)
        lock_h = body_h
        lock_w = shape.shape[1] + body_w
        lock_mask = np.zeros((lock_h, lock_w), dtype=bool)
        if side == "left":
            lock_mask[:, shape.shape[1] :] = True
            lock_mask[offset : offset + shape.shape[0], : shape.shape[1]] |= shape
            body_offset = (0, shape.shape[1])
        else:
            lock_mask[:, :body_w] = True
            lock_mask[offset : offset + shape.shape[0], body_w:] |= shape
            body_offset = (0, 0)

    rot_k = random.randint(0, 3)
    key_mask = np.rot90(shape, rot_k)
    body_mask = np.ones((body_h, body_w), dtype=bool)

    assert len(_decompose_lock(lock_mask)) == 1
    assert _matches_under_rotation(shape, key_mask)

    return {
        "color": int(color),
        "lock_mask": lock_mask,
        "body_mask": body_mask,
        "body_offset": body_offset,
        "key_mask": key_mask,
    }


def _can_place(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    grid_h, grid_w = occupied.shape
    if top < 1 or left < 1 or top + h > grid_h - 1 or left + w > grid_w - 1:
        return False
    r0 = max(0, top - PLACEMENT_PAD)
    c0 = max(0, left - PLACEMENT_PAD)
    r1 = min(grid_h, top + h + PLACEMENT_PAD)
    c1 = min(grid_w, left + w + PLACEMENT_PAD)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _reserve(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> None:
    h, w = shape
    grid_h, grid_w = occupied.shape
    r0 = max(0, top - PLACEMENT_PAD)
    c0 = max(0, left - PLACEMENT_PAD)
    r1 = min(grid_h, top + h + PLACEMENT_PAD)
    c1 = min(grid_w, left + w + PLACEMENT_PAD)
    occupied[r0:r1, c0:c1] = True


def _stamp(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != 0
    region[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(COLORS, pair_count)

    pair_specs = []
    for color in colors:
        spec = _build_pair_spec(color)
        if spec is None:
            return None
        pair_specs.append(spec)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    lock_order = sorted(range(pair_count), key=lambda i: -int(pair_specs[i]["lock_mask"].sum()))
    lock_positions: dict[int, tuple[int, int]] = {}
    for idx in lock_order:
        lock_mask = pair_specs[idx]["lock_mask"]
        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(1, grid_h - lock_mask.shape[0] - 1)
            left = random.randint(1, grid_w - lock_mask.shape[1] - 1)
            if not _can_place(occupied, top, left, lock_mask.shape):
                continue
            lock_positions[idx] = (top, left)
            _reserve(occupied, top, left, lock_mask.shape)
            _stamp(grid, _paint_mask(lock_mask, pair_specs[idx]["color"]), top, left)
            bro, bco = pair_specs[idx]["body_offset"]
            _stamp(
                expected,
                _paint_mask(pair_specs[idx]["body_mask"], pair_specs[idx]["color"]),
                top + bro,
                left + bco,
            )
            placed = True
            break
        if not placed:
            return None

    key_order = sorted(range(pair_count), key=lambda i: -int(pair_specs[i]["key_mask"].sum()))
    for idx in key_order:
        key_mask = pair_specs[idx]["key_mask"]
        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(1, grid_h - key_mask.shape[0] - 1)
            left = random.randint(1, grid_w - key_mask.shape[1] - 1)
            if not _can_place(occupied, top, left, key_mask.shape):
                continue
            _reserve(occupied, top, left, key_mask.shape)
            _stamp(grid, _paint_mask(key_mask, pair_specs[idx]["color"]), top, left)
            placed = True
            break
        if not placed:
            return None

    if int(np.count_nonzero(grid)) * 2 >= grid.size:
        return None

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
    raise RuntimeError("failed to generate a valid lock/key grid")
