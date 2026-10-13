from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_foreground(grid: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox(np.asarray(grid) != int(bg))
    assert box is not None, "expected foreground"
    return _crop_bbox(grid, box)


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _component_counts_by_color(grid: np.ndarray, bg: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        counts[int(color)] = len(_components_of_mask(np.asarray(grid) == int(color)))
    return counts


def _shift_side(grid: np.ndarray, cut: int, move_top: bool, dx: int, bg: int) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, int(bg))

    if move_top:
        out[cut + 1 :, :] = g[cut + 1 :, :]
        coords = np.argwhere(g[: cut + 1, :] != int(bg))
    else:
        out[: cut + 1, :] = g[: cut + 1, :]
        tail = np.argwhere(g[cut + 1 :, :] != int(bg))
        coords = tail.copy()
        coords[:, 0] += cut + 1

    if coords.size == 0:
        return None
    shifted_cols = coords[:, 1] + int(dx)
    if np.any((shifted_cols < 0) | (shifted_cols >= w)):
        return None
    out[coords[:, 0], shifted_cols] = g[coords[:, 0], coords[:, 1]]
    return out


def _candidate_records(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    before_counts = _component_counts_by_color(g, bg)
    records: list[dict] = []

    for cut in range(h - 1):
        for move_top in (True, False):
            for dx in range(-(w - 1), w):
                if dx == 0:
                    continue
                restored = _shift_side(g, cut, move_top, dx, bg)
                if restored is None:
                    continue
                seam_mask = (restored[cut, :] != int(bg)) & (restored[cut, :] == restored[cut + 1, :])
                seam_pairs = int(np.count_nonzero(seam_mask))
                if seam_pairs == 0:
                    continue
                seam_colors = {
                    int(color)
                    for color in restored[cut, seam_mask]
                }
                after_counts = _component_counts_by_color(restored, bg)
                all_colors = set(before_counts) | set(after_counts)
                gains = {
                    color: int(before_counts.get(color, 0) - after_counts.get(color, 0))
                    for color in all_colors
                }
                component_gain = sum(value for value in gains.values() if value > 0)
                color_gain = sum(1 for value in gains.values() if value > 0)
                if component_gain <= 0:
                    continue
                records.append(
                    {
                        "cut": int(cut),
                        "move_top": bool(move_top),
                        "dx": int(dx),
                        "restored": restored,
                        "crop": _crop_foreground(restored, bg),
                        "score": (
                            int(component_gain),
                            int(color_gain),
                            int(len(seam_colors)),
                            int(seam_pairs),
                        ),
                    }
                )
    return records


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Undo the unique hidden horizontal fault shift and crop the restored foreground.

    Input structure:
    - A black background contains several separated colored objects.
    - The picture was cut between two rows, and every foreground cell on exactly one side of that invisible cut was shifted horizontally by a nonzero amount.
    - Some objects cross the cut and therefore appear broken into mismatched upper and lower pieces, while other complete single-side objects act as distractors.

    Rule:
    - Search for the unique cut position, side choice, and horizontal undo-shift that makes the broken cross-cut objects reconnect across the cut.
    - Apply that restoring shift to the moved side only.
    - Return the tight bounding-box crop of all restored foreground cells.

    Key insight:
    - The cut itself is never drawn; it is inferred because multiple apparent fragments can all be healed by the same row split and the same horizontal realignment.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)

    records = _candidate_records(g, bg)
    assert records, "expected at least one valid realignment"
    best_score = max(record["score"] for record in records)
    best = [record for record in records if record["score"] == best_score]
    if len(best) == 1:
        return best[0]["crop"]

    crop_keys = {
        (record["crop"].shape, tuple(map(int, record["crop"].ravel()))): record["crop"]
        for record in best
    }
    assert len(crop_keys) == 1, "expected a unique restored crop"
    return next(iter(crop_keys.values()))


def _dilate_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True
    return mask


def _grow_connected(
    allowed: set[tuple[int, int]],
    seeds: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    cells = set(seeds)
    while len(cells) < int(target_size):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nbr = (r + dr, c + dc)
                if nbr in allowed and nbr not in cells:
                    frontier.add(nbr)
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _make_cross_shape(top_h: int, bottom_h: int, width: int, seam_w: int) -> np.ndarray | None:
    seam_c0 = random.randint(0, width - seam_w)
    top_seed = {(top_h - 1, seam_c0 + dc) for dc in range(seam_w)}
    bottom_seed = {(top_h, seam_c0 + dc) for dc in range(seam_w)}

    top_extra_max = max(2, top_h * width // 2)
    bottom_extra_max = max(2, bottom_h * width // 2)
    top_target = len(top_seed) + random.randint(2, top_extra_max)
    bottom_target = len(bottom_seed) + random.randint(2, bottom_extra_max)

    top_allowed = {(r, c) for r in range(top_h - 1) for c in range(width)}
    bottom_allowed = {(r, c) for r in range(top_h + 1, top_h + bottom_h) for c in range(width)}

    top_cells = _grow_connected(top_allowed, top_seed, top_target)
    bottom_cells = _grow_connected(bottom_allowed, bottom_seed, bottom_target)
    if top_cells is None or bottom_cells is None:
        return None

    cells = np.array(sorted(top_cells | bottom_cells), dtype=int)
    if len(_components_of_mask(_coords_to_mask(cells, (top_h + bottom_h, width)))) != 1:
        return None
    return cells


def _make_blob(height: int, width: int, area: int) -> np.ndarray | None:
    seed = {(random.randrange(height), random.randrange(width))}
    allowed = {(r, c) for r in range(height) for c in range(width)}
    cells = _grow_connected(allowed, seed, area)
    if cells is None:
        return None
    return np.array(sorted(cells), dtype=int)


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if coords.size:
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _globalize(coords: np.ndarray, r0: int, c0: int) -> np.ndarray:
    out = coords.copy()
    out[:, 0] += int(r0)
    out[:, 1] += int(c0)
    return out


def _fault_coords(coords: np.ndarray, cut: int, move_top: bool, fault_dx: int) -> np.ndarray:
    out = coords.copy()
    if move_top:
        moved = out[:, 0] <= int(cut)
    else:
        moved = out[:, 0] > int(cut)
    out[moved, 1] += int(fault_dx)
    return out


def _can_place(coords: np.ndarray, blocked: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= h) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= w):
        return False
    return not bool(blocked[coords[:, 0], coords[:, 1]].any())


def _paint_object(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _has_multicolor_touch(grid: np.ndarray, bg: int) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == int(bg):
                continue
            if r + 1 < h and int(grid[r + 1, c]) not in (int(bg), color):
                return True
            if c + 1 < w and int(grid[r, c + 1]) not in (int(bg), color):
                return True
    return False


def _build_faulted_grid(restored: np.ndarray, cut: int, move_top: bool, fault_dx: int, bg: int) -> np.ndarray:
    faulted = _shift_side(restored, cut, move_top, fault_dx, bg)
    assert faulted is not None, "generator produced an out-of-bounds fault"
    return faulted


def _actual_restore_record(records: list[dict], cut: int, move_top: bool, fault_dx: int) -> dict | None:
    for record in records:
        if record["cut"] == int(cut) and record["move_top"] == bool(move_top) and record["dx"] == int(-fault_dx):
            return record
    return None


def _place_cross_object(
    blocked_out: np.ndarray,
    blocked_in: np.ndarray,
    shape: tuple[int, int],
    cut: int,
    move_top: bool,
    fault_dx: int,
) -> np.ndarray | None:
    h, w = shape
    max_top_h = min(5, cut + 1)
    max_bottom_h = min(5, h - cut - 1)
    if max_top_h < 2 or max_bottom_h < 2:
        return None

    for _ in range(120):
        top_h = random.randint(2, max_top_h)
        bottom_h = random.randint(2, max_bottom_h)
        obj_w = random.randint(4, min(8, w - 2))
        seam_w = random.randint(1, min(3, abs(fault_dx), obj_w - 1))
        local = _make_cross_shape(top_h, bottom_h, obj_w, seam_w)
        if local is None:
            continue

        min_c0 = 1 + max(0, -fault_dx)
        max_c0 = w - obj_w - 1 - max(0, fault_dx)
        if min_c0 > max_c0:
            continue
        c0 = random.randint(min_c0, max_c0)
        r0 = cut - top_h + 1
        global_coords = _globalize(local, r0, c0)
        input_coords = _fault_coords(global_coords, cut, move_top, fault_dx)
        if _can_place(global_coords, blocked_out, shape) and _can_place(input_coords, blocked_in, shape):
            blocked_out |= _dilate_coords(global_coords, shape)
            blocked_in |= _dilate_coords(input_coords, shape)
            return global_coords
    return None


def _place_anchor(
    blocked_out: np.ndarray,
    blocked_in: np.ndarray,
    shape: tuple[int, int],
    cut: int,
    move_top: bool,
    fault_dx: int,
) -> np.ndarray | None:
    h, w = shape
    stationary_top = not move_top
    touch_right = fault_dx > 0
    obj_h = random.randint(2, 4)
    obj_w = random.randint(2, 4)
    local = np.array([(r, c) for r in range(obj_h) for c in range(obj_w)], dtype=int)
    c0 = w - obj_w if touch_right else 0
    if stationary_top:
        max_r0 = cut - obj_h + 1
        if max_r0 < 0:
            return None
        rows = list(range(0, max_r0 + 1))
    else:
        min_r0 = cut + 1
        max_r0 = h - obj_h
        if min_r0 > max_r0:
            return None
        rows = list(range(min_r0, max_r0 + 1))
    random.shuffle(rows)
    for r0 in rows:
        global_coords = _globalize(local, r0, c0)
        input_coords = _fault_coords(global_coords, cut, move_top, fault_dx)
        if _can_place(global_coords, blocked_out, shape) and _can_place(input_coords, blocked_in, shape):
            blocked_out |= _dilate_coords(global_coords, shape)
            blocked_in |= _dilate_coords(input_coords, shape)
            return global_coords
    return None


def _place_distractor(
    blocked_out: np.ndarray,
    blocked_in: np.ndarray,
    shape: tuple[int, int],
    cut: int,
    move_top: bool,
    fault_dx: int,
) -> np.ndarray | None:
    h, w = shape
    for _ in range(120):
        side_top = bool(random.getrandbits(1))
        obj_h = random.randint(2, 4)
        obj_w = random.randint(2, 4)
        area = random.randint(max(3, obj_h + obj_w - 2), obj_h * obj_w)
        local = _make_blob(obj_h, obj_w, area)
        if local is None:
            continue

        if side_top:
            max_r0 = cut - obj_h + 1
            if max_r0 < 0:
                continue
            r0 = random.randint(0, max_r0)
        else:
            min_r0 = cut + 1
            max_r0 = h - obj_h
            if min_r0 > max_r0:
                continue
            r0 = random.randint(min_r0, max_r0)

        min_c0 = 1
        max_c0 = w - obj_w - 1
        if side_top == move_top and fault_dx < 0:
            min_c0 = max(min_c0, -fault_dx)
        if side_top == move_top and fault_dx > 0:
            max_c0 = min(max_c0, w - obj_w - fault_dx)
        if min_c0 > max_c0:
            continue
        c0 = random.randint(min_c0, max_c0)

        global_coords = _globalize(local, r0, c0)
        input_coords = _fault_coords(global_coords, cut, move_top, fault_dx)
        if _can_place(global_coords, blocked_out, shape) and _can_place(input_coords, blocked_in, shape):
            blocked_out |= _dilate_coords(global_coords, shape)
            blocked_in |= _dilate_coords(input_coords, shape)
            return global_coords
    return None


def _generate_case(return_meta: bool = False):
    for _ in range(400):
        h = random.randint(16, 26)
        w = random.randint(18, 28)
        cut = random.randint(4, h - 6)
        max_shift = min(5, w - 6)
        fault_dx = random.choice([-1, 1]) * random.randint(2, max_shift)
        move_top = bool(random.getrandbits(1))
        cross_count = random.randint(2, 4)
        distractor_count = random.randint(1, 3)
        total_objects = 1 + cross_count + distractor_count
        if total_objects > len(PALETTE):
            continue
        colors = random.sample(PALETTE, total_objects)

        blocked_out = np.zeros((h, w), dtype=bool)
        blocked_in = np.zeros((h, w), dtype=bool)
        objects: list[tuple[int, np.ndarray]] = []

        anchor = _place_anchor(blocked_out, blocked_in, (h, w), cut, move_top, fault_dx)
        if anchor is None:
            continue
        objects.append((colors.pop(), anchor))

        failed = False
        for _idx in range(cross_count):
            coords = _place_cross_object(blocked_out, blocked_in, (h, w), cut, move_top, fault_dx)
            if coords is None:
                failed = True
                break
            objects.append((colors.pop(), coords))
        if failed:
            continue

        for _idx in range(distractor_count):
            coords = _place_distractor(blocked_out, blocked_in, (h, w), cut, move_top, fault_dx)
            if coords is None:
                failed = True
                break
            objects.append((colors.pop(), coords))
        if failed:
            continue

        restored = np.zeros((h, w), dtype=int)
        for color, coords in objects:
            _paint_object(restored, coords, color)

        faulted = _build_faulted_grid(restored, cut, move_top, fault_dx, BG)
        if _has_multicolor_touch(faulted, BG):
            continue

        expected = _crop_foreground(restored, BG)
        records = _candidate_records(faulted, BG)
        actual = _actual_restore_record(records, cut, move_top, fault_dx)
        if actual is None:
            continue
        best_score = max(record["score"] for record in records)
        best = [record for record in records if record["score"] == best_score]
        if len(best) != 1:
            continue
        if best[0] is not actual:
            continue
        if not np.array_equal(best[0]["crop"], expected):
            continue
        if not np.array_equal(main(faulted), expected):
            continue

        if return_meta:
            return faulted, {
                "cut": int(cut),
                "move_top": bool(move_top),
                "fault_dx": int(fault_dx),
                "expected": expected,
                "best_score": best_score,
            }
        return faulted
    raise RuntimeError("failed to generate a unique hidden-fault ARC case")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
