from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


RED = 2
GRAY = 5
BG_CHOICES = [0, 1, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _local_axis_kind(mask: np.ndarray) -> str:
    h, w = mask.shape
    anti = np.flipud(np.fliplr(mask)).T
    flags = []
    if h % 2 == 1 and np.array_equal(mask, mask[::-1, :]):
        flags.append("H")
    if w % 2 == 1 and np.array_equal(mask, mask[:, ::-1]):
        flags.append("V")
    if h == w and np.array_equal(mask, mask.T):
        flags.append("D1")
    if h == w and np.array_equal(mask, anti):
        flags.append("D2")
    assert len(flags) == 1, "expected exactly one usable symmetry axis"
    return flags[0]


def _axis_key(coords: np.ndarray) -> tuple[str, int]:
    bbox = _bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    mask = _component_mask(coords, bbox)
    kind = _local_axis_kind(mask)
    if kind == "H":
        return kind, int(r0 + mask.shape[0] // 2)
    if kind == "V":
        return kind, int(c0 + mask.shape[1] // 2)
    if kind == "D1":
        return kind, int(r0 - c0)
    return kind, int(r0 + c1)


def _coord_on_axis(coord: tuple[int, int], axis: tuple[str, int]) -> bool:
    r, c = map(int, coord)
    kind, value = axis
    if kind == "H":
        return r == value
    if kind == "V":
        return c == value
    if kind == "D1":
        return r - c == value
    return r + c == value


def _axis_cells(coords: np.ndarray, axis: tuple[str, int]) -> np.ndarray:
    kind, value = axis
    if kind == "H":
        mask = coords[:, 0] == value
    elif kind == "V":
        mask = coords[:, 1] == value
    elif kind == "D1":
        mask = coords[:, 0] - coords[:, 1] == value
    else:
        mask = coords[:, 0] + coords[:, 1] == value
    rail = coords[mask]
    assert rail.size > 0, "object must intersect its own rail"
    return rail


def _rail_scalar(coords: np.ndarray, axis: tuple[str, int]) -> np.ndarray:
    kind, _ = axis
    if kind == "H":
        return coords[:, 1]
    return coords[:, 0]


def _rail_interval(coords: np.ndarray, axis: tuple[str, int]) -> tuple[int, int]:
    vals = _rail_scalar(_axis_cells(coords, axis), axis)
    return int(vals.min()), int(vals.max())


def _dock_scalar(dock: tuple[int, int], axis: tuple[str, int]) -> int:
    return int(dock[1] if axis[0] == "H" else dock[0])


def _shift_for_axis(axis: tuple[str, int], delta: int) -> tuple[int, int]:
    kind = axis[0]
    if kind == "H":
        return 0, int(delta)
    if kind == "V":
        return int(delta), 0
    if kind == "D1":
        return int(delta), int(delta)
    return int(delta), -int(delta)


def _analyze_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    colors = set(map(int, np.unique(g)))
    assert colors <= {bg, RED, GRAY}, "unexpected colors"

    red_components = _find_components(g == RED)
    multi = []
    singletons = []
    for coords in red_components:
        if len(coords) == 1:
            singletons.append(coords)
            continue
        axis = _axis_key(coords)
        multi.append(
            {
                "coords": coords,
                "axis": axis,
                "interval": _rail_interval(coords, axis),
                "area": int(coords.shape[0]),
            }
        )
    assert multi, "expected at least one non-singleton red object"

    gray_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == GRAY)]
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for obj in multi:
        groups[obj["axis"]].append(obj)

    used_axes = set(groups)
    for axis, objects in groups.items():
        dock_candidates = [p for p in gray_coords if _coord_on_axis(p, axis)]
        assert len(dock_candidates) == 1, "expected exactly one dock on each used axis"
        dock = dock_candidates[0]
        ds = _dock_scalar(dock, axis)
        directions = []
        for obj in objects:
            lo, hi = obj["interval"]
            if hi < ds:
                directions.append(1)
            elif lo > ds:
                directions.append(-1)
            else:
                raise AssertionError("dock must lie strictly beyond the whole object")
        assert len(set(directions)) == 1, "same-axis objects must lie on the same dock side"

    for p in gray_coords:
        hits = sum(_coord_on_axis(p, axis) for axis in used_axes)
        assert hits <= 1, "gray distractors must stay off every object axis"

    return {
        "bg": bg,
        "gray_coords": gray_coords,
        "objects": multi,
        "singletons": singletons,
        "groups": groups,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Slide each symmetric red object along its own symmetry axis toward the dock pixel on that axis.

    Input structure:
    - The grid has one dominant background color, several red 4-connected objects,
      some gray single pixels, and a few isolated red singleton noise pixels.
    - Every non-singleton red object has exactly one usable reflection axis:
      horizontal, vertical, main-diagonal, or anti-diagonal.
    - Objects that share an axis also share one gray dock on that axis, and that
      dock lies on the same side of all those objects.

    Rule:
    - Erase isolated red singleton noise pixels.
    - For each non-singleton red object, find its unique symmetry axis and treat
      that axis as a rail.
    - On each rail, move the closest object until its rail-front stops with exactly
      one background cell before the gray dock; then park the remaining same-rail
      objects one by one behind it, always leaving one background cell between
      consecutive rail tails and fronts.
    - Keep every gray pixel fixed, including distractors that are not on any used axis.

    Key insight:
    - The symmetry axis is not just a label: it is the motion direction, and shared-axis
      objects must be packed by chained stopping positions rather than moved independently.
    """
    g = np.asarray(input_grid, dtype=int)
    info = _analyze_instance(g)
    bg = info["bg"]
    out = np.full_like(g, bg)
    for r, c in info["gray_coords"]:
        out[r, c] = GRAY

    occupied = set(info["gray_coords"])
    for axis, objects in info["groups"].items():
        dock = next(p for p in info["gray_coords"] if _coord_on_axis(p, axis))
        dock_s = _dock_scalar(dock, axis)
        first = objects[0]["interval"]
        sign = 1 if first[1] < dock_s else -1
        if sign == 1:
            ordered = sorted(objects, key=lambda obj: obj["interval"][1], reverse=True)
        else:
            ordered = sorted(objects, key=lambda obj: obj["interval"][0])

        target_front = dock_s - 2 * sign
        for obj in ordered:
            lo, hi = obj["interval"]
            cur_front = hi if sign == 1 else lo
            cur_tail = lo if sign == 1 else hi
            delta = target_front - cur_front
            dr, dc = _shift_for_axis(axis, delta)
            shifted = obj["coords"] + np.array([dr, dc], dtype=int)
            assert shifted[:, 0].min() >= 0 and shifted[:, 0].max() < g.shape[0]
            assert shifted[:, 1].min() >= 0 and shifted[:, 1].max() < g.shape[1]
            for rr, cc in map(tuple, shifted):
                assert (rr, cc) not in occupied, "parked objects must not overwrite docks or each other"
            out[shifted[:, 0], shifted[:, 1]] = RED
            occupied.update(map(tuple, shifted))
            placed_tail = cur_tail + delta
            target_front = placed_tail - 2 * sign

    return out


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _random_h_mask() -> np.ndarray:
    for _ in range(120):
        half_h = random.randint(1, 2)
        width = random.randint(4, 8)
        mask = np.zeros((2 * half_h + 1, width), dtype=bool)
        axis_r = half_h
        mask[axis_r, :] = True
        branch_count = random.randint(1, max(1, min(width - 2, 4)))
        cols = random.sample(list(range(1, width - 1)), branch_count)
        for c in cols:
            depth = random.randint(1, half_h)
            for d in range(1, depth + 1):
                mask[axis_r - d, c] = True
                mask[axis_r + d, c] = True
        try:
            if _local_axis_kind(mask) == "H":
                return mask
        except AssertionError:
            pass
    raise RuntimeError("failed to build horizontal mask")


def _random_v_mask() -> np.ndarray:
    for _ in range(120):
        mask = _random_h_mask().T
        try:
            if _local_axis_kind(mask) == "V":
                return mask
        except AssertionError:
            pass
    raise RuntimeError("failed to build vertical mask")


def _random_d1_mask() -> np.ndarray:
    for _ in range(120):
        n = random.randint(4, 7)
        mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            mask[i, i] = True
        for i in range(n - 1):
            mask[i, i + 1] = True
            mask[i + 1, i] = True
        extra_count = random.randint(0, max(1, n - 3))
        if n >= 3 and extra_count:
            for i in random.sample(list(range(n - 2)), extra_count):
                mask[i, i + 2] = True
                mask[i + 2, i] = True
        try:
            if _local_axis_kind(mask) == "D1":
                return mask
        except AssertionError:
            pass
    raise RuntimeError("failed to build diagonal mask")


def _random_d2_mask() -> np.ndarray:
    for _ in range(120):
        mask = np.fliplr(_random_d1_mask())
        try:
            if _local_axis_kind(mask) == "D2":
                return mask
        except AssertionError:
            pass
    raise RuntimeError("failed to build anti-diagonal mask")


def _random_object_mask(kind: str) -> np.ndarray:
    if kind == "H":
        return _random_h_mask()
    if kind == "V":
        return _random_v_mask()
    if kind == "D1":
        return _random_d1_mask()
    return _random_d2_mask()


def _relative_place(mask: np.ndarray, kind: str, sign: int, front_scalar: int) -> np.ndarray:
    coords = _mask_to_coords(mask)
    axis = _axis_key(coords)
    lo, hi = _rail_interval(coords, axis)
    front_local = hi if sign == 1 else lo
    r0, c0, r1, c1 = _bbox_coords(coords)
    if kind == "H":
        top = -axis[1]
        left = front_scalar - front_local
    elif kind == "V":
        top = front_scalar - front_local
        left = -axis[1]
    elif kind == "D1":
        top = front_scalar - front_local
        left = top
    else:
        axis_sum = axis[1]
        top = front_scalar - front_local
        left = -axis_sum - top
    placed = coords + np.array([top - r0, left - c0], dtype=int)
    return placed


def _rail_length(mask: np.ndarray) -> int:
    coords = _mask_to_coords(mask)
    axis = _axis_key(coords)
    lo, hi = _rail_interval(coords, axis)
    return int(hi - lo + 1)


def _build_relative_group(kind: str, sign: int, count: int) -> dict:
    masks = [_random_object_mask(kind) for _ in range(count)]
    lengths = [_rail_length(mask) for mask in masks]
    final_fronts = []
    current_front = -2 * sign
    for length in lengths:
        final_fronts.append(current_front)
        tail = current_front - sign * (length - 1)
        current_front = tail - 2 * sign

    start_fronts = []
    extra = random.randint(2, 4)
    for front in final_fronts:
        extra += random.randint(1, 4)
        start_fronts.append(front - sign * extra)

    rel_objects = []
    for mask, front in zip(masks, start_fronts):
        rel_objects.append(_relative_place(mask, kind, sign, front))

    all_coords = np.concatenate(rel_objects, axis=0)
    dock = np.array([[0, 0]], dtype=int)
    union = np.concatenate([all_coords, dock], axis=0)
    r0, c0, r1, c1 = _bbox_coords(union)
    return {
        "kind": kind,
        "sign": sign,
        "dock_rel": (0, 0),
        "objects_rel": rel_objects,
        "bbox_rel": (r0, c0, r1, c1),
    }


def _abs_axis_from_dock(kind: str, dock: tuple[int, int]) -> tuple[str, int]:
    r, c = dock
    if kind == "H":
        return kind, int(r)
    if kind == "V":
        return kind, int(c)
    if kind == "D1":
        return kind, int(r - c)
    return kind, int(r + c)


def _coords_hit_axis(coords: np.ndarray, axis: tuple[str, int]) -> bool:
    if axis[0] == "H":
        return bool(np.any(coords[:, 0] == axis[1]))
    if axis[0] == "V":
        return bool(np.any(coords[:, 1] == axis[1]))
    if axis[0] == "D1":
        return bool(np.any(coords[:, 0] - coords[:, 1] == axis[1]))
    return bool(np.any(coords[:, 0] + coords[:, 1] == axis[1]))


def _try_place_group(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    red_coords: list[tuple[int, int]],
    docks: list[tuple[int, int]],
    axes: list[tuple[str, int]],
    kind: str,
    count: int,
) -> dict | None:
    sign = random.choice([-1, 1])
    rel = _build_relative_group(kind, sign, count)
    r0, c0, r1, c1 = rel["bbox_rel"]
    h, w = shape
    dr_min, dr_max = -r0, h - 1 - r1
    dc_min, dc_max = -c0, w - 1 - c1
    if dr_min > dr_max or dc_min > dc_max:
        return None

    existing_red = np.array(red_coords, dtype=int) if red_coords else np.zeros((0, 2), dtype=int)
    for _ in range(80):
        dr = random.randint(dr_min, dr_max)
        dc = random.randint(dc_min, dc_max)
        dock = (dr, dc)
        axis = _abs_axis_from_dock(kind, dock)
        if axis in axes:
            continue
        if any(_coord_on_axis(dock, old_axis) for old_axis in axes):
            continue
        if any(_coord_on_axis(old_dock, axis) for old_dock in docks):
            continue

        abs_objects = [coords + np.array([dr, dc], dtype=int) for coords in rel["objects_rel"]]
        flat = np.concatenate(abs_objects, axis=0)
        if any(tuple(map(int, rc)) in occupied for rc in flat):
            continue
        if any(_coords_hit_axis(flat, old_axis) for old_axis in axes):
            continue
        if existing_red.size and _coords_hit_axis(existing_red, axis):
            continue
        return {"dock": dock, "axis": axis, "objects": abs_objects}
    return None


def _add_singletons(grid: np.ndarray, axes: list[tuple[str, int]], count: int) -> bool:
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    added = 0
    for r, c in cells:
        if grid[r, c] != get_bg(grid):
            continue
        if any(_coord_on_axis((r, c), axis) for axis in axes):
            continue
        ok = True
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if grid[rr, cc] == RED:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        grid[r, c] = RED
        added += 1
        if added == count:
            return True
    return False


def _add_gray_distractors(grid: np.ndarray, axes: list[tuple[str, int]], count: int) -> bool:
    bg = get_bg(grid)
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    added = 0
    for r, c in cells:
        if grid[r, c] != bg:
            continue
        if any(_coord_on_axis((r, c), axis) for axis in axes):
            continue
        grid[r, c] = GRAY
        added += 1
        if added == count:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 29)
    w = random.randint(18, 29)
    bg = random.choice(BG_CHOICES)
    grid = np.full((h, w), bg, dtype=int)

    group_count = random.randint(2, 4)
    kinds = ["H", "V", "D1", "D2"]
    random.shuffle(kinds)
    chosen_kinds = [random.choice(kinds) for _ in range(group_count)]
    if not any(kind.startswith("D") for kind in chosen_kinds) and random.random() < 0.55:
        chosen_kinds[random.randrange(group_count)] = random.choice(["D1", "D2"])

    counts = [random.randint(2, 3)]
    counts.extend(random.randint(1, 2) for _ in range(group_count - 1))

    occupied: set[tuple[int, int]] = set()
    red_coords: list[tuple[int, int]] = []
    docks: list[tuple[int, int]] = []
    axes: list[tuple[str, int]] = []

    for kind, count in zip(chosen_kinds, counts):
        placed = None
        for _ in range(60):
            placed = _try_place_group((h, w), occupied, red_coords, docks, axes, kind, count)
            if placed is not None:
                break
        if placed is None:
            return None

        dr, dc = placed["dock"]
        grid[dr, dc] = GRAY
        occupied.add((dr, dc))
        docks.append((dr, dc))
        axes.append(placed["axis"])
        for coords in placed["objects"]:
            grid[coords[:, 0], coords[:, 1]] = RED
            tuples = list(map(tuple, coords))
            occupied.update(tuples)
            red_coords.extend(tuples)

    if not _add_singletons(grid, axes, random.randint(1, 4)):
        return None
    if not _add_gray_distractors(grid, axes, random.randint(1, 4)):
        return None

    try:
        info = _analyze_instance(grid)
    except AssertionError:
        return None
    if len(info["groups"]) < 2:
        return None
    if not any(len(v) >= 2 for v in info["groups"].values()):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    try:
        out_info = _analyze_instance(out)
    except AssertionError:
        return None
    if set(info["groups"]) != set(out_info["groups"]):
        return None
    if sorted(obj["area"] for obj in info["objects"]) != sorted(obj["area"] for obj in out_info["objects"]):
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == RED)) <= int(np.count_nonzero(grid == RED) // 3):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
