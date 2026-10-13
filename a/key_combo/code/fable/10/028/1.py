from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVE_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
H_RANGE = (14, 20)
W_RANGE = (14, 20)
STATIC_COUNT_RANGE = (1, 2)
MOVING_COUNT_RANGE = (2, 4)
SHAPE_AREA_RANGE = (5, 12)
SHAPE_SIZE_RANGE = (3, 6)
MAX_SHIFT = 7
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], coords: np.ndarray) -> bool:
    h, w = shape
    return bool(
        coords.size == 0
        or (
            coords[:, 0].min() >= 0
            and coords[:, 1].min() >= 0
            and coords[:, 0].max() < h
            and coords[:, 1].max() < w
        )
    )


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = set()
    stack = [tuple(map(int, pts[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "bbox": _bbox_from_coords(arr)})
    return comps


def _mirror_cell(axis: str, cell: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    h, w = shape
    if axis == "v":
        return r, w - 1 - c
    return h - 1 - r, c


def _classify_broken(mask: np.ndarray, axis: str) -> dict | None:
    h, w = mask.shape
    mismatches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        mr, mc = _mirror_cell(axis, (int(r), int(c)), (h, w))
        if not mask[mr, mc]:
            mismatches.append(((int(r), int(c)), (int(mr), int(mc))))
    if len(mismatches) != 1:
        return None

    twin, missing = mismatches[0]
    if twin == missing or mask[missing]:
        return None

    fixed = mask.copy()
    fixed[missing] = True
    if axis == "v":
        if not np.array_equal(fixed, np.fliplr(fixed)):
            return None
        direction = (0, -1 if missing[1] < twin[1] else 1)
    else:
        if not np.array_equal(fixed, np.flipud(fixed)):
            return None
        direction = (-1, 0) if missing[0] < twin[0] else (1, 0)

    return {
        "kind": "moving",
        "axis": axis,
        "direction": direction,
        "missing": missing,
        "twin": twin,
    }


def _classify_mask(mask: np.ndarray) -> dict:
    vertical = np.array_equal(mask, np.fliplr(mask))
    horizontal = np.array_equal(mask, np.flipud(mask))
    axes = []
    if vertical:
        axes.append("v")
    if horizontal:
        axes.append("h")
    if axes:
        return {"kind": "static", "axes": tuple(axes)}

    candidates = []
    for axis in ("v", "h"):
        candidate = _classify_broken(mask, axis)
        if candidate is not None:
            candidates.append(candidate)
    assert len(candidates) == 1, "each non-symmetric object must be a unique one-pixel symmetry defect"
    return candidates[0]


def _try_classify_mask(mask: np.ndarray) -> dict | None:
    try:
        return _classify_mask(mask)
    except AssertionError:
        return None


def _classify_component(comp: dict) -> dict:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    info = _classify_mask(mask)
    enriched = {
        "color": int(comp["color"]),
        "coords": comp["coords"].copy(),
        "bbox": comp["bbox"],
        "mask": mask,
    }
    enriched.update(info)
    if info["kind"] == "moving":
        enriched["missing_abs"] = np.array([r0 + info["missing"][0], c0 + info["missing"][1]], dtype=int)
    return enriched


def _try_classify_component(comp: dict) -> dict | None:
    try:
        return _classify_component(comp)
    except AssertionError:
        return None


def _shift_limit(coords: np.ndarray, direction: tuple[int, int], occupied: np.ndarray) -> int:
    h, w = occupied.shape
    delta = np.array(direction, dtype=int)
    shift = 0
    while True:
        moved = coords + (shift + 1) * delta
        if not _in_bounds((h, w), moved):
            break
        if np.any(occupied[moved[:, 0], moved[:, 1]]):
            break
        shift += 1
    return shift


def _sort_key(info: dict) -> int:
    r0, c0, r1, c1 = info["bbox"]
    direction = tuple(info["direction"])
    if direction == (0, -1):
        return c0
    if direction == (0, 1):
        return -c1
    if direction == (-1, 0):
        return r0
    return -r1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each one-pixel-broken symmetric object toward its missing side, then complete it after it stops.

    Input structure:
    - The background is the most common color.
    - Non-background cells form several pairwise non-touching 4-connected monochrome objects.
    - Some cropped objects are already mirror-symmetric about a vertical or horizontal mid-axis.
    - Every remaining object differs from such a symmetric shape by exactly one missing pixel whose mirror twin is present.

    Rule:
    - Keep every fully symmetric object fixed in place as an obstacle.
    - For each broken object, infer the motion direction from the solid twin toward the missing pixel, which is perpendicular to that object's symmetry axis.
    - Slide the broken object rigidly as far as possible until one more step would hit the grid boundary, a fixed symmetric object, or an already settled moved object.
    - After stopping, add the missing pixel at the translated location so the object becomes fully mirror-symmetric.

    Key insight:
    - The only motion cue is internal: the lone asymmetry says both which symmetry axis is intended and which side the object wants to move toward before repairing itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)

    infos = [_classify_component(comp) for comp in _find_components(g, bg)]
    out = np.full_like(g, bg)
    occupied = np.zeros_like(g, dtype=bool)

    movers = []
    for info in infos:
        if info["kind"] == "static":
            coords = info["coords"]
            out[coords[:, 0], coords[:, 1]] = info["color"]
            occupied[coords[:, 0], coords[:, 1]] = True
        else:
            movers.append(info)

    for info in sorted(movers, key=_sort_key):
        shift = _shift_limit(info["coords"], tuple(info["direction"]), occupied)
        delta = np.array(info["direction"], dtype=int)
        final_coords = info["coords"] + shift * delta
        missing = info["missing_abs"] + shift * delta
        assert _in_bounds(g.shape, final_coords), "translated object must stay in bounds"
        assert 0 <= missing[0] < g.shape[0] and 0 <= missing[1] < g.shape[1], "completed pixel must stay in bounds"
        assert not np.any(occupied[final_coords[:, 0], final_coords[:, 1]]), "moved object may not overlap prior obstacles"
        assert not occupied[missing[0], missing[1]], "completed pixel may not overlap prior obstacles"
        out[final_coords[:, 0], final_coords[:, 1]] = info["color"]
        out[missing[0], missing[1]] = info["color"]
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
        occupied[missing[0], missing[1]] = True

    return out


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    return np.argwhere(mask).astype(int) + np.array([top, left], dtype=int)


def _overlaps(coords: np.ndarray, occ: np.ndarray) -> bool:
    return bool(np.any(occ[coords[:, 0], coords[:, 1]]))


def _touches_occupied(coords: np.ndarray, occ: np.ndarray) -> bool:
    h, w = occ.shape
    for r, c in map(tuple, coords):
        if occ[r, c]:
            return True
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and occ[nr, nc]:
                return True
    return False


def _mark(occ: np.ndarray, coords: np.ndarray) -> None:
    occ[coords[:, 0], coords[:, 1]] = True


def _orbit(axis: str, cell: tuple[int, int], shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    other = _mirror_cell(axis, cell, shape)
    if other == cell:
        return (cell,)
    return tuple(sorted((cell, other)))


def _gen_symmetric_mask(axis: str, require_single_axis: bool) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(*SHAPE_SIZE_RANGE)
        w = random.randint(*SHAPE_SIZE_RANGE)
        if axis == "v" and w < 2:
            continue
        if axis == "h" and h < 2:
            continue

        mask = np.zeros((h, w), dtype=bool)
        if axis == "v":
            if w % 2 == 1:
                mask[random.randrange(h), w // 2] = True
            else:
                r = random.randrange(h)
                mask[r, w // 2 - 1] = True
                mask[r, w // 2] = True
        else:
            if h % 2 == 1:
                mask[h // 2, random.randrange(w)] = True
            else:
                c = random.randrange(w)
                mask[h // 2 - 1, c] = True
                mask[h // 2, c] = True

        while True:
            area = int(mask.sum())
            candidates: list[tuple[tuple[int, int], ...]] = []
            for r, c in map(tuple, np.argwhere(mask)):
                for dr, dc in DIRS4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    cells = tuple(cell for cell in _orbit(axis, (nr, nc), (h, w)) if not mask[cell])
                    if not cells or area + len(cells) > SHAPE_AREA_RANGE[1]:
                        continue
                    candidates.append(cells)
            candidates = list(dict.fromkeys(candidates))
            if area >= SHAPE_AREA_RANGE[0] and (
                area >= SHAPE_AREA_RANGE[1] or not candidates or random.random() < 0.35
            ):
                break
            if not candidates:
                break
            for rr, cc in random.choice(candidates):
                mask[rr, cc] = True

        if int(mask.sum()) < SHAPE_AREA_RANGE[0]:
            continue
        mask = _crop_bool(mask)
        if not _is_connected(mask):
            continue
        cls = _try_classify_mask(mask)
        if cls is None:
            continue
        if cls["kind"] != "static" or axis not in cls["axes"]:
            continue
        if require_single_axis and tuple(cls["axes"]) != (axis,):
            continue
        if min(mask.shape) < 2:
            continue
        return mask
    return None


def _gen_broken_pair(direction: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    axis = "v" if direction[0] == 0 else "h"
    for _ in range(250):
        full = _gen_symmetric_mask(axis, require_single_axis=True)
        if full is None:
            continue
        candidates = list(map(tuple, np.argwhere(full)))
        random.shuffle(candidates)
        for cell in candidates:
            twin = _mirror_cell(axis, cell, full.shape)
            if twin == cell:
                continue
            if axis == "v":
                side = -1 if cell[1] < twin[1] else 1
                if (0, side) != direction:
                    continue
            else:
                side = -1 if cell[0] < twin[0] else 1
                if (side, 0) != direction:
                    continue
            broken = full.copy()
            broken[cell] = False
            if not _is_connected(broken):
                continue
            cls = _try_classify_mask(broken)
            if cls is None:
                continue
            if cls["kind"] != "moving":
                continue
            if cls["axis"] != axis or tuple(cls["direction"]) != direction:
                continue
            if tuple(cls["missing"]) != tuple(cell):
                continue
            return full, broken
    return None


def _random_directions(count: int) -> list[tuple[int, int]]:
    choices = []
    if count >= 1:
        choices.append(random.choice([(0, -1), (0, 1)]))
    if count >= 2:
        choices.append(random.choice([(-1, 0), (1, 0)]))
    while len(choices) < count:
        choices.append(random.choice(MOVE_DIRS))
    random.shuffle(choices)
    return choices


def _render_input(
    shape: tuple[int, int],
    bg: int,
    static_objects: list[dict],
    moving_objects: list[dict],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for obj in static_objects:
        coords = _coords_from_mask(obj["mask"], obj["top"], obj["left"])
        grid[coords[:, 0], coords[:, 1]] = obj["color"]
    for obj in moving_objects:
        coords = _coords_from_mask(obj["broken"], obj["input_top"], obj["input_left"])
        grid[coords[:, 0], coords[:, 1]] = obj["color"]
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    if np.sum(grid != bg) >= grid.size // 2:
        return False

    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == bg:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg and int(grid[nr, nc]) != int(grid[r, c]):
                    return False

    infos = []
    for comp in _find_components(grid, bg):
        info = _try_classify_component(comp)
        if info is None:
            return False
        infos.append(info)
    static_infos = [info for info in infos if info["kind"] == "static"]
    moving_infos = [info for info in infos if info["kind"] == "moving"]
    if not static_infos or not moving_infos:
        return False

    static_occ = np.zeros_like(grid, dtype=bool)
    for info in static_infos:
        _mark(static_occ, info["coords"])

    path_occ = np.zeros_like(grid, dtype=bool)
    expected = np.full_like(grid, bg)
    for info in static_infos:
        expected[info["coords"][:, 0], info["coords"][:, 1]] = info["color"]

    for info in moving_infos:
        shift = _shift_limit(info["coords"], tuple(info["direction"]), static_occ)
        if shift <= 0:
            return False
        delta = np.array(info["direction"], dtype=int)
        final_coords = info["coords"] + shift * delta
        next_coords = final_coords + delta
        if _in_bounds(grid.shape, next_coords) and not _overlaps(next_coords, static_occ):
            return False
        object_path_occ = np.zeros_like(grid, dtype=bool)
        for s in range(shift + 1):
            step_coords = info["coords"] + s * delta
            if _overlaps(step_coords, static_occ) or _overlaps(step_coords, path_occ):
                return False
            _mark(object_path_occ, step_coords)
        path_occ |= object_path_occ
        missing = info["missing_abs"] + shift * delta
        full_coords = np.vstack([final_coords, missing.reshape(1, 2)])
        if not _in_bounds(grid.shape, full_coords):
            return False
        if np.any(expected[full_coords[:, 0], full_coords[:, 1]] != bg):
            return False
        expected[full_coords[:, 0], full_coords[:, 1]] = info["color"]

    out = main(grid)
    return not np.array_equal(out, grid) and np.array_equal(out, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*H_RANGE)
    w = random.randint(*W_RANGE)
    bg = random.randrange(10)

    static_count = random.randint(*STATIC_COUNT_RANGE)
    moving_count = random.randint(*MOVING_COUNT_RANGE)
    total = static_count + moving_count
    if total > 9:
        return None
    colors = random.sample([c for c in range(10) if c != bg], total)

    input_occ = np.zeros((h, w), dtype=bool)
    static_occ = np.zeros((h, w), dtype=bool)
    moving_output_occ = np.zeros((h, w), dtype=bool)
    reserved_path_occ = np.zeros((h, w), dtype=bool)

    static_objects = []
    for i in range(static_count):
        axis = random.choice(["v", "h"])
        mask = _gen_symmetric_mask(axis, require_single_axis=True)
        if mask is None:
            return None
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = random.randint(0, h - mask.shape[0])
            left = random.randint(0, w - mask.shape[1])
            coords = _coords_from_mask(mask, top, left)
            if _touches_occupied(coords, input_occ):
                continue
            _mark(input_occ, coords)
            _mark(static_occ, coords)
            static_objects.append({"mask": mask, "top": top, "left": left, "color": colors[i]})
            placed = True
            break
        if not placed:
            return None

    stop_preferences = ["boundary", "static"] if moving_count >= 2 else [random.choice(["boundary", "static"])]
    while len(stop_preferences) < moving_count:
        stop_preferences.append(random.choice(["boundary", "static"]))
    random.shuffle(stop_preferences)

    moving_objects = []
    for color, direction, stop_type in zip(colors[static_count:], _random_directions(moving_count), stop_preferences):
        pair = _gen_broken_pair(direction)
        if pair is None:
            return None
        full, broken = pair
        dr, dc = direction
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            final_top = random.randint(0, h - full.shape[0])
            final_left = random.randint(0, w - full.shape[1])
            final_full = _coords_from_mask(full, final_top, final_left)
            final_broken = _coords_from_mask(broken, final_top, final_left)
            if _overlaps(final_full, static_occ) or _overlaps(final_full, moving_output_occ):
                continue
            if _touches_occupied(final_full, moving_output_occ) or _overlaps(final_full, reserved_path_occ):
                continue
            next_broken = final_broken + np.array([dr, dc], dtype=int)
            if stop_type == "boundary":
                if _in_bounds((h, w), next_broken):
                    continue
            else:
                if not _in_bounds((h, w), next_broken) or not _overlaps(next_broken, static_occ):
                    continue

            shift = random.randint(1, MAX_SHIFT)
            input_top = final_top - shift * dr
            input_left = final_left - shift * dc
            input_broken = _coords_from_mask(broken, input_top, input_left)
            if not _in_bounds((h, w), input_broken):
                continue
            if _touches_occupied(input_broken, input_occ):
                continue

            path_cells = []
            ok = True
            for s in range(shift + 1):
                step_coords = _coords_from_mask(broken, input_top + s * dr, input_left + s * dc)
                if _overlaps(step_coords, static_occ) or _overlaps(step_coords, reserved_path_occ) or _overlaps(step_coords, moving_output_occ):
                    ok = False
                    break
                path_cells.append(step_coords)
            if not ok:
                continue

            _mark(input_occ, input_broken)
            _mark(moving_output_occ, final_full)
            for step_coords in path_cells:
                _mark(reserved_path_occ, step_coords)
            moving_objects.append(
                {
                    "full": full,
                    "broken": broken,
                    "input_top": input_top,
                    "input_left": input_left,
                    "final_top": final_top,
                    "final_left": final_left,
                    "direction": direction,
                    "color": color,
                    "stop_type": stop_type,
                }
            )
            placed = True
            break
        if not placed:
            return None

    grid = _render_input((h, w), bg, static_objects, moving_objects)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-defect navigation input")
