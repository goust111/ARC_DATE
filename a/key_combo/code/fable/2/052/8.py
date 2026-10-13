from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 160
SCENARIOS = [
    ("h", "v", "d"),
    ("h", "d", "d"),
    ("v", "d", "d"),
    ("h", "h", "d"),
    ("v", "v", "d"),
    ("h", "v", "d", "d"),
    ("h", "v", "h", "d"),
    ("h", "v", "v", "d"),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_set(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _simple_open_curve(coords: np.ndarray) -> bool:
    pts = {tuple(map(int, rc)) for rc in coords}
    endpoints = 0
    for r, c in pts:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        if deg == 1:
            endpoints += 1
        elif deg == 2:
            continue
        else:
            return False
    return endpoints == 2


def _classify_transformable(coords: np.ndarray) -> dict | None:
    pts = {tuple(map(int, rc)) for rc in coords}
    degrees: dict[tuple[int, int], int] = {}
    endpoints: list[tuple[int, int]] = []
    for r, c in pts:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        degrees[(r, c)] = deg
        if deg == 1:
            endpoints.append((r, c))
        elif deg == 2:
            continue
        else:
            return None

    if len(endpoints) != 2:
        return None

    a, b = endpoints
    interior = [rc for rc in pts if rc not in {a, b}]
    if not interior:
        return None

    if a[0] == b[0]:
        axis_kind = "h"
        axis = a[0]
        side_values = {np.sign(r - axis) for r, _ in interior}
        if side_values != {-1} and side_values != {1}:
            return None
        ordered = tuple(sorted(endpoints, key=lambda rc: rc[1]))
        return {"axis_kind": axis_kind, "axis": axis, "side": int(next(iter(side_values))), "endpoints": ordered}

    if a[1] == b[1]:
        axis_kind = "v"
        axis = a[1]
        side_values = {np.sign(c - axis) for _, c in interior}
        if side_values != {-1} and side_values != {1}:
            return None
        ordered = tuple(sorted(endpoints, key=lambda rc: rc[0]))
        return {"axis_kind": axis_kind, "axis": axis, "side": int(next(iter(side_values))), "endpoints": ordered}

    return None


def _reflect_coords(coords: set[tuple[int, int]], axis_kind: str, axis: int) -> set[tuple[int, int]]:
    if axis_kind == "h":
        return {(2 * axis - r, c) for r, c in coords}
    return {(r, 2 * axis - c) for r, c in coords}


def _chord_coords(endpoints: tuple[tuple[int, int], tuple[int, int]]) -> set[tuple[int, int]]:
    (r0, c0), (r1, c1) = endpoints
    if r0 == r1:
        return {(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)}
    return {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}


def _interior_coords(boundary_coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not boundary_coords:
        return set()

    r0, c0, r1, c1 = _bbox_from_set(boundary_coords)
    r0 -= 1
    c0 -= 1
    r1 += 1
    c1 += 1
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    boundary = np.zeros((h, w), dtype=bool)
    for r, c in boundary_coords:
        boundary[r - r0, c - c0] = True

    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(rr: int, cc: int) -> None:
        if 0 <= rr < h and 0 <= cc < w and not boundary[rr, cc] and not exterior[rr, cc]:
            exterior[rr, cc] = True
            queue.append((rr, cc))

    for rr in range(h):
        add(rr, 0)
        add(rr, w - 1)
    for cc in range(w):
        add(0, cc)
        add(h - 1, cc)

    while queue:
        rr, cc = queue.popleft()
        for nr, nc in _neighbors4(rr, cc):
            add(nr, nc)

    inside = ~boundary & ~exterior
    return {(int(rr + r0), int(cc + c0)) for rr, cc in np.argwhere(inside)}


def _paint_coords(grid: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    h, w = grid.shape
    for r, c in coords:
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = color


def _render_trigger_coords(curve_coords: set[tuple[int, int]], desc: dict) -> dict:
    mirror_coords = _reflect_coords(curve_coords, desc["axis_kind"], desc["axis"])
    chord = _chord_coords(desc["endpoints"])
    fill = _interior_coords(curve_coords | mirror_coords | chord)
    return {"curve": curve_coords, "mirror": mirror_coords, "chord": chord, "fill": fill}


def _normalize_coord_sets(sets: dict[str, set[tuple[int, int]]]) -> tuple[dict[str, set[tuple[int, int]]], tuple[int, int]]:
    union = set()
    for coords in sets.values():
        union |= coords
    r0, c0, _, _ = _bbox_from_set(union)
    shifted = {}
    for key, coords in sets.items():
        shifted[key] = {(r - r0, c - c0) for r, c in coords}
    return shifted, (r0, c0)


def _build_patch_from_sets(sets: dict[str, set[tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray]:
    union = set()
    for coords in sets.values():
        union |= coords
    r0, c0, r1, c1 = _bbox_from_set(union)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for r, c in sets["curve"]:
        inp[r - r0, c - c0] = GREY
        out[r - r0, c - c0] = GREY
    for r, c in sets.get("mirror", set()):
        out[r - r0, c - c0] = GREY
    for r, c in sets.get("fill", set()):
        out[r - r0, c - c0] = YELLOW
    for r, c in sets.get("chord", set()):
        out[r - r0, c - c0] = RED
    return inp, out


def _completion_for_component(coords: np.ndarray) -> dict | None:
    desc = _classify_transformable(coords)
    if desc is None:
        return None
    curve_coords = {tuple(map(int, rc)) for rc in coords}
    rendered = _render_trigger_coords(curve_coords, desc)
    return {"desc": desc, **rendered}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each endpoint-aligned grey open curve across its invisible endpoint chord, then close and fill the two chambers.

    Input structure:
    - The background is black and the foreground consists of separate one-cell-thick grey open curves.
    - Each curve is a single 4-connected path with exactly two endpoints.
    - Some curves have endpoints on the same row or the same column, while other similar curves end on different rows and columns.

    Rule:
    - For a curve whose two endpoints are aligned horizontally or vertically, use the segment between those endpoints as an invisible mirror axis.
    - Reflect the whole grey curve across that endpoint chord, draw the chord itself in red, and fill every enclosed cavity between the chord and the two grey halves with yellow.
    - Leave every curve whose endpoints are not aligned, or whose interior is not strictly on one side of the endpoint chord, unchanged.

    Key insight:
    - The axis is not drawn in the input; it must be inferred from an open curve's own aligned endpoints, and only those aligned curves become closed symmetric chambers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()

    colors = sorted(int(v) for v in np.unique(g) if int(v) != bg)
    if not colors:
        return out
    curve_color = GREY if GREY in colors else colors[0]

    for comp in _components(g == curve_color):
        completion = _completion_for_component(comp)
        if completion is None:
            continue

        _paint_coords(out, completion["mirror"], curve_color)
        h, w = out.shape
        for r, c in completion["fill"]:
            if 0 <= r < h and 0 <= c < w and out[r, c] == bg:
                out[r, c] = YELLOW
        _paint_coords(out, completion["chord"], RED)

    return out


def _step_toward(value: int, target: int) -> int:
    if value < target:
        return value + 1
    if value > target:
        return value - 1
    return value


def _append_segment(coords: list[tuple[int, int]], row: int, col: int, target_row: int, target_col: int) -> tuple[int, int]:
    while row != target_row:
        row = _step_toward(row, target_row)
        coords.append((row, col))
    while col != target_col:
        col = _step_toward(col, target_col)
        coords.append((row, col))
    return row, col


def _build_columnwise_path(row_values: list[int]) -> set[tuple[int, int]]:
    coords = [(row_values[0], 0)]
    row = row_values[0]
    col = 0
    for x in range(len(row_values) - 1):
        target_row = row_values[x + 1]
        if row == 0 and target_row != 0:
            order = "vh"
        elif target_row == 0 and row != 0:
            order = "hv"
        else:
            order = random.choice(("vh", "hv"))
        if order == "vh":
            row, col = _append_segment(coords, row, col, target_row, col)
            row, col = _append_segment(coords, row, col, row, col + 1)
        else:
            row, col = _append_segment(coords, row, col, row, col + 1)
            row, col = _append_segment(coords, row, col, target_row, col)
    return set(coords)


def _transform_coords(coords: set[tuple[int, int]], mode: str) -> set[tuple[int, int]]:
    if mode == "id":
        return {(r, c) for r, c in coords}
    if mode == "rot90":
        return {(c, -r) for r, c in coords}
    if mode == "rot180":
        return {(-r, -c) for r, c in coords}
    if mode == "rot270":
        return {(-c, r) for r, c in coords}
    if mode == "flip_h":
        return {(-r, c) for r, c in coords}
    if mode == "flip_v":
        return {(r, -c) for r, c in coords}
    if mode == "diag":
        return {(c, r) for r, c in coords}
    if mode == "anti":
        return {(-c, -r) for r, c in coords}
    raise ValueError(mode)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr0 = max(0, dr)
            rr1 = h + min(0, dr)
            cc0 = max(0, dc)
            cc1 = w + min(0, dc)
            out[rr0:rr1, cc0:cc1] |= mask[rr0 - dr : rr1 - dr, cc0 - dc : cc1 - dc]
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != transparent
        view[mask] = patch[mask]


def _sample_trigger_object(axis_kind: str) -> dict:
    for _ in range(OBJECT_ATTEMPTS):
        span = random.randint(5, 9)
        max_depth = random.randint(2, 5)
        interior = [random.randint(1, max_depth) for _ in range(span - 1)]
        if sum(interior) < 5:
            continue
        if max(interior) < 2:
            continue
        if random.random() < 0.5:
            interior.reverse()
        side = random.choice((-1, 1))
        row_values = [0] + [side * v for v in interior] + [0]
        coords = _build_columnwise_path(row_values)
        if axis_kind == "v":
            coords = {(c, r) for r, c in coords}

        comp = np.array(sorted(coords), dtype=int)
        desc = _classify_transformable(comp)
        if desc is None or desc["axis_kind"] != axis_kind:
            continue

        rendered = _render_trigger_coords(coords, desc)
        if axis_kind == "h":
            fills_a = [1 for r, _ in rendered["fill"] if r < desc["axis"]]
            fills_b = [1 for r, _ in rendered["fill"] if r > desc["axis"]]
        else:
            fills_a = [1 for _, c in rendered["fill"] if c < desc["axis"]]
            fills_b = [1 for _, c in rendered["fill"] if c > desc["axis"]]
        if not fills_a or not fills_b:
            continue

        shifted, origin = _normalize_coord_sets(rendered)
        inp, out = _build_patch_from_sets(shifted)
        change_mask = out != inp
        reserve_mask = _dilate8(out != BLACK)
        return {
            "kind": axis_kind,
            "desc": desc,
            "axis_local": desc["axis"] - (origin[0] if axis_kind == "h" else origin[1]),
            "input_patch": inp,
            "output_patch": out,
            "change_mask": change_mask,
            "reserve_mask": reserve_mask,
            "red_mask": out == RED,
            "yellow_mask": out == YELLOW,
        }
    raise RuntimeError(f"failed to sample trigger object: {axis_kind}")


def _sample_decoy_object() -> dict:
    transforms = ("id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "diag", "anti")
    for _ in range(OBJECT_ATTEMPTS):
        span = random.randint(5, 9)
        end_row = random.choice((-1, 1)) * random.randint(2, 5)
        low = min(0, end_row) - random.randint(0, 3)
        high = max(0, end_row) + random.randint(0, 3)
        row_values = [0]
        for _col in range(1, span):
            row_values.append(random.randint(low, high))
        row_values.append(end_row)
        if all(v == 0 for v in row_values[1:-1]):
            continue
        if len(set(row_values[1:-1])) < 2 and row_values[1:-1].count(end_row) == len(row_values) - 2:
            continue

        coords = _build_columnwise_path(row_values)
        coords = _transform_coords(coords, random.choice(transforms))
        comp = np.array(sorted(coords), dtype=int)
        if not _simple_open_curve(comp):
            continue
        if _classify_transformable(comp) is not None:
            continue

        inp, out = _build_patch_from_sets({"curve": coords})
        reserve_mask = _dilate8(out != BLACK)
        return {
            "kind": "d",
            "input_patch": inp,
            "output_patch": out,
            "change_mask": np.zeros_like(inp, dtype=bool),
            "reserve_mask": reserve_mask,
            "red_mask": out == RED,
            "yellow_mask": out == YELLOW,
        }
    raise RuntimeError("failed to sample decoy object")


def _sample_canvas(objects: list[dict]) -> tuple[int, int]:
    max_h = max(obj["output_patch"].shape[0] for obj in objects)
    max_w = max(obj["output_patch"].shape[1] for obj in objects)
    base_h = max(18, max_h + 4)
    base_w = max(18, max_w + 4)
    return random.randint(base_h, 30), random.randint(base_w, 30)


def _place_objects(objects: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    ordered = sorted(objects, key=lambda obj: int(np.count_nonzero(obj["reserve_mask"])), reverse=True)
    h, w = _sample_canvas(ordered)
    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for obj in ordered:
        patch_h, patch_w = obj["output_patch"].shape
        if patch_h > h or patch_w > w:
            return None

        row_lo = 1 if h - patch_h >= 2 else 0
        row_hi = h - patch_h - 1 if h - patch_h >= 2 else h - patch_h
        col_lo = 1 if w - patch_w >= 2 else 0
        col_hi = w - patch_w - 1 if w - patch_w >= 2 else w - patch_w

        placed_ok = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            reserve_mask = obj["reserve_mask"]
            target = reserved[r0 : r0 + patch_h, c0 : c0 + patch_w]
            if np.any(target & reserve_mask):
                continue

            _paste(inp, obj["input_patch"], r0, c0, transparent=BLACK)
            _paste(expected, obj["output_patch"], r0, c0, transparent=BLACK)
            target |= reserve_mask

            local = dict(obj)
            local["r0"] = r0
            local["c0"] = c0
            local["patch_h"] = patch_h
            local["patch_w"] = patch_w
            placed.append(local)
            placed_ok = True
            break

        if not placed_ok:
            return None

    return inp, expected, placed


def _absolute_mask(shape: tuple[int, int], obj: dict, key: str) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    local = obj[key]
    h, w = local.shape
    out[obj["r0"] : obj["r0"] + h, obj["c0"] : obj["c0"] + w] |= local
    return out


def _validate_scene(inp: np.ndarray, expected: np.ndarray, placed: list[dict]) -> None:
    assert any(obj["kind"] != "d" for obj in placed), "need at least one transformable curve"
    assert any(obj["kind"] == "d" for obj in placed), "need at least one decoy"
    assert not np.array_equal(inp, expected), "scene must change"

    allowed_changes = np.zeros_like(inp, dtype=bool)
    for obj in placed:
        allowed_changes |= _absolute_mask(inp.shape, obj, "change_mask")
        red_mask = _absolute_mask(inp.shape, obj, "red_mask")
        yellow_mask = _absolute_mask(inp.shape, obj, "yellow_mask")
        if obj["kind"] == "d":
            patch_in = inp[obj["r0"] : obj["r0"] + obj["patch_h"], obj["c0"] : obj["c0"] + obj["patch_w"]]
            patch_out = expected[obj["r0"] : obj["r0"] + obj["patch_h"], obj["c0"] : obj["c0"] + obj["patch_w"]]
            own_mask = obj["input_patch"] != BLACK
            assert np.array_equal(patch_in[own_mask], patch_out[own_mask]), "decoy pixels should remain unchanged"
            assert not red_mask.any() and not yellow_mask.any(), "decoys must not create helper colors"
            continue

        assert red_mask.any(), "triggered object must draw a red chord"
        assert yellow_mask.any(), "triggered object must fill a cavity"

        desc = obj["desc"]
        if desc["axis_kind"] == "h":
            axis_abs = obj["axis_local"] + obj["r0"]
            rows, _cols = np.where(yellow_mask)
            assert np.any(rows < axis_abs) and np.any(rows > axis_abs), "both sides of a horizontal chord must fill"
        else:
            axis_abs = obj["axis_local"] + obj["c0"]
            _rows, cols = np.where(yellow_mask)
            assert np.any(cols < axis_abs) and np.any(cols > axis_abs), "both sides of a vertical chord must fill"

    assert np.array_equal(inp != expected, allowed_changes), "only intended cells may change"
    actual = main(inp)
    assert np.array_equal(actual, expected), "scene oracle must match solver output"


def _sample_scene() -> np.ndarray | None:
    scenario = random.choice(SCENARIOS)
    objects = []
    for kind in scenario:
        if kind == "h":
            objects.append(_sample_trigger_object("h"))
        elif kind == "v":
            objects.append(_sample_trigger_object("v"))
        else:
            objects.append(_sample_decoy_object())

    placed = _place_objects(objects)
    if placed is None:
        return None
    inp, expected, info = placed

    comps = _components(inp == GREY)
    if len(comps) != len(objects):
        return None
    if sum(_classify_transformable(comp) is not None for comp in comps) != sum(obj["kind"] != "d" for obj in info):
        return None

    _validate_scene(inp, expected, info)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-chord-axis ARC input")
