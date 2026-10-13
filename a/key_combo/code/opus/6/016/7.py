from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
GUIDE = 3
WALL = 5
PALETTE = (1, 2, 4, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
MIN_SIDE = 16
MAX_SIDE = 22

LINE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _component_top_left(coords: np.ndarray) -> tuple[int, int]:
    mins = coords.min(axis=0)
    return int(mins[0]), int(mins[1])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bboxes_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return bool(a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _mirror_coords(coords: np.ndarray, axis_kind: str, axis_idx: int) -> np.ndarray:
    mirrored = coords.copy()
    if axis_kind == "h":
        mirrored[:, 0] = 2 * axis_idx - mirrored[:, 0]
    else:
        mirrored[:, 1] = 2 * axis_idx - mirrored[:, 1]
    return mirrored


def _full_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segment must be axis-aligned")


def _path_candidates(
    start: tuple[int, int],
    stop: tuple[int, int],
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    if start == stop:
        return [((start[0], start[1]), [start])]
    if start[0] == stop[0] or start[1] == stop[1]:
        bend = (min(start[0], stop[0]), min(start[1], stop[1]))
        return [(bend, _full_segment(start, stop))]

    bend_a = (start[0], stop[1])
    bend_b = (stop[0], start[1])
    path_a = _full_segment(start, bend_a) + _full_segment(bend_a, stop)[1:]
    path_b = _full_segment(start, bend_b) + _full_segment(bend_b, stop)[1:]
    return [(bend_a, path_a), (bend_b, path_b)]


def _path_clear(path: list[tuple[int, int]], canvas: np.ndarray, color: int) -> bool:
    allowed = {BACKGROUND, GUIDE, color}
    return all(int(canvas[r, c]) in allowed for r, c in path)


def _best_connector(comp_a: np.ndarray, comp_b: np.ndarray, canvas: np.ndarray, color: int) -> dict | None:
    best: dict | None = None
    for start_raw in comp_a:
        start = (int(start_raw[0]), int(start_raw[1]))
        for stop_raw in comp_b:
            stop = (int(stop_raw[0]), int(stop_raw[1]))
            for bend, path in _path_candidates(start, stop):
                if not _path_clear(path, canvas, color):
                    continue
                record = {
                    "start": start,
                    "stop": stop,
                    "bend": (int(bend[0]), int(bend[1])),
                    "path": path,
                    "length": len(path) - 1,
                }
                key = (
                    record["length"],
                    record["bend"][0],
                    record["bend"][1],
                    record["start"][0],
                    record["start"][1],
                    record["stop"][0],
                    record["stop"][1],
                    tuple(record["path"]),
                )
                if best is None or key < best["key"]:
                    record["key"] = key
                    best = record
    return best


def _detect_axis(grid: np.ndarray) -> tuple[str, int]:
    row_axes = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GUIDE)]
    col_axes = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GUIDE)]
    if len(row_axes) == 1 and not col_axes:
        return "h", int(row_axes[0])
    if len(col_axes) == 1 and not row_axes:
        return "v", int(col_axes[0])
    raise AssertionError("expected exactly one full green guide line")


def _source_components(grid: np.ndarray) -> tuple[str, int, str, list[dict]]:
    axis_kind, axis_idx = _detect_axis(grid)
    source_mask = ~np.isin(grid, [BACKGROUND, GUIDE, WALL])

    if axis_kind == "h":
        top_has = bool(source_mask[:axis_idx, :].any())
        bottom_has = bool(source_mask[axis_idx + 1 :, :].any())
        if top_has == bottom_has:
            raise AssertionError("exactly one half must contain source components")
        source_side = "top" if top_has else "bottom"
        half_mask = np.zeros_like(source_mask, dtype=bool)
        if source_side == "top":
            half_mask[:axis_idx, :] = True
        else:
            half_mask[axis_idx + 1 :, :] = True
    else:
        left_has = bool(source_mask[:, :axis_idx].any())
        right_has = bool(source_mask[:, axis_idx + 1 :].any())
        if left_has == right_has:
            raise AssertionError("exactly one half must contain source components")
        source_side = "left" if left_has else "right"
        half_mask = np.zeros_like(source_mask, dtype=bool)
        if source_side == "left":
            half_mask[:, :axis_idx] = True
        else:
            half_mask[:, axis_idx + 1 :] = True

    colors = [int(color) for color in np.unique(grid[half_mask & source_mask])]
    if not 2 <= len(colors) <= 3:
        raise AssertionError("expected 2-3 source colors")

    comps: list[dict] = []
    for color in colors:
        pieces = _component_coords((grid == color) & half_mask)
        if len(pieces) != 1:
            raise AssertionError("each source color must form exactly one component")
        coords = pieces[0]
        if not 2 <= len(coords) <= 5:
            raise AssertionError("source components must have 2-5 cells")
        comps.append({"color": color, "coords": coords, "top_left": _component_top_left(coords)})

    comps.sort(key=lambda item: item["top_left"])
    return axis_kind, axis_idx, source_side, comps


def _solve_details(grid: np.ndarray) -> dict:
    axis_kind, axis_idx, _source_side, comps = _source_components(grid)
    out = np.array(grid, dtype=int, copy=True)
    ordered: list[dict] = []
    for item in comps:
        color = int(item["color"])
        mirrored = _mirror_coords(item["coords"], axis_kind, axis_idx)
        if np.any(mirrored < 0) or np.any(mirrored[:, 0] >= grid.shape[0]) or np.any(mirrored[:, 1] >= grid.shape[1]):
            raise AssertionError("mirrored component leaves the grid")
        for r, c in mirrored:
            if int(out[r, c]) != BACKGROUND:
                raise AssertionError("mirrored component overlaps an existing object")
            out[r, c] = color
        ordered.append({"color": color, "source": item["coords"], "mirror": mirrored, "top_left": item["top_left"]})

    paths: list[dict] = []
    for item in ordered:
        color = int(item["color"])
        best = _best_connector(item["source"], item["mirror"], out, color)
        if best is None:
            raise AssertionError(f"no connector found for color {color}")
        for r, c in best["path"]:
            cell = int(out[r, c])
            if cell not in (BACKGROUND, GUIDE, color):
                raise AssertionError("connector crossed a blocker")
            if cell != GUIDE:
                out[r, c] = color
        paths.append({**item, **best})

    return {"axis_kind": axis_kind, "axis_idx": axis_idx, "components": ordered, "paths": paths, "output": out}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the source-side components across the green guide line, then route each original-copy color pair with the shortest clear one-bend connector.

    Input structure:
    - The background is black, and exactly one full green line spans an entire row or an entire column.
    - Only one half of the grid contains 2-3 small 4-connected colored source components, each with a distinct non-black, non-green, non-grey color.
    - Grey wall segments of length 1-3 may appear on both halves; the opposite half contains no source components.

    Rule:
    - Use the guide line as the mirror axis: a horizontal guide means top-bottom reflection, and a vertical guide means left-right reflection.
    - Reflect every source component to the empty half with the same color, while leaving grey walls unchanged.
    - Process source colors by the row-major order of their original top-left cells, and for each original/mirrored pair choose the shortest clear orthogonal path with at most one bend.
    - Paths may pass through green guide cells but may not overwrite them, and they may not cross grey walls, other colors, or earlier connectors of other colors; equal-length choices break ties by smaller bend row, then smaller bend column.

    Key insight:
    - The guide line is not itself the output pattern: it only tells you how to create the missing mirrored targets, after which the real task is obstacle-aware pairwise routing.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _shape_coords(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> np.ndarray:
    return np.array([(r0 + dr, c0 + dc) for dr, dc in shape], dtype=int)


def _path_signature(path: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in path)


def _wall_components(grid: np.ndarray) -> list[np.ndarray]:
    return _component_coords(grid == WALL)


def _wall_layout_ok(grid: np.ndarray) -> bool:
    axis_kind, axis_idx = _detect_axis(grid)
    comps = _wall_components(grid)
    if not comps:
        return False
    near_a = False
    near_b = False
    for comp in comps:
        if not 1 <= len(comp) <= 3:
            return False
        rows = {int(v) for v in comp[:, 0]}
        cols = {int(v) for v in comp[:, 1]}
        if len(rows) != 1 and len(cols) != 1:
            return False
        if axis_kind == "v":
            if int(comp[:, 1].max()) < axis_idx:
                near_a = True
            elif int(comp[:, 1].min()) > axis_idx:
                near_b = True
        else:
            if int(comp[:, 0].max()) < axis_idx:
                near_a = True
            elif int(comp[:, 0].min()) > axis_idx:
                near_b = True
    return near_a and near_b


def _choose_axis_col(w: int) -> int:
    low = max(7, w // 2 - 1)
    high = min(w - 8, w // 2 + 1)
    return random.randint(low, high)


def _place_component(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    for r, c in coords:
        if int(grid[r, c]) != BACKGROUND:
            raise AssertionError("component overlap")
        grid[r, c] = color


def _candidate_rows(h: int, color_count: int) -> list[int] | None:
    tops = list(range(1, h - 1))
    random.shuffle(tops)
    chosen: list[int] = []
    for top in sorted(tops):
        if top + 1 >= h:
            continue
        if any(abs(top - other) < 3 for other in chosen):
            continue
        chosen.append(top)
        if len(chosen) == color_count:
            return chosen
    return None


def _build_canonical_case() -> np.ndarray | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    axis = _choose_axis_col(w)
    if axis < 7 or w - axis - 1 < 7:
        return None

    color_count = random.choice((2, 3))
    tops = _candidate_rows(h, color_count)
    if tops is None:
        return None
    grid = np.zeros((h, w), dtype=int)
    grid[:, axis] = GUIDE

    colors = random.sample(PALETTE, color_count)
    c0 = random.randint(1, axis - 3)
    placed: list[tuple[int, np.ndarray, np.ndarray]] = []
    route_rows: list[int] = []
    blocker_cells: list[tuple[int, int]] = []

    for idx, (color, top) in enumerate(zip(colors, tops)):
        coords = _shape_coords(LINE_SHAPES[0], int(top), c0)
        mirrored = _mirror_coords(coords, "v", axis)
        if np.any(mirrored[:, 1] >= w):
            return None
        box_src = _bbox(coords)
        box_mir = _bbox(mirrored)
        if not _bboxes_far(box_src, box_mir, gap=1):
            return None
        if any(not _bboxes_far(box_src, _bbox(prev[1]), gap=1) for prev in placed):
            return None
        if any(not _bboxes_far(box_src, _bbox(prev[2]), gap=1) for prev in placed):
            return None
        if any(not _bboxes_far(box_mir, _bbox(prev[1]), gap=1) for prev in placed):
            return None
        if any(not _bboxes_far(box_mir, _bbox(prev[2]), gap=1) for prev in placed):
            return None
        placed.append((color, coords, mirrored))

        top_row = int(coords[:, 0].min())
        bottom_row = int(coords[:, 0].max())
        mirror_col = int(mirrored[:, 1].min())
        force_lower = idx == 0 or random.random() < 0.5
        route_rows.append(bottom_row if force_lower else top_row)
        if force_lower:
            blocker_row = top_row
        else:
            blocker_row = bottom_row
        if idx % 2 == 0:
            blocker_col = random.randint(c0 + 1, axis - 1)
        else:
            blocker_col = random.randint(axis + 1, mirror_col - 1)
        blocker_cells.append((blocker_row, blocker_col))

    for color, coords, _mirrored in placed:
        _place_component(grid, coords, color)

    for r, c in blocker_cells:
        if int(grid[r, c]) != BACKGROUND:
            return None
        grid[r, c] = WALL

    try:
        details = _solve_details(grid)
    except AssertionError:
        return None

    if len(details["paths"]) != color_count:
        return None
    for item, expected_row in zip(details["paths"], route_rows):
        if item["bend"][0] != expected_row:
            return None

    no_walls = grid.copy()
    no_walls[no_walls == WALL] = BACKGROUND
    try:
        no_wall_details = _solve_details(no_walls)
    except AssertionError:
        return None
    if not any(
        _path_signature(a["path"]) != _path_signature(b["path"])
        for a, b in zip(details["paths"], no_wall_details["paths"])
    ):
        return None

    reserved = details["output"] != BACKGROUND
    extra_attempts = random.randint(1, 3)
    for _ in range(extra_attempts):
        _add_safe_wall_segment(grid, reserved, axis)

    return grid


def _add_safe_wall_segment(grid: np.ndarray, reserved: np.ndarray, axis_idx: int) -> bool:
    h, w = grid.shape
    for _ in range(120):
        length = random.randint(1, 3)
        orient = random.choice(("h", "v"))
        side = random.choice(("left", "right"))
        if orient == "h":
            r = random.randint(0, h - 1)
            if side == "left":
                c0 = random.randint(0, axis_idx - length)
            else:
                c0 = random.randint(axis_idx + 1, w - length)
            seg = [(r, c0 + dc) for dc in range(length)]
        else:
            c = random.randint(0, w - 1)
            if c == axis_idx:
                continue
            if side == "left" and c >= axis_idx:
                continue
            if side == "right" and c <= axis_idx:
                continue
            r0 = random.randint(0, h - length)
            seg = [(r0 + dr, c) for dr in range(length)]

        if any(reserved[r, c] or int(grid[r, c]) != BACKGROUND for r, c in seg):
            continue
        seg_set = set(seg)
        ok = True
        for r, c in seg:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seg_set and int(grid[nr, nc]) == WALL:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for r, c in seg:
            grid[r, c] = WALL
        return True
    return False


def _transform_case(grid: np.ndarray) -> np.ndarray:
    mode = random.choice(("identity", "flip_lr", "rot90", "rot90_flip_ud"))
    if mode == "identity":
        return np.array(grid, dtype=int, copy=True)
    if mode == "flip_lr":
        return np.fliplr(grid).copy()
    rotated = np.rot90(grid, k=1).copy()
    if mode == "rot90":
        return rotated
    return np.flipud(rotated).copy()


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        details = _solve_details(grid)
    except AssertionError:
        return False
    if not _wall_layout_ok(grid):
        return False
    if len(details["paths"]) < 2:
        return False
    no_walls = grid.copy()
    no_walls[no_walls == WALL] = BACKGROUND
    try:
        no_wall_details = _solve_details(no_walls)
    except AssertionError:
        return False
    if not any(
        _path_signature(a["path"]) != _path_signature(b["path"])
        for a, b in zip(details["paths"], no_wall_details["paths"])
    ):
        return False
    return bool(np.any(details["output"] != grid))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        canonical = _build_canonical_case()
        if canonical is None:
            continue
        grid = _transform_case(canonical)
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
