from __future__ import annotations

import random
from typing import Iterable

import numpy as np


GREY = 5
ORIENTATIONS = ("horizontal", "vertical", "diag_down", "diag_up")
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]

GENERATION_RETRIES = 400
OBJECT_COUNT_RANGE = (1, 4)
GRID_MIN_SIZE, GRID_MAX_SIZE = 18, 30
LARGE_OBJECT_LENGTH_RANGE = (3, 5)
SMALL_OBJECT_LENGTH_RANGE = (3, 6)
PATTERN_DEPTH_RANGE = (2, 3)
PATTERN_SIZE_RANGE = (3, 7)
PLACEMENT_MARGIN = 1

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    assert connectivity in {4, 8}
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            out.append(np.array(cells, dtype=int))

    return out


def _infer_mirror(coords: np.ndarray, color: int) -> dict[str, object]:
    coords = np.asarray(coords, dtype=int)
    assert coords.ndim == 2 and coords.shape[1] == 2
    rows = coords[:, 0]
    cols = coords[:, 1]
    length = int(len(coords))
    assert length >= 3

    if np.all(rows == rows[0]):
        row = int(rows[0])
        ordered = np.sort(cols)
        assert np.array_equal(ordered, np.arange(int(ordered[0]), int(ordered[-1]) + 1))
        return {
            "color": int(color),
            "kind": "horizontal",
            "row": row,
            "c0": int(ordered[0]),
            "c1": int(ordered[-1]),
            "coords": {(int(r), int(c)) for r, c in coords},
            "sort_key": (row, int(ordered[0]), 0),
        }

    if np.all(cols == cols[0]):
        col = int(cols[0])
        ordered = np.sort(rows)
        assert np.array_equal(ordered, np.arange(int(ordered[0]), int(ordered[-1]) + 1))
        return {
            "color": int(color),
            "kind": "vertical",
            "col": col,
            "r0": int(ordered[0]),
            "r1": int(ordered[-1]),
            "coords": {(int(r), int(c)) for r, c in coords},
            "sort_key": (int(ordered[0]), col, 1),
        }

    diffs = rows - cols
    if np.all(diffs == diffs[0]):
        order = np.argsort(rows)
        ordered_rows = rows[order]
        ordered_cols = cols[order]
        assert np.array_equal(ordered_rows, np.arange(int(ordered_rows[0]), int(ordered_rows[-1]) + 1))
        assert np.array_equal(ordered_cols, np.arange(int(ordered_cols[0]), int(ordered_cols[-1]) + 1))
        return {
            "color": int(color),
            "kind": "diag_down",
            "delta": int(diffs[0]),
            "coords": {(int(r), int(c)) for r, c in coords},
            "sort_key": (int(ordered_rows[0]), int(ordered_cols[0]), 2),
        }

    sums = rows + cols
    if np.all(sums == sums[0]):
        order = np.argsort(rows)
        ordered_rows = rows[order]
        ordered_cols = cols[order]
        assert np.array_equal(ordered_rows, np.arange(int(ordered_rows[0]), int(ordered_rows[-1]) + 1))
        assert np.array_equal(ordered_cols, np.arange(int(ordered_cols[0]), int(ordered_cols[-1]) - 1, -1))
        return {
            "color": int(color),
            "kind": "diag_up",
            "total": int(sums[0]),
            "coords": {(int(r), int(c)) for r, c in coords},
            "sort_key": (int(ordered_rows[0]), int(ordered_cols[0]), 3),
        }

    raise AssertionError(f"color {color} is not a valid mirror segment")


def _side_value(mirror: dict[str, object], r: int, c: int) -> int:
    kind = mirror["kind"]
    if kind == "horizontal":
        return int(r - int(mirror["row"]))
    if kind == "vertical":
        return int(c - int(mirror["col"]))
    if kind == "diag_down":
        return int(r - c - int(mirror["delta"]))
    if kind == "diag_up":
        return int(r + c - int(mirror["total"]))
    raise AssertionError(f"unknown mirror kind: {kind}")


def _reflect_point(mirror: dict[str, object], r: int, c: int) -> tuple[int, int]:
    kind = mirror["kind"]
    if kind == "horizontal":
        return 2 * int(mirror["row"]) - r, c
    if kind == "vertical":
        return r, 2 * int(mirror["col"]) - c
    if kind == "diag_down":
        delta = int(mirror["delta"])
        return c + delta, r - delta
    if kind == "diag_up":
        total = int(mirror["total"])
        return total - c, total - r
    raise AssertionError(f"unknown mirror kind: {kind}")


def _touches_mirror(component: np.ndarray, mirror_coords: set[tuple[int, int]]) -> bool:
    for rr_raw, cc_raw in component:
        rr = int(rr_raw)
        cc = int(cc_raw)
        for dr, dc in NEIGHBORS8:
            if (rr + dr, cc + dc) in mirror_coords:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each grey source pattern across its adjacent colored mirror segment.

    Every non-background, non-grey color is a straight mirror segment. The grey
    pattern next to that mirror is copied to the opposite side by geometric
    reflection, but the reflected copy is recolored with the mirror's color.
    Existing non-background cells are never overwritten.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape

    bg = _dominant_color(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert GREY in colors, "expected grey source pixels"

    mirror_colors = [color for color in colors if color != GREY]
    assert mirror_colors, "expected at least one mirror"

    mirrors = []
    for color in mirror_colors:
        coords = np.argwhere(g == color)
        mirrors.append(_infer_mirror(coords, color))
    mirrors.sort(key=lambda item: item["sort_key"])

    grey_components = _components(g == GREY, connectivity=8)
    assert grey_components, "expected grey source components"

    assigned: list[list[np.ndarray]] = [[] for _ in mirrors]
    for component in grey_components:
        touched = [idx for idx, mirror in enumerate(mirrors) if _touches_mirror(component, mirror["coords"])]
        assert len(touched) == 1, "each grey component must touch exactly one mirror"
        assigned[touched[0]].append(component)

    out = g.copy()
    for mirror, groups in zip(mirrors, assigned):
        assert groups, "each mirror must have an adjacent grey source pattern"
        source = np.concatenate(groups, axis=0)
        side_signs = {1 if _side_value(mirror, int(r), int(c)) > 0 else -1 for r, c in source}
        raw_values = {_side_value(mirror, int(r), int(c)) for r, c in source}
        assert 0 not in raw_values, "source pixels cannot lie on the mirror"
        assert len(side_signs) == 1, "source pixels for one mirror must stay on one side"

        for r_raw, c_raw in source:
            rr, cc = _reflect_point(mirror, int(r_raw), int(c_raw))
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == bg:
                out[rr, cc] = int(mirror["color"])

    return out


def _local_neighbors(a: int, b: int, length: int, depth: int) -> Iterable[tuple[int, int]]:
    for da, db in NEIGHBORS4:
        na = a + da
        nb = b + db
        if 0 <= na < length and 0 <= nb < depth:
            yield na, nb


def _generate_local_pattern(length: int, depth: int) -> list[tuple[int, int]] | None:
    max_cells = min(PATTERN_SIZE_RANGE[1], length * depth)
    if max_cells < PATTERN_SIZE_RANGE[0]:
        return None

    for _ in range(80):
        count = random.randint(PATTERN_SIZE_RANGE[0], max_cells)
        if count == length * depth and length > 1 and depth > 1:
            continue

        seed = (random.randrange(length), 0)
        cells = {seed}
        frontier = set(_local_neighbors(seed[0], seed[1], length, depth))
        while len(cells) < count:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in cells:
                continue
            cells.add(cell)
            frontier.update(_local_neighbors(cell[0], cell[1], length, depth))
            frontier.difference_update(cells)

        if len(cells) != count:
            continue

        a_values = {a for a, _ in cells}
        b_values = {b for _, b in cells}
        if len(a_values) < 2 or len(b_values) < 2:
            continue

        return sorted((a, b + 1) for a, b in cells)

    return None


def _map_local_point(kind: str, length: int, a: int, b: int, sigma: int) -> tuple[int, int]:
    if kind == "horizontal":
        return sigma * b, a
    if kind == "vertical":
        return a, sigma * b
    if kind == "diag_down":
        return a + sigma * b, a - sigma * b
    if kind == "diag_up":
        return a + sigma * b, (length - 1 - a) + sigma * b
    raise AssertionError(f"unknown kind: {kind}")


def _line_points(kind: str, length: int) -> list[tuple[int, int]]:
    if kind == "horizontal":
        return [(0, a) for a in range(length)]
    if kind == "vertical":
        return [(a, 0) for a in range(length)]
    if kind == "diag_down":
        return [(a, a) for a in range(length)]
    if kind == "diag_up":
        return [(a, length - 1 - a) for a in range(length)]
    raise AssertionError(f"unknown kind: {kind}")


def _shift_coords(coords: Iterable[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    pts = np.array(list(coords), dtype=int)
    mins = pts.min(axis=0)
    shifted = pts - mins
    return shifted, (int(mins[0]), int(mins[1]))


def _build_object(kind: str, color: int, crowded: bool) -> dict[str, object] | None:
    length_lo, length_hi = LARGE_OBJECT_LENGTH_RANGE if crowded else SMALL_OBJECT_LENGTH_RANGE
    length = random.randint(length_lo, length_hi)
    depth = random.randint(*PATTERN_DEPTH_RANGE)
    pattern = _generate_local_pattern(length, depth)
    if pattern is None:
        return None

    side = random.choice((-1, 1))
    line = _line_points(kind, length)
    source = [_map_local_point(kind, length, a, b, side) for a, b in pattern]
    reflected = [_map_local_point(kind, length, a, b, -side) for a, b in pattern]
    shifted, _offset = _shift_coords(line + source + reflected)

    n_line = len(line)
    n_source = len(source)
    line_arr = shifted[:n_line]
    source_arr = shifted[n_line : n_line + n_source]
    reflected_arr = shifted[n_line + n_source :]

    height = int(shifted[:, 0].max()) + 1
    width = int(shifted[:, 1].max()) + 1
    return {
        "kind": kind,
        "color": int(color),
        "line": line_arr,
        "source": source_arr,
        "reflected": reflected_arr,
        "height": height,
        "width": width,
    }


def _bbox_clear(
    occupied: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    margin: int,
) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, top + height + margin)
    c1 = min(w, left + width + margin)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_objects(objects: list[dict[str, object]], height: int, width: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(objects)), key=lambda idx: objects[idx]["height"] * objects[idx]["width"], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(objects)

    for idx in order:
        obj = objects[idx]
        obj_h = int(obj["height"])
        obj_w = int(obj["width"])
        if obj_h > height or obj_w > width:
            return None

        candidates = []
        for top in range(height - obj_h + 1):
            for left in range(width - obj_w + 1):
                if _bbox_clear(occupied, top, left, obj_h, obj_w, PLACEMENT_MARGIN):
                    candidates.append((top, left))

        if not candidates:
            return None

        top, left = random.choice(candidates)
        placements[idx] = (top, left)
        occupied[top : top + obj_h, left : left + obj_w] = True

    return [placement for placement in placements if placement is not None]


def _render_scene(
    objects: list[dict[str, object]],
    placements: list[tuple[int, int]],
    shape: tuple[int, int],
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)

    for obj, (top, left) in zip(objects, placements):
        color = int(obj["color"])

        line = np.asarray(obj["line"], dtype=int)
        source = np.asarray(obj["source"], dtype=int)
        reflected = np.asarray(obj["reflected"], dtype=int)

        inp[top + line[:, 0], left + line[:, 1]] = color
        inp[top + source[:, 0], left + source[:, 1]] = GREY

        out[top + line[:, 0], left + line[:, 1]] = color
        out[top + source[:, 0], left + source[:, 1]] = GREY
        for rr_raw, cc_raw in reflected:
            rr = top + int(rr_raw)
            cc = left + int(cc_raw)
            if int(out[rr, cc]) == bg:
                out[rr, cc] = color

    return inp, out


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    bg = random.choice(COLOR_POOL)
    mirror_colors = random.sample([c for c in COLOR_POOL if c != bg], object_count)
    orientations = random.sample(ORIENTATIONS, k=object_count)

    objects = []
    crowded = object_count >= 3
    for kind, color in zip(orientations, mirror_colors):
        obj = _build_object(kind, color, crowded=crowded)
        if obj is None:
            return None
        objects.append(obj)

    max_h = max(int(obj["height"]) for obj in objects)
    max_w = max(int(obj["width"]) for obj in objects)
    total_area = sum((int(obj["height"]) + 1) * (int(obj["width"]) + 1) for obj in objects)
    min_side = max(GRID_MIN_SIZE, max_h, max_w, int(np.ceil(np.sqrt(total_area))) + 2)
    if min_side > GRID_MAX_SIZE:
        return None

    for _ in range(60):
        height = random.randint(min_side, GRID_MAX_SIZE)
        width = random.randint(min_side, GRID_MAX_SIZE)
        placements = _place_objects(objects, height, width)
        if placements is None:
            continue
        return _render_scene(objects, placements, (height, width), bg)

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        scene = _generate_scene()
        if scene is None:
            continue

        inp, expected = scene
        try:
            produced = main(inp)
        except AssertionError:
            continue

        if np.array_equal(produced, expected):
            return inp

    raise RuntimeError("failed to generate a valid mirror-reflection task")
