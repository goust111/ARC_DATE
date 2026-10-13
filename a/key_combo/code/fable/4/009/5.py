from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BACKGROUND = 0
MARKER = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MARKER_KINDS = ("h", "v", "r")
GENERATION_ATTEMPTS = 700
PLACE_ATTEMPTS = 700


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(row: int, col: int):
    yield row - 1, col
    yield row + 1, col
    yield row, col - 1
    yield row, col + 1


def _neighbors8(row: int, col: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield row + dr, col + dc


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(
    grid: np.ndarray,
    *,
    bg: int,
    colors: set[int] | None = None,
    connectivity: int = 8,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    neighbor_fn = _neighbors4 if int(connectivity) == 4 else _neighbors8

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == int(bg):
                continue
            if colors is not None and color not in colors:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbor_fn(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "id": len(comps),
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _has_two_mirror_axes(component: dict) -> bool:
    mask = _component_mask(component)
    return bool(np.array_equal(mask, np.flipud(mask)) and np.array_equal(mask, np.fliplr(mask)))


def _marker_kind(component: dict) -> str:
    coords = component["coords"]
    if int(coords.shape[0]) == 1:
        return "r"
    assert int(coords.shape[0]) == 2, "gray marker must be one pixel or a domino"
    r0, c0, r1, c1 = component["bbox"]
    if r0 == r1 and c1 == c0 + 1:
        return "h"
    if c0 == c1 and r1 == r0 + 1:
        return "v"
    raise AssertionError("gray domino marker must be horizontal or vertical")


def _touches4(a: np.ndarray, b: np.ndarray) -> bool:
    b_cells = {(int(r), int(c)) for r, c in b}
    for r, c in a:
        for nr, nc in _neighbors4(int(r), int(c)):
            if (nr, nc) in b_cells:
                return True
    return False


def _transform_coords(coords: np.ndarray, kind: str, row_sum: int, col_sum: int) -> np.ndarray:
    out = coords.copy()
    if kind == "h":
        out[:, 1] = int(col_sum) - out[:, 1]
    elif kind == "v":
        out[:, 0] = int(row_sum) - out[:, 0]
    elif kind == "r":
        out[:, 0] = int(row_sum) - out[:, 0]
        out[:, 1] = int(col_sum) - out[:, 1]
    else:
        raise AssertionError(f"unknown marker kind {kind!r}")
    return out


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict], dict, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)

    marker_components = _components(g, bg=bg, colors={MARKER}, connectivity=4)
    assert marker_components, "expected gray direction markers"
    for marker in marker_components:
        _marker_kind(marker)

    object_components = [
        comp
        for comp in _components(g, bg=bg, connectivity=8)
        if int(comp["color"]) != MARKER
    ]
    symmetric = [comp for comp in object_components if _has_two_mirror_axes(comp)]
    assert len(symmetric) == 1, "expected exactly one object with both mirror axes"
    axis_object = symmetric[0]
    axis_r0, axis_c0, axis_r1, axis_c1 = axis_object["bbox"]
    row_sum = int(axis_r0 + axis_r1)
    col_sum = int(axis_c0 + axis_c1)

    asymmetric = [comp for comp in object_components if comp["id"] != axis_object["id"]]
    assert asymmetric, "expected marked non-symmetric objects"
    for comp in asymmetric:
        assert not _has_two_mirror_axes(comp), "marked objects must not also define both axes"

    return bg, marker_components, asymmetric, axis_object, row_sum, col_sum


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique two-axis symmetric object as global axes, then copy each marked asymmetric object by its marker-selected transform.

    Input structure:
    - A dominant background contains one monochrome object whose own bounding-box mask
      has both horizontal and vertical mirror symmetry.
    - Several separated monochrome asymmetric objects are each side-adjacent to exactly
      one gray marker component.
    - Every gray marker is either a horizontal domino, a vertical domino, or a single
      pixel, and no marker touches two objects.

    Rule:
    - Borrow the symmetric object's horizontal and vertical axes as the whole-grid
      reference axes.
    - A horizontal gray domino beside an object creates only that object's left-right
      mirror copy across the borrowed vertical axis.
    - A vertical gray domino creates only the top-bottom mirror copy across the borrowed
      horizontal axis; a single gray pixel creates only the 180-degree rotated copy
      around the borrowed axes' intersection.
    - Keep the original colored objects and the symmetric axis object, paint each copy
      in its source color, and erase all gray markers back to the background.

    Key insight:
    - The symmetric object is only an axis ruler; the local marker shape chooses one
      transform for each asymmetric object instead of completing every reflection orbit.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg, markers, objects, _axis_object, row_sum, col_sum = _parse_scene(g)

    marker_to_object: list[tuple[dict, dict, str]] = []
    used_objects: set[int] = set()
    for marker in markers:
        touching = [obj for obj in objects if _touches4(marker["coords"], obj["coords"])]
        assert len(touching) == 1, "each marker must touch exactly one asymmetric object"
        obj = touching[0]
        assert obj["id"] not in used_objects, "each asymmetric object must have one marker"
        used_objects.add(int(obj["id"]))
        marker_to_object.append((marker, obj, _marker_kind(marker)))
    assert len(used_objects) == len(objects), "every asymmetric object must be marked"

    out = g.copy()
    out[g == MARKER] = int(bg)
    for _marker, obj, kind in marker_to_object:
        targets = _transform_coords(obj["coords"], kind, row_sum, col_sum)
        assert np.all((0 <= targets[:, 0]) & (targets[:, 0] < height))
        assert np.all((0 <= targets[:, 1]) & (targets[:, 1] < width))
        out[targets[:, 0], targets[:, 1]] = int(obj["color"])
    return out


# ---------------- generator ----------------


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    cells = {(int(r), int(c)) for r, c in coords}
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_asymmetric_mask(mask: np.ndarray) -> bool:
    return not (
        np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(np.fliplr(mask)))
    )


def _symmetric_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if random.random() >= 0.35:
                continue
            orbit = {
                (r, c),
                (height - 1 - r, c),
                (r, width - 1 - c),
                (height - 1 - r, width - 1 - c),
            }
            if all(0 < rr < height - 1 and 0 < cc < width - 1 for rr, cc in orbit):
                trial = mask.copy()
                for rr, cc in orbit:
                    trial[rr, cc] = False
                if _mask_connected(trial):
                    mask = trial
    assert _mask_connected(mask)
    return mask


def _random_asymmetric_mask() -> np.ndarray:
    for _ in range(500):
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        target_area = random.randint(4, min(8, height * width))
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < target_area:
            r, c = random.choice(tuple(cells))
            options = [
                (nr, nc)
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells
            ]
            if not options:
                continue
            cells.add(random.choice(options))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if min(mask.shape) < 2:
            continue
        if not _mask_connected(mask):
            continue
        if not _is_asymmetric_mask(mask):
            continue
        return mask
    raise RuntimeError("failed to sample asymmetric object")


def _marker_options(mask: np.ndarray, kind: str) -> list[set[Coord]]:
    obj = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    options: list[set[Coord]] = []
    for r, c in obj:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            anchor = (r + dr, c + dc)
            if anchor in obj:
                continue
            if kind == "r":
                candidates = [{anchor}]
            elif kind == "h":
                ar, ac = anchor
                candidates = [{(ar, ac), (ar, ac + 1)}, {(ar, ac - 1), (ar, ac)}]
            elif kind == "v":
                ar, ac = anchor
                candidates = [{(ar, ac), (ar + 1, ac)}, {(ar - 1, ac), (ar, ac)}]
            else:
                raise AssertionError(f"unknown marker kind {kind!r}")

            for marker in candidates:
                if marker & obj:
                    continue
                if len(marker) != (1 if kind == "r" else 2):
                    continue
                options.append(marker)
    return options


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords.astype(int)


def _translated(cells: set[Coord], top: int, left: int) -> np.ndarray:
    return np.asarray([(int(top) + r, int(left) + c) for r, c in cells], dtype=int)


def _in_bounds(coords: np.ndarray, height: int, width: int) -> bool:
    return bool(
        coords.size
        and np.all((0 <= coords[:, 0]) & (coords[:, 0] < height))
        and np.all((0 <= coords[:, 1]) & (coords[:, 1] < width))
    )


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    coords = np.argwhere(mask)
    h, w = mask.shape
    for r, c in coords:
        for rr in range(max(0, int(r) - radius), min(h, int(r) + radius + 1)):
            for cc in range(max(0, int(c) - radius), min(w, int(c) + radius + 1)):
                out[rr, cc] = True
    return out


def _clear(coords: np.ndarray, occupied: np.ndarray, *, padding: int = 0) -> bool:
    blocked = _dilate(occupied, padding) if padding else occupied
    return bool(not blocked[coords[:, 0], coords[:, 1]].any())


def _coord_set(coords: np.ndarray) -> frozenset[Coord]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _branch_targets_distinct(coords: np.ndarray, row_sum: int, col_sum: int) -> bool:
    original = _coord_set(coords)
    targets = [_coord_set(_transform_coords(coords, kind, row_sum, col_sum)) for kind in MARKER_KINDS]
    return original not in targets and len(set(targets)) == len(targets)


def _place_one(
    height: int,
    width: int,
    kind: str,
    color: int,
    row_sum: int,
    col_sum: int,
    input_occ: np.ndarray,
    output_occ: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(PLACE_ATTEMPTS):
        obj_mask = _random_asymmetric_mask()
        marker_rel = random.choice(_marker_options(obj_mask, kind))
        obj_rel = {(int(r), int(c)) for r, c in np.argwhere(obj_mask)}
        combined = obj_rel | marker_rel
        min_r = min(r for r, _c in combined)
        max_r = max(r for r, _c in combined)
        min_c = min(c for _r, c in combined)
        max_c = max(c for _r, c in combined)

        top_min = -min_r + 1
        top_max = height - 2 - max_r
        left_min = -min_c + 1
        left_max = width - 2 - max_c
        if top_min > top_max or left_min > left_max:
            continue

        top = random.randint(top_min, top_max)
        left = random.randint(left_min, left_max)
        obj_coords = _coords_from_mask(obj_mask, top, left)
        marker_coords = _translated(marker_rel, top, left)
        combined_coords = np.vstack([obj_coords, marker_coords])
        if not _in_bounds(combined_coords, height, width):
            continue
        if not _clear(combined_coords, input_occ, padding=1):
            continue

        if not _branch_targets_distinct(obj_coords, row_sum, col_sum):
            continue
        target_coords = _transform_coords(obj_coords, kind, row_sum, col_sum)
        if not _in_bounds(target_coords, height, width):
            continue

        candidate_input_occ = input_occ.copy()
        candidate_input_occ[combined_coords[:, 0], combined_coords[:, 1]] = True
        blocked_for_target = output_occ | candidate_input_occ
        if not _clear(target_coords, blocked_for_target, padding=1):
            continue

        # The two unselected transforms should remain visibly absent, ruling out
        # the shortcut of completing every reflection orbit.
        selected = _coord_set(target_coords)
        for other_kind in MARKER_KINDS:
            if other_kind == kind:
                continue
            other = _coord_set(_transform_coords(obj_coords, other_kind, row_sum, col_sum))
            if other == selected:
                return None
            if any(output_occ[r, c] for r, c in other):
                return None

        colored_obj = np.column_stack([obj_coords, np.full(len(obj_coords), int(color), dtype=int)])
        return obj_coords, marker_coords, target_coords, colored_obj
    return None


def _expected_from_input(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        bg, markers, objects, axis_object, row_sum, col_sum = _parse_scene(inp)
    except Exception:
        return False

    if len(markers) < 3 or { _marker_kind(marker) for marker in markers } != set(MARKER_KINDS):
        return False
    if np.any(out == MARKER):
        return False
    if not np.array_equal(out[axis_object["coords"][:, 0], axis_object["coords"][:, 1]], inp[axis_object["coords"][:, 0], axis_object["coords"][:, 1]]):
        return False

    for marker in markers:
        touching = [obj for obj in objects if _touches4(marker["coords"], obj["coords"])]
        if len(touching) != 1:
            return False
        obj = touching[0]
        kind = _marker_kind(marker)
        selected = _coord_set(_transform_coords(obj["coords"], kind, row_sum, col_sum))
        for other_kind in MARKER_KINDS:
            target = _coord_set(_transform_coords(obj["coords"], other_kind, row_sum, col_sum))
            if other_kind != kind and target == selected:
                return False
    return bool(np.count_nonzero(inp != out) >= len(markers) + 3)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(23, 30)
    width = random.randint(23, 30)
    sym_h_choices = [size for size in (3, 4, 5, 6, 7) if size <= height and size % 2 == height % 2]
    sym_w_choices = [size for size in (3, 4, 5, 6, 7) if size <= width and size % 2 == width % 2]
    sym_h = random.choice(sym_h_choices)
    sym_w = random.choice(sym_w_choices)
    sym_top = (height - sym_h) // 2
    sym_left = (width - sym_w) // 2
    row_sum = 2 * sym_top + sym_h - 1
    col_sum = 2 * sym_left + sym_w - 1
    assert row_sum == height - 1 and col_sum == width - 1

    object_count = random.randint(3, 5)
    kinds = list(MARKER_KINDS)
    while len(kinds) < object_count:
        kinds.append(random.choice(MARKER_KINDS))
    random.shuffle(kinds)

    colors = random.sample(list(OBJECT_COLORS), object_count + 1)
    sym_color = colors[0]
    object_colors = colors[1:]

    inp = np.full((height, width), BACKGROUND, dtype=int)
    expected = inp.copy()
    input_occ = np.zeros((height, width), dtype=bool)
    output_occ = np.zeros((height, width), dtype=bool)

    sym_mask = _symmetric_mask(sym_h, sym_w)
    sym_coords = _coords_from_mask(sym_mask, sym_top, sym_left)
    _paint(inp, sym_coords, sym_color)
    _paint(expected, sym_coords, sym_color)
    input_occ[sym_coords[:, 0], sym_coords[:, 1]] = True
    output_occ[sym_coords[:, 0], sym_coords[:, 1]] = True

    for kind, color in zip(kinds, object_colors):
        placed = _place_one(height, width, kind, color, row_sum, col_sum, input_occ, output_occ)
        if placed is None:
            return None
        obj_coords, marker_coords, target_coords, _colored_obj = placed
        _paint(inp, obj_coords, color)
        _paint(inp, marker_coords, MARKER)
        _paint(expected, obj_coords, color)
        _paint(expected, target_coords, color)

        input_occ[obj_coords[:, 0], obj_coords[:, 1]] = True
        input_occ[marker_coords[:, 0], marker_coords[:, 1]] = True
        output_occ[obj_coords[:, 0], obj_coords[:, 1]] = True
        output_occ[target_coords[:, 0], target_coords[:, 1]] = True

    try:
        solved = _expected_from_input(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid borrowed-axis marker-transform input")
